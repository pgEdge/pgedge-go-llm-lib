//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package httpclient

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// RetryEvent describes a single retry attempt that just FAILED. It
// is emitted via RetryConfig.OnRetry just before the retry layer
// sleeps and tries again.
type RetryEvent struct {
	// Attempt is the 1-indexed attempt number that just failed.
	// The next attempt (if any) is Attempt+1.
	Attempt int

	// StatusCode is the HTTP status of the failed attempt. Zero if
	// the attempt failed with a network error before a response.
	StatusCode int

	// Err is the network error from the failed attempt, or nil if
	// the failure was a retryable HTTP status.
	Err error

	// Wait is the duration the retry layer will sleep before the
	// next attempt.
	Wait time.Duration
}

// RetryConfig configures HTTP retry behaviour. This struct mirrors
// the public llm.RetryConfig — it is duplicated here (rather than
// imported) to avoid a circular dependency between the internal
// httpclient package and the public llm package.
type RetryConfig struct {
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Disabled       bool
	OnRetry        func(RetryEvent)

	// PerAttemptTimeout, when > 0, bounds each individual attempt. A
	// stalled attempt that exceeds it is abandoned and (budget
	// permitting) retried, so a slow upstream becomes retryable rather
	// than consuming the caller's whole deadline. Zero disables it.
	PerAttemptTimeout time.Duration
}

// ErrRetryBudgetExhausted is returned when MaxRetries has been
// reached and the upstream is still failing.
var ErrRetryBudgetExhausted = errors.New("retry budget exhausted")

// RetryTransport is an http.RoundTripper that retries idempotent
// requests on transient failures: network errors, 429, and
// 500/502/503/504/529. Retry-After headers are honoured. The request
// body is buffered upfront so it can be replayed.
//
// When a retryable response is returned to the caller after the retry
// budget is exhausted (i.e. the upstream kept failing), the response
// body has already been drained and closed by the retry loop. Callers
// that need the body should inspect StatusCode and not rely on Body.
type RetryTransport struct {
	Inner  http.RoundTripper
	Config RetryConfig
}

// RoundTrip implements http.RoundTripper.
func (t *RetryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.Config.Disabled || t.Config.MaxRetries <= 0 {
		return roundTripWithTimeout(t.Inner, req, t.Config.PerAttemptTimeout)
	}

	var bodyBytes []byte
	if req.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
	}

	backoff := t.Config.InitialBackoff
	if backoff <= 0 {
		backoff = 2 * time.Second
	}
	maxBackoff := t.Config.MaxBackoff
	if maxBackoff <= 0 {
		maxBackoff = 60 * time.Second
	}

	for attempt := 0; attempt <= t.Config.MaxRetries; attempt++ {
		if err := req.Context().Err(); err != nil {
			return nil, err
		}

		if bodyBytes != nil {
			req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}

		resp, err := roundTripWithTimeout(t.Inner, req, t.Config.PerAttemptTimeout)
		if err != nil {
			if attempt == t.Config.MaxRetries {
				return nil, err
			}
			if t.Config.OnRetry != nil {
				t.Config.OnRetry(RetryEvent{
					Attempt:    attempt + 1,
					StatusCode: 0,
					Err:        err,
					Wait:       backoff,
				})
			}
			if !sleep(req.Context(), backoff) {
				return nil, req.Context().Err()
			}
			backoff = capDuration(backoff*2, maxBackoff)
			continue
		}

		switch resp.StatusCode {
		case http.StatusTooManyRequests:
			_ = resp.Body.Close()
			if attempt == t.Config.MaxRetries {
				return resp, nil
			}
			retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
			wait := backoff
			if retryAfter > 0 {
				wait = retryAfter
			}
			if t.Config.OnRetry != nil {
				t.Config.OnRetry(RetryEvent{
					Attempt:    attempt + 1,
					StatusCode: resp.StatusCode,
					Wait:       wait,
				})
			}
			if !sleep(req.Context(), wait) {
				return nil, req.Context().Err()
			}
			backoff = capDuration(backoff*2, maxBackoff)
			continue

		case http.StatusInternalServerError,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
			529:
			_ = resp.Body.Close()
			if attempt == t.Config.MaxRetries {
				return resp, nil
			}
			if t.Config.OnRetry != nil {
				t.Config.OnRetry(RetryEvent{
					Attempt:    attempt + 1,
					StatusCode: resp.StatusCode,
					Wait:       backoff,
				})
			}
			if !sleep(req.Context(), backoff) {
				return nil, req.Context().Err()
			}
			backoff = capDuration(backoff*2, maxBackoff)
			continue

		default:
			return resp, nil
		}
	}

	// Unreachable: every iteration of the loop above either returns
	// or continues. This satisfies the compiler.
	return nil, ErrRetryBudgetExhausted
}

// roundTripWithTimeout runs a single attempt of req through inner. When
// timeout > 0 the attempt is bounded by a context derived from the
// request's own context: a stalled attempt fails with
// context.DeadlineExceeded WITHOUT cancelling the caller's context, so
// the retry loop is free to try again. When timeout <= 0 the attempt is
// passed straight through.
//
// On a successful round trip the per-attempt timer is stopped and its
// cancellation is deferred to the response Body's Close, so the timeout
// never interrupts reads of a (possibly long-lived, streaming) body —
// matching RequestTimeout's documented streaming semantics.
func roundTripWithTimeout(inner http.RoundTripper, req *http.Request, timeout time.Duration) (*http.Response, error) {
	if timeout <= 0 {
		return inner.RoundTrip(req)
	}

	ctx, cancel := context.WithCancel(req.Context())
	timer := time.AfterFunc(timeout, cancel)

	resp, err := inner.RoundTrip(req.WithContext(ctx))

	// timer.Stop reports false iff the timer already fired — i.e. the
	// per-attempt budget elapsed and cancel() ran. We only ever call
	// Stop once, so false unambiguously means "this attempt timed out".
	timedOut := !timer.Stop()

	if err != nil {
		cancel()
		// A per-attempt timeout surfaces as the caller's context being
		// fine but our derived context being cancelled. Re-shape it as
		// DeadlineExceeded so it is treated as a retryable timeout
		// rather than a caller cancellation.
		if timedOut && req.Context().Err() == nil {
			return nil, fmt.Errorf("per-attempt timeout after %s: %w", timeout, context.DeadlineExceeded)
		}
		return nil, err
	}

	if timedOut {
		// Boundary race: the timer fired just as the headers arrived, so
		// the returned body is tied to a cancelled context. Treat it as a
		// timeout rather than handing back a doomed body.
		cancel()
		if req.Context().Err() == nil {
			return nil, fmt.Errorf("per-attempt timeout after %s: %w", timeout, context.DeadlineExceeded)
		}
		return nil, req.Context().Err()
	}

	resp.Body = &cancelBody{ReadCloser: resp.Body, cancel: cancel}
	return resp, nil
}

// cancelBody wraps a response body so that closing it releases the
// per-attempt context. context.CancelFunc is safe to call more than
// once, so no additional guarding is needed.
type cancelBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

// Close closes the underlying body and then releases the per-attempt
// context.
func (b *cancelBody) Close() error {
	err := b.ReadCloser.Close()
	b.cancel()
	return err
}

// parseRetryAfter parses the Retry-After header value. Accepts both
// delta-seconds and HTTP-date formats. Returns 0 if the value is
// empty or unparseable.
func parseRetryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		return time.Duration(seconds) * time.Second
	}
	if t, err := http.ParseTime(value); err == nil {
		return time.Until(t)
	}
	return 0
}

// sleep waits the given duration, returning false if the context
// cancels before the timer fires.
func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func capDuration(d, ceiling time.Duration) time.Duration {
	if d > ceiling {
		return ceiling
	}
	return d
}
