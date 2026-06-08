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
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:     3,
		InitialBackoff: 5 * time.Millisecond,
		MaxBackoff:     20 * time.Millisecond,
	}
}

func TestRetryTransportSucceedsOnFirstAttempt(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: testRetryConfig()}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp.Body.Close()
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Errorf("attempts = %d, want 1", got)
	}
}

func TestRetryTransportRetriesOn429(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: testRetryConfig()}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("attempts = %d, want 3", got)
	}
}

func TestRetryTransportRetriesOn5xx(t *testing.T) {
	codes := []int{500, 502, 503, 504, 529}
	for _, code := range codes {
		t.Run(http.StatusText(code), func(t *testing.T) {
			var attempts int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := atomic.AddInt32(&attempts, 1)
				if n < 2 {
					w.WriteHeader(code)
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: testRetryConfig()}}
			resp, err := client.Get(srv.URL)
			if err != nil {
				t.Fatalf("get: %v", err)
			}
			resp.Body.Close()
			if got := atomic.LoadInt32(&attempts); got != 2 {
				t.Errorf("status %d: attempts = %d, want 2", code, got)
			}
		})
	}
}

func TestRetryTransportDoesNotRetryOn4xxOtherThan429(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: testRetryConfig()}}
	resp, _ := client.Get(srv.URL)
	if resp != nil {
		resp.Body.Close()
	}
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Errorf("attempts = %d, want 1", got)
	}
}

func TestRetryTransportRespectsRetryAfterSeconds(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := testRetryConfig()
	cfg.InitialBackoff = 10 * time.Millisecond
	cfg.MaxBackoff = 5 * time.Second

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: cfg}}
	start := time.Now()
	resp, err := client.Get(srv.URL)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp.Body.Close()
	if elapsed < 900*time.Millisecond {
		t.Errorf("elapsed = %v, want >= ~1s (Retry-After honoured)", elapsed)
	}
}

func TestRetryTransportRespectsContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	cfg := testRetryConfig()
	cfg.InitialBackoff = 50 * time.Millisecond
	cfg.MaxRetries = 100

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: cfg}}
	_, err := client.Do(req)
	if err == nil {
		t.Fatal("want error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want DeadlineExceeded", err)
	}
}

func TestRetryTransportReplaysRequestBody(t *testing.T) {
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(body))
		if len(bodies) < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: testRetryConfig()}}
	resp, err := client.Post(srv.URL, "application/json", strings.NewReader(`{"hi":1}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if len(bodies) != 2 {
		t.Fatalf("got %d attempts, want 2", len(bodies))
	}
	if bodies[0] != `{"hi":1}` || bodies[1] != `{"hi":1}` {
		t.Errorf("body not replayed correctly: %v", bodies)
	}
}

func TestParseRetryAfterSeconds(t *testing.T) {
	if got := parseRetryAfter("3"); got != 3*time.Second {
		t.Errorf("parseRetryAfter(\"3\") = %v, want 3s", got)
	}
}

func TestParseRetryAfterEmpty(t *testing.T) {
	if got := parseRetryAfter(""); got != 0 {
		t.Errorf("parseRetryAfter(\"\") = %v, want 0", got)
	}
}

func TestParseRetryAfterHTTPDate(t *testing.T) {
	future := time.Now().Add(2 * time.Second).UTC().Format(http.TimeFormat)
	got := parseRetryAfter(future)
	if got < 1*time.Second || got > 3*time.Second {
		t.Errorf("parseRetryAfter(<HTTP-date 2s in future>) = %v, want ~2s", got)
	}
}

func TestParseRetryAfterGarbage(t *testing.T) {
	if got := parseRetryAfter("not a duration"); got != 0 {
		t.Errorf("parseRetryAfter(garbage) = %v, want 0", got)
	}
}

func TestRetryTransportInvokesOnRetry(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	var events []RetryEvent
	cfg := testRetryConfig()
	cfg.OnRetry = func(e RetryEvent) {
		events = append(events, e)
	}

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: cfg}}
	resp, _ := client.Get(srv.URL)
	if resp != nil {
		resp.Body.Close()
	}

	if len(events) != 2 {
		t.Fatalf("got %d retry events, want 2", len(events))
	}
	if events[0].Attempt != 1 || events[0].StatusCode != 503 {
		t.Errorf("event 0: %+v", events[0])
	}
	if events[1].Attempt != 2 || events[1].StatusCode != 503 {
		t.Errorf("event 1: %+v", events[1])
	}
	// Both events should report a non-zero Wait (the configured InitialBackoff
	// or doubled).
	if events[0].Wait == 0 || events[1].Wait == 0 {
		t.Errorf("Wait should be non-zero: %+v %+v", events[0], events[1])
	}
}

func TestCapDuration(t *testing.T) {
	cases := []struct {
		d, ceiling, want time.Duration
	}{
		{500 * time.Millisecond, time.Second, 500 * time.Millisecond}, // below ceiling
		{2 * time.Second, time.Second, time.Second},                   // above ceiling
		{time.Second, time.Second, time.Second},                       // equal
	}
	for _, tc := range cases {
		got := capDuration(tc.d, tc.ceiling)
		if got != tc.want {
			t.Errorf("capDuration(%v, %v) = %v, want %v", tc.d, tc.ceiling, got, tc.want)
		}
	}
}

func TestRoundTripContextCancelledMidBackoff(t *testing.T) {
	// Server always 503s; the caller cancels its context during the
	// first backoff. RoundTrip must abort with the context error
	// rather than burning through the retry budget.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner: http.DefaultTransport,
		Config: RetryConfig{
			MaxRetries:     3,
			InitialBackoff: 200 * time.Millisecond,
			MaxBackoff:     time.Second,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected context cancellation error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestRoundTripDisabledShortCircuits(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner:  http.DefaultTransport,
		Config: RetryConfig{Disabled: true, MaxRetries: 5},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
	if hits != 1 {
		t.Errorf("Disabled config retried %d times; want exactly 1 hit", hits)
	}
}

func TestRoundTripRetriesNetworkErrors(t *testing.T) {
	// First call fails with a transport error (server closed mid-handshake);
	// second succeeds.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	var calls int32
	rt := &RetryTransport{
		Inner: roundTripFn(func(req *http.Request) (*http.Response, error) {
			n := atomic.AddInt32(&calls, 1)
			if n == 1 {
				return nil, fmt.Errorf("simulated network error")
			}
			return http.DefaultTransport.RoundTrip(req)
		}),
		Config: RetryConfig{MaxRetries: 2, InitialBackoff: time.Millisecond, MaxBackoff: 5 * time.Millisecond},
	}

	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed after retry: %v", err)
	}
	resp.Body.Close()
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("expected 2 attempts, got %d", calls)
	}
}

func TestRoundTripRetriesNetworkErrorFiresHook(t *testing.T) {
	var events []RetryEvent
	rt := &RetryTransport{
		Inner: roundTripFn(func(_ *http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("net error")
		}),
		Config: RetryConfig{
			MaxRetries:     1,
			InitialBackoff: time.Millisecond,
			MaxBackoff:     5 * time.Millisecond,
			OnRetry:        func(e RetryEvent) { events = append(events, e) },
		},
	}
	req, _ := http.NewRequest(http.MethodGet, "http://example.invalid", nil)
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected network error after retry budget exhausted")
	}
	if len(events) != 1 {
		t.Fatalf("OnRetry events = %d, want 1", len(events))
	}
	if events[0].StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 for network-error event", events[0].StatusCode)
	}
	if events[0].Err == nil {
		t.Errorf("Err should be non-nil on network-error event")
	}
}

func TestRoundTripRetriesRateLimitHonoursRetryAfter(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			w.Header().Set("Retry-After", "0") // 0 seconds → fall through to backoff
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	var events []RetryEvent
	rt := &RetryTransport{
		Inner: http.DefaultTransport,
		Config: RetryConfig{
			MaxRetries:     2,
			InitialBackoff: time.Millisecond,
			MaxBackoff:     5 * time.Millisecond,
			OnRetry:        func(e RetryEvent) { events = append(events, e) },
		},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed after retry: %v", err)
	}
	resp.Body.Close()
	if calls != 2 {
		t.Errorf("expected 2 calls (1 ratelimited + 1 success), got %d", calls)
	}
	if len(events) != 1 {
		t.Fatalf("OnRetry events = %d, want 1", len(events))
	}
	if events[0].StatusCode != http.StatusTooManyRequests {
		t.Errorf("StatusCode = %d, want 429", events[0].StatusCode)
	}
}

func TestRoundTripRateLimitBudgetExhausted(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner: http.DefaultTransport,
		Config: RetryConfig{
			MaxRetries:     1,
			InitialBackoff: time.Millisecond,
			MaxBackoff:     5 * time.Millisecond,
		},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip should return the final 429 response, got error: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429 returned after budget exhausted", resp.StatusCode)
	}
	if calls != 2 {
		t.Errorf("expected 2 calls, got %d", calls)
	}
}

func TestRoundTrip5xxBudgetExhaustedReturnsFinalResponse(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner: http.DefaultTransport,
		Config: RetryConfig{
			MaxRetries:     1,
			InitialBackoff: time.Millisecond,
			MaxBackoff:     5 * time.Millisecond,
		},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("expected final 5xx response, got error: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 returned after budget exhausted", resp.StatusCode)
	}
	if calls != 2 {
		t.Errorf("expected 2 calls, got %d", calls)
	}
}

func TestRoundTripBodyReadErrorPropagates(t *testing.T) {
	rt := &RetryTransport{
		Inner:  http.DefaultTransport,
		Config: RetryConfig{MaxRetries: 1, InitialBackoff: time.Millisecond, MaxBackoff: 5 * time.Millisecond},
	}
	req, _ := http.NewRequest(http.MethodPost, "http://example.invalid", &errorReader{})
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected body-read error, got nil")
	}
	if !strings.Contains(err.Error(), "body explode") {
		t.Errorf("err = %v, want one mentioning the reader failure", err)
	}
}

func TestRoundTripContextAlreadyCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	rt := &RetryTransport{
		Inner: roundTripFn(func(_ *http.Request) (*http.Response, error) {
			t.Fatal("inner round-trip should not run when ctx is already cancelled")
			return nil, nil
		}),
		Config: RetryConfig{MaxRetries: 2, InitialBackoff: time.Millisecond, MaxBackoff: 5 * time.Millisecond},
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	_, err := rt.RoundTrip(req)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestRoundTripBackoffDefaultsApplied(t *testing.T) {
	// Drive RoundTrip with InitialBackoff/MaxBackoff zero so it falls
	// through the internal default branches. We use a context that
	// cancels almost immediately so the test still runs fast.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()

	rt := &RetryTransport{
		Inner:  http.DefaultTransport,
		Config: RetryConfig{MaxRetries: 5}, // no backoff overrides → defaults applied
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected context deadline error")
	}
}

// roundTripFn adapts a function to http.RoundTripper.
type roundTripFn func(*http.Request) (*http.Response, error)

func (f roundTripFn) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// errorReader is an io.Reader that always returns an error.
type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, fmt.Errorf("body explode") }
func (errorReader) Close() error             { return nil }

func TestRetryTransportRetriesPerAttemptTimeout(t *testing.T) {
	// The first attempt stalls past the per-attempt timeout; the retry
	// layer must abandon it and retry, with the second attempt
	// succeeding. This is the Gemini batchEmbedContents failure mode: a
	// slow request that previously consumed the whole budget and could
	// not be retried.
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n == 1 {
			select {
			case <-r.Context().Done():
			case <-time.After(2 * time.Second):
			}
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := testRetryConfig()
	cfg.PerAttemptTimeout = 50 * time.Millisecond

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: cfg}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if got := atomic.LoadInt32(&attempts); got < 2 {
		t.Errorf("attempts = %d, want >= 2 (per-attempt timeout should be retried)", got)
	}
}

func TestRetryTransportPerAttemptTimeoutExhaustsBudget(t *testing.T) {
	// Every attempt stalls past the per-attempt timeout. Once the retry
	// budget is exhausted the final error must surface the timeout.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	cfg := RetryConfig{
		MaxRetries:        2,
		InitialBackoff:    time.Millisecond,
		MaxBackoff:        5 * time.Millisecond,
		PerAttemptTimeout: 30 * time.Millisecond,
	}

	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	rt := &RetryTransport{Inner: http.DefaultTransport, Config: cfg}
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("want timeout error after budget exhausted, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want DeadlineExceeded", err)
	}
}

func TestRetryTransportPerAttemptTimeoutDoesNotInterruptBody(t *testing.T) {
	// Headers arrive quickly (within the per-attempt timeout) but the
	// response body trickles in with a gap LONGER than the per-attempt
	// timeout. A correct implementation detaches the per-attempt timer
	// once the response is returned, so the body read is not killed —
	// this is the streaming/SSE case.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter is not a Flusher")
			return
		}
		_, _ = io.WriteString(w, "data: first\n")
		fl.Flush()
		select {
		case <-r.Context().Done():
			return
		case <-time.After(200 * time.Millisecond):
		}
		_, _ = io.WriteString(w, "data: second\n")
		fl.Flush()
	}))
	defer srv.Close()

	cfg := testRetryConfig()
	cfg.PerAttemptTimeout = 60 * time.Millisecond

	client := &http.Client{Transport: &RetryTransport{Inner: http.DefaultTransport, Config: cfg}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v (per-attempt timeout should not interrupt body)", err)
	}
	if !strings.Contains(string(body), "second") {
		t.Errorf("body truncated by per-attempt timeout: %q", body)
	}
}

func TestRoundTripPerAttemptTimeoutWithRetriesDisabled(t *testing.T) {
	// With retries disabled but a per-attempt timeout set, a single
	// attempt must still be bounded by the timeout.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner:  http.DefaultTransport,
		Config: RetryConfig{Disabled: true, PerAttemptTimeout: 30 * time.Millisecond},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("want timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want DeadlineExceeded", err)
	}
}

func TestRoundTripZeroMaxRetriesShortCircuits(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	rt := &RetryTransport{
		Inner:  http.DefaultTransport,
		Config: RetryConfig{MaxRetries: 0},
	}
	req, _ := http.NewRequest(http.MethodGet, srv.URL, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp.Body.Close()
	if hits != 1 {
		t.Errorf("MaxRetries=0 retried %d times; want exactly 1 hit", hits)
	}
}
