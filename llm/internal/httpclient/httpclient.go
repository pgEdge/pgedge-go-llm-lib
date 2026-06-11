//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package httpclient is an internal helper package shared by the
// provider implementations. It assembles an *http.Client wired with
// retry middleware, custom headers, and request timeouts; provides
// JSON request/response and SSE-streaming helpers; and validates
// caller-supplied base URLs.
//
// This package is internal — its surface is not part of the public
// library API and may change without notice.
package httpclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const defaultTimeout = 120 * time.Second

// HeaderTransport wraps an http.RoundTripper and injects custom
// headers into every request without overriding existing ones.
type HeaderTransport struct {
	inner   http.RoundTripper
	headers map[string]string
}

// NewHeaderTransport returns a HeaderTransport that wraps the given
// inner RoundTripper and injects the given headers. If inner is nil,
// http.DefaultTransport is used. If headers is nil or empty, requests
// pass through to inner unmodified.
func NewHeaderTransport(inner http.RoundTripper, headers map[string]string) *HeaderTransport {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &HeaderTransport{
		inner:   inner,
		headers: headers,
	}
}

// RoundTrip implements http.RoundTripper.
func (t *HeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(t.headers) == 0 {
		return t.inner.RoundTrip(req)
	}
	clone := req.Clone(req.Context())
	for k, v := range t.headers {
		if clone.Header.Get(k) == "" {
			clone.Header.Set(k, v)
		}
	}
	return t.inner.RoundTrip(clone)
}

// New creates an *http.Client. If base is non-nil, its Transport
// becomes the inner of the new transport stack. Otherwise
// http.DefaultTransport is used as the inner. The retry middleware
// (when not disabled) and header middleware (when headers is
// non-empty) are layered on top.
//
// timeout, when > 0, overrides the client's Timeout field. When
// timeout is 0 and the resulting client has no timeout set, the
// library's 120-second default is applied. If base already carries a
// non-zero Timeout and timeout is 0, the caller's timeout is
// preserved.
//
// New does not mutate base — it returns a copy.
func New(base *http.Client, headers map[string]string, retry RetryConfig, timeout time.Duration) *http.Client {
	// `inner` is typed as the interface so the subsequent reassignment
	// from base.Transport (also an http.RoundTripper) compiles cleanly.
	var inner http.RoundTripper //nolint:staticcheck,revive // ST1023: explicit interface type is intentional
	inner = http.DefaultTransport
	if base != nil && base.Transport != nil {
		inner = base.Transport
	}

	// The RetryTransport also enforces per-attempt timeouts, so it is
	// installed whenever retries are enabled OR a per-attempt timeout is
	// configured (even with retries disabled).
	if (!retry.Disabled && retry.MaxRetries > 0) || retry.PerAttemptTimeout > 0 {
		inner = &RetryTransport{Inner: inner, Config: retry}
	}
	if len(headers) > 0 {
		inner = NewHeaderTransport(inner, headers)
	}

	out := base
	if out == nil {
		out = &http.Client{}
	} else {
		// Don't mutate caller's client.
		clone := *base
		out = &clone
	}
	out.Transport = inner
	if timeout > 0 {
		out.Timeout = timeout
	} else if out.Timeout == 0 {
		out.Timeout = defaultTimeout
	}
	return out
}

// DoJSON sends an HTTP request with a JSON body and decodes the JSON
// response. It returns the status code, raw response body, and any
// error. For non-2xx responses, the response is not decoded into dest
// but the raw body and status code are still returned so the caller
// can extract provider-specific error details.
func DoJSON(ctx context.Context, client *http.Client, method, url string, headers map[string]string, reqBody any, dest any) (int, []byte, error) {
	var bodyReader io.Reader
	if reqBody != nil {
		data, err := json.Marshal(reqBody)
		if err != nil {
			return 0, nil, fmt.Errorf("marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return 0, nil, fmt.Errorf("create request: %w", err)
	}

	if bodyReader != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode >= 200 && resp.StatusCode < 300 && dest != nil {
		if err := json.Unmarshal(body, dest); err != nil {
			return resp.StatusCode, body, fmt.Errorf("decode response: %w", err)
		}
	}

	return resp.StatusCode, body, nil
}

// DoSSERequest sends an HTTP request and returns the raw response
// for SSE streaming. The caller is responsible for closing the body.
func DoSSERequest(ctx context.Context, client *http.Client, method, url string, headers map[string]string, reqBody any) (*http.Response, error) {
	var bodyReader io.Reader
	if reqBody != nil {
		data, err := json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	return client.Do(req)
}

// SSEScanner reads server-sent events from an io.Reader.
type SSEScanner struct {
	scanner *bufio.Scanner
	data    string
}

// NewSSEScanner creates a scanner for SSE event streams.
func NewSSEScanner(r io.Reader) *SSEScanner {
	return &SSEScanner{scanner: bufio.NewScanner(r)}
}

// Scan advances to the next SSE data event. Returns false when done.
func (s *SSEScanner) Scan() bool {
	for s.scanner.Scan() {
		line := s.scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			s.data = strings.TrimPrefix(line, "data: ")
			return true
		}
	}
	return false
}

// Data returns the data from the most recent event.
func (s *SSEScanner) Data() string {
	return s.data
}

// Err returns the first non-EOF error encountered by the underlying
// bufio.Scanner. Callers should check this after Scan returns false to
// distinguish a clean end-of-stream from a read/parse failure.
func (s *SSEScanner) Err() error {
	return s.scanner.Err()
}

// ValidateBaseURL trims whitespace and a trailing slash, then checks
// that the URL has an http or https scheme and a non-empty host.
// providerName is included in error messages so callers can identify
// which provider's BaseURL was malformed.
func ValidateBaseURL(baseURL, providerName string) (string, error) {
	baseURL = strings.TrimSpace(baseURL)
	baseURL = strings.TrimSuffix(baseURL, "/")

	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid %s base URL: %w", providerName, err)
	}
	if parsed.Scheme != "https" && parsed.Scheme != "http" {
		return "", fmt.Errorf("%s base URL must use http or https scheme, got: %q", providerName, parsed.Scheme)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("%s base URL must include a host", providerName)
	}
	return baseURL, nil
}

// IsLocalBaseURL reports whether the URL points at a local/loopback host
// (localhost, 127.0.0.0/8, ::1, or a *.local name). Used to auto-select
// compact tool descriptions for local models.
func IsLocalBaseURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	switch {
	case host == "localhost", strings.HasSuffix(host, ".local"):
		return true
	}
	// net.ParseIP covers loopback addresses, including IPv4 127.0.0.0/8
	// and the IPv6 ::1 form.
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}
