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

// DefaultMaxResponseBytes caps a response body when the caller sets no
// limit. It is far above any legitimate provider response (an OpenAI
// embedding batch of 2048 inputs at 3072 dimensions is about 125 MB of
// JSON), but stops a misbehaving or hostile endpoint from exhausting
// memory before the request timeout fires.
const DefaultMaxResponseBytes int64 = 256 << 20

// limitTransport wraps an http.RoundTripper and caps how many bytes of
// each response body can be read.
type limitTransport struct {
	inner http.RoundTripper
	max   int64
}

// RoundTrip implements http.RoundTripper.
func (t *limitTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		return resp, err
	}
	resp.Body = &limitedBody{rc: resp.Body, remaining: t.max, max: t.max}
	return resp, nil
}

// CloseIdleConnections passes through to the inner transport, so
// http.Client.CloseIdleConnections still reaches the connection pool.
func (t *limitTransport) CloseIdleConnections() {
	if ci, ok := t.inner.(interface{ CloseIdleConnections() }); ok {
		ci.CloseIdleConnections()
	}
}

// limitedBody reads at most max bytes from rc and fails with an error,
// rather than a silent EOF, once the body proves to be longer, so a
// truncated body is never mistaken for a complete one.
type limitedBody struct {
	rc        io.ReadCloser
	remaining int64
	max       int64
}

func (b *limitedBody) Read(p []byte) (int, error) {
	if b.remaining <= 0 {
		// Probe for one more byte to tell a body of exactly max bytes
		// from a longer one.
		var probe [1]byte
		n, err := b.rc.Read(probe[:])
		if n > 0 {
			return 0, fmt.Errorf("response body exceeds the %d-byte limit (see Options.MaxResponseBytes)", b.max)
		}
		return 0, err
	}
	if int64(len(p)) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.rc.Read(p)
	b.remaining -= int64(n)
	return n, err
}

func (b *limitedBody) Close() error { return b.rc.Close() }

// refuseRedirect is the CheckRedirect installed when the caller's client
// has none. Provider APIs do not redirect, and following one would replay
// the request body (the prompt) and any API key sent in a custom header,
// which Go does not strip on a cross-host redirect, to another server.
func refuseRedirect(req *http.Request, _ []*http.Request) error {
	status := 0
	if req.Response != nil {
		status = req.Response.StatusCode
	}
	return fmt.Errorf("refusing to follow HTTP %d redirect to host %q", status, req.URL.Host)
}

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
// maxResponse caps the bytes read from any response body; zero or less
// applies DefaultMaxResponseBytes. Reading past the cap fails with an
// error. Redirects are refused unless base sets its own CheckRedirect.
//
// New does not mutate base — it returns a copy.
func New(base *http.Client, headers map[string]string, retry RetryConfig, timeout time.Duration, maxResponse int64) *http.Client {
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
	if maxResponse <= 0 {
		maxResponse = DefaultMaxResponseBytes
	}
	inner = &limitTransport{inner: inner, max: maxResponse}

	out := base
	if out == nil {
		out = &http.Client{}
	} else {
		// Don't mutate caller's client.
		clone := *base
		out = &clone
	}
	out.Transport = inner
	if out.CheckRedirect == nil {
		out.CheckRedirect = refuseRedirect
	}
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
// but the first 64 KiB of the body and the status code are still
// returned so the caller can extract provider-specific error details.
//
// A failure to read a 2xx body (including one over the client's size
// limit) returns a zero status, like a transport failure, because the
// response is unusable and callers treat a non-zero 2xx status as
// success.
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

	req, release := withHardCancel(req)

	resp, err := client.Do(req)
	// The watchdog only guards the request-body write, which is complete
	// once Do returns (whether it succeeded or failed). Release it now,
	// before reading the response body: the transport can return the
	// connection to its idle pool as soon as that body reaches EOF, and
	// a watchdog still holding a reference could then force-close a
	// connection already reassigned to another request. Response reads
	// are covered by ordinary context cancellation.
	release()
	if err != nil {
		return 0, nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, ReadErrorBody(resp.Body), nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("read response: %w", err)
	}

	if dest != nil {
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

	req, release := withHardCancel(req)
	resp, err := client.Do(req)
	// Release the watchdog as soon as Do returns: by then the response
	// headers have arrived, so the request-body write it guards is
	// complete. The caller reads the stream body afterwards, but those
	// reads are covered by ordinary context cancellation; keeping the
	// watchdog armed through the stream would risk force-closing a
	// connection the transport has since returned to its pool.
	release()
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// maxErrorBody bounds how much of a non-2xx response body ReadErrorBody
// keeps; provider error payloads are small JSON documents.
const maxErrorBody = 64 << 10

// ReadErrorBody reads a non-2xx response body to EOF, up to 64 KiB, so a
// JSON error split across several reads is still whole when parsed. A
// read error keeps whatever arrived before it.
func ReadErrorBody(r io.Reader) []byte {
	body, _ := io.ReadAll(io.LimitReader(r, maxErrorBody))
	return body
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
