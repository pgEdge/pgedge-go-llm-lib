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
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeConn is a minimal net.Conn test double that only tracks whether
// Close was called. The watchdog in withHardCancel only ever calls
// Close on the captured connection, so nothing else needs to work.
// closedCh closes when Close is called, so other test doubles (e.g.
// blockingRoundTripper in retry_test.go) can block until it happens.
type fakeConn struct {
	net.Conn
	closed   int32
	closedCh chan struct{}
}

func newFakeConn() *fakeConn {
	return &fakeConn{closedCh: make(chan struct{})}
}

func (f *fakeConn) Close() error {
	if atomic.CompareAndSwapInt32(&f.closed, 0, 1) {
		close(f.closedCh)
	}
	return nil
}

func (f *fakeConn) isClosed() bool {
	return atomic.LoadInt32(&f.closed) == 1
}

// gotConnTrace extracts the httptrace.ClientTrace installed by
// withHardCancel on req's context, so a test can drive GotConn
// directly without needing a real network round trip.
func gotConnTrace(t *testing.T, req *http.Request) *httptrace.ClientTrace {
	t.Helper()
	trace := httptrace.ContextClientTrace(req.Context())
	if trace == nil || trace.GotConn == nil {
		t.Fatal("expected withHardCancel to install a ClientTrace with GotConn")
	}
	return trace
}

func waitUntil(t *testing.T, within time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("condition not met within %v", within)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestWithHardCancel_ClosesConnOnContextDone is the core regression
// test for issue #22, exercised deterministically rather than by
// trying to recreate the exact OS-level TCP buffer sizes that make a
// real write block (which vary by platform and are inherently flaky
// to depend on in a test). It drives the watchdog's own logic
// directly: capture a connection via the same httptrace hook the real
// code path uses, then confirm the connection is force-closed once the
// request's context is done — this is exactly the mechanism that
// releases a request whose body write is genuinely stuck at the OS
// level, since context cancellation alone does not reliably interrupt
// that.
func TestWithHardCancel_ClosesConnOnContextDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}

	wrapped, release := withHardCancel(req)
	defer release()

	trace := gotConnTrace(t, wrapped)
	fc := newFakeConn()
	trace.GotConn(httptrace.GotConnInfo{Conn: fc})

	if fc.isClosed() {
		t.Fatal("connection should not be closed before the context is done")
	}

	cancel()

	waitUntil(t, time.Second, fc.isClosed)
}

// TestWithHardCancel_DoesNotCloseOnCleanRelease verifies the common
// case is left alone: a round trip that completes normally must not
// have its connection touched by the watchdog, since that would break
// keep-alive connection reuse for every ordinary request.
func TestWithHardCancel_DoesNotCloseOnCleanRelease(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}

	wrapped, release := withHardCancel(req)
	trace := gotConnTrace(t, wrapped)
	fc := newFakeConn()
	trace.GotConn(httptrace.GotConnInfo{Conn: fc})

	release() // the round trip "finished" before ctx was ever done

	time.Sleep(50 * time.Millisecond) // give a buggy watchdog a chance to fire anyway
	if fc.isClosed() {
		t.Error("connection was closed even though release() ran before the context was ever done")
	}
}

// TestWithHardCancel_ReleaseIsIdempotent verifies release() tolerates
// being called more than once. It is invoked from Close() on two
// response-body wrappers (releaseOnCloseBody, cancelBody), and Close()
// on an io.ReadCloser is conventionally safe to call repeatedly (e.g.
// an explicit early Close() plus a deferred one) — release must honour
// that same convention rather than panicking on a second call.
func TestWithHardCancel_ReleaseIsIdempotent(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}

	_, release := withHardCancel(req)

	release()
	release() // must not panic
}

// TestWithHardCancel_GotConnAfterContextAlreadyDone covers the race
// where the transport delivers a connection only after ctx is already
// cancelled: Transport.getConn races connection delivery against
// ctx.Done() with no ordering guarantee, so GotConn firing after
// cancellation is a real, if narrow, possibility — not just this
// test's artificial ordering. If GotConn didn't check for that itself,
// the watchdog goroutine could already have observed ctx.Done() with
// no connection captured yet and exited, leaving this connection
// permanently unwatched.
func TestWithHardCancel_GotConnAfterContextAlreadyDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}

	wrapped, release := withHardCancel(req)
	defer release()
	trace := gotConnTrace(t, wrapped)

	cancel()
	waitUntil(t, time.Second, func() bool { return ctx.Err() != nil })

	// Give the watchdog goroutine a moment to observe ctx.Done() with
	// no connection yet captured, and exit — simulating it having lost
	// the race, before GotConn ever fires.
	time.Sleep(50 * time.Millisecond)

	fc := newFakeConn()
	trace.GotConn(httptrace.GotConnInfo{Conn: fc})

	waitUntil(t, time.Second, fc.isClosed)
}

// TestWithHardCancel_NilConnIsSafe verifies the watchdog doesn't panic
// if the context expires before any connection was ever captured (e.g.
// the request failed to even acquire a connection).
func TestWithHardCancel_NilConnIsSafe(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.invalid", nil)
	if err != nil {
		t.Fatal(err)
	}

	_, release := withHardCancel(req)
	defer release()

	cancel() // no GotConn was ever fired
	time.Sleep(50 * time.Millisecond)
	// No panic is the assertion.
}

// TestDoJSON_SmallRequestConnectionReusable is a regression guard: the
// hard-cancel wiring must not interfere with the normal case. A
// request that completes well within its context deadline must behave
// exactly as before — same success path, same ability to serve
// multiple requests through the same client.
func TestDoJSON_SmallRequestConnectionReusable(t *testing.T) {
	var handled int
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		handled++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	runtime.GC()
	before := runtime.NumGoroutine()

	for i := 0; i < 3; i++ {
		var resp map[string]any
		_, _, err := DoJSON(ctx, client, http.MethodPost, srv.URL, nil, map[string]string{"q": "hi"}, &resp)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
	}

	mu.Lock()
	if handled != 3 {
		t.Errorf("handled = %d, want 3", handled)
	}
	mu.Unlock()

	// Each request's watchdog goroutine (see withHardCancel) must exit
	// once DoJSON returns on the normal, successful path — it must not
	// accumulate one per call. CloseIdleConnections first, since a
	// kept-alive connection's own readLoop/writeLoop goroutines are
	// expected to still be running otherwise and are unrelated to what
	// this checks.
	client.CloseIdleConnections()
	waitUntil(t, time.Second, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= before
	})
}

// TestDoSSERequest_WrapsBodyForDeferredRelease verifies DoSSERequest's
// specific wiring: unlike DoJSON (which reads the whole response
// before returning, so it can release the watchdog via a plain
// defer), DoSSERequest hands the caller a live stream that is read
// long after this function returns. The watchdog must stay armed
// until the caller closes the body, not until DoSSERequest itself
// returns — otherwise a slow-to-start stream's connection would not
// be protected by hard-cancel for the rest of its lifetime.
func TestDoSSERequest_WrapsBodyForDeferredRelease(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter is not a Flusher")
			return
		}
		io.WriteString(w, "data: hello\n\n")
		fl.Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	resp, err := DoSSERequest(ctx, client, http.MethodGet, srv.URL, nil, nil)
	if err != nil {
		t.Fatalf("DoSSERequest failed: %v", err)
	}

	if _, ok := resp.Body.(*releaseOnCloseBody); !ok {
		t.Fatalf("expected resp.Body to be wrapped in *releaseOnCloseBody so the watchdog outlives this function, got %T", resp.Body)
	}

	closed := make(chan struct{})
	go func() {
		resp.Body.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("resp.Body.Close() did not return within 1s")
	}
}

// TestDoSSERequest_ErrorDoesNotLeakWatchdog verifies that when the
// underlying client.Do fails outright (no response ever comes back),
// DoSSERequest still releases the watchdog goroutine rather than
// leaving it running forever waiting for a context that may never be
// canceled.
func TestDoSSERequest_ErrorDoesNotLeakWatchdog(t *testing.T) {
	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	before := runtime.NumGoroutine()

	// Port 1 is reserved and nothing listens there; Do fails immediately
	// with a connection error, never reaching GotConn.
	_, err := DoSSERequest(ctx, client, http.MethodGet, "http://127.0.0.1:1", nil, nil)
	if err == nil {
		t.Fatal("expected a connection error, got nil")
	}

	waitUntil(t, time.Second, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= before
	})
}

// realTCPConnPair dials a listener and returns the client's own
// *net.TCPConn plus its accepted server-side counterpart (closed via
// t.Cleanup), for tests that need a genuine *net.TCPConn to unwrap to.
func realTCPConnPair(t *testing.T) *net.TCPConn {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })

	accepted := make(chan net.Conn, 1)
	go func() {
		c, acceptErr := ln.Accept()
		if acceptErr == nil {
			accepted <- c
		}
	}()

	clientConn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { clientConn.Close() })

	select {
	case serverConn := <-accepted:
		t.Cleanup(func() { serverConn.Close() })
	case <-time.After(time.Second):
		t.Fatal("listener never accepted the connection")
	}

	tc, ok := clientConn.(*net.TCPConn)
	if !ok {
		t.Fatalf("net.Dial(\"tcp\", ...) returned %T, want *net.TCPConn", clientConn)
	}
	return tc
}

// TestUnderlyingTCPConn_UnwrapsTLSConn is a regression test for the
// code path every real provider actually hits: every LLM provider is
// HTTPS, so the connection httptrace hands withHardCancel is a
// *tls.Conn, not a raw *net.TCPConn — SetLinger only exists on
// *net.TCPConn. Every other test in this file uses a plain
// httptest.NewServer (HTTP, not HTTPS), so this path was previously
// never exercised at all.
//
// tls.Client wraps an arbitrary net.Conn without performing a
// handshake, so this only needs a real *net.TCPConn underneath, not a
// full TLS round trip — NetConn() just returns whatever was wrapped.
func TestUnderlyingTCPConn_UnwrapsTLSConn(t *testing.T) {
	realConn := realTCPConnPair(t)
	tlsConn := tls.Client(realConn, &tls.Config{})

	got, ok := underlyingTCPConn(tlsConn)
	if !ok {
		t.Fatal("expected underlyingTCPConn to unwrap a *tls.Conn to its *net.TCPConn")
	}
	if got != realConn {
		t.Error("underlyingTCPConn returned a different *net.TCPConn than the one wrapped")
	}
}

// TestUnderlyingTCPConn_PlainTCPConn confirms the non-TLS path (used
// by tests and any http:// base URL, e.g. local model servers): no
// unwrapping needed, the conn is returned as-is.
func TestUnderlyingTCPConn_PlainTCPConn(t *testing.T) {
	realConn := realTCPConnPair(t)

	got, ok := underlyingTCPConn(realConn)
	if !ok {
		t.Fatal("expected underlyingTCPConn to recognise a plain *net.TCPConn")
	}
	if got != realConn {
		t.Error("underlyingTCPConn returned a different *net.TCPConn")
	}
}

// TestUnderlyingTCPConn_NeitherIsSafe confirms a connection type that
// is neither *tls.Conn nor *net.TCPConn (e.g. this package's own
// fakeConn, or in production a Unix socket) is reported as "no
// SetLinger available" rather than panicking.
func TestUnderlyingTCPConn_NeitherIsSafe(t *testing.T) {
	fc := newFakeConn()
	_, ok := underlyingTCPConn(fc)
	if ok {
		t.Error("expected ok=false for a connection type with no underlying *net.TCPConn")
	}
}

// captureConn issues a real request through client to url and returns
// whatever connection httptrace saw handle it, plus the response's
// negotiated protocol major version, for tests that need a genuinely
// negotiated (not merely constructed) TLS connection.
func captureConn(t *testing.T, client *http.Client, url string) (net.Conn, *http.Response) {
	t.Helper()
	var conn net.Conn
	trace := &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) { conn = info.Conn },
	}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if conn == nil {
		t.Fatal("GotConn never fired")
	}
	return conn, resp
}

// TestIsMultiplexed_HTTP2Connection is a regression test for the
// HTTP/2 collateral-damage concern: many concurrent requests can share
// one HTTP/2 connection, so hardCloseConn must recognise one and
// refuse to force-close it — closing it to unstick a single stalled
// request would take down every other request sharing it too.
func TestIsMultiplexed_HTTP2Connection(t *testing.T) {
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	srv.EnableHTTP2 = true // httptest defaults to HTTP/1.1-only otherwise
	srv.StartTLS()
	defer srv.Close()

	conn, resp := captureConn(t, srv.Client(), srv.URL)
	if resp.ProtoMajor != 2 {
		t.Fatalf("expected the test server to negotiate HTTP/2 with EnableHTTP2 set, got proto major %d", resp.ProtoMajor)
	}

	if !isMultiplexed(conn) {
		t.Error("expected isMultiplexed to report true for a negotiated HTTP/2 connection")
	}
	if err := hardCloseConn(conn); err != nil {
		t.Errorf("hardCloseConn on a multiplexed connection should be a no-op returning nil, got: %v", err)
	}

	// The real assertion: hardCloseConn returning nil isn't proof of
	// anything by itself (closing a live connection "succeeds" either
	// way). What matters is whether the connection actually survived —
	// verified by reusing the same client for a second request and
	// checking it rides the same underlying connection rather than
	// having to establish a new one.
	conn2, _ := captureConn(t, srv.Client(), srv.URL)
	if conn2 != conn {
		t.Error("hardCloseConn tore down the multiplexed connection instead of leaving it alone")
	}
}

// TestIsMultiplexed_HTTP1TLSConnection confirms a plain HTTP/1.1-over-
// TLS connection (not multiplexed, one request per connection) is not
// mistaken for HTTP/2 — hardCloseConn must still be free to force-close
// these, since that's the whole point of the fix.
func TestIsMultiplexed_HTTP1TLSConnection(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	defer srv.Close()

	client := srv.Client()
	// Force HTTP/1.1 by disabling the client's HTTP/2 upgrade.
	client.Transport.(*http.Transport).TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}

	conn, resp := captureConn(t, client, srv.URL)
	if resp.ProtoMajor != 1 {
		t.Fatalf("expected HTTP/1.1 with TLSNextProto disabled, got proto major %d", resp.ProtoMajor)
	}
	if isMultiplexed(conn) {
		t.Error("expected isMultiplexed to report false when HTTP/2 is disabled")
	}
}

// TestIsMultiplexed_PlainTCPConn confirms a non-TLS connection (used
// by tests and any http:// base URL, e.g. local model servers) is
// never considered multiplexed — Go's HTTP/2 support only runs over
// TLS.
func TestIsMultiplexed_PlainTCPConn(t *testing.T) {
	realConn := realTCPConnPair(t)
	if isMultiplexed(realConn) {
		t.Error("expected isMultiplexed to report false for a plain (non-TLS) connection")
	}
}
