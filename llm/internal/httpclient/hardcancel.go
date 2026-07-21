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
	"crypto/tls"
	"net"
	"net/http"
	"net/http/httptrace"
	"sync"
)

// withHardCancel returns a copy of req carrying an httptrace that
// captures whatever connection ends up handling the round trip, plus a
// release func. Call release() once the round trip has returned
// (success or failure); if req's context is done before that, a
// background goroutine force-closes the captured connection.
//
// This exists because context cancellation reliably aborts a request
// that is waiting for a response, but does not reliably interrupt a
// request body write that is blocked at the OS level — specifically
// when the body is large enough to exceed the OS socket buffers and
// the peer never reads it, so nothing ever drains the connection's
// receive window. Without this, such a request's connection stays
// open indefinitely even after the caller's context expires, since
// net/http's own cancellation only reliably covers the response-wait
// phase, not a genuinely blocked write.
//
// Force-closing (rather than, say, setting a deadline) is deliberate:
// it guarantees the connection is never handed back to the pool for
// reuse. After a client-side timeout the peer may still be mid-write
// of a now-orphaned response, and reusing the connection before that
// stale data is drained would corrupt framing for whatever request
// reuses it next.
func withHardCancel(req *http.Request) (*http.Request, func()) {
	ctx := req.Context()

	var mu sync.Mutex
	var conn net.Conn
	trace := &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) {
			mu.Lock()
			conn = info.Conn
			mu.Unlock()
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(ctx, trace))

	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			mu.Lock()
			c := conn
			mu.Unlock()
			if c != nil {
				_ = hardCloseConn(c)
			}
		case <-done:
		}
	}()

	return req, func() { close(done) }
}

// hardCloseConn forces an abortive close (RST) rather than a graceful
// one. A plain Close() still has to flush any unsent, already-buffered
// data to the peer before it can send FIN — which stalls exactly when
// the peer has stopped reading and will never ACK it, defeating the
// whole point of forcing the connection closed. SetLinger(0) tells the
// kernel to discard unsent data and reset the connection immediately
// instead.
func hardCloseConn(conn net.Conn) error {
	if tc, ok := underlyingTCPConn(conn); ok {
		_ = tc.SetLinger(0)
	}
	return conn.Close()
}

// underlyingTCPConn extracts the *net.TCPConn SetLinger lives on, if
// there is one. Real provider connections are TLS, so conn is normally
// a *tls.Conn; unwrap it via NetConn to reach the *net.TCPConn
// underneath. Split out from hardCloseConn so this unwrapping logic —
// the part that differs between the HTTP and HTTPS cases every real
// provider actually uses — can be tested directly, independent of
// Close()'s own OS-level, timing-dependent behaviour.
func underlyingTCPConn(conn net.Conn) (*net.TCPConn, bool) {
	if tc, ok := conn.(*tls.Conn); ok {
		conn = tc.NetConn()
	}
	tc, ok := conn.(*net.TCPConn)
	return tc, ok
}
