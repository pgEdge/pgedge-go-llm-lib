//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package proxy_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/proxy"
)

// redactionTestKey is synthetic. It has never been a working key.
const redactionTestKey = "sk-proj-T3stK3yNotReal0000000000000000000000000000000000AbCd"

// postAndReadBody drives one request against p and returns the
// response body as a string.
func postAndReadBody(t *testing.T, p *proxy.Proxy, path, body string) string {
	t.Helper()

	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	out, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(out)
}

// TestErrorResponseDoesNotLeakProviderKey covers the second layer: even
// if a provider error reaches the proxy with the configured credential
// still in it, the proxy must not put it on the wire.
func TestErrorResponseDoesNotLeakProviderKey(t *testing.T) {
	setFake(&fakeProvider{
		chatErr: fmt.Errorf("upstream refused key %s outright", redactionTestKey),
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha", APIKey: redactionTestKey}},
	})

	body := postAndReadBody(t, p, "/v1/chat", `{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

	if strings.Contains(body, redactionTestKey) {
		t.Errorf("the provider API key leaked into the response body: %s", body)
	}
	if !strings.Contains(body, "upstream refused key") {
		t.Errorf("diagnostic context was lost: %s", body)
	}
	assertRedacted(t, body)
}

// TestAuthErrorDoesNotLeakClientToken covers what the proxy layer
// catches first-hand rather than as a second line of defence: an
// Authorize hook is a natural place for a caller's own token to end up
// spliced into an error message, and the proxy holds no secret to
// compare against, so the shape-based patterns have to carry it.
func TestAuthErrorDoesNotLeakClientToken(t *testing.T) {
	const clientToken = "abcdef0123456789wxyz"

	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		Authorize: func(*http.Request) error {
			return &proxy.AuthError{
				Err: fmt.Errorf("rejected Authorization: Bearer %s", clientToken),
			}
		},
	})

	body := postAndReadBody(t, p, "/v1/chat", `{}`)

	if strings.Contains(body, clientToken) {
		t.Errorf("a client token leaked into the response body: %s", body)
	}
	if !strings.Contains(strings.ToLower(body), "bearer") {
		t.Errorf("the scheme should survive for diagnosis: %s", body)
	}
	assertRedacted(t, body)
}

// TestStreamErrorDoesNotLeakProviderKey covers the SSE error event,
// which is a separate write path from writeError and was leaking
// independently of it.
func TestStreamErrorDoesNotLeakProviderKey(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			return nil, fmt.Errorf("upstream refused key %s outright", redactionTestKey)
		},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha", APIKey: redactionTestKey}},
	})

	body := postAndReadBody(t, p, "/v1/chat/stream", `{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

	if strings.Contains(body, redactionTestKey) {
		t.Errorf("the provider API key leaked into the stream response: %s", body)
	}
	assertRedacted(t, body)
}

// TestHealthErrorDoesNotLeakProviderKey covers the health endpoint,
// which builds its own per-provider error strings rather than going
// through writeError.
func TestHealthErrorDoesNotLeakProviderKey(t *testing.T) {
	setFake(&fakeProvider{
		pingErr: fmt.Errorf("ping rejected for key %s", redactionTestKey),
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha", APIKey: redactionTestKey}},
	})

	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	out, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	if strings.Contains(string(out), redactionTestKey) {
		t.Errorf("the provider API key leaked into the health response: %s", out)
	}
	assertRedacted(t, string(out))
}

// assertRedacted fails if body carries no redaction placeholder. Every
// test in this file feeds a credential through a real error path, so a
// body with nothing redacted means the assertion above passed for the
// wrong reason: the text never reached the response at all, and the
// test has stopped proving anything.
func assertRedacted(t *testing.T, body string) {
	t.Helper()
	if !strings.Contains(body, "[REDACTED]") {
		t.Errorf("no redaction placeholder in the response; this test is no longer exercising the redaction path: %s", body)
	}
}
