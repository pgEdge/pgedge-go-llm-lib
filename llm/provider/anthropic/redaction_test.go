//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// redactionTestKey is synthetic. It has never been a working key.
const redactionTestKey = "sk-ant-api03-T3stK3yNotReal000000000000000000000000000000WxYz"

// TestAuthErrorDoesNotLeakAPIKey asserts that a provider which quotes
// the submitted credential back at us cannot get that credential into
// the error we hand our caller. Anthropic's own 401 body does not
// currently echo the key, but that is a property of today's API rather
// than a guarantee, and it is not one we should depend on.
func TestAuthErrorDoesNotLeakAPIKey(t *testing.T) {
	const echoed = "sk-ant-api03-T3stK3y****************************WxYz"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error",` +
			`"message":"invalid credential ` + echoed + ` supplied"}}`))
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  redactionTestKey,
		Model:   "claude-sonnet-4-20250514",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	_, err = c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("hello")},
	})
	if err == nil {
		t.Fatal("expected an error from a 401 response")
	}

	assertNoKeyFragments(t, err.Error(), redactionTestKey)

	assertRedacted(t, err.Error())

	if !strings.Contains(err.Error(), "invalid credential") {
		t.Errorf("diagnostic context was lost: %q", err.Error())
	}
}

// assertNoKeyFragments fails if msg contains the key or any run of it
// long enough to be useful to an attacker.
func assertNoKeyFragments(t *testing.T, msg, key string) {
	t.Helper()

	if strings.Contains(msg, key) {
		t.Fatalf("the full API key leaked into the error: %q", msg)
	}
	const minRun = 4
	for i := 0; i+minRun <= len(key); i++ {
		if frag := key[i : i+minRun]; strings.Contains(msg, frag) {
			t.Errorf("a %d-character run of the API key (%q) leaked into the error: %q",
				minRun, frag, msg)
			return
		}
	}
}

// assertRedacted fails if msg carries no redaction placeholder. Each
// test that calls this feeds a credential through a real provider error
// path, so a message with nothing redacted means the leak assertions
// passed for the wrong reason and the test has stopped proving
// anything.
func assertRedacted(t *testing.T, msg string) {
	t.Helper()
	if !strings.Contains(msg, "[REDACTED]") {
		t.Errorf("no redaction placeholder in the message; this test is no longer exercising the redaction path: %q", msg)
	}
}
