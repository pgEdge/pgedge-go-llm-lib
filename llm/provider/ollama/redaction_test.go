//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package ollama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// TestErrorDoesNotLeakCredentialShapedText covers the Ollama case.
// Ollama runs locally and authenticates with no key, so there is no
// configured secret to compare against, but a gateway or proxy in front
// of it may quote a credential of its own back at us. Shape-based
// redaction is all we have here, and it must still fire.
func TestErrorDoesNotLeakCredentialShapedText(t *testing.T) {
	const token = "abcdef0123456789xyzQ"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"gateway rejected upstream Authorization: Bearer ` +
			token + ` for this model"}`))
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		Model:   "llama3",
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

	if strings.Contains(err.Error(), token) {
		t.Errorf("a credential-shaped token leaked into the error: %q", err.Error())
	}
	assertRedacted(t, err.Error())

	if !strings.Contains(err.Error(), "gateway rejected upstream") {
		t.Errorf("diagnostic context was lost: %q", err.Error())
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
