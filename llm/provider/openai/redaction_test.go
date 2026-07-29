//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// redactionTestKey is synthetic. It has never been a working key.
const redactionTestKey = "sk-proj-T3stK3yNotReal0000000000000000000000000000000000AbCd"

// TestAuthErrorDoesNotLeakAPIKey covers the reported vulnerability: on
// an authentication failure OpenAI quotes a partially masked form of
// the submitted key in the error message, and relaying that message
// verbatim handed a fragment of the operator's real key to anyone who
// could see the resulting error.
func TestAuthErrorDoesNotLeakAPIKey(t *testing.T) {
	// Mirrors the shape of a real 401 body: a leading fragment of the
	// submitted key and its final characters survive the masking.
	const echoed = "sk-proj-T3stK3y****************************************AbCd"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"Incorrect API key provided: ` + echoed +
			`. You can find your API key at https://platform.openai.com/account/api-keys.","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  redactionTestKey,
		Model:   "gpt-4o",
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

	// The rest of the message must survive, or we have traded a leak
	// for an undiagnosable error.
	assertRedacted(t, err.Error())

	if !strings.Contains(err.Error(), "Incorrect API key provided") {
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
