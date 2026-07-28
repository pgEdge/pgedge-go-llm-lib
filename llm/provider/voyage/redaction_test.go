//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package voyage

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// redactionTestKey is synthetic. It has never been a working key.
const redactionTestKey = "pa-T3stK3yNotReal000000000000000000000000Qq77"

// TestAuthErrorDoesNotLeakAPIKey covers both halves of the Voyage fix:
// the credential must not survive into the error, and the raw response
// body must no longer be dumped into it wholesale. The body carries a
// distinctive marker outside the error field; if that marker appears in
// the message, we are still relaying bytes we have not reasoned about.
func TestAuthErrorDoesNotLeakAPIKey(t *testing.T) {
	const (
		echoed = "pa-T3stK3y****************************Qq77"
		marker = "unreasoned-body-marker"
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail":"Provided API key ` + echoed + ` is invalid.",` +
			`"trace":"` + marker + `"}`))
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  redactionTestKey,
		Model:   "voyage-3.5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	_, err = c.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected an error from a 401 response")
	}

	assertNoKeyFragments(t, err.Error(), redactionTestKey)

	if strings.Contains(err.Error(), marker) {
		t.Errorf("the raw response body is still being relayed: %q", err.Error())
	}
	assertRedacted(t, err.Error())

	if !strings.Contains(err.Error(), "is invalid") {
		t.Errorf("diagnostic context was lost: %q", err.Error())
	}
}

// TestUnparseableErrorBodyIsNotRelayed asserts that a body we could not
// parse degrades to the status code rather than being quoted, since an
// unparseable body is precisely the case where we cannot know what it
// contains.
func TestUnparseableErrorBodyIsNotRelayed(t *testing.T) {
	const junk = "not json at all, and here is pa-T3stK3yNotReal000000000000000000000000Qq77"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(junk))
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  redactionTestKey,
		Model:   "voyage-3.5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	_, err = c.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected an error from a 500 response")
	}

	if strings.Contains(err.Error(), "not json at all") {
		t.Errorf("an unparseable body was relayed verbatim: %q", err.Error())
	}
	assertNoKeyFragments(t, err.Error(), redactionTestKey)
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
