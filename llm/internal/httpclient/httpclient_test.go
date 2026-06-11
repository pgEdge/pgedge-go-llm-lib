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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHeaderTransportInjectsHeaders(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.WriteHeader(200)
	}))
	defer server.Close()

	transport := NewHeaderTransport(nil, map[string]string{
		"X-Custom-Key": "custom-value",
	})
	client := &http.Client{Transport: transport}

	req, _ := http.NewRequest("GET", server.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	if receivedHeaders.Get("X-Custom-Key") != "custom-value" {
		t.Errorf("expected custom header, got %q", receivedHeaders.Get("X-Custom-Key"))
	}
}

func TestHeaderTransportDoesNotOverride(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.WriteHeader(200)
	}))
	defer server.Close()

	transport := NewHeaderTransport(nil, map[string]string{
		"Authorization": "Bearer custom-token",
	})
	client := &http.Client{Transport: transport}

	req, _ := http.NewRequest("GET", server.URL, nil)
	req.Header.Set("Authorization", "Bearer provider-token")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	if receivedHeaders.Get("Authorization") != "Bearer provider-token" {
		t.Errorf("expected provider token preserved, got %q", receivedHeaders.Get("Authorization"))
	}
}

func TestHeaderTransportNilHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer server.Close()

	transport := NewHeaderTransport(nil, nil)
	client := &http.Client{Transport: transport}

	req, _ := http.NewRequest("GET", server.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestDoJSON(t *testing.T) {
	type testReq struct {
		Name string `json:"name"`
	}
	type testResp struct {
		Greeting string `json:"greeting"`
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Content-Type") != "application/json" {
			t.Error("expected Content-Type application/json")
		}
		body, _ := io.ReadAll(r.Body)
		var req testReq
		json.Unmarshal(body, &req)
		resp := testResp{Greeting: "Hello " + req.Name}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	var resp testResp
	statusCode, body, err := DoJSON(context.Background(), client, "POST", server.URL, nil, testReq{Name: "World"}, &resp)
	if err != nil {
		t.Fatalf("DoJSON failed: %v", err)
	}
	if statusCode != 200 {
		t.Errorf("expected 200, got %d", statusCode)
	}
	if resp.Greeting != "Hello World" {
		t.Errorf("expected 'Hello World', got %q", resp.Greeting)
	}
	_ = body
}

func TestDoJSONWithHeaders(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("{}"))
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	var resp map[string]any
	_, _, err := DoJSON(context.Background(), client, "POST", server.URL, headers, nil, &resp)
	if err != nil {
		t.Fatalf("DoJSON failed: %v", err)
	}
	if receivedHeaders.Get("x-api-key") != "test-key" {
		t.Errorf("expected x-api-key header")
	}
	if receivedHeaders.Get("anthropic-version") != "2023-06-01" {
		t.Errorf("expected anthropic-version header")
	}
}

func TestDoJSONErrorStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error":"internal"}`))
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	var resp map[string]any
	statusCode, body, err := DoJSON(context.Background(), client, "POST", server.URL, nil, nil, &resp)
	if err != nil {
		t.Fatalf("DoJSON failed: %v", err)
	}
	if statusCode != 500 {
		t.Errorf("expected 500, got %d", statusCode)
	}
	if len(body) == 0 {
		t.Error("expected body for error response")
	}
}

// TestNewWithHeaders verifies that New returns a client with a HeaderTransport
// when a non-empty headers map is provided.
func TestNewWithHeaders(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.WriteHeader(200)
	}))
	defer server.Close()

	client := New(nil, map[string]string{"X-Test-Header": "hello"}, RetryConfig{Disabled: true}, 0)
	if _, ok := client.Transport.(*HeaderTransport); !ok {
		t.Fatalf("expected *HeaderTransport, got %T", client.Transport)
	}

	req, _ := http.NewRequest("GET", server.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	if receivedHeaders.Get("X-Test-Header") != "hello" {
		t.Errorf("expected X-Test-Header=hello, got %q", receivedHeaders.Get("X-Test-Header"))
	}
}

// TestDoJSONNilBody verifies that when reqBody is nil no Content-Type header is set.
func TestDoJSONNilBody(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("{}"))
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	var dest map[string]any
	statusCode, _, err := DoJSON(context.Background(), client, "GET", server.URL, nil, nil, &dest)
	if err != nil {
		t.Fatalf("DoJSON failed: %v", err)
	}
	if statusCode != 200 {
		t.Errorf("expected 200, got %d", statusCode)
	}
	if receivedHeaders.Get("Content-Type") != "" {
		t.Errorf("expected no Content-Type for nil body, got %q", receivedHeaders.Get("Content-Type"))
	}
}

// TestDoJSONBadURL verifies that an invalid URL causes DoJSON to return an error.
func TestDoJSONBadURL(t *testing.T) {
	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	_, _, err := DoJSON(context.Background(), client, "GET", "://bad-url", nil, nil, nil)
	if err == nil {
		t.Fatal("expected error for bad URL, got nil")
	}
}

// TestDoJSONMarshalError verifies that a non-marshalable body returns an error.
func TestDoJSONMarshalError(t *testing.T) {
	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	// channels cannot be marshalled to JSON
	badBody := make(chan int)
	_, _, err := DoJSON(context.Background(), client, "POST", "http://localhost", nil, badBody, nil)
	if err == nil {
		t.Fatal("expected marshal error, got nil")
	}
}

// TestDoSSERequest verifies that DoSSERequest makes a successful HTTP request
// and returns the response body for streaming.
func TestDoSSERequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, "data: hello")
		fmt.Fprintln(w, "data: world")
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	resp, err := DoSSERequest(context.Background(), client, "POST", server.URL,
		map[string]string{"Accept": "text/event-stream"}, nil)
	if err != nil {
		t.Fatalf("DoSSERequest failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "data: hello") {
		t.Errorf("expected SSE data in body, got %q", string(body))
	}
}

// TestDoSSERequestWithBody verifies that DoSSERequest marshals a request body correctly.
func TestDoSSERequestWithBody(t *testing.T) {
	type reqPayload struct {
		Stream bool `json:"stream"`
	}

	var gotContentType string
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, "data: ok")
	}))
	defer server.Close()

	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	resp, err := DoSSERequest(context.Background(), client, "POST", server.URL,
		nil, reqPayload{Stream: true})
	if err != nil {
		t.Fatalf("DoSSERequest failed: %v", err)
	}
	resp.Body.Close()

	// DoSSERequest does not set Content-Type (unlike DoJSON), so we only check
	// that the body was forwarded correctly.
	_ = gotContentType
	var parsed reqPayload
	if err := json.Unmarshal(gotBody, &parsed); err != nil {
		t.Fatalf("could not parse forwarded body: %v", err)
	}
	if !parsed.Stream {
		t.Error("expected stream=true in forwarded body")
	}
}

// TestDoSSERequestMarshalError verifies that a non-marshalable body returns an error.
func TestDoSSERequestMarshalError(t *testing.T) {
	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	badBody := make(chan int)
	_, err := DoSSERequest(context.Background(), client, "POST", "http://localhost", nil, badBody)
	if err == nil {
		t.Fatal("expected marshal error, got nil")
	}
}

// TestDoSSERequestBadURL verifies that an invalid URL returns an error.
func TestDoSSERequestBadURL(t *testing.T) {
	client := New(nil, nil, RetryConfig{Disabled: true}, 0)
	_, err := DoSSERequest(context.Background(), client, "GET", "://bad-url", nil, nil)
	if err == nil {
		t.Fatal("expected error for bad URL, got nil")
	}
}

// TestNewSSEScanner verifies that NewSSEScanner returns a usable scanner.
func TestNewSSEScanner(t *testing.T) {
	r := strings.NewReader("data: test\n")
	scanner := NewSSEScanner(r)
	if scanner == nil {
		t.Fatal("expected non-nil SSEScanner")
	}
}

// TestSSEScannerScan verifies that Scan reads data: lines and skips non-data lines.
func TestSSEScannerScan(t *testing.T) {
	input := "event: update\ndata: first\n\ndata: second\n: comment\ndata: third\n"
	scanner := NewSSEScanner(strings.NewReader(input))

	var results []string
	for scanner.Scan() {
		results = append(results, scanner.Data())
	}

	expected := []string{"first", "second", "third"}
	if len(results) != len(expected) {
		t.Fatalf("expected %d results, got %d: %v", len(expected), len(results), results)
	}
	for i, v := range expected {
		if results[i] != v {
			t.Errorf("result[%d]: expected %q, got %q", i, v, results[i])
		}
	}
}

// TestSSEScannerScanEmpty verifies that Scan returns false immediately on an empty stream.
func TestSSEScannerScanEmpty(t *testing.T) {
	scanner := NewSSEScanner(strings.NewReader(""))
	if scanner.Scan() {
		t.Error("expected Scan to return false on empty input")
	}
}

// TestSSEScannerScanNoDataLines verifies Scan returns false when there are no data: lines.
func TestSSEScannerScanNoDataLines(t *testing.T) {
	scanner := NewSSEScanner(strings.NewReader("event: ping\n: comment\n\n"))
	if scanner.Scan() {
		t.Error("expected Scan to return false when no data: lines are present")
	}
}

// TestSSEScannerData verifies that Data returns the correct value after each Scan call.
func TestSSEScannerData(t *testing.T) {
	scanner := NewSSEScanner(strings.NewReader("data: hello world\n"))
	if !scanner.Scan() {
		t.Fatal("expected Scan to return true")
	}
	if scanner.Data() != "hello world" {
		t.Errorf("expected 'hello world', got %q", scanner.Data())
	}
}

// TestSSEScannerErrNil verifies Err returns nil on a clean stream.
func TestSSEScannerErrNil(t *testing.T) {
	scanner := NewSSEScanner(strings.NewReader("data: ok\n\n"))
	for scanner.Scan() {
	}
	if err := scanner.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

// TestSSEScannerErrTooLong verifies Err surfaces bufio.ErrTooLong when
// a single SSE line exceeds bufio.Scanner's 64KiB default buffer.
func TestSSEScannerErrTooLong(t *testing.T) {
	big := "data: " + strings.Repeat("A", 70*1024) + "\n\n"
	scanner := NewSSEScanner(strings.NewReader(big))
	for scanner.Scan() {
	}
	if err := scanner.Err(); err == nil {
		t.Fatal("expected non-nil Err for an over-long SSE line")
	}
}

func TestNewInstallsRetryTransport(t *testing.T) {
	cfg := RetryConfig{MaxRetries: 1, InitialBackoff: time.Millisecond, MaxBackoff: time.Millisecond}

	c := New(nil, nil, cfg, 0)
	if c.Transport == nil {
		t.Fatal("Transport is nil; expected RetryTransport")
	}
	if _, ok := c.Transport.(*RetryTransport); !ok {
		t.Errorf("Transport = %T, want *RetryTransport", c.Transport)
	}
}

func TestNewInstallsRetryTransportForPerAttemptTimeout(t *testing.T) {
	// Even with retries disabled, a per-attempt timeout requires the
	// RetryTransport so the timeout is enforced per attempt.
	cfg := RetryConfig{Disabled: true, PerAttemptTimeout: time.Second}
	c := New(nil, nil, cfg, 0)
	if _, ok := c.Transport.(*RetryTransport); !ok {
		t.Errorf("Transport = %T, want *RetryTransport when PerAttemptTimeout is set", c.Transport)
	}
}

func TestNewSkipsRetryWhenDisabled(t *testing.T) {
	c := New(nil, nil, RetryConfig{Disabled: true}, 0)
	if _, ok := c.Transport.(*RetryTransport); ok {
		t.Errorf("got RetryTransport when retry was disabled")
	}
}

func TestNewWithHeadersComposesRetryAndHeaders(t *testing.T) {
	cfg := RetryConfig{MaxRetries: 1, InitialBackoff: time.Millisecond, MaxBackoff: time.Millisecond}
	c := New(nil, map[string]string{"X-Foo": "bar"}, cfg, 0)

	ht, ok := c.Transport.(*HeaderTransport)
	if !ok {
		t.Fatalf("outer transport = %T, want *HeaderTransport", c.Transport)
	}
	if _, ok := ht.inner.(*RetryTransport); !ok {
		t.Errorf("HeaderTransport.inner = %T, want *RetryTransport", ht.inner)
	}
}

func TestValidateBaseURLTrimsAndNormalises(t *testing.T) {
	got, err := ValidateBaseURL("  https://api.example.com/  ", "test")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got != "https://api.example.com" {
		t.Errorf("got %q, want %q", got, "https://api.example.com")
	}
}

func TestValidateBaseURLAcceptsHTTP(t *testing.T) {
	got, err := ValidateBaseURL("http://localhost:11434", "ollama")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got != "http://localhost:11434" {
		t.Errorf("got %q", got)
	}
}

func TestValidateBaseURLRejectsMissingScheme(t *testing.T) {
	_, err := ValidateBaseURL("api.example.com", "test")
	if err == nil {
		t.Fatal("want error for missing scheme")
	}
	if !strings.Contains(err.Error(), "test") {
		t.Errorf("error should name the provider: %v", err)
	}
}

func TestValidateBaseURLRejectsBogusScheme(t *testing.T) {
	_, err := ValidateBaseURL("ftp://api.example.com", "test")
	if err == nil {
		t.Fatal("want error for ftp scheme")
	}
	if !strings.Contains(err.Error(), "ftp") {
		t.Errorf("error should report the bad scheme: %v", err)
	}
}

func TestValidateBaseURLRejectsEmptyHost(t *testing.T) {
	_, err := ValidateBaseURL("https://", "test")
	if err == nil {
		t.Fatal("want error for empty host")
	}
	if !strings.Contains(err.Error(), "host") {
		t.Errorf("error should mention the host: %v", err)
	}
}

func TestIsLocalBaseURL(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want bool
	}{
		{"localhost", "http://localhost:11434", true},
		{"loopback v4", "http://127.0.0.1:8080", true},
		{"loopback v6", "http://[::1]:1234", true},
		{"dot-local name", "http://host.local", true},
		{"openai remote", "https://api.openai.com/v1", false},
		{"anthropic remote", "https://api.anthropic.com", false},
		{"private but routable", "http://10.0.0.5", false},
		{"malformed", "http://[::1", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsLocalBaseURL(tt.raw); got != tt.want {
				t.Errorf("IsLocalBaseURL(%q) = %v, want %v", tt.raw, got, tt.want)
			}
		})
	}
}
