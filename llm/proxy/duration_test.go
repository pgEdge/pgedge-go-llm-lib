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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/proxy"
)

// testUpstreamDelay is the deterministic delay the fake provider applies
// to each upstream call in these timing tests. time.Sleep guarantees the
// goroutine pauses for at least this long, so a Duration measured across
// the call is always >= testUpstreamDelay.
const testUpstreamDelay = 5 * time.Millisecond

// TestChatResponseInfoCarriesDuration verifies that a successful
// non-streaming chat surfaces a non-zero upstream-call Duration on the
// ResponseInfo passed to OnResponse, covering at least the time the
// provider spent servicing the call.
func TestChatResponseInfoCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		delay: testUpstreamDelay,
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: llm.BlockText, Text: "ok"}},
			StopReason: "stop",
		},
	})

	var info proxy.ResponseInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnResponse:      func(_ *http.Request, ri proxy.ResponseInfo) { info = ri },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.Duration < testUpstreamDelay {
		t.Errorf("ResponseInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}

// TestChatUpstreamErrorCarriesDuration verifies that when the upstream
// Chat call fails, the ErrorInfo passed to OnError carries the
// wall-clock time spent on that failed provider call.
func TestChatUpstreamErrorCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		delay:   testUpstreamDelay,
		chatErr: errors.New("upstream boom"),
	})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.StatusCode != http.StatusBadGateway {
		t.Fatalf("OnError StatusCode = %d, want 502", info.StatusCode)
	}
	if info.Duration < testUpstreamDelay {
		t.Errorf("ErrorInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}

// TestChatPreUpstreamErrorHasZeroDuration verifies that an error raised
// before any upstream call (here, a request-body parse failure) leaves
// ErrorInfo.Duration at zero, because no provider request was made.
func TestChatPreUpstreamErrorHasZeroDuration(t *testing.T) {
	setFake(&fakeProvider{delay: testUpstreamDelay})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json",
		strings.NewReader("not-valid-json"))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.StatusCode != http.StatusBadRequest {
		t.Fatalf("OnError StatusCode = %d, want 400", info.StatusCode)
	}
	if info.Duration != 0 {
		t.Errorf("ErrorInfo.Duration = %v, want 0 for a pre-upstream failure", info.Duration)
	}
}

// TestChatStreamResponseInfoCarriesDuration verifies that a successful
// streaming chat surfaces a Duration covering the full stream, from
// initiating the call through to the final chunk.
func TestChatStreamResponseInfoCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk)
			errCh := make(chan error, 1)
			go func() {
				// Sleep mid-stream so the measured Duration reflects
				// time spent consuming the stream, not just setup.
				time.Sleep(testUpstreamDelay)
				chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "hi"}
				chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 1}}
				close(chunks)
				close(errCh)
			}()
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})

	var info proxy.ResponseInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnResponse:      func(_ *http.Request, ri proxy.ResponseInfo) { info = ri },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if !info.Stream {
		t.Errorf("ResponseInfo.Stream = false, want true")
	}
	if info.Duration < testUpstreamDelay {
		t.Errorf("ResponseInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}

// TestChatStreamMidStreamErrorCarriesDuration verifies that a stream
// that errors mid-flight surfaces the elapsed upstream time on both the
// ResponseInfo (partial response) and the ErrorInfo.
func TestChatStreamMidStreamErrorCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk)
			errCh := make(chan error, 1)
			go func() {
				time.Sleep(testUpstreamDelay)
				errCh <- errors.New("mid-stream boom")
				close(chunks)
				close(errCh)
			}()
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})

	var respInfo proxy.ResponseInfo
	var errInfo proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnResponse:      func(_ *http.Request, ri proxy.ResponseInfo) { respInfo = ri },
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { errInfo = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if respInfo.Duration < testUpstreamDelay {
		t.Errorf("ResponseInfo.Duration = %v, want >= %v", respInfo.Duration, testUpstreamDelay)
	}
	if errInfo.Err == nil {
		t.Fatal("OnError did not fire on mid-stream error")
	}
	if errInfo.Duration < testUpstreamDelay {
		t.Errorf("ErrorInfo.Duration = %v, want >= %v", errInfo.Duration, testUpstreamDelay)
	}
}

// TestEmbedUpstreamErrorCarriesDuration verifies that a failed upstream
// EmbedBatch call surfaces its elapsed time on the ErrorInfo. The embed
// handler does not fire OnResponse, so the error path is the only place
// Duration is observable for this endpoint.
func TestEmbedUpstreamErrorCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		delay:    testUpstreamDelay,
		embedErr: errors.New("embed boom"),
	})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.EmbedRequest{Provider: "fake", Input: []string{"x"}})
	resp, err := http.Post(srv.URL+"/v1/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.Duration < testUpstreamDelay {
		t.Errorf("ErrorInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}

// TestEmbedPreUpstreamErrorHasZeroDuration verifies that an embed
// request rejected before any upstream call (empty input) reports a
// zero Duration on OnError.
func TestEmbedPreUpstreamErrorHasZeroDuration(t *testing.T) {
	setFake(&fakeProvider{delay: testUpstreamDelay})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.EmbedRequest{Provider: "fake", Input: []string{}})
	resp, err := http.Post(srv.URL+"/v1/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.StatusCode != http.StatusBadRequest {
		t.Fatalf("OnError StatusCode = %d, want 400", info.StatusCode)
	}
	if info.Duration != 0 {
		t.Errorf("ErrorInfo.Duration = %v, want 0 for a pre-upstream failure", info.Duration)
	}
}

// TestRerankUpstreamErrorCarriesDuration verifies that a failed upstream
// Rerank call surfaces its elapsed time on the ErrorInfo.
func TestRerankUpstreamErrorCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		delay:     testUpstreamDelay,
		rerankErr: errors.New("rerank boom"),
	})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.RerankRequest{
		Provider: "fake", Query: "q", Documents: []string{"a"},
	})
	resp, err := http.Post(srv.URL+"/v1/rerank", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.Duration < testUpstreamDelay {
		t.Errorf("ErrorInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}

// TestEmbedMultimodalUpstreamErrorCarriesDuration verifies that a failed
// upstream EmbedMultimodal call surfaces its elapsed time on the
// ErrorInfo.
func TestEmbedMultimodalUpstreamErrorCarriesDuration(t *testing.T) {
	setFake(&fakeProvider{
		delay:         testUpstreamDelay,
		multimodalErr: errors.New("multimodal boom"),
	})

	var info proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, ei proxy.ErrorInfo) { info = ei },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.EmbedMultimodalRequest{
		Provider: "fake",
		Inputs: []proxy.MultimodalInputRequest{
			{Content: []proxy.MultimodalContentRequest{{Type: "text", Text: "x"}}},
		},
	})
	resp, err := http.Post(srv.URL+"/v1/embed/multimodal", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	if info.Duration < testUpstreamDelay {
		t.Errorf("ErrorInfo.Duration = %v, want >= %v", info.Duration, testUpstreamDelay)
	}
}
