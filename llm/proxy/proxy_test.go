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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/proxy"

	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
)

func TestNewWithNilProvidersDoesNotPanic(t *testing.T) {
	p := proxy.New(proxy.Config{})
	if p == nil {
		t.Fatal("New returned nil")
	}
	if p.Handler() == nil {
		t.Fatal("Handler returned nil")
	}
}

func TestHandlerReturnsNotFoundForUnknownPath(t *testing.T) {
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/nope")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestHandleProviders(t *testing.T) {
	p := proxy.New(proxy.Config{
		DefaultProvider: "anthropic",
		Providers: map[string]llm.Options{
			"anthropic": {APIKey: "k", Model: "claude-x"},
			"openai":    {APIKey: "k", Model: "gpt-x"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/providers")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var body proxy.ProvidersResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if body.DefaultProvider != "anthropic" {
		t.Errorf("default = %q, want anthropic", body.DefaultProvider)
	}
	if len(body.Providers) != 2 {
		t.Fatalf("got %d providers, want 2", len(body.Providers))
	}

	names := map[string]proxy.ProviderInfo{}
	for _, info := range body.Providers {
		names[info.Name] = info
	}
	if names["anthropic"].Model != "claude-x" || !names["anthropic"].Default {
		t.Errorf("anthropic info = %+v", names["anthropic"])
	}
	if names["openai"].Model != "gpt-x" || names["openai"].Default {
		t.Errorf("openai info = %+v", names["openai"])
	}
}

func TestNewDefensivelyCopiesProvidersMap(t *testing.T) {
	original := map[string]llm.Options{
		"a": {Model: "m1"},
	}
	p := proxy.New(proxy.Config{Providers: original})

	// Mutate the original map after construction.
	original["b"] = llm.Options{Model: "m2"}
	delete(original, "a")

	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/providers")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	var body proxy.ProvidersResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}

	names := make(map[string]bool, len(body.Providers))
	for _, info := range body.Providers {
		names[info.Name] = true
	}
	if !names["a"] {
		t.Errorf("proxy lost 'a' after caller mutated the input map")
	}
	if names["b"] {
		t.Errorf("proxy picked up 'b' after caller mutated the input map")
	}
}

func TestHandleProvidersEmpty(t *testing.T) {
	p := proxy.New(proxy.Config{})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/providers")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	// Must serialise as [] not null so JS clients can iterate.
	if !strings.Contains(string(raw), `"providers":[]`) {
		t.Errorf("expected providers:[] in body, got %s", raw)
	}

	var body proxy.ProvidersResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.DefaultProvider != "" {
		t.Errorf("DefaultProvider = %q, want empty", body.DefaultProvider)
	}
	if len(body.Providers) != 0 {
		t.Errorf("Providers length = %d, want 0", len(body.Providers))
	}
}

func TestHandleModels(t *testing.T) {
	setFake(&fakeProvider{models: []string{"alpha", "beta"}})

	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{
			"fake": {Model: "alpha"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models?provider=fake")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var body proxy.ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Models) != 2 || body.Models[0] != "alpha" || body.Models[1] != "beta" {
		t.Errorf("models = %v", body.Models)
	}
}

func TestHandleModelsRejectsUnknownProvider(t *testing.T) {
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{"fake": {}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models?provider=does-not-exist")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleModelsRequiresProviderQuery(t *testing.T) {
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{"fake": {}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatRoundtrips(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: "text", Text: "hello back"}},
			StopReason: "stop",
			Usage:      llm.TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
		},
	})

	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers: map[string]llm.Options{
			"fake": {Model: "alpha"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var got proxy.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.StopReason != "stop" {
		t.Errorf("stop_reason = %q", got.StopReason)
	}
	if len(got.Content) != 1 || got.Content[0].Text != "hello back" {
		t.Errorf("content = %+v", got.Content)
	}
	if got.Usage.TotalTokens != 7 {
		t.Errorf("usage.total = %d", got.Usage.TotalTokens)
	}
}

func TestHandleChatPropagatesToolDescriptions(t *testing.T) {
	f := &fakeProvider{
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: "text", Text: "ok"}},
			StopReason: "stop",
		},
	}
	setFake(f)

	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers: map[string]llm.Options{
			"fake": {Model: "alpha"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages:         []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
		ToolDescriptions: llm.ToolDescriptionCompact,
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	f.mu.RLock()
	got := f.chatReq
	f.mu.RUnlock()
	if got == nil {
		t.Fatal("provider did not receive a Chat request")
	}
	if got.ToolDescriptions != llm.ToolDescriptionCompact {
		t.Errorf("ToolDescriptions = %q, want %q", got.ToolDescriptions, llm.ToolDescriptionCompact)
	}
}

func TestHandleChatHonoursPerRequestProviderAndModel(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: "text", Text: "ok"}},
			StopReason: "stop",
		},
	})
	// fakeProvider records opts.Model into f.model on construction;
	// we read it back from the singleton after the request.
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers: map[string]llm.Options{
			"fake": {Model: "default-model"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
		Provider: "fake",
		Model:    "override-model",
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	if got := fakeInstance.Model(); got != "override-model" {
		t.Errorf("model passed to provider = %q, want override-model", got)
	}
}

func TestHandleChatRejectsUnknownProvider(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{"fake": {}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
		Provider: "does-not-exist",
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatInvokesHooks(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: "text", Text: "ok"}},
			StopReason: "stop",
			Usage:      llm.TokenUsage{TotalTokens: 3},
		},
	})

	var reqInfo proxy.RequestInfo
	var respInfo proxy.ResponseInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnRequest:       func(_ *http.Request, info proxy.RequestInfo) { reqInfo = info },
		OnResponse:      func(_ *http.Request, info proxy.ResponseInfo) { respInfo = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	resp.Body.Close()

	if reqInfo.Provider != "fake" || reqInfo.Stream {
		t.Errorf("OnRequest got %+v", reqInfo)
	}
	if respInfo.Provider != "fake" || respInfo.StatusCode != 200 || respInfo.Usage.TotalTokens != 3 {
		t.Errorf("OnResponse got %+v", respInfo)
	}
}

func TestHandleChatStream(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(ctx context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk, 3)
			errCh := make(chan error, 1)
			chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "hello "}
			chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "world"}
			chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 9}}
			close(chunks)
			close(errCh)
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})

	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", got)
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	body2 := string(raw)
	for _, want := range []string{
		`data: {"type":"text","text":"hello "}`,
		`data: {"type":"text","text":"world"}`,
		`event: done`,
	} {
		if !strings.Contains(body2, want) {
			t.Errorf("response missing %q\nfull body:\n%s", want, body2)
		}
	}
}

func TestHandleChatStreamHookCarriesFinalUsage(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(ctx context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk, 1)
			errCh := make(chan error, 1)
			chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 42}}
			close(chunks)
			close(errCh)
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
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if !info.Stream {
		t.Errorf("ResponseInfo.Stream = false")
	}
	if info.Usage.TotalTokens != 42 {
		t.Errorf("ResponseInfo.Usage.TotalTokens = %d", info.Usage.TotalTokens)
	}
}

func TestOnErrorFiresOnInvalidJSONBody(t *testing.T) {
	var got proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, info proxy.ErrorInfo) { got = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json",
		strings.NewReader("not-valid-json"))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
	if got.StatusCode != http.StatusBadRequest {
		t.Errorf("OnError StatusCode = %d, want 400", got.StatusCode)
	}
	if got.Err == nil {
		t.Error("OnError Err is nil, want non-nil")
	}
	if got.Provider != "" {
		t.Errorf("OnError Provider = %q, want empty (no parse context)", got.Provider)
	}
}

func TestOnErrorFiresOnUpstreamFailureWithContext(t *testing.T) {
	setFake(&fakeProvider{chatErr: errors.New("upstream 401")})

	var got proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, info proxy.ErrorInfo) { got = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
	if got.StatusCode != http.StatusBadGateway {
		t.Errorf("OnError StatusCode = %d, want 502", got.StatusCode)
	}
	if got.Err == nil || got.Err.Error() != "upstream 401" {
		t.Errorf("OnError Err = %v, want 'upstream 401'", got.Err)
	}
	if got.Provider != "fake" {
		t.Errorf("OnError Provider = %q, want 'fake'", got.Provider)
	}
	if got.Model != "alpha" {
		t.Errorf("OnError Model = %q, want 'alpha'", got.Model)
	}
	if got.Stream {
		t.Error("OnError Stream = true, want false")
	}
}

func TestOnErrorFiresOnMidStreamError(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(ctx context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk)
			errCh := make(chan error, 1)
			errCh <- errors.New("network died mid-stream")
			close(chunks)
			close(errCh)
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})

	var got proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		OnError:         func(_ *http.Request, info proxy.ErrorInfo) { got = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if got.StatusCode != http.StatusBadGateway {
		t.Errorf("OnError StatusCode = %d, want 502", got.StatusCode)
	}
	if !got.Stream {
		t.Error("OnError Stream = false, want true")
	}
	if got.Provider != "fake" {
		t.Errorf("OnError Provider = %q, want 'fake'", got.Provider)
	}
}

func TestHandleModelsWithMetadata(t *testing.T) {
	setFake(&fakeProvider{models: []string{"alpha", "beta"}})

	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models?provider=fake&metadata=true")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var body proxy.ModelsMetadataResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Models) != 2 || body.Models[0].ID != "alpha" {
		t.Errorf("models = %+v", body.Models)
	}
}

func TestHandleHealth(t *testing.T) {
	setFake(&fakeProvider{models: []string{"alpha"}})
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{
			"fake": {Model: "alpha"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d", resp.StatusCode)
	}

	var body proxy.HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "ok" {
		t.Errorf("status = %q, want ok", body.Status)
	}
	if body.Providers["fake"].Status != "ok" {
		t.Errorf("fake provider status = %+v", body.Providers["fake"])
	}
}

func TestHandleHealthEmptyProviders(t *testing.T) {
	p := proxy.New(proxy.Config{})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	var body proxy.HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// With no providers, allOK stays true → status "ok".
	if body.Status != "ok" {
		t.Errorf("status = %q, want ok", body.Status)
	}
	if len(body.Providers) != 0 {
		t.Errorf("expected empty Providers, got %+v", body.Providers)
	}
}

func TestEndToEndStreamThroughOpenAIProvider(t *testing.T) {
	// Stand up a fake "OpenAI" server that returns a real SSE stream
	// in OpenAI's chat-completion wire format.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"alpha"},"finish_reason":null}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":" beta"},"finish_reason":null}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	p := proxy.New(proxy.Config{
		DefaultProvider: "openai",
		Providers: map[string]llm.Options{
			"openai": {APIKey: "k", BaseURL: upstream.URL, Model: "test"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	body2 := string(raw)

	for _, want := range []string{
		`"text":"alpha"`,
		`"text":" beta"`,
		`event: done`,
	} {
		if !strings.Contains(body2, want) {
			t.Errorf("response missing %q\nfull body:\n%s", want, body2)
		}
	}
}

func TestAuthorizeRejectsWithDefaultStatus(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		Authorize: func(r *http.Request) error {
			return fmt.Errorf("nope")
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestAuthorizeRejectsWithCustomStatus(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		Authorize: func(r *http.Request) error {
			if r.Header.Get("X-Auth") != "good" {
				return &proxy.AuthError{Err: fmt.Errorf("nope"), Status: http.StatusForbidden}
			}
			return nil
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("status = %d, want 403", resp.StatusCode)
	}
}

// TestAuthorizeRejectionForwardsRequestID verifies that an Authorize
// rejection threads the request ID into OnError, matching the
// behaviour of every other error path.
func TestAuthorizeRejectionForwardsRequestID(t *testing.T) {
	setFake(&fakeProvider{})
	var gotReqID string
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		Authorize: func(*http.Request) error {
			return fmt.Errorf("nope")
		},
		OnError: func(_ *http.Request, info proxy.ErrorInfo) {
			gotReqID = info.RequestID
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "auth-reqid-42")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
	if gotReqID != "auth-reqid-42" {
		t.Errorf("OnError RequestID = %q, want auth-reqid-42", gotReqID)
	}
}

func TestAuthorizeAllowsWhenNil(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "ok"}}, StopReason: llm.StopReasonEndTurn},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		// Authorize: nil
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
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

func TestRequestIDEcho(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "ok"}}, StopReason: llm.StopReasonEndTurn},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "test-id-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if got := resp.Header.Get("X-Request-ID"); got != "test-id-123" {
		t.Errorf("X-Request-ID = %q, want test-id-123", got)
	}
}

func TestRequestIDGeneratedWhenMissing(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "ok"}}, StopReason: llm.StopReasonEndTurn},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	resp.Body.Close()
	id := resp.Header.Get("X-Request-ID")
	if id == "" {
		t.Errorf("X-Request-ID empty; expected generated value")
	}
	if len(id) != 32 {
		t.Errorf("generated request ID has wrong length: %q (len=%d)", id, len(id))
	}
}

func TestRequestIDDisabled(t *testing.T) {
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "ok"}}, StopReason: llm.StopReasonEndTurn},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		RequestIDHeader: "-",
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	resp.Body.Close()
	if id := resp.Header.Get("X-Request-ID"); id != "" {
		t.Errorf("X-Request-ID = %q, want empty when disabled", id)
	}
}

func TestAuthErrorUnwrapAndDefaultStatus(t *testing.T) {
	inner := errors.New("denied")
	e := &proxy.AuthError{Err: inner}
	if got := e.Error(); got != "denied" {
		t.Errorf("Error() = %q, want %q", got, "denied")
	}
	if !errors.Is(e, inner) {
		t.Errorf("errors.Is(e, inner) = false; Unwrap is not exposing the inner error")
	}
	if got := e.HTTPStatus(); got != http.StatusUnauthorized {
		t.Errorf("HTTPStatus() with zero Status = %d, want %d", got, http.StatusUnauthorized)
	}
	withStatus := &proxy.AuthError{Err: inner, Status: http.StatusForbidden}
	if got := withStatus.HTTPStatus(); got != http.StatusForbidden {
		t.Errorf("HTTPStatus() with Status set = %d, want %d", got, http.StatusForbidden)
	}
}

func TestRequestIDFromContext(t *testing.T) {
	// A bare context carries no ID.
	if got := proxy.RequestIDFromContext(context.Background()); got != "" {
		t.Errorf("RequestIDFromContext on bare ctx = %q, want empty", got)
	}

	// When the proxy receives a request with X-Request-ID set, hooks
	// see the same ID via the context.
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{
			Content:    []llm.ContentBlock{{Type: "text", Text: "ok"}},
			StopReason: "stop",
		},
	})
	var seen string
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		OnRequest: func(r *http.Request, _ proxy.RequestInfo) {
			seen = proxy.RequestIDFromContext(r.Context())
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	})
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "abc-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	if seen != "abc-123" {
		t.Errorf("RequestIDFromContext in OnRequest = %q, want %q", seen, "abc-123")
	}
}

func TestHandleModelsUpstreamError(t *testing.T) {
	setFake(&fakeProvider{listModelsErr: errors.New("upstream down")})
	p := proxy.New(proxy.Config{Providers: map[string]llm.Options{"fake": {Model: "m"}}})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models?provider=fake")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "upstream down") {
		t.Errorf("body missing upstream error: %s", body)
	}
}

func TestHandleModelsMetadataUpstreamError(t *testing.T) {
	setFake(&fakeProvider{listModelsErr: errors.New("metadata fetch failed")})
	p := proxy.New(proxy.Config{Providers: map[string]llm.Options{"fake": {Model: "m"}}})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models?provider=fake&metadata=true")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
}

func TestHandleHealthMixedDownAndOk(t *testing.T) {
	// Two providers configured: "fake" responds OK, "down" returns
	// a Ping error. The aggregate status should be "degraded" with
	// a 503 response.
	setFake(&fakeProvider{pingErr: errors.New("offline")})
	p := proxy.New(proxy.Config{Providers: map[string]llm.Options{"fake": {Model: "m"}}})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	var got proxy.HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Status != "degraded" {
		t.Errorf("Status = %q, want degraded", got.Status)
	}
	if h, ok := got.Providers["fake"]; !ok || h.Status != "down" || h.Error == "" {
		t.Errorf("fake health = %+v, want {down, non-empty error}", h)
	}
}

func TestHandleChatRejectsInvalidJSONBody(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "invalid request body") {
		t.Errorf("body missing expected text: %s", body)
	}
}

func TestHandleChatNoProviderConfigured(t *testing.T) {
	p := proxy.New(proxy.Config{}) // no DefaultProvider, no Providers
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatStreamUpstreamError(t *testing.T) {
	// streamFn returns an error from ChatStream — that path is the
	// 502 BadGateway branch in handleChatStream.
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			return nil, errors.New("upstream stream failed")
		},
	})
	var sawError proxy.ErrorInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		OnError:         func(_ *http.Request, info proxy.ErrorInfo) { sawError = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
	if sawError.StatusCode != http.StatusBadGateway || !sawError.Stream {
		t.Errorf("OnError got %+v, want stream=true status=502", sawError)
	}
}

func TestHandleChatStreamMidStreamErrorFiresHooks(t *testing.T) {
	// The provider returns a *Stream that immediately produces an
	// error from Recv. handleChatStream should: emit `event: error`
	// via writeSSE, fire OnError, and pass the streamErr through to
	// OnResponse with status 502.
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk)
			errCh := make(chan error, 1)
			errCh <- errors.New("mid-stream boom")
			close(chunks)
			close(errCh)
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})
	var sawError, sawResponse bool
	var responseStatus int
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		OnError:         func(_ *http.Request, _ proxy.ErrorInfo) { sawError = true },
		OnResponse: func(_ *http.Request, info proxy.ResponseInfo) {
			sawResponse = true
			responseStatus = info.StatusCode
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if !sawError {
		t.Errorf("OnError did not fire on mid-stream error")
	}
	if !sawResponse || responseStatus != http.StatusBadGateway {
		t.Errorf("OnResponse got status=%d, want 502 (sawResponse=%v)", responseStatus, sawResponse)
	}
}

func TestAuthorizeRejectsAllEndpoints(t *testing.T) {
	// A single Authorize hook that rejects every request must short-
	// circuit /v1/providers, /v1/models, /v1/health, /v1/chat,
	// /v1/chat/stream, /v1/embed, /v1/embed/multimodal, and
	// /v1/rerank alike.
	setFake(&fakeProvider{})
	rejected := errors.New("nope")
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		Authorize:       func(*http.Request) error { return rejected },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	type call struct {
		method, path, body string
	}
	calls := []call{
		{http.MethodGet, "/v1/providers", ""},
		{http.MethodGet, "/v1/models?provider=fake", ""},
		{http.MethodGet, "/v1/health", ""},
		{http.MethodPost, "/v1/chat", `{"messages":[]}`},
		{http.MethodPost, "/v1/chat/stream", `{"messages":[]}`},
		{http.MethodPost, "/v1/embed", `{"provider":"fake","input":["x"]}`},
		{http.MethodPost, "/v1/embed/multimodal", `{"provider":"fake","inputs":[]}`},
		{http.MethodPost, "/v1/rerank", `{"provider":"fake","query":"q","documents":["a"]}`},
	}
	for _, c := range calls {
		req, _ := http.NewRequest(c.method, srv.URL+c.path, strings.NewReader(c.body))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: request failed: %v", c.method, c.path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s %s: status = %d, want 401", c.method, c.path, resp.StatusCode)
		}
	}
}

func TestNewClientErrorsAcrossEndpoints(t *testing.T) {
	// Configure a provider name that is NOT registered. llm.NewClient
	// returns an error; the proxy must surface that as 500 on
	// /v1/models, /v1/chat, and /v1/chat/stream, and as a "down"
	// entry on /v1/health.
	setFake(&fakeProvider{}) // never invoked, but setFake is required for any "fake"-typed init
	const unregistered = "no-such-provider"
	p := proxy.New(proxy.Config{
		DefaultProvider: unregistered,
		Providers:       map[string]llm.Options{unregistered: {Model: "m"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	t.Run("models", func(t *testing.T) {
		resp, err := http.Get(srv.URL + "/v1/models?provider=" + unregistered)
		if err != nil {
			t.Fatalf("get failed: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500", resp.StatusCode)
		}
	})

	t.Run("chat", func(t *testing.T) {
		body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
		resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("post failed: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500", resp.StatusCode)
		}
	})

	t.Run("chat stream", func(t *testing.T) {
		body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
		resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("post failed: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500", resp.StatusCode)
		}
	})

	t.Run("health reports down", func(t *testing.T) {
		resp, err := http.Get(srv.URL + "/v1/health")
		if err != nil {
			t.Fatalf("get failed: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503", resp.StatusCode)
		}
		var got proxy.HealthResponse
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		h := got.Providers[unregistered]
		if h.Status != "down" || h.Error == "" {
			t.Errorf("provider entry = %+v, want {down, non-empty error}", h)
		}
	})
}

func TestHandleChatStreamRejectsInvalidJSONBody(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHookRequestAndResponseExposePayloads(t *testing.T) {
	// Verify that OnRequest sees the messages, tools, and stop
	// sequences the proxy is about to send, and OnResponse sees the
	// content blocks (text + tool-use) the provider returned.
	setFake(&fakeProvider{
		chatResp: &llm.ChatResponse{
			Content: []llm.ContentBlock{
				{Type: llm.BlockText, Text: "calling get_weather"},
				{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
					ID:    "tu_resp",
					Name:  "get_weather",
					Input: []byte(`{"city":"London"}`),
				}},
			},
			StopReason: llm.StopReasonToolUse,
			Usage:      llm.TokenUsage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20},
		},
	})

	var reqInfo proxy.RequestInfo
	var respInfo proxy.ResponseInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		OnRequest:       func(_ *http.Request, info proxy.RequestInfo) { reqInfo = info },
		OnResponse:      func(_ *http.Request, info proxy.ResponseInfo) { respInfo = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages:      []llm.Message{llm.UserText("Weather in London?")},
		SystemPrompt:  "You are a helpful assistant.",
		Tools:         []llm.Tool{{Name: "get_weather", Description: "get weather"}},
		StopSequences: []string{"</answer>"},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	resp.Body.Close()

	// OnRequest received the resolved request payload.
	if reqInfo.Request == nil {
		t.Fatal("RequestInfo.Request is nil")
	}
	if len(reqInfo.Request.Messages) != 1 || len(reqInfo.Request.Messages[0].Content) == 0 {
		t.Errorf("Request.Messages not populated: %+v", reqInfo.Request.Messages)
	}
	if reqInfo.Request.Messages[0].Content[0].Text != "Weather in London?" {
		t.Errorf("user text = %q, want %q", reqInfo.Request.Messages[0].Content[0].Text, "Weather in London?")
	}
	if reqInfo.Request.SystemPrompt != "You are a helpful assistant." {
		t.Errorf("SystemPrompt = %q", reqInfo.Request.SystemPrompt)
	}
	if len(reqInfo.Request.Tools) != 1 || reqInfo.Request.Tools[0].Name != "get_weather" {
		t.Errorf("Tools not populated: %+v", reqInfo.Request.Tools)
	}
	if len(reqInfo.Request.StopSequences) != 1 || reqInfo.Request.StopSequences[0] != "</answer>" {
		t.Errorf("StopSequences not populated: %+v", reqInfo.Request.StopSequences)
	}

	// OnResponse received the full response, including the tool call
	// and the token-usage breakdown.
	if respInfo.Response == nil {
		t.Fatal("ResponseInfo.Response is nil")
	}
	if respInfo.Response.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason = %q", respInfo.Response.StopReason)
	}
	var sawText, sawTool bool
	for _, b := range respInfo.Response.Content {
		switch b.Type {
		case llm.BlockText:
			sawText = b.Text == "calling get_weather"
		case llm.BlockToolUse:
			if b.ToolUse != nil && b.ToolUse.Name == "get_weather" && string(b.ToolUse.Input) == `{"city":"London"}` {
				sawTool = true
			}
		}
	}
	if !sawText || !sawTool {
		t.Errorf("Response.Content missing text/tool: sawText=%v sawTool=%v full=%+v",
			sawText, sawTool, respInfo.Response.Content)
	}
	if respInfo.Usage.PromptTokens != 12 || respInfo.Usage.CompletionTokens != 8 || respInfo.Usage.TotalTokens != 20 {
		t.Errorf("Usage = %+v, want {12,8,20,...}", respInfo.Usage)
	}
}

func TestEmbedHappyPath(t *testing.T) {
	setFake(&fakeProvider{
		embedVec: [][]float64{{0.1, 0.2}},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	body := bytes.NewReader([]byte(`{"provider":"fake","input":["hello"]}`))
	req := httptest.NewRequest(http.MethodPost, "/v1/embed", body)
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 1 || resp.Embeddings[0][0] != 0.1 {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestEmbedUnsupported(t *testing.T) {
	setFake(&fakeProvider{
		embedErr: &llm.ProviderError{Err: llm.ErrNotSupported, Message: "fake says no", Provider: "fake"},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed",
		bytes.NewReader([]byte(`{"provider":"fake","input":["x"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", rec.Code)
	}
}

func TestEmbedMultimodalHappyPath(t *testing.T) {
	setFake(&fakeProvider{
		multimodalVec: [][]float64{{0.5, 0.6}},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	body := bytes.NewReader([]byte(`{"provider":"fake","inputs":[{"content":[{"type":"text","text":"hi"}]}]}`))
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal", body)
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 1 || resp.Embeddings[0][0] != 0.5 {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestEmbedMultimodalUnsupportedReturns501(t *testing.T) {
	setFake(&fakeProvider{
		multimodalErr: &llm.ProviderError{Err: llm.ErrNotSupported, Message: "no", Provider: "fake"},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
		bytes.NewReader([]byte(`{"provider":"fake","inputs":[{"content":[{"type":"text","text":"hi"}]}]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", rec.Code)
	}
}

func TestRerankHappyPath(t *testing.T) {
	setFake(&fakeProvider{
		rerankResp: &llm.RerankResponse{
			Results: []llm.RerankResult{
				{Index: 1, RelevanceScore: 0.9},
				{Index: 0, RelevanceScore: 0.4},
			},
			Usage: llm.TokenUsage{TotalTokens: 7},
		},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	body := bytes.NewReader([]byte(`{
		"provider":"fake",
		"query":"q",
		"documents":["a","b"]
	}`))
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank", body)
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"results"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 2 || resp.Results[0].Index != 1 {
		t.Fatalf("unexpected response %s", rec.Body.String())
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("usage TotalTokens = %d", resp.Usage.TotalTokens)
	}
}

func TestRerankUnsupportedReturns501(t *testing.T) {
	setFake(&fakeProvider{
		rerankErr: &llm.ProviderError{Err: llm.ErrNotSupported, Message: "no", Provider: "fake"},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
		bytes.NewReader([]byte(`{"provider":"fake","query":"q","documents":["a"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", rec.Code)
	}
}

func TestModelsCapabilityFilter(t *testing.T) {
	setFake(&fakeProvider{
		modelInfos: []llm.ModelInfo{
			{ID: "chat-model", Capabilities: []llm.ModelCapability{llm.ModelCapabilityChat}},
			{ID: "embed-model", Capabilities: []llm.ModelCapability{llm.ModelCapabilityEmbeddings}},
		},
	})
	p := proxy.New(proxy.Config{
		Providers: map[string]llm.Options{"fake": {Model: "chat-model"}},
	})
	req := httptest.NewRequest(http.MethodGet, "/v1/models?provider=fake&metadata=true&capability=embeddings", nil)
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	var resp proxy.ModelsMetadataResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	for _, m := range resp.Models {
		found := false
		for _, c := range m.Capabilities {
			if c == llm.ModelCapabilityEmbeddings {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("model %s leaked through filter (caps=%v)", m.ID, m.Capabilities)
		}
	}
	if len(resp.Models) == 0 {
		t.Error("expected at least one embeddings model in response, got none")
	}
}

func TestHookStreamingResponseAssembledFromChunks(t *testing.T) {
	// The streaming OnResponse receives a *ChatResponse assembled from
	// SSE chunks — text deltas concatenated, tool-use deltas folded
	// into their start block — so consumers don't have to re-implement
	// Stream.Collect-style buffering in their hook.
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			chunks := make(chan llm.StreamChunk, 6)
			errCh := make(chan error, 1)
			chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "Hello "}
			chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "world."}
			chunks <- llm.StreamChunk{Type: llm.ChunkToolUseStart, ToolUse: &llm.ToolUse{ID: "tu_s", Name: "ping"}}
			chunks <- llm.StreamChunk{Type: llm.ChunkToolUseDelta, Partial: `{"hosts":[`}
			chunks <- llm.StreamChunk{Type: llm.ChunkToolUseDelta, Partial: `"a","b"]}`}
			chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 33}}
			close(chunks)
			close(errCh)
			return &llm.Stream{Chunks: chunks, Err: errCh}, nil
		},
	})

	var respInfo proxy.ResponseInfo
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "m"}},
		OnResponse:      func(_ *http.Request, info proxy.ResponseInfo) { respInfo = info },
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if respInfo.Response == nil {
		t.Fatal("streaming OnResponse did not receive a Response")
	}
	if len(respInfo.Response.Content) != 2 {
		t.Fatalf("expected 2 blocks (text + tool_use), got %+v", respInfo.Response.Content)
	}
	if respInfo.Response.Content[0].Text != "Hello world." {
		t.Errorf("text = %q, want %q", respInfo.Response.Content[0].Text, "Hello world.")
	}
	tu := respInfo.Response.Content[1].ToolUse
	if tu == nil || tu.Name != "ping" || string(tu.Input) != `{"hosts":["a","b"]}` {
		t.Errorf("tool-use block = %+v", respInfo.Response.Content[1])
	}
	if respInfo.Usage.TotalTokens != 33 {
		t.Errorf("Usage.TotalTokens = %d, want 33", respInfo.Usage.TotalTokens)
	}
}

func TestEmbedBadRequest(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed",
		bytes.NewReader([]byte(`{"provider":"fake","input":[]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (body=%s)", rec.Code, rec.Body.String())
	}
}

func TestEmbedMultimodalBadRequest(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
		bytes.NewReader([]byte(`{"provider":"fake","inputs":[]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (body=%s)", rec.Code, rec.Body.String())
	}
}

func TestRerankBadRequest(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
		bytes.NewReader([]byte(`{"provider":"fake","query":"q","documents":[]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (body=%s)", rec.Code, rec.Body.String())
	}
}

func newEmbedProxy(t *testing.T) http.Handler {
	t.Helper()
	setFake(&fakeProvider{embedVec: [][]float64{{0.1}}})
	return proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	}).Handler()
}

func TestEmbedInvalidJSON(t *testing.T) {
	h := newEmbedProxy(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/embed", bytes.NewReader([]byte(`{not json`)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestEmbedUnknownProvider(t *testing.T) {
	h := newEmbedProxy(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/embed",
		bytes.NewReader([]byte(`{"provider":"nope","input":["x"]}`)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestEmbedUpstreamError(t *testing.T) {
	setFake(&fakeProvider{embedErr: errors.New("upstream boom")})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed",
		bytes.NewReader([]byte(`{"provider":"fake","input":["x"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rec.Code)
	}
}

func TestEmbedRespectsModelOverride(t *testing.T) {
	setFake(&fakeProvider{embedVec: [][]float64{{0.1}}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "default-model"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed",
		bytes.NewReader([]byte(`{"provider":"fake","model":"override","input":["x"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	if got := fakeInstance.Model(); got != "override" {
		t.Errorf("model = %q, want %q", got, "override")
	}
}

func TestRerankInvalidJSON(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank", bytes.NewReader([]byte(`{not json`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestRerankUnknownProvider(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
		bytes.NewReader([]byte(`{"provider":"nope","query":"q","documents":["a"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestRerankUpstreamError(t *testing.T) {
	setFake(&fakeProvider{rerankErr: errors.New("upstream boom")})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
		bytes.NewReader([]byte(`{"provider":"fake","query":"q","documents":["a"]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rec.Code)
	}
}

func TestEmbedMultimodalInvalidJSON(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal", bytes.NewReader([]byte(`{not json`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestEmbedMultimodalUnknownProvider(t *testing.T) {
	setFake(&fakeProvider{})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
		bytes.NewReader([]byte(`{"provider":"nope","inputs":[{"content":[{"type":"text","text":"hi"}]}]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestEmbedMultimodalBadBase64(t *testing.T) {
	setFake(&fakeProvider{multimodalVec: [][]float64{{0.1}}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
		bytes.NewReader([]byte(`{"provider":"fake","inputs":[{"content":[{"type":"image_base64","image_base64":"!!!not-base64!!!"}]}]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (body=%s)", rec.Code, rec.Body.String())
	}
}

func TestEmbedMultimodalOversizedBase64(t *testing.T) {
	setFake(&fakeProvider{multimodalVec: [][]float64{{0.1}}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	// Build a base64 string whose decoded size exceeds 10 MiB. ~14 MiB of 'A'
	// characters base64-decodes to ~10.5 MiB.
	big := strings.Repeat("A", 14<<20)
	body := fmt.Sprintf(`{"provider":"fake","inputs":[{"content":[{"type":"image_base64","image_base64":%q}]}]}`, big)
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for oversized image, got %d", rec.Code)
	}
}

func TestEmbedMultimodalImageURLPath(t *testing.T) {
	// Exercises the no-base64 branch in the content-type loop.
	setFake(&fakeProvider{multimodalVec: [][]float64{{0.1}}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	body := `{"provider":"fake","inputs":[{"content":[{"type":"image_url","image_url":"https://example.com/x.jpg"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestEmbedMultimodalUpstreamError(t *testing.T) {
	setFake(&fakeProvider{multimodalErr: errors.New("upstream boom")})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
		bytes.NewReader([]byte(`{"provider":"fake","inputs":[{"content":[{"type":"text","text":"hi"}]}]}`)))
	rec := httptest.NewRecorder()
	p.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rec.Code)
	}
}

// TestNewAppliesConfigDefaults locks the default /v1 prefix behaviour
// before Task 8 makes the prefix configurable. It verifies that a
// Proxy created with an empty PathPrefix still serves /v1/health at
// the expected path.
func TestNewAppliesConfigDefaults(t *testing.T) {
	setFake(&fakeProvider{models: []string{"alpha"}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/v1/health status = %d, want 200", resp.StatusCode)
	}
}

// TestHandlerHonoursPathPrefix verifies that when PathPrefix is set to
// a non-default value, Handler() registers routes under that prefix and
// the old default prefix (/v1) returns 404.
func TestHandlerHonoursPathPrefix(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}})
	p := proxy.New(proxy.Config{
		PathPrefix:      "/api/llm",
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	r1, err := http.Get(srv.URL + "/api/llm/health")
	if err != nil {
		t.Fatal(err)
	}
	r1.Body.Close()
	if r1.StatusCode != http.StatusOK {
		t.Fatalf("/api/llm/health = %d, want 200", r1.StatusCode)
	}
	r2, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	r2.Body.Close()
	if r2.StatusCode != http.StatusNotFound {
		t.Fatalf("/v1/health = %d, want 404 when prefix overridden", r2.StatusCode)
	}
}

// TestHandlerWithSlashPrefixFallsBackToDefault verifies that a
// PathPrefix of "/" (or any all-slashes string) normalises to the
// default "/v1" and that /v1/health is reachable.
func TestHandlerWithSlashPrefixFallsBackToDefault(t *testing.T) {
	setFake(&fakeProvider{models: []string{"alpha"}})
	p := proxy.New(proxy.Config{
		PathPrefix:      "/",
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/v1/health with PathPrefix='/' status = %d, want 200", resp.StatusCode)
	}
}

// TestChatRejectsOversizedBody verifies that a MaxBodyBytes limit is
// enforced for POST /v1/chat, returning 400 when the body exceeds the
// configured ceiling.
func TestChatRejectsOversizedBody(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    64,
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	big := make([]byte, 0, 512)
	big = append(big, []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"`)...)
	for i := 0; i < 400; i++ {
		big = append(big, 'a')
	}
	big = append(big, []byte(`"}]}]}`)...)

	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for oversized body", resp.StatusCode)
	}
}

// TestChatStreamRejectsOversizedBody verifies the MaxBodyBytes limit on
// POST /v1/chat/stream.
func TestChatStreamRejectsOversizedBody(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    64,
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	big := make([]byte, 0, 512)
	big = append(big, []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"`)...)
	for i := 0; i < 400; i++ {
		big = append(big, 'a')
	}
	big = append(big, []byte(`"}]}]}`)...)

	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for oversized body", resp.StatusCode)
	}
}

// TestEmbedRejectsOversizedBody verifies the MaxBodyBytes limit on
// POST /v1/embed.
func TestEmbedRejectsOversizedBody(t *testing.T) {
	setFake(&fakeProvider{embedVec: [][]float64{{0.1}}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    64,
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	big := make([]byte, 0, 512)
	big = append(big, []byte(`{"provider":"fake","input":["`)...)
	for i := 0; i < 400; i++ {
		big = append(big, 'x')
	}
	big = append(big, []byte(`"]}`)...)

	resp, err := http.Post(srv.URL+"/v1/embed", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for oversized embed body", resp.StatusCode)
	}
}

// TestRerankRejectsOversizedBody verifies the MaxBodyBytes limit on
// POST /v1/rerank.
func TestRerankRejectsOversizedBody(t *testing.T) {
	setFake(&fakeProvider{rerankResp: &llm.RerankResponse{
		Results: []llm.RerankResult{{Index: 0, RelevanceScore: 0.9}},
	}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    64,
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	big := make([]byte, 0, 512)
	big = append(big, []byte(`{"provider":"fake","query":"q","documents":["`)...)
	for i := 0; i < 400; i++ {
		big = append(big, 'd')
	}
	big = append(big, []byte(`"]}`)...)

	resp, err := http.Post(srv.URL+"/v1/rerank", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for oversized rerank body", resp.StatusCode)
	}
}

// TestMaxBodyBytesZeroAllowsNormalRequest verifies that when MaxBodyBytes
// is 0 (unlimited), a normal-sized request succeeds with 200.
func TestMaxBodyBytesZeroAllowsNormalRequest(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    0,
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hello")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 when MaxBodyBytes=0 (unlimited)", resp.StatusCode)
	}
}

// TestMaxBodyBytesGenerousLimitAllowsNormalRequest verifies that a
// generous MaxBodyBytes limit does not block a normal-sized request.
func TestMaxBodyBytesGenerousLimitAllowsNormalRequest(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}})
	p := proxy.New(proxy.Config{
		MaxBodyBytes:    1 << 20, // 1 MiB — generous for a normal request
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()

	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{llm.UserText("hello")},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 with generous MaxBodyBytes", resp.StatusCode)
	}
}

// TestTransformRequestMutatesChat verifies the TransformRequest hook can
// rewrite the request and that the mutation reaches the provider's Chat
// call. The fake captures the dispatched request in f.chatReq.
func TestTransformRequestMutatesChat(t *testing.T) {
	f := &fakeProvider{chatResp: &llm.ChatResponse{
		Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, StopReason: "stop",
	}}
	setFake(f)
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		TransformRequest: func(_ *http.Request, req *llm.ChatRequest) error {
			req.SystemPrompt = "INJECTED"
			return nil
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	f.mu.RLock()
	got := f.chatReq
	f.mu.RUnlock()
	if got == nil {
		t.Fatal("provider did not receive a Chat request")
	}
	if got.SystemPrompt != "INJECTED" {
		t.Fatalf("provider saw SystemPrompt = %q, want INJECTED", got.SystemPrompt)
	}
}

type transformErr struct{ status int }

func (transformErr) Error() string     { return "denied" }
func (e transformErr) HTTPStatus() int { return e.status }

// TestTransformRequestRejectsWithStatus verifies a TransformRequest error
// carrying an HTTPStatus() method rejects the request with that status.
func TestTransformRequestRejectsWithStatus(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{StopReason: "stop"}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		TransformRequest: func(_ *http.Request, _ *llm.ChatRequest) error {
			return transformErr{status: http.StatusForbidden}
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", resp.StatusCode)
	}
}

// TestTransformRequestRejectsDefaultStatus verifies a plain error from
// TransformRequest (no HTTPStatus method) yields the default 400.
func TestTransformRequestRejectsDefaultStatus(t *testing.T) {
	setFake(&fakeProvider{chatResp: &llm.ChatResponse{StopReason: "stop"}})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		TransformRequest: func(_ *http.Request, _ *llm.ChatRequest) error {
			return errors.New("nope")
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat", "application/json", bytes.NewReader(body))
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

// TestTransformRequestMutatesStream verifies the hook also fires on the
// streaming path and that the mutation reaches ChatStream. The streamFn
// captures the request the proxy dispatched.
func TestTransformRequestMutatesStream(t *testing.T) {
	var mu sync.Mutex
	var seen string
	f := &fakeProvider{
		streamFn: func(_ context.Context, req llm.ChatRequest) (*llm.Stream, error) {
			mu.Lock()
			seen = req.SystemPrompt
			mu.Unlock()
			return nil, llm.ErrNotSupported
		},
	}
	setFake(f)
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		TransformRequest: func(_ *http.Request, req *llm.ChatRequest) error {
			req.SystemPrompt = "STREAM-INJECTED"
			return nil
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, err := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	mu.Lock()
	got := seen
	mu.Unlock()
	if got != "STREAM-INJECTED" {
		t.Fatalf("ChatStream saw SystemPrompt = %q, want STREAM-INJECTED", got)
	}
}

func TestHandleProvidersIncludesDisplayName(t *testing.T) {
	p := proxy.New(proxy.Config{
		DefaultProvider: "anthropic",
		Providers: map[string]llm.Options{
			"anthropic":   {Model: "claude"},
			"weirdcustom": {Model: "x"},
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/providers")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var pr proxy.ProvidersResponse
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		t.Fatal(err)
	}
	byName := map[string]proxy.ProviderInfo{}
	for _, info := range pr.Providers {
		byName[info.Name] = info
	}
	if byName["anthropic"].DisplayName != "Anthropic" {
		t.Fatalf("anthropic display = %q, want Anthropic", byName["anthropic"].DisplayName)
	}
	if byName["weirdcustom"].DisplayName != "weirdcustom" {
		t.Fatalf("unknown provider display name = %q, want raw name %q",
			byName["weirdcustom"].DisplayName, "weirdcustom")
	}
}

// TestTransformRequestStreamRejectsWithStatus verifies that a rejection
// before the stream starts yields a non-200 status on the streaming
// endpoint, because the transform runs before SSE headers are written.
func TestTransformRequestStreamRejectsWithStatus(t *testing.T) {
	setFake(&fakeProvider{
		streamFn: func(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
			return nil, llm.ErrNotSupported
		},
	})
	p := proxy.New(proxy.Config{
		DefaultProvider: "fake",
		Providers:       map[string]llm.Options{"fake": {Model: "alpha"}},
		TransformRequest: func(_ *http.Request, _ *llm.ChatRequest) error {
			return transformErr{status: http.StatusForbidden}
		},
	})
	srv := httptest.NewServer(p.Handler())
	defer srv.Close()
	body, _ := json.Marshal(proxy.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	resp, _ := http.Post(srv.URL+"/v1/chat/stream", "application/json", bytes.NewReader(body))
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", resp.StatusCode)
	}
	// The rejection happens before any SSE framing begins, so the
	// response must be a JSON error rather than an event stream.
	if ct := resp.Header.Get("Content-Type"); ct == "text/event-stream" {
		t.Fatalf("Content-Type = %q, want non-SSE (SSE framing must not begin)", ct)
	} else if ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
}
