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
	"sync"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// fakeProvider is a test double registered under the name "fake".
// Tests prepare it via setFake(...) before each request; subsequent
// tasks (8, 9) reuse it for chat and streaming behaviour.
//
// IMPORTANT: every test that drives the fake must call setFake at
// the top to reset its state. The proxy's per-request constructor
// will overwrite the model field with opts.Model, so anything you
// place in fakeProvider.model via setFake is replaced when the
// proxy creates its client — read fakeInstance.model AFTER the
// request to verify what the proxy passed in.
type fakeProvider struct {
	mu            sync.RWMutex
	model         string
	models        []string
	chatResp      *llm.ChatResponse
	chatErr       error
	streamFn      func(context.Context, llm.ChatRequest) (*llm.Stream, error)
	listModelsErr error // when set, ListModels and ListModelsWithMetadata return this error
	pingErr       error // when set, Ping returns this error
}

var (
	fakeMu       sync.Mutex
	fakeInstance *fakeProvider
)

func setFake(f *fakeProvider) {
	fakeMu.Lock()
	defer fakeMu.Unlock()
	fakeInstance = f
}

func init() {
	llm.RegisterProvider("fake", func(opts llm.Options) (llm.Client, error) {
		fakeMu.Lock()
		defer fakeMu.Unlock()
		if fakeInstance == nil {
			return nil, fmt.Errorf("fake provider not initialised; call setFake() in your test before driving the proxy")
		}
		fakeInstance.mu.Lock()
		fakeInstance.model = opts.Model
		fakeInstance.mu.Unlock()
		return fakeInstance, nil
	})
}

func (f *fakeProvider) Chat(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.chatErr != nil {
		return nil, f.chatErr
	}
	return f.chatResp, nil
}

func (f *fakeProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	f.mu.RLock()
	streamFn := f.streamFn
	f.mu.RUnlock()
	if streamFn != nil {
		return streamFn(ctx, req)
	}
	return nil, llm.ErrNotSupported
}

func (f *fakeProvider) Embed(context.Context, string) ([]float64, error) {
	return nil, llm.ErrNotSupported
}
func (f *fakeProvider) EmbedBatch(context.Context, []string) ([][]float64, error) {
	return nil, llm.ErrNotSupported
}
func (f *fakeProvider) ListModels(_ context.Context, _ ...llm.ListModelsOption) ([]string, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.listModelsErr != nil {
		return nil, f.listModelsErr
	}
	return f.models, nil
}

func (f *fakeProvider) ListModelsWithMetadata(_ context.Context, _ ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.listModelsErr != nil {
		return nil, f.listModelsErr
	}
	out := make([]llm.ModelInfo, len(f.models))
	for i, m := range f.models {
		out[i] = llm.ModelInfo{ID: m, Capabilities: []llm.ModelCapability{llm.ModelCapabilityChat}}
	}
	return out, nil
}
func (f *fakeProvider) Provider() string { return "fake" }
func (f *fakeProvider) Model() string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.model
}
func (f *fakeProvider) Usage() llm.TokenUsage { return llm.TokenUsage{} }

func (f *fakeProvider) Ping(context.Context) error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.pingErr
}

func (f *fakeProvider) ResetUsage() {
	f.mu.Lock()
	defer f.mu.Unlock()
	// fakeProvider doesn't track usage; nothing to reset.
}

func (f *fakeProvider) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "fake does not support reranking",
		Provider: "fake",
	}
}
