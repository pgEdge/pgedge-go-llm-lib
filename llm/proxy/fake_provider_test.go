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
	modelInfos    []llm.ModelInfo // when set, ListModelsWithMetadata returns these (ignoring models)
	chatResp      *llm.ChatResponse
	chatErr       error
	streamFn      func(context.Context, llm.ChatRequest) (*llm.Stream, error)
	listModelsErr error               // when set, ListModels and ListModelsWithMetadata return this error
	pingErr       error               // when set, Ping returns this error
	embedVec      [][]float64         // when set, Embed/EmbedBatch return this
	embedErr      error               // when set, Embed/EmbedBatch return this error
	multimodalVec [][]float64         // when set, EmbedMultimodal returns this
	multimodalErr error               // when set, EmbedMultimodal returns this error
	rerankResp    *llm.RerankResponse // when set, Rerank returns this
	rerankErr     error               // when set, Rerank returns this error
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

func (f *fakeProvider) Embed(_ context.Context, text string) ([]float64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.embedErr != nil {
		return nil, f.embedErr
	}
	if len(f.embedVec) > 0 {
		return f.embedVec[0], nil
	}
	return nil, llm.ErrNotSupported
}

func (f *fakeProvider) EmbedBatch(_ context.Context, _ []string) ([][]float64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.embedErr != nil {
		return nil, f.embedErr
	}
	if f.embedVec != nil {
		return f.embedVec, nil
	}
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

func (f *fakeProvider) ListModelsWithMetadata(_ context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.listModelsErr != nil {
		return nil, f.listModelsErr
	}
	var infos []llm.ModelInfo
	if len(f.modelInfos) > 0 {
		infos = f.modelInfos
	} else {
		infos = make([]llm.ModelInfo, len(f.models))
		for i, m := range f.models {
			infos[i] = llm.ModelInfo{ID: m, Capabilities: []llm.ModelCapability{llm.ModelCapabilityChat}}
		}
	}
	var cfg llm.ListModelsConfig
	for _, o := range opts {
		o(&cfg)
	}
	return llm.FilterModelInfos(infos, cfg), nil
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
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.rerankErr != nil {
		return nil, f.rerankErr
	}
	if f.rerankResp != nil {
		return f.rerankResp, nil
	}
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "fake does not support reranking",
		Provider: "fake",
	}
}

func (f *fakeProvider) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.multimodalErr != nil {
		return nil, f.multimodalErr
	}
	if f.multimodalVec != nil {
		return f.multimodalVec, nil
	}
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "fake does not support multimodal embeddings",
		Provider: "fake",
	}
}
