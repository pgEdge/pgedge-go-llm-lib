//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package llm provides a unified Go interface for chatting with
// multiple large-language-model providers (Anthropic, OpenAI, Gemini,
// and Ollama) through a single Client interface.
//
// Provider packages register themselves at import time via init().
// Import either the convenience package llm/all (for all four built-in
// providers) or individual provider packages, then call NewClient to
// construct a client.
//
// The package surface includes streaming via Stream, tool calling via
// Tool/ToolUse, multimodal images via ImageBlock/ImageURLBlock,
// document input (e.g. PDFs) via DocumentBlock/DocumentURLBlock, JSON
// mode via ResponseFormat, retries via RetryConfig, and observability
// via OnRetry/Usage/Ping.
package llm

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// Client is the unified interface for interacting with LLM providers.
// All four built-in providers (Anthropic, OpenAI, Gemini, Ollama)
// implement this interface.
//
// Create a Client with NewClient; import provider packages (or
// llm/all) to register them.
type Client interface {
	// Chat sends a single-turn or multi-turn chat request and returns
	// the complete response. Use ChatStream for real-time streaming.
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// ChatStream sends a chat request and returns a Stream for
	// real-time chunk-by-chunk reading. Use Stream.Recv() to iterate,
	// or Stream.Collect() to drain and assemble a ChatResponse.
	ChatStream(ctx context.Context, req ChatRequest) (*Stream, error)

	// Embed generates an embedding vector for a single text string.
	// Returns ErrNotSupported for providers that do not support
	// embeddings (e.g. Anthropic).
	Embed(ctx context.Context, text string) ([]float64, error)

	// EmbedBatch generates embedding vectors for multiple texts.
	// The returned slice is in the same order as the input.
	// Returns ErrNotSupported for providers that do not support
	// embeddings.
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)

	// ListModels returns the names of chat-capable models available
	// from the provider. Each provider filters its list to relevant
	// models only.
	ListModels(ctx context.Context) ([]string, error)

	// ListModelsWithMetadata returns ModelInfo for each available
	// model, including context-window size, max output, capabilities,
	// and deprecation status. Fields are best-effort; providers
	// populate what their APIs expose.
	ListModelsWithMetadata(ctx context.Context) ([]ModelInfo, error)

	// Ping checks provider connectivity with a lightweight request
	// (typically a HEAD or models-list call). Returns nil if the
	// provider is reachable and the API key is valid.
	Ping(ctx context.Context) error

	// Provider returns the provider name (e.g. "anthropic", "openai").
	Provider() string

	// Model returns the model name configured in Options.
	Model() string

	// Usage returns cumulative token usage accumulated since the
	// client was created (or since the last ResetUsage call).
	Usage() TokenUsage

	// ResetUsage zeroes the cumulative token usage counter.
	ResetUsage()
}

// ProviderConstructor is a function that creates a Client for a
// given provider. Provider packages register one in their init()
// function via RegisterProvider.
type ProviderConstructor func(opts Options) (Client, error)

var (
	providersMu sync.RWMutex
	providers   = make(map[string]ProviderConstructor)
)

// RegisterProvider registers a provider constructor under the given
// name. This is called by provider init() functions. Registering the
// same name twice overwrites the previous constructor.
func RegisterProvider(name string, constructor ProviderConstructor) {
	providersMu.Lock()
	defer providersMu.Unlock()
	providers[name] = constructor
}

// RegisteredProviders returns the names of all currently-registered
// providers, sorted alphabetically. The returned slice is safe to
// mutate.
func RegisteredProviders() []string {
	providersMu.RLock()
	defer providersMu.RUnlock()
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// NewClient creates a new LLM client for the named provider.
// opts.WithDefaults() is applied before the provider constructor
// is called. Returns *ProviderError wrapping ErrInvalidRequest
// if provider is empty or not registered.
func NewClient(provider string, opts Options) (Client, error) {
	if provider == "" {
		return nil, &ProviderError{
			Err:     ErrInvalidRequest,
			Message: "provider name is required",
		}
	}

	providersMu.RLock()
	constructor, ok := providers[provider]
	providersMu.RUnlock()

	if !ok {
		return nil, &ProviderError{
			Err:      ErrInvalidRequest,
			Message:  fmt.Sprintf("unknown provider: %s", provider),
			Provider: provider,
		}
	}

	opts = opts.WithDefaults()
	return constructor(opts)
}
