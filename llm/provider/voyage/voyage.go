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
	"errors"
	"net/http"
	"os"
	"sync"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

const (
	defaultBaseURL = "https://api.voyageai.com/v1"
	providerName   = "voyage"
)

// New constructs a Voyage client. Registered with llm.RegisterProvider
// at package init time, so callers normally invoke
// llm.NewClient("voyage", opts).
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("VOYAGE_API_KEY")
	}
	if opts.APIKey == "" {
		return nil, &llm.ProviderError{
			Err:      llm.ErrInvalidRequest,
			Message:  "Voyage API key is required (Options.APIKey or VOYAGE_API_KEY)",
			Provider: providerName,
		}
	}

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		var err error
		baseURL, err = httpclient.ValidateBaseURL(baseURL, providerName)
		if err != nil {
			return nil, err
		}
	}

	retryCfg := httpclient.RetryConfig{
		MaxRetries:     opts.Retry.MaxRetries,
		InitialBackoff: opts.Retry.InitialBackoff,
		MaxBackoff:     opts.Retry.MaxBackoff,
		Disabled:       opts.Retry.Disabled,
	}
	if opts.OnRetry != nil {
		hook := opts.OnRetry
		retryCfg.OnRetry = func(e httpclient.RetryEvent) {
			hook(llm.RetryEvent{
				Attempt:    e.Attempt,
				StatusCode: e.StatusCode,
				Err:        e.Err,
				Wait:       e.Wait,
			})
		}
	}

	return &client{
		httpClient: httpclient.New(opts.HTTPClient, opts.CustomHeaders, retryCfg, opts.RequestTimeout),
		apiKey:     opts.APIKey,
		baseURL:    baseURL,
		model:      opts.Model,
		opts:       opts,
	}, nil
}

func init() {
	llm.RegisterProvider(providerName, New)
}

type client struct {
	httpClient *http.Client
	apiKey     string
	baseURL    string
	model      string
	opts       llm.Options

	usageMu sync.Mutex
	usage   llm.TokenUsage
}

func (c *client) Provider() string { return providerName }
func (c *client) Model() string    { return c.model }

func (c *client) Usage() llm.TokenUsage {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	return c.usage
}

func (c *client) ResetUsage() {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	c.usage = llm.TokenUsage{}
}

func (c *client) addUsage(u llm.TokenUsage) {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	c.usage.PromptTokens += u.PromptTokens
	c.usage.CompletionTokens += u.CompletionTokens
	c.usage.TotalTokens += u.TotalTokens
}

// ---------- Chat (not supported) ----------

func (c *client) Chat(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Voyage does not support chat completions",
		Provider: providerName,
	}
}

func (c *client) ChatStream(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Voyage does not support chat completions",
		Provider: providerName,
	}
}

// ---------- Ping ----------
//
// Voyage exposes no dedicated ping endpoint. We send a minimal one-token
// embeddings request to the configured model (or voyage-3.5-lite if no
// model is configured). A 2xx response is success; an auth failure
// (401 / 403) propagates as an error; any other 4xx is treated as
// reachable (the API answered, just rejected this particular request).
func (c *client) Ping(ctx context.Context) error {
	model := c.model
	if model == "" {
		model = "voyage-3.5-lite"
	}
	_, err := c.embed(ctx, []string{"."}, model, nil)
	if err == nil {
		return nil
	}
	var pe *llm.ProviderError
	if errors.As(err, &pe) {
		if pe.StatusCode == http.StatusUnauthorized || pe.StatusCode == http.StatusForbidden {
			return err
		}
		if pe.StatusCode >= 400 && pe.StatusCode < 500 {
			return nil
		}
	}
	return err
}

// ---------- ListModels ----------

func (c *client) ListModels(ctx context.Context, opts ...llm.ListModelsOption) ([]string, error) {
	infos, err := c.ListModelsWithMetadata(ctx, opts...)
	if err != nil {
		return nil, err
	}
	names := make([]string, len(infos))
	for i, info := range infos {
		names[i] = info.ID
	}
	return names, nil
}

func (c *client) ListModelsWithMetadata(_ context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	cfg := llm.ListModelsConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return llm.FilterModelInfos(voyageCatalog(), cfg), nil
}

// voyageCatalog returns the hard-coded list of Voyage models. Voyage
// exposes no /models endpoint, so we maintain this list manually.
func voyageCatalog() []llm.ModelInfo {
	embed := []llm.ModelCapability{llm.ModelCapabilityEmbeddings}
	multimodal := []llm.ModelCapability{llm.ModelCapabilityEmbeddings, llm.ModelCapabilityMultimodalEmbeddings}
	rerank := []llm.ModelCapability{llm.ModelCapabilityReranking}
	return []llm.ModelInfo{
		{ID: "voyage-3.5", Capabilities: embed, ContextWindow: 32000},
		{ID: "voyage-3.5-lite", Capabilities: embed, ContextWindow: 32000},
		{ID: "voyage-3-large", Capabilities: embed, ContextWindow: 32000},
		{ID: "voyage-code-3", Capabilities: embed, ContextWindow: 32000},
		{ID: "voyage-finance-2", Capabilities: embed, ContextWindow: 32000},
		{ID: "voyage-law-2", Capabilities: embed, ContextWindow: 16000},
		{ID: "voyage-multimodal-3", Capabilities: multimodal, ContextWindow: 32000},
		{ID: "rerank-2.5", Capabilities: rerank, ContextWindow: 8000},
		{ID: "rerank-2.5-lite", Capabilities: rerank, ContextWindow: 8000},
	}
}

// embed, Embed, EmbedBatch, EmbedMultimodal, Rerank are filled in by
// later tasks. The stubs below keep the package compiling so the
// skeleton's tests can run.

func (c *client) embed(_ context.Context, _ []string, _ string, _ *Extension) ([][]float64, error) {
	return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: embed not implemented yet", Provider: providerName}
}

func (c *client) Embed(_ context.Context, _ string) ([]float64, error) {
	return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: Embed not implemented yet", Provider: providerName}
}

func (c *client) EmbedBatch(_ context.Context, _ []string) ([][]float64, error) {
	return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: EmbedBatch not implemented yet", Provider: providerName}
}

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: EmbedMultimodal not implemented yet", Provider: providerName}
}

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: Rerank not implemented yet", Provider: providerName}
}

// findExtension locates a voyage.Extension in a generic
// []llm.ProviderExtension. (llm.FindExtension only operates on
// ChatRequest; we have multiple non-chat request types that need this.)
func findExtension(exts []llm.ProviderExtension) *Extension {
	for _, e := range exts {
		if e.ProviderName() == providerName {
			if ext, ok := e.(Extension); ok {
				return &ext
			}
			if ext, ok := e.(*Extension); ok {
				return ext
			}
		}
	}
	return nil
}
