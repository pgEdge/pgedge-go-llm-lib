//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package typesafe implements the TypeSafe provider, which serves the
// Jev decision models. It supports Decide, model listing and Ping only.
package typesafe

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/redact"
)

const (
	defaultBaseURL = "https://api.typesafe.ai/v1"
	providerName   = "typesafe"
)

// DefaultModel is the Jev model used when neither DecisionRequest.Model
// nor Options.Model is set.
const DefaultModel = "jev-latest"

// New constructs a TypeSafe client. Registered with llm.RegisterProvider
// at package init time, so callers normally invoke
// llm.NewClient("typesafe", opts).
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("TYPESAFE_API_KEY")
	}
	if opts.APIKey == "" {
		return nil, &llm.ProviderError{
			Err:      llm.ErrInvalidRequest,
			Message:  "TypeSafe API key is required (Options.APIKey, Options.APIKeyFile or TYPESAFE_API_KEY)",
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

		PerAttemptTimeout: opts.PerAttemptTimeout,
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
	c.usage.Add(u)
}

// ---------- Unsupported methods ----------

func (c *client) notSupported(what string) error {
	return &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "TypeSafe does not support " + what,
		Provider: providerName,
	}
}

func (c *client) Chat(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	return nil, c.notSupported("chat completions")
}

func (c *client) ChatStream(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
	return nil, c.notSupported("chat completions")
}

func (c *client) Embed(_ context.Context, _ string) ([]float64, error) {
	return nil, c.notSupported("embeddings")
}

func (c *client) EmbedBatch(_ context.Context, _ []string) ([][]float64, error) {
	return nil, c.notSupported("embeddings")
}

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, c.notSupported("multimodal embeddings")
}

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, c.notSupported("reranking")
}

// ---------- Models and ping ----------

type modelsResponse struct {
	Models []struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		ReleaseDate string `json:"release_date"`
	} `json:"models"`
}

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

// ListModelsWithMetadata lists the Jev models from GET /models. Every
// TypeSafe model is a decision model. llm.ModelInfo has no description
// or release date field, so those are not carried over.
func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	cfg := llm.ListModelsConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	var resp modelsResponse
	if err := c.doJSON(ctx, http.MethodGet, "/models", nil, &resp); err != nil {
		return nil, err
	}
	infos := make([]llm.ModelInfo, 0, len(resp.Models))
	for _, m := range resp.Models {
		infos = append(infos, llm.ModelInfo{
			ID:           m.Name,
			Capabilities: []llm.ModelCapability{llm.ModelCapabilityDecisions},
		})
	}
	return llm.FilterModelInfos(infos, cfg), nil
}

// Ping lists the models, which needs a valid key and costs no tokens.
func (c *client) Ping(ctx context.Context) error {
	_, err := c.ListModelsWithMetadata(ctx)
	return err
}

// ---------- HTTP helpers ----------

func (c *client) doJSON(ctx context.Context, method, path string, body, out any) error {
	headers := map[string]string{"Authorization": "Bearer " + c.apiKey}
	status, respBody, err := httpclient.DoJSON(ctx, c.httpClient, method, c.baseURL+path, headers, body, out)
	if err != nil {
		return &llm.ProviderError{
			Err:        llm.ErrProviderError,
			Message:    redact.Message(err.Error(), c.apiKey),
			Provider:   providerName,
			StatusCode: status,
		}
	}
	if status < 200 || status >= 300 {
		return &llm.ProviderError{
			Err:        statusToErr(status),
			Message:    c.errorMessage(status, respBody),
			Provider:   providerName,
			StatusCode: status,
		}
	}
	return nil
}

// errorResponse covers TypeSafe's error bodies, whose "detail" is a
// string, an object with a message, or (for request validation
// failures) a list of {loc, msg} entries.
type errorResponse struct {
	Detail json.RawMessage `json:"detail"`
}

type errorDetail struct {
	Message string `json:"message"`
}

type validationDetail struct {
	Loc []any  `json:"loc"`
	Msg string `json:"msg"`
}

// errorMessage builds the message for a non-2xx response. As in the
// voyage provider, a body we cannot parse is never quoted, and any
// text we do use is redacted.
func (c *client) errorMessage(status int, respBody []byte) string {
	var resp errorResponse
	_ = json.Unmarshal(respBody, &resp) // best-effort; fall back to the status below
	detail := redact.Message(detailText(resp.Detail), c.apiKey)
	if detail == "" {
		return fmt.Sprintf("typesafe: HTTP %d", status)
	}
	return fmt.Sprintf("typesafe: HTTP %d: %s", status, detail)
}

func detailText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj errorDetail
	if json.Unmarshal(raw, &obj) == nil && obj.Message != "" {
		return obj.Message
	}
	var list []validationDetail
	if json.Unmarshal(raw, &list) == nil && len(list) > 0 {
		parts := make([]string, len(list[0].Loc))
		for i, p := range list[0].Loc {
			parts[i] = fmt.Sprint(p)
		}
		if len(parts) == 0 {
			return list[0].Msg
		}
		return strings.Join(parts, ".") + ": " + list[0].Msg
	}
	return ""
}

func statusToErr(status int) error {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return llm.ErrAuthentication
	case http.StatusTooManyRequests:
		return llm.ErrRateLimit
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return llm.ErrInvalidRequest
	default:
		return llm.ErrProviderError
	}
}
