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
	"encoding/base64"
	"errors"
	"fmt"
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
	c.usage.Add(u)
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

// ---------- Embed / EmbedBatch ----------

type embeddingsRequest struct {
	Input           []string `json:"input"`
	Model           string   `json:"model"`
	InputType       string   `json:"input_type,omitempty"`
	Truncation      *bool    `json:"truncation,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	OutputDtype     string   `json:"output_dtype,omitempty"`
}

type embeddingsResponseDatum struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingsResponse struct {
	Data  []embeddingsResponseDatum `json:"data"`
	Model string                    `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (c *client) Embed(ctx context.Context, text string) ([]float64, error) {
	vecs, err := c.embed(ctx, []string{text}, c.model, nil)
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "empty embedding response", Provider: providerName}
	}
	return vecs[0], nil
}

func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	return c.embed(ctx, texts, c.model, nil)
}

func (c *client) embed(ctx context.Context, texts []string, model string, ext *Extension) ([][]float64, error) {
	if model == "" {
		return nil, &llm.ProviderError{
			Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName,
		}
	}
	req := embeddingsRequest{Input: texts, Model: model}
	if ext != nil {
		req.InputType = string(ext.InputType)
		req.Truncation = ext.Truncation
		req.OutputDimension = ext.OutputDimension
		req.OutputDtype = string(ext.OutputDtype)
	}
	var resp embeddingsResponse
	if err := c.postJSON(ctx, "/embeddings", req, &resp); err != nil {
		return nil, err
	}
	out := make([][]float64, len(texts))
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "embedding index out of range", Provider: providerName}
		}
		out[d.Index] = d.Embedding
	}
	c.addUsage(llm.TokenUsage{TotalTokens: resp.Usage.TotalTokens})
	return out, nil
}

type multimodalContentWire struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
}

type multimodalInputWire struct {
	Content []multimodalContentWire `json:"content"`
}

type multimodalEmbeddingsRequest struct {
	Inputs          []multimodalInputWire `json:"inputs"`
	Model           string                `json:"model"`
	InputType       string                `json:"input_type,omitempty"`
	Truncation      *bool                 `json:"truncation,omitempty"`
	OutputDimension int                   `json:"output_dimension,omitempty"`
	OutputDtype     string                `json:"output_dtype,omitempty"`
}

func (c *client) EmbedMultimodal(ctx context.Context, req llm.MultimodalEmbedRequest) ([][]float64, error) {
	if c.model == "" {
		return nil, &llm.ProviderError{Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName}
	}
	wireInputs := make([]multimodalInputWire, len(req.Inputs))
	for i, in := range req.Inputs {
		wireContent := make([]multimodalContentWire, len(in.Content))
		for j, mc := range in.Content {
			wireContent[j] = contentToWire(mc)
		}
		wireInputs[i] = multimodalInputWire{Content: wireContent}
	}
	wire := multimodalEmbeddingsRequest{Inputs: wireInputs, Model: c.model}
	if ext := findExtension(req.Extensions); ext != nil {
		wire.InputType = string(ext.InputType)
		wire.Truncation = ext.Truncation
		wire.OutputDimension = ext.OutputDimension
		wire.OutputDtype = string(ext.OutputDtype)
	}
	var resp embeddingsResponse
	if err := c.postJSON(ctx, "/multimodalembeddings", wire, &resp); err != nil {
		return nil, err
	}
	out := make([][]float64, len(req.Inputs))
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "embedding index out of range", Provider: providerName}
		}
		out[d.Index] = d.Embedding
	}
	c.addUsage(llm.TokenUsage{TotalTokens: resp.Usage.TotalTokens})
	return out, nil
}

func contentToWire(mc llm.MultimodalContent) multimodalContentWire {
	switch mc.Type {
	case llm.MultimodalContentText:
		return multimodalContentWire{Type: "text", Text: mc.Text}
	case llm.MultimodalContentImageURL:
		return multimodalContentWire{Type: "image_url", ImageURL: mc.ImageURL}
	case llm.MultimodalContentImageData:
		return multimodalContentWire{Type: "image_base64", ImageBase64: base64.StdEncoding.EncodeToString(mc.ImageData)}
	default:
		return multimodalContentWire{}
	}
}

type rerankRequestWire struct {
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	Model           string   `json:"model"`
	TopK            *int     `json:"top_k,omitempty"`
	ReturnDocuments *bool    `json:"return_documents,omitempty"`
	Truncation      *bool    `json:"truncation,omitempty"`
}

type rerankResponseWire struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
		Document       string  `json:"document,omitempty"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (c *client) Rerank(ctx context.Context, req llm.RerankRequest) (*llm.RerankResponse, error) {
	if c.model == "" {
		return nil, &llm.ProviderError{Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName}
	}
	wire := rerankRequestWire{Query: req.Query, Documents: req.Documents, Model: c.model, TopK: req.TopK}
	if ext := findExtension(req.Extensions); ext != nil {
		wire.ReturnDocuments = ext.ReturnDocuments
		wire.Truncation = ext.Truncation
	}
	var raw rerankResponseWire
	if err := c.postJSON(ctx, "/rerank", wire, &raw); err != nil {
		return nil, err
	}
	out := &llm.RerankResponse{
		Results: make([]llm.RerankResult, len(raw.Data)),
		Usage:   llm.TokenUsage{TotalTokens: raw.Usage.TotalTokens},
	}
	for i, d := range raw.Data {
		out.Results[i] = llm.RerankResult{Index: d.Index, RelevanceScore: d.RelevanceScore, Document: d.Document}
	}
	c.addUsage(out.Usage)
	return out, nil
}

// ---------- HTTP helpers ----------

func (c *client) postJSON(ctx context.Context, path string, body, out any) error {
	headers := map[string]string{
		"Authorization": "Bearer " + c.apiKey,
		"Content-Type":  "application/json",
	}
	status, respBody, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+path, headers, body, out)
	if err != nil {
		return &llm.ProviderError{Err: llm.ErrProviderError, Message: err.Error(), Provider: providerName, StatusCode: status}
	}
	if status < 200 || status >= 300 {
		return &llm.ProviderError{
			Err:        statusToErr(status),
			Message:    fmt.Sprintf("voyage: HTTP %d: %s", status, string(respBody)),
			Provider:   providerName,
			StatusCode: status,
		}
	}
	return nil
}

func statusToErr(status int) error {
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return llm.ErrAuthentication
	case status == http.StatusTooManyRequests:
		return llm.ErrRateLimit
	case status == http.StatusBadRequest:
		return llm.ErrInvalidRequest
	default:
		return llm.ErrProviderError
	}
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
