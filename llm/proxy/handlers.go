//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package proxy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// providerDisplayNames maps well-known provider registration names to
// human-readable labels for use in UI pickers and similar surfaces.
var providerDisplayNames = map[string]string{
	"anthropic": "Anthropic",
	"openai":    "OpenAI",
	"gemini":    "Google Gemini",
	"ollama":    "Ollama",
	"voyage":    "Voyage AI",
}

// displayNameFor returns a human-readable label for a provider, falling back
// to the raw registration name for unrecognised providers.
func displayNameFor(name string) string {
	if d, ok := providerDisplayNames[name]; ok {
		return d
	}
	return name
}

func (p *Proxy) handleProviders(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}

	infos := make([]ProviderInfo, 0, len(p.cfg.Providers))
	for name, opts := range p.cfg.Providers {
		infos = append(infos, ProviderInfo{
			Name:        name,
			DisplayName: displayNameFor(name),
			Model:       opts.Model,
			Default:     name == p.cfg.DefaultProvider,
		})
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })

	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, ProvidersResponse{
		Providers:       infos,
		DefaultProvider: p.cfg.DefaultProvider,
	}); err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
	}
}

func (p *Proxy) handleModels(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}

	provider := r.URL.Query().Get("provider")
	if provider == "" {
		p.writeError(w, r, ErrorInfo{
			StatusCode: http.StatusBadRequest,
			Err:        fmt.Errorf("missing required query parameter 'provider'"),
			RequestID:  reqID,
		})
		return
	}

	opts, ok := p.cfg.Providers[provider]
	if !ok {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			StatusCode: http.StatusBadRequest,
			Err:        fmt.Errorf("provider %q is not configured", provider),
			RequestID:  reqID,
		})
		return
	}

	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			StatusCode: http.StatusInternalServerError,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	var listOpts []llm.ListModelsOption
	if caps := r.URL.Query()["capability"]; len(caps) > 0 {
		typed := make([]llm.ModelCapability, 0, len(caps))
		for _, c := range caps {
			typed = append(typed, llm.ModelCapability(c))
		}
		listOpts = append(listOpts, llm.WithCapabilities(typed...))
	}

	if r.URL.Query().Get("metadata") == "true" {
		models, mdErr := client.ListModelsWithMetadata(r.Context(), listOpts...)
		if mdErr != nil {
			p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadGateway, Err: mdErr, RequestID: reqID})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if encErr := encodeJSON(w, ModelsMetadataResponse{Models: models}); encErr != nil {
			p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: encErr, RequestID: reqID})
		}
		return
	}

	models, err := client.ListModels(r.Context(), listOpts...)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			StatusCode: http.StatusBadGateway,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, ModelsResponse{Models: models}); err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			StatusCode: http.StatusInternalServerError,
			Err:        err,
			RequestID:  reqID,
		})
	}
}

func (p *Proxy) handleChat(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}
	if p.cfg.MaxBodyBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, p.cfg.MaxBodyBytes)
	}

	req, opts, provider, model, err := p.parseChatRequest(r)
	if err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}

	llmReq := buildLLMRequest(req)
	if !p.transform(w, r, ErrorInfo{Provider: provider, Model: model, Stream: false, RequestID: reqID}, &llmReq) {
		return
	}
	if p.cfg.OnRequest != nil {
		p.cfg.OnRequest(r, RequestInfo{
			Provider:  provider,
			Model:     model,
			Stream:    false,
			RequestID: reqID,
			Request:   &llmReq,
		})
	}

	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     false,
			StatusCode: http.StatusInternalServerError,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	start := time.Now()
	chatResp, err := client.Chat(r.Context(), llmReq)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     false,
			StatusCode: http.StatusBadGateway,
			Err:        err,
			RequestID:  reqID,
			Duration:   time.Since(start),
		})
		return
	}
	dur := time.Since(start)

	out := ChatResponse{
		Content:    chatResp.Content,
		StopReason: chatResp.StopReason,
		Usage:      chatResp.Usage,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, out); err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     false,
			StatusCode: http.StatusInternalServerError,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	if p.cfg.OnResponse != nil {
		p.cfg.OnResponse(r, ResponseInfo{
			Provider:   provider,
			Model:      model,
			Stream:     false,
			Usage:      chatResp.Usage,
			Response:   chatResp,
			StatusCode: http.StatusOK,
			RequestID:  reqID,
			Duration:   dur,
		})
	}
}

// buildLLMRequest projects the proxy wire request onto the library's
// llm.ChatRequest. Shared by handleChat and handleChatStream.
func buildLLMRequest(req ChatRequest) llm.ChatRequest {
	return llm.ChatRequest{
		Messages:         req.Messages,
		Tools:            req.Tools,
		SystemPrompt:     req.SystemPrompt,
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		ResponseFormat:   req.ResponseFormat,
		ToolChoice:       req.ToolChoice,
		StopSequences:    req.StopSequences,
		ToolDescriptions: req.ToolDescriptions,
	}
}

// parseChatRequest decodes the request body and resolves the provider
// and model, applying per-request overrides on top of the configured
// defaults. The returned llm.Options has Model already set to the
// effective model.
func (p *Proxy) parseChatRequest(r *http.Request) (ChatRequest, llm.Options, string, string, error) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return req, llm.Options{}, "", "", fmt.Errorf("invalid request body: %w", err)
	}

	provider := req.Provider
	if provider == "" {
		provider = p.cfg.DefaultProvider
	}
	if provider == "" {
		return req, llm.Options{}, "", "", fmt.Errorf("no provider specified and no default configured")
	}

	opts, ok := p.cfg.Providers[provider]
	if !ok {
		return req, llm.Options{}, "", "", fmt.Errorf("provider %q is not configured", provider)
	}

	if req.Model != "" {
		opts.Model = req.Model
	}
	return req, opts, provider, opts.Model, nil
}

func (p *Proxy) handleHealth(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}

	out := HealthResponse{Providers: map[string]ProviderHealth{}}
	allOK := true

	for name, opts := range p.cfg.Providers {
		c, err := llm.NewClient(name, opts)
		if err != nil {
			out.Providers[name] = ProviderHealth{Status: "down", Error: err.Error()}
			allOK = false
			continue
		}
		// Use a short-deadline context so a hanging provider doesn't
		// block the health check.
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		err = c.Ping(ctx)
		cancel()
		if err != nil {
			out.Providers[name] = ProviderHealth{Status: "down", Error: err.Error()}
			allOK = false
		} else {
			out.Providers[name] = ProviderHealth{Status: "ok"}
		}
	}

	if allOK {
		out.Status = "ok"
	} else {
		out.Status = "degraded"
	}

	w.Header().Set("Content-Type", "application/json")
	if !allOK {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	_ = encodeJSON(w, out)
}

func (p *Proxy) handleChatStream(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}
	if p.cfg.MaxBodyBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, p.cfg.MaxBodyBytes)
	}

	req, opts, provider, model, err := p.parseChatRequest(r)
	if err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}

	llmReq := buildLLMRequest(req)
	if !p.transform(w, r, ErrorInfo{Provider: provider, Model: model, Stream: true, RequestID: reqID}, &llmReq) {
		return
	}
	if p.cfg.OnRequest != nil {
		p.cfg.OnRequest(r, RequestInfo{
			Provider:  provider,
			Model:     model,
			Stream:    true,
			RequestID: reqID,
			Request:   &llmReq,
		})
	}

	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     true,
			StatusCode: http.StatusInternalServerError,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	start := time.Now()
	stream, err := client.ChatStream(r.Context(), llmReq)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     true,
			StatusCode: http.StatusBadGateway,
			Err:        err,
			RequestID:  reqID,
			Duration:   time.Since(start),
		})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	streamResp, streamErr := writeSSE(w, stream)
	dur := time.Since(start)

	if p.cfg.OnResponse != nil {
		status := http.StatusOK
		if streamErr != nil {
			status = http.StatusBadGateway
		}
		p.cfg.OnResponse(r, ResponseInfo{
			Provider:   provider,
			Model:      model,
			Stream:     true,
			Usage:      streamResp.Usage,
			StatusCode: status,
			RequestID:  reqID,
			Response:   streamResp,
			Duration:   dur,
		})
	}

	if streamErr != nil && p.cfg.OnError != nil {
		p.cfg.OnError(r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     true,
			StatusCode: http.StatusBadGateway,
			Err:        streamErr,
			RequestID:  reqID,
			Duration:   dur,
		})
	}
}

func (p *Proxy) handleEmbed(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}
	if p.cfg.MaxBodyBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, p.cfg.MaxBodyBytes)
	}
	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}
	if len(req.Input) == 0 {
		p.writeError(w, r, ErrorInfo{
			StatusCode: http.StatusBadRequest,
			Err:        fmt.Errorf("input must contain at least one string"),
			RequestID:  reqID,
		})
		return
	}
	provider := req.Provider
	if provider == "" {
		provider = p.cfg.DefaultProvider
	}
	opts, ok := p.cfg.Providers[provider]
	if !ok {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
			Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
		return
	}
	if req.Model != "" {
		opts.Model = req.Model
	}
	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
		return
	}
	start := time.Now()
	vecs, err := client.EmbedBatch(r.Context(), req.Input)
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, llm.ErrNotSupported) {
			status = http.StatusNotImplemented
		}
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID, Duration: time.Since(start)})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, EmbedResponse{Embeddings: vecs}); err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
	}
}

func (p *Proxy) handleRerank(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}
	if p.cfg.MaxBodyBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, p.cfg.MaxBodyBytes)
	}
	var req RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}
	if req.Query == "" || len(req.Documents) == 0 {
		p.writeError(w, r, ErrorInfo{
			StatusCode: http.StatusBadRequest,
			Err:        fmt.Errorf("query and documents are required"),
			RequestID:  reqID,
		})
		return
	}
	provider := req.Provider
	if provider == "" {
		provider = p.cfg.DefaultProvider
	}
	opts, ok := p.cfg.Providers[provider]
	if !ok {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
			Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
		return
	}
	if req.Model != "" {
		opts.Model = req.Model
	}
	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
		return
	}
	start := time.Now()
	libResp, err := client.Rerank(r.Context(), llm.RerankRequest{
		Query: req.Query, Documents: req.Documents, TopK: req.TopK,
	})
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, llm.ErrNotSupported) {
			status = http.StatusNotImplemented
		}
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID, Duration: time.Since(start)})
		return
	}
	out := RerankResponse{Usage: RerankUsage{TotalTokens: libResp.Usage.TotalTokens}}
	out.Results = make([]RerankResult, len(libResp.Results))
	for i, res := range libResp.Results {
		out.Results[i] = RerankResult{Index: res.Index, RelevanceScore: res.RelevanceScore, Document: res.Document}
	}
	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, out); err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
	}
}

const (
	maxEmbedMultimodalBodyBytes = 16 << 20 // 16 MiB
	maxDecodedImageBytes        = 10 << 20 // 10 MiB per image
)

func (p *Proxy) handleEmbedMultimodal(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r, ErrorInfo{RequestID: reqID}) {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxEmbedMultimodalBodyBytes)
	var req EmbedMultimodalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}
	if len(req.Inputs) == 0 {
		p.writeError(w, r, ErrorInfo{
			StatusCode: http.StatusBadRequest,
			Err:        fmt.Errorf("inputs must contain at least one item"),
			RequestID:  reqID,
		})
		return
	}
	provider := req.Provider
	if provider == "" {
		provider = p.cfg.DefaultProvider
	}
	opts, ok := p.cfg.Providers[provider]
	if !ok {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
			Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
		return
	}
	if req.Model != "" {
		opts.Model = req.Model
	}
	client, err := llm.NewClient(provider, opts)
	if err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
		return
	}
	libReq := llm.MultimodalEmbedRequest{
		Inputs: make([]llm.MultimodalInput, len(req.Inputs)),
	}
	for i, in := range req.Inputs {
		contents := make([]llm.MultimodalContent, len(in.Content))
		for j, c := range in.Content {
			mc := llm.MultimodalContent{
				Type:     llm.MultimodalContentType(c.Type),
				Text:     c.Text,
				ImageURL: c.ImageURL,
				MIMEType: c.MIMEType,
			}
			if c.ImageBase64 != "" {
				if base64.StdEncoding.DecodedLen(len(c.ImageBase64)) > maxDecodedImageBytes {
					p.writeError(w, r, ErrorInfo{
						Provider:   provider,
						StatusCode: http.StatusBadRequest,
						Err:        fmt.Errorf("image_base64 exceeds max decoded size"),
						RequestID:  reqID,
					})
					return
				}
				data, decErr := base64.StdEncoding.DecodeString(c.ImageBase64)
				if decErr != nil {
					p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest, Err: decErr, RequestID: reqID})
					return
				}
				mc.ImageData = data
			}
			contents[j] = mc
		}
		libReq.Inputs[i] = llm.MultimodalInput{Content: contents}
	}
	start := time.Now()
	vecs, err := client.EmbedMultimodal(r.Context(), libReq)
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, llm.ErrNotSupported) {
			status = http.StatusNotImplemented
		}
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID, Duration: time.Since(start)})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, EmbedMultimodalResponse{Embeddings: vecs}); err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
	}
}
