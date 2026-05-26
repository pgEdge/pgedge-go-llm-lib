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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

func (p *Proxy) handleProviders(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r) {
		return
	}

	infos := make([]ProviderInfo, 0, len(p.cfg.Providers))
	for name, opts := range p.cfg.Providers {
		infos = append(infos, ProviderInfo{
			Name:    name,
			Model:   opts.Model,
			Default: name == p.cfg.DefaultProvider,
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
	if !p.authorize(w, r) {
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

	if r.URL.Query().Get("metadata") == "true" {
		models, mdErr := client.ListModelsWithMetadata(r.Context())
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

	models, err := client.ListModels(r.Context())
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
	if !p.authorize(w, r) {
		return
	}

	req, opts, provider, model, err := p.parseChatRequest(r)
	if err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}

	llmReq := buildLLMRequest(req)
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

	chatResp, err := client.Chat(r.Context(), llmReq)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     false,
			StatusCode: http.StatusBadGateway,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

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
		})
	}
}

// buildLLMRequest projects the proxy wire request onto the library's
// llm.ChatRequest. Shared by handleChat and handleChatStream.
func buildLLMRequest(req ChatRequest) llm.ChatRequest {
	return llm.ChatRequest{
		Messages:       req.Messages,
		Tools:          req.Tools,
		SystemPrompt:   req.SystemPrompt,
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		ResponseFormat: req.ResponseFormat,
		ToolChoice:     req.ToolChoice,
		StopSequences:  req.StopSequences,
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
	if !p.authorize(w, r) {
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
	if !p.authorize(w, r) {
		return
	}

	req, opts, provider, model, err := p.parseChatRequest(r)
	if err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
		return
	}

	llmReq := buildLLMRequest(req)
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

	stream, err := client.ChatStream(r.Context(), llmReq)
	if err != nil {
		p.writeError(w, r, ErrorInfo{
			Provider:   provider,
			Model:      model,
			Stream:     true,
			StatusCode: http.StatusBadGateway,
			Err:        err,
			RequestID:  reqID,
		})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	streamResp, streamErr := writeSSE(w, stream)

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
		})
	}
}

func (p *Proxy) handleEmbed(w http.ResponseWriter, r *http.Request) {
	r, reqID := p.ensureRequestID(r)
	if reqID != "" {
		w.Header().Set(p.requestIDHeaderName(), reqID)
	}
	if !p.authorize(w, r) {
		return
	}
	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
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
	vecs, err := client.EmbedBatch(r.Context(), req.Input)
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, llm.ErrNotSupported) {
			status = http.StatusNotImplemented
		}
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := encodeJSON(w, EmbedResponse{Embeddings: vecs}); err != nil {
		p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
	}
}
