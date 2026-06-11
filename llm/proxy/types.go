//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package proxy exposes the pgedge-go-llm-lib provider abstraction
// over an HTTP API, including SSE streaming.
//
// The proxy is intentionally minimal: it does not implement
// authentication, authorization, rate limiting, or persistent
// telemetry. Consumers wrap the returned http.Handler with their own
// middleware and use the Config.OnRequest, OnResponse, and OnError
// hooks to plumb in tracing.
package proxy

import "github.com/pgEdge/pgedge-go-llm-lib/llm"

// ChatRequest is the wire shape for POST /v1/chat and POST /v1/chat/stream.
//
// Provider and Model override the proxy's defaults for this request
// only. If empty, the proxy's DefaultProvider and the per-provider
// configured Model are used.
//
// Each Message.Content must be a JSON array of typed content blocks
// — text, tool_use, tool_result, or image. The legacy form
// (`"content": "string"`) is no longer accepted; clients must always
// send the array form. See llm.ContentBlock for the per-block fields.
type ChatRequest struct {
	Messages       []llm.Message       `json:"messages"`
	Tools          []llm.Tool          `json:"tools,omitempty"`
	SystemPrompt   string              `json:"system_prompt,omitempty"`
	MaxTokens      *int                `json:"max_tokens,omitempty"`
	Temperature    *float64            `json:"temperature,omitempty"`
	Provider       string              `json:"provider,omitempty"`
	Model          string              `json:"model,omitempty"`
	ResponseFormat *llm.ResponseFormat `json:"response_format,omitempty"`
	ToolChoice     *llm.ToolChoice     `json:"tool_choice,omitempty"`
	StopSequences  []string            `json:"stop_sequences,omitempty"`

	// ToolDescriptions selects which tool description text the provider
	// sends on the wire; see llm.ToolDescriptionMode. Empty (the default)
	// auto-selects compact descriptions for local provider base URLs.
	ToolDescriptions llm.ToolDescriptionMode `json:"tool_descriptions,omitempty"`
}

// ChatResponse is the wire shape for POST /v1/chat.
type ChatResponse struct {
	Content    []llm.ContentBlock `json:"content"`
	StopReason llm.StopReason     `json:"stop_reason"`
	Usage      llm.TokenUsage     `json:"usage"`
}

// ProvidersResponse is the wire shape for GET /v1/providers.
type ProvidersResponse struct {
	Providers       []ProviderInfo `json:"providers"`
	DefaultProvider string         `json:"default_provider,omitempty"`
}

// ProviderInfo describes a configured provider.
type ProviderInfo struct {
	Name    string `json:"name"`
	Model   string `json:"model,omitempty"`
	Default bool   `json:"default,omitempty"`
}

// ModelsResponse is the wire shape for GET /v1/models?provider=X.
// Models lists the model identifiers available from the named
// provider, as reported by that provider's ListModels method.
type ModelsResponse struct {
	Models []string `json:"models"`
}

// ModelsMetadataResponse is the wire shape for GET /v1/models?provider=X&metadata=true.
type ModelsMetadataResponse struct {
	Models []llm.ModelInfo `json:"models"`
}

// HealthResponse is the wire shape for GET /v1/health.
type HealthResponse struct {
	Status    string                    `json:"status"` // "ok" | "degraded"
	Providers map[string]ProviderHealth `json:"providers"`
}

// ProviderHealth describes the health of a single provider.
type ProviderHealth struct {
	Status string `json:"status"` // "ok" | "down"
	Error  string `json:"error,omitempty"`
}

// EmbedRequest is the JSON body of POST /v1/embed. Input may contain
// one or many strings; the proxy chooses Embed vs EmbedBatch
// accordingly.
type EmbedRequest struct {
	Provider string   `json:"provider,omitempty"`
	Model    string   `json:"model,omitempty"`
	Input    []string `json:"input"`
}

// EmbedResponse is the JSON body returned by POST /v1/embed.
type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// MultimodalContentRequest describes one content item within a
// multimodal embedding input. Type is required; the remaining fields
// are type-dependent (text for "text", image_url / image_base64 /
// mime_type for image types).
type MultimodalContentRequest struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
	MIMEType    string `json:"mime_type,omitempty"`
}

// MultimodalInputRequest is a single multimodal input to embed,
// comprising one or more content items (text and/or image).
type MultimodalInputRequest struct {
	Content []MultimodalContentRequest `json:"content"`
}

// EmbedMultimodalRequest is the JSON body for POST /v1/embed/multimodal.
type EmbedMultimodalRequest struct {
	Provider string                   `json:"provider,omitempty"`
	Model    string                   `json:"model,omitempty"`
	Inputs   []MultimodalInputRequest `json:"inputs"`
}

// EmbedMultimodalResponse is the JSON body returned by POST /v1/embed/multimodal.
type EmbedMultimodalResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// RerankRequest is the JSON body for POST /v1/rerank.
type RerankRequest struct {
	Provider  string   `json:"provider,omitempty"`
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopK      *int     `json:"top_k,omitempty"`
}

// RerankResult is one ranked document within a RerankResponse.
type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
	Document       string  `json:"document,omitempty"`
}

// RerankUsage carries token-usage information for a rerank call.
type RerankUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// RerankResponse is the JSON body returned by POST /v1/rerank.
type RerankResponse struct {
	Results []RerankResult `json:"results"`
	Usage   RerankUsage    `json:"usage"`
}

// ErrorResponse is the wire shape for any non-streaming error.
type ErrorResponse struct {
	Error string `json:"error"`
}

// SSE Wire Format
//
// POST /v1/chat/stream returns Server-Sent Events. The contract is:
//
//	data: <chunk-json>\n\n           — one event per non-done chunk
//	event: done\ndata: <chunk-json>\n\n  — exactly one terminator;
//	                                       chunk-json is a StreamChunk
//	                                       with Type=done and (when
//	                                       available) Usage populated.
//	event: error\ndata: <json>\n\n   — emitted on stream error in
//	                                    lieu of done; json is
//	                                    {"error": "..."}.
//
// A consumer is guaranteed to receive either an `event: done` or an
// `event: error` before the underlying connection closes, even if
// the upstream provider terminates abnormally — in that case a
// synthetic done with zero Usage is emitted.
//
// Each chunk-json is the JSON marshalling of llm.StreamChunk:
//
//	{"type":"text","text":"..."}                  — incremental text
//	{"type":"tool_use_start","tool_use":{...}}    — tool call started
//	{"type":"tool_use_delta","partial":"..."}     — tool args fragment
//	{"type":"done","usage":{...}}                 — terminator
