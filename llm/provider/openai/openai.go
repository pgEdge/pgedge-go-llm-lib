//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package openai implements the OpenAI provider for the LLM client.
package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	providerName   = "openai"
)

func init() {
	llm.RegisterProvider(providerName, func(opts llm.Options) (llm.Client, error) {
		return New(opts)
	})
}

// client implements llm.Client for OpenAI.
type client struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	opts       llm.Options

	mu              sync.Mutex
	cumulativeUsage llm.TokenUsage
}

// New creates a new OpenAI client.
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		var err error
		baseURL, err = httpclient.ValidateBaseURL(baseURL, "openai")
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
		model:      opts.Model,
		baseURL:    baseURL,
		opts:       opts,
	}, nil
}

func (c *client) Provider() string { return providerName }
func (c *client) Model() string    { return c.model }

func (c *client) Usage() llm.TokenUsage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cumulativeUsage
}

func (c *client) addUsage(u llm.TokenUsage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cumulativeUsage.Add(u)
}

// Ping calls ListModels as a lightweight liveness probe. Returns
// nil if the provider responds, or the underlying error otherwise.
func (c *client) Ping(ctx context.Context) error {
	_, err := c.ListModels(ctx)
	return err
}

// ResetUsage zeroes the cumulative usage counter.
func (c *client) ResetUsage() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cumulativeUsage = llm.TokenUsage{}
}

func (c *client) headers() map[string]string {
	h := map[string]string{
		"Content-Type": "application/json",
	}
	if c.apiKey != "" {
		h["Authorization"] = "Bearer " + c.apiKey
	}
	return h
}

// rejectDocumentBlocks returns ErrNotSupported if any message in the
// request carries a document content block. OpenAI's Chat Completions
// API does not accept document inputs (PDF and similar) inline; the
// caller must either pre-extract the text or use a provider that
// natively supports document blocks (Anthropic, Gemini).
func rejectDocumentBlocks(req llm.ChatRequest) error {
	for _, m := range req.Messages {
		for _, b := range m.Content {
			if b.Type == llm.BlockDocument {
				return &llm.ProviderError{
					Err:      llm.ErrNotSupported,
					Message:  "OpenAI Chat Completions does not support document content blocks; pre-extract text or use a provider with native document support (Anthropic, Gemini)",
					Provider: providerName,
				}
			}
		}
	}
	return nil
}

// useMaxCompletionTokens returns true for models that require
// max_completion_tokens instead of max_tokens.
func useMaxCompletionTokens(model string) bool {
	return strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "gpt-5")
}

// ---------- Chat ----------

// openaiChatRequest is the request body for /chat/completions.
type openaiChatRequest struct {
	Model               string                `json:"model"`
	Messages            []openaiMessage       `json:"messages"`
	MaxTokens           *int                  `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                  `json:"max_completion_tokens,omitempty"`
	Temperature         *float64              `json:"temperature,omitempty"`
	Tools               []openaiTool          `json:"tools,omitempty"`
	ToolChoice          any                   `json:"tool_choice,omitempty"`
	Stop                []string              `json:"stop,omitempty"`
	Stream              bool                  `json:"stream,omitempty"`
	StreamOptions       *streamOptions        `json:"stream_options,omitempty"`
	ResponseFormat      *openaiResponseFormat `json:"response_format,omitempty"`
}

type openaiResponseFormat struct {
	Type       string                  `json:"type"`
	JSONSchema *openaiJSONSchemaConfig `json:"json_schema,omitempty"`
}

type openaiJSONSchemaConfig struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// openaiContentPart is one element of OpenAI's array-form content
// (used for messages mixing text + images). Text-only messages use
// the legacy "content": "string" form for compatibility.
type openaiContentPart struct {
	Type     string             `json:"type"`
	Text     string             `json:"text,omitempty"`
	ImageURL *openaiImageURLRef `json:"image_url,omitempty"`
}

type openaiImageURLRef struct {
	URL string `json:"url"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string        `json:"type"`
	Function openaiToolDef `json:"function"`
}

type openaiToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// openaiChatResponse is the response from /chat/completions.
type openaiChatResponse struct {
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Message      openaiRespMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openaiRespMessage struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []openaiToolCall `json:"tool_calls,omitempty"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (c *client) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if useResponsesAPI(c.model, c.opts.Extensions) {
		return c.chatResponses(ctx, req)
	}
	if err := rejectDocumentBlocks(req); err != nil {
		return nil, err
	}
	oaiReq := c.buildChatRequest(req, false)

	var oaiResp openaiChatResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/chat/completions", c.headers(), oaiReq, &oaiResp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	resp := c.parseChatResponse(&oaiResp)
	c.addUsage(resp.Usage)
	return resp, nil
}

func (c *client) buildChatRequest(req llm.ChatRequest, stream bool) openaiChatRequest {
	oaiReq := openaiChatRequest{
		Model:    c.model,
		Messages: c.convertMessages(req),
		Stream:   stream,
	}

	if stream {
		oaiReq.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	// Determine max tokens: per-request → client default → omit.
	var maxTokens *int
	if req.MaxTokens != nil {
		maxTokens = req.MaxTokens
	} else {
		maxTokens = c.opts.MaxTokens
	}
	if maxTokens != nil {
		if useMaxCompletionTokens(c.model) {
			oaiReq.MaxCompletionTokens = maxTokens
		} else {
			oaiReq.MaxTokens = maxTokens
		}
	}

	// Temperature: per-request → client default → omit (use provider default).
	if req.Temperature != nil {
		oaiReq.Temperature = req.Temperature
	} else {
		oaiReq.Temperature = c.opts.Temperature
	}

	// Tools.
	if len(req.Tools) > 0 {
		useCompact := req.UseCompactDescriptions(c.baseURL)
		oaiReq.Tools = make([]openaiTool, len(req.Tools))
		for i, t := range req.Tools {
			oaiReq.Tools[i] = openaiTool{
				Type: "function",
				Function: openaiToolDef{
					Name:        t.Name,
					Description: llm.EffectiveToolDescription(t, useCompact),
					Parameters:  t.InputSchema,
				},
			}
		}
	}

	// ToolChoice: map the normalised ToolChoice onto OpenAI's wire format.
	// OpenAI accepts "auto", "none", "required", or
	// {type:"function", function:{name:...}} for a specific function.
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case llm.ToolChoiceAuto:
			oaiReq.ToolChoice = "auto"
		case llm.ToolChoiceNone:
			oaiReq.ToolChoice = "none"
		case llm.ToolChoiceRequired:
			oaiReq.ToolChoice = "required"
		case llm.ToolChoiceSpecific:
			oaiReq.ToolChoice = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": req.ToolChoice.Name},
			}
		}
	}

	// ResponseFormat: OpenAI supports native JSON mode and structured outputs.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			oaiReq.ResponseFormat = &openaiResponseFormat{Type: "json_object"}
		case llm.ResponseFormatJSONSchema:
			oaiReq.ResponseFormat = &openaiResponseFormat{
				Type: "json_schema",
				JSONSchema: &openaiJSONSchemaConfig{
					Name:   "response",
					Schema: req.ResponseFormat.JSONSchema,
					Strict: true,
				},
			}
		}
	}

	// StopSequences: passed directly as the stop array.
	if len(req.StopSequences) > 0 {
		oaiReq.Stop = req.StopSequences
	}

	return oaiReq
}

func (c *client) convertMessages(req llm.ChatRequest) []openaiMessage {
	var msgs []openaiMessage

	// System prompt: per-request only (no client-level default).
	if req.SystemPrompt != "" {
		msgs = append(msgs, openaiMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, m := range req.Messages {
		msgs = append(msgs, convertMessage(m)...)
	}
	return msgs
}

// convertMessage maps an llm.Message to one or more openaiMessage
// values. A single tool-result Message can produce multiple OpenAI
// messages (one per BlockToolResult), since OpenAI's wire format
// requires one role:"tool" message per tool_call_id.
func convertMessage(m llm.Message) []openaiMessage {
	switch m.Role {
	case llm.RoleAssistant:
		return []openaiMessage{convertAssistantMessage(m)}
	case llm.RoleTool:
		return convertToolMessage(m)
	default:
		// user, system
		return []openaiMessage{convertUserMessage(m)}
	}
}

// convertUserMessage emits a user/system message. Text-only content
// uses the legacy "content": "string" wire form for compatibility;
// mixed text+image content uses OpenAI's array-of-parts form.
func convertUserMessage(m llm.Message) openaiMessage {
	msg := openaiMessage{Role: string(m.Role)}

	hasImage := false
	for _, b := range m.Content {
		if b.Type == llm.BlockImage {
			hasImage = true
			break
		}
	}

	if !hasImage {
		// Text-only: concatenate into a single string for the
		// legacy wire form (`"content": "..."`).
		var sb strings.Builder
		for _, b := range m.Content {
			if b.Type == llm.BlockText {
				sb.WriteString(b.Text)
			}
		}
		msg.Content = sb.String()
		return msg
	}

	// Mixed text + image: emit array-of-parts form.
	var parts []openaiContentPart
	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText:
			parts = append(parts, openaiContentPart{Type: "text", Text: b.Text})
		case llm.BlockImage:
			if b.Image == nil {
				continue
			}
			url := b.Image.URL
			if url == "" && len(b.Image.Data) > 0 {
				// Build a data URL: data:<media_type>;base64,<b64>.
				mediaType := b.Image.MediaType
				if mediaType == "" {
					mediaType = "application/octet-stream"
				}
				url = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(b.Image.Data)
			}
			if url == "" {
				continue
			}
			parts = append(parts, openaiContentPart{
				Type:     "image_url",
				ImageURL: &openaiImageURLRef{URL: url},
			})
		}
	}
	msg.Content = parts
	return msg
}

// convertAssistantMessage builds an assistant message from typed
// content. Text blocks are concatenated into Content; tool-use blocks
// become tool_calls. A nil Content with non-empty ToolCalls is the
// expected shape for assistant messages that ONLY make tool calls.
func convertAssistantMessage(m llm.Message) openaiMessage {
	msg := openaiMessage{Role: "assistant"}

	var textParts []string
	var toolCalls []openaiToolCall

	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText:
			textParts = append(textParts, b.Text)
		case llm.BlockToolUse:
			if b.ToolUse != nil {
				toolCalls = append(toolCalls, openaiToolCall{
					ID:   b.ToolUse.ID,
					Type: "function",
					Function: openaiToolFunction{
						Name:      b.ToolUse.Name,
						Arguments: string(b.ToolUse.Input),
					},
				})
			}
		}
	}

	if len(textParts) > 0 {
		msg.Content = strings.Join(textParts, "")
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	return msg
}

// convertToolMessage emits one role:"tool" OpenAI message per
// BlockToolResult in the input message.
func convertToolMessage(m llm.Message) []openaiMessage {
	out := make([]openaiMessage, 0, len(m.Content))
	for _, b := range m.Content {
		if b.Type != llm.BlockToolResult {
			continue
		}
		out = append(out, openaiMessage{
			Role:       "tool",
			Content:    b.Text,
			ToolCallID: b.ToolUseID,
		})
	}
	return out
}

func (c *client) parseChatResponse(oaiResp *openaiChatResponse) *llm.ChatResponse {
	resp := &llm.ChatResponse{
		Usage: llm.TokenUsage{
			PromptTokens:     oaiResp.Usage.PromptTokens,
			CompletionTokens: oaiResp.Usage.CompletionTokens,
			TotalTokens:      oaiResp.Usage.TotalTokens,
		},
	}

	if len(oaiResp.Choices) == 0 {
		return resp
	}

	choice := oaiResp.Choices[0]
	resp.StopReason = normalizeStopReason(choice.FinishReason)

	// Build content blocks.
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		resp.Content = append(resp.Content, llm.ContentBlock{
			Type: llm.BlockText,
			Text: *choice.Message.Content,
		})
	}

	for _, tc := range choice.Message.ToolCalls {
		resp.Content = append(resp.Content, llm.ContentBlock{
			Type: llm.BlockToolUse,
			ToolUse: &llm.ToolUse{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			},
		})
	}

	return resp
}

func normalizeStopReason(reason string) llm.StopReason {
	switch reason {
	case "stop":
		return llm.StopReasonEndTurn
	case "length":
		return llm.StopReasonMaxTokens
	case "tool_calls", "function_call":
		return llm.StopReasonToolUse
	case "content_filter":
		return llm.StopReasonContentFilter
	default:
		return llm.StopReasonEndTurn
	}
}

// ---------- ChatStream ----------

// streamDelta is a partial representation of the streaming response.
type streamChunkResponse struct {
	Choices []streamChoice `json:"choices"`
	Usage   *openaiUsage   `json:"usage,omitempty"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type streamDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   *string               `json:"content,omitempty"`
	ToolCalls []streamDeltaToolCall `json:"tool_calls,omitempty"`
}

type streamDeltaToolCall struct {
	Index    int                  `json:"index"`
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function *streamDeltaFunction `json:"function,omitempty"`
}

type streamDeltaFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func (c *client) ChatStream(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	if useResponsesAPI(c.model, c.opts.Extensions) {
		return c.chatStreamResponses(ctx, req)
	}
	if err := rejectDocumentBlocks(req); err != nil {
		return nil, err
	}
	oaiReq := c.buildChatRequest(req, true)

	resp, err := httpclient.DoSSERequest(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/chat/completions", c.headers(), oaiReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body := make([]byte, 4096)
		n, _ := resp.Body.Read(body)
		return nil, mapError(resp.StatusCode, body[:n])
	}

	chunks := make(chan llm.StreamChunk, 64)
	errCh := make(chan error, 1)

	go func() {
		defer resp.Body.Close()
		defer close(chunks)
		defer close(errCh)

		scanner := httpclient.NewSSEScanner(resp.Body)

		// Track tool calls being built across deltas.
		type toolCallState struct {
			id   string
			name string
			args strings.Builder
		}
		toolCalls := make(map[int]*toolCallState)
		var lastUsage *openaiUsage

		for scanner.Scan() {
			data := scanner.Data()
			if data == "[DONE]" {
				break
			}

			var chunk streamChunkResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				errCh <- fmt.Errorf("decode stream chunk: %w", err)
				return
			}

			if chunk.Usage != nil {
				lastUsage = chunk.Usage
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			delta := chunk.Choices[0].Delta

			// Text content.
			if delta.Content != nil && *delta.Content != "" {
				chunks <- llm.StreamChunk{
					Type: llm.ChunkText,
					Text: *delta.Content,
				}
			}

			// Tool calls.
			for _, tc := range delta.ToolCalls {
				state, exists := toolCalls[tc.Index]
				if !exists {
					state = &toolCallState{}
					toolCalls[tc.Index] = state
				}

				if tc.ID != "" {
					state.id = tc.ID
				}
				if tc.Function != nil && tc.Function.Name != "" {
					state.name = tc.Function.Name
					// Emit tool_use_start.
					chunks <- llm.StreamChunk{
						Type: llm.ChunkToolUseStart,
						ToolUse: &llm.ToolUse{
							ID:   state.id,
							Name: state.name,
						},
					}
				}
				if tc.Function != nil && tc.Function.Arguments != "" {
					state.args.WriteString(tc.Function.Arguments)
					chunks <- llm.StreamChunk{
						Type:    llm.ChunkToolUseDelta,
						Partial: tc.Function.Arguments,
					}
				}
			}

			// Finish reason.
			if chunk.Choices[0].FinishReason != nil && *chunk.Choices[0].FinishReason != "" {
				usage := &llm.TokenUsage{}
				if lastUsage != nil {
					usage.PromptTokens = lastUsage.PromptTokens
					usage.CompletionTokens = lastUsage.CompletionTokens
					usage.TotalTokens = lastUsage.TotalTokens
					// Only accumulate cumulative usage when the provider
					// reported real counts; never accumulate zero-token
					// "phantom" usage from streams without usage data.
					c.addUsage(*usage)
				}
				chunks <- llm.StreamChunk{
					Type:  llm.ChunkDone,
					Usage: usage,
				}
			}
		}
	}()

	return &llm.Stream{
		Chunks: chunks,
		Err:    errCh,
	}, nil
}

// ---------- Embed ----------

type openaiEmbedRequest struct {
	Model      string `json:"model"`
	Input      any    `json:"input"`
	Dimensions int    `json:"dimensions,omitempty"`
}

type openaiEmbedResponse struct {
	Data  []openaiEmbedData `json:"data"`
	Usage openaiUsage       `json:"usage"`
}

type openaiEmbedData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

func (c *client) Embed(ctx context.Context, text string) ([]float64, error) {
	reqBody := openaiEmbedRequest{
		Model: c.model,
		Input: text,
	}
	if ext := findExtension(c.opts.Extensions); ext != nil {
		reqBody.Dimensions = ext.EmbeddingDimensions
	}

	var resp openaiEmbedResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/embeddings", c.headers(), reqBody, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	if len(resp.Data) == 0 {
		return nil, &llm.ProviderError{
			Err:      llm.ErrProviderError,
			Message:  "no embedding data returned",
			Provider: providerName,
		}
	}
	// Embeddings have no completion tokens; only prompt/total apply.
	c.addUsage(llm.TokenUsage{
		PromptTokens: resp.Usage.PromptTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	})
	return resp.Data[0].Embedding, nil
}

func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	reqBody := openaiEmbedRequest{
		Model: c.model,
		Input: texts,
	}
	if ext := findExtension(c.opts.Extensions); ext != nil {
		reqBody.Dimensions = ext.EmbeddingDimensions
	}

	var resp openaiEmbedResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/embeddings", c.headers(), reqBody, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	// Sort by index to maintain order.
	result := make([][]float64, len(texts))
	for _, d := range resp.Data {
		if d.Index < len(result) {
			result[d.Index] = d.Embedding
		}
	}
	// Embeddings have no completion tokens; only prompt/total apply.
	c.addUsage(llm.TokenUsage{
		PromptTokens: resp.Usage.PromptTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	})
	return result, nil
}

// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "OpenAI does not support reranking",
		Provider: "openai",
	}
}

// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "OpenAI does not support multimodal embeddings",
		Provider: "openai",
	}
}

// ---------- ListModels ----------

type openaiModelsResponse struct {
	Data []openaiModelData `json:"data"`
}

type openaiModelData struct {
	ID string `json:"id"`
}

// openaiEmbeddingDimensions maps known OpenAI embedding model IDs to their
// native (default) vector dimensions. Models not listed here have unknown
// dimensions and are left at 0 in ModelInfo.Dimensions.
var openaiEmbeddingDimensions = map[string]int{
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
	"text-embedding-ada-002": 1536,
}

// filterPrefixes lists model ID prefixes for non-chat models.
var filterPrefixes = []string{
	"text-embedding",
	"embedding",
	"tts",
	"whisper",
	"dall-e",
	"davinci",
	"babbage",
	"text-moderation",
	"text-search",
	"text-similarity",
	"code-search",
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

func shouldFilterModel(id string) bool {
	lower := strings.ToLower(id)
	for _, prefix := range filterPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

// ---------- ListModelsWithMetadata ----------

var openaiModelCapabilities = map[string][]llm.ModelCapability{
	"gpt-4o":          {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gpt-4o-mini":     {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gpt-4-turbo":     {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gpt-4":           {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gpt-3.5-turbo":   {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"o1":              {llm.ModelCapabilityChat, llm.ModelCapabilityStreaming},
	"text-embedding-": {llm.ModelCapabilityEmbeddings},
}

func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	var resp openaiModelsResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodGet,
		c.baseURL+"/models", c.headers(), nil, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	cfg := llm.ListModelsConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	wantEmbeddings := false
	for _, cap := range cfg.Capabilities {
		if cap == llm.ModelCapabilityEmbeddings {
			wantEmbeddings = true
			break
		}
	}

	var infos []llm.ModelInfo
	for _, m := range resp.Data {
		caps := lookupOpenAICapabilities(m.ID)
		isEmbedding := openaiEmbeddingModel(m.ID)
		if isEmbedding {
			// Embedding models are only included when the caller explicitly
			// requests ModelCapabilityEmbeddings; they are excluded from the
			// default (chat-focused) model list.
			if wantEmbeddings {
				infos = append(infos, llm.ModelInfo{
					ID:           m.ID,
					Capabilities: caps,
					Dimensions:   openaiEmbeddingDimensions[m.ID],
				})
			}
		} else if !shouldFilterModel(m.ID) {
			infos = append(infos, llm.ModelInfo{ID: m.ID, Capabilities: caps})
		}
	}

	return llm.FilterModelInfos(infos, cfg), nil
}

// openaiEmbeddingModel reports whether id identifies an OpenAI
// text-embedding model.
func openaiEmbeddingModel(id string) bool {
	return strings.HasPrefix(id, "text-embedding-") ||
		id == "text-embedding-ada-002"
}

func lookupOpenAICapabilities(modelID string) []llm.ModelCapability {
	// Use longest-prefix matching to ensure "gpt-4o-mini" takes priority
	// over the shorter "gpt-4o" entry when both could match.
	bestLen := -1
	var bestCaps []llm.ModelCapability
	for prefix, caps := range openaiModelCapabilities {
		if strings.HasPrefix(modelID, prefix) && len(prefix) > bestLen {
			bestLen = len(prefix)
			bestCaps = caps
		}
	}
	if bestLen >= 0 {
		return bestCaps
	}
	return []llm.ModelCapability{llm.ModelCapabilityChat, llm.ModelCapabilityStreaming}
}

// ---------- Error mapping ----------

type openaiErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func mapError(status int, body []byte) error {
	var errResp openaiErrorResponse
	_ = json.Unmarshal(body, &errResp) // best-effort; fall back to status-based message below

	msg := errResp.Error.Message
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", status)
	}

	var sentinel error
	switch {
	case status == 401 || status == 403:
		sentinel = llm.ErrAuthentication
	case status == 429:
		sentinel = llm.ErrRateLimit
	case status == 400:
		sentinel = llm.ErrInvalidRequest
	default:
		sentinel = llm.ErrProviderError
	}

	return &llm.ProviderError{
		Err:        sentinel,
		StatusCode: status,
		Message:    msg,
		Provider:   providerName,
	}
}
