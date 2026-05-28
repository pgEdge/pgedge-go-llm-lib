//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package anthropic implements the Anthropic provider for the LLM client.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

const (
	defaultBaseURL = "https://api.anthropic.com/v1"
	providerName   = "anthropic"
)

func init() {
	llm.RegisterProvider(providerName, func(opts llm.Options) (llm.Client, error) {
		return New(opts)
	})
}

// client implements llm.Client for Anthropic.
type client struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	opts       llm.Options

	mu              sync.Mutex
	cumulativeUsage llm.TokenUsage
}

// New creates a new Anthropic client.
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		var err error
		baseURL, err = httpclient.ValidateBaseURL(baseURL, "anthropic")
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
		"Content-Type":      "application/json",
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "prompt-caching-2024-07-31,pdfs-2024-09-25",
	}
	if c.apiKey != "" {
		h["x-api-key"] = c.apiKey
	}
	return h
}

// ---------- Chat ----------

// anthropicToolChoice is the wire representation of Anthropic's
// tool_choice object.
//
//   - type "auto"  — model decides (default).
//   - type "any"   — model must call one of the available tools (maps to ToolChoiceRequired).
//   - type "tool"  — model must call the named tool.
//
// Anthropic has no native "none" mode. When the caller sends
// ToolChoiceNone we fall back to "auto" (see buildChatRequest).
// If the intent is truly "no tool calls", the caller should omit the
// Tools array rather than relying on ToolChoiceNone.
type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// anthropicChatRequest is the request body for /messages.
type anthropicChatRequest struct {
	Model    string             `json:"model"`
	Messages []anthropicMessage `json:"messages"`

	// MaxTokens is required by the Anthropic Messages API. Unlike
	// other providers' wire formats, this is a plain int (not *int
	// with omitempty) — Anthropic returns HTTP 400 if the field is
	// missing. The constant anthropicDefaultMaxTokens fills it when
	// neither client nor request specifies a value.
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	System        []systemBlock        `json:"system,omitempty"`
	Tools         []anthropicTool      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Thinking      *anthropicThinking   `json:"thinking,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
}

// anthropicThinking is the wire shape for Anthropic's extended-thinking
// mode. Type is currently always "enabled".
type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

// anthropic's max_tokens fallback when both client and request leave it unset.
const anthropicDefaultMaxTokens = 4096

type systemBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicMessage struct {
	Role    string                  `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  json.RawMessage        `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicChatResponse is the response from /messages.
type anthropicChatResponse struct {
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
	Usage      anthropicUsage          `json:"usage"`
}

// anthropicContentBlock is the wire representation of a single
// content block for Anthropic's Messages API. The set of populated
// fields depends on the block Type:
//
//	"text"         -> Text
//	"image"        -> Source
//	"document"     -> Source, optional Title
//	"tool_use"     -> ID, Name, Input
//	"tool_result"  -> ToolUseID, Content, IsError
type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// Source — populated for "image" and "document".
	Source *anthropicSource `json:"source,omitempty"`

	// Title — optional human-readable label populated for "document"
	// (e.g., a filename surfaced to the model).
	Title string `json:"title,omitempty"`

	// Tool result — populated for "tool_result".
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`

	// Cache control — Anthropic prompt-caching marker.
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicSource is the wire representation of an image- or
// document-block source on Anthropic's Messages API.
type anthropicSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // for base64
	Data      []byte `json:"data,omitempty"`       // for base64 (auto base64-encoded by encoding/json)
	URL       string `json:"url,omitempty"`        // for url
}

type anthropicCacheControl struct {
	Type string `json:"type"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

func (c *client) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	aReq := c.buildChatRequest(req, false)

	var aResp anthropicChatResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/messages", c.headers(), aReq, &aResp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	resp := c.parseChatResponse(&aResp)
	c.addUsage(resp.Usage)
	return resp, nil
}

func (c *client) buildChatRequest(req llm.ChatRequest, stream bool) anthropicChatRequest {
	aReq := anthropicChatRequest{
		Model:    c.model,
		Messages: c.convertMessages(req),
		Stream:   stream,
	}

	// MaxTokens: per-request → client default → hardcoded fallback.
	// Anthropic requires max_tokens in every request; omitting it returns HTTP 400.
	if req.MaxTokens != nil {
		aReq.MaxTokens = *req.MaxTokens
	} else if c.opts.MaxTokens != nil {
		aReq.MaxTokens = *c.opts.MaxTokens
	} else {
		aReq.MaxTokens = anthropicDefaultMaxTokens
	}

	// Temperature: per-request → client default → omit (use provider default).
	if req.Temperature != nil {
		aReq.Temperature = req.Temperature
	} else {
		aReq.Temperature = c.opts.Temperature
	}

	// System prompt: per-request only (no client-level default).
	if req.SystemPrompt != "" {
		aReq.System = []systemBlock{
			{Type: "text", Text: req.SystemPrompt},
		}
	}

	// Tools.
	if len(req.Tools) > 0 {
		aReq.Tools = make([]anthropicTool, len(req.Tools))
		for i, t := range req.Tools {
			aReq.Tools[i] = anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.InputSchema,
			}
		}
	}

	// Anthropic-specific extension lookup. The CacheSystem branch is
	// deferred until after ResponseFormat below, which may append
	// further system blocks — the cache_control marker must land on the
	// last block on the wire.
	ext := llm.FindExtension[Extension](req, "anthropic")
	if ext != nil {
		if ext.CacheToolsThrough >= 0 && ext.CacheToolsThrough < len(aReq.Tools) {
			aReq.Tools[ext.CacheToolsThrough].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
		}
		if ext.ExtendedThinking {
			aReq.Thinking = &anthropicThinking{
				Type:         "enabled",
				BudgetTokens: ext.BudgetTokens,
			}
		}
	}

	// ToolChoice: map the normalised ToolChoice onto Anthropic's wire format.
	// Anthropic accepts {type: "auto"|"any"|"tool", name?: ...}.
	// NOTE: Anthropic has no "none" mode. ToolChoiceNone falls back to "auto"
	// because that is the closest available option. This does NOT prevent tool
	// calls — Anthropic's "auto" still permits the model to call tools.
	// The most reliable way to suppress tool calls with Anthropic is to omit
	// the Tools array from the request entirely.
	if req.ToolChoice != nil {
		tc := &anthropicToolChoice{}
		switch req.ToolChoice.Mode {
		case llm.ToolChoiceAuto:
			tc.Type = "auto"
		case llm.ToolChoiceNone:
			// Anthropic has no "none" — fall back to "auto".
			tc.Type = "auto"
		case llm.ToolChoiceRequired:
			tc.Type = "any"
		case llm.ToolChoiceSpecific:
			tc.Type = "tool"
			tc.Name = req.ToolChoice.Name
		}
		aReq.ToolChoice = tc
	}

	// ResponseFormat: Anthropic has no native JSON mode. We fall back to
	// prompt-engineering by appending a directive to the system prompt.
	// This is best-effort; there is no wire-level enforcement.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			systemSuffix := "Respond ONLY with valid JSON. No prose, no markdown fences."
			aReq.System = append(aReq.System, systemBlock{Type: "text", Text: systemSuffix})
		case llm.ResponseFormatJSONSchema:
			systemSuffix := fmt.Sprintf("Respond ONLY with valid JSON conforming to this schema:\n%s\nNo prose, no markdown fences.", string(req.ResponseFormat.JSONSchema))
			aReq.System = append(aReq.System, systemBlock{Type: "text", Text: systemSuffix})
		}
	}

	// System prompt caching is applied after ResponseFormat so the
	// cache_control marker always lands on the final system block.
	if ext != nil && ext.CacheSystem && len(aReq.System) > 0 {
		aReq.System[len(aReq.System)-1].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
	}

	// StopSequences: passed directly as stop_sequences.
	if len(req.StopSequences) > 0 {
		aReq.StopSequences = req.StopSequences
	}

	return aReq
}

func (c *client) convertMessages(req llm.ChatRequest) []anthropicMessage {
	msgs := make([]anthropicMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, convertMessage(m))
	}
	return msgs
}

// convertMessage maps an llm.Message to Anthropic's wire format.
//
// Role mapping: assistant -> "assistant"; user/system -> "user";
// tool -> "user" (Anthropic represents tool results as user-role
// messages containing tool_result content blocks).
func convertMessage(m llm.Message) anthropicMessage {
	var role string
	switch m.Role {
	case llm.RoleAssistant:
		role = "assistant"
	case llm.RoleTool:
		// Anthropic represents tool results as user-role messages
		// with tool_result content blocks. Map RoleTool -> "user".
		role = "user"
	default:
		role = "user"
	}

	blocks := make([]anthropicContentBlock, 0, len(m.Content))
	for _, b := range m.Content {
		blocks = append(blocks, convertBlock(b))
	}
	return anthropicMessage{Role: role, Content: blocks}
}

// convertBlock maps a single llm.ContentBlock to Anthropic's wire
// shape, dispatching on the block's Type tag.
func convertBlock(b llm.ContentBlock) anthropicContentBlock {
	out := anthropicContentBlock{Type: string(b.Type)}
	switch b.Type {
	case llm.BlockText:
		out.Text = b.Text
	case llm.BlockImage:
		if b.Image != nil {
			if b.Image.URL != "" {
				out.Source = &anthropicSource{
					Type: "url",
					URL:  b.Image.URL,
				}
			} else {
				out.Source = &anthropicSource{
					Type:      "base64",
					MediaType: b.Image.MediaType,
					Data:      b.Image.Data,
				}
			}
		}
	case llm.BlockDocument:
		if b.Document != nil {
			if b.Document.URL != "" {
				out.Source = &anthropicSource{
					Type: "url",
					URL:  b.Document.URL,
				}
			} else {
				out.Source = &anthropicSource{
					Type:      "base64",
					MediaType: b.Document.MediaType,
					Data:      b.Document.Data,
				}
			}
			out.Title = b.Document.Filename
		}
	case llm.BlockToolUse:
		if b.ToolUse != nil {
			out.ID = b.ToolUse.ID
			out.Name = b.ToolUse.Name
			out.Input = b.ToolUse.Input
		}
	case llm.BlockToolResult:
		out.ToolUseID = b.ToolUseID
		out.Content = b.Text
		out.IsError = b.IsError
	}
	if b.CacheControl != nil {
		out.CacheControl = &anthropicCacheControl{Type: b.CacheControl.Type}
	}
	return out
}

func (c *client) parseChatResponse(aResp *anthropicChatResponse) *llm.ChatResponse {
	usage := llm.TokenUsage{
		PromptTokens:             aResp.Usage.InputTokens,
		CompletionTokens:         aResp.Usage.OutputTokens,
		TotalTokens:              aResp.Usage.InputTokens + aResp.Usage.OutputTokens,
		CacheCreationInputTokens: aResp.Usage.CacheCreationInputTokens,
		CacheReadInputTokens:     aResp.Usage.CacheReadInputTokens,
	}

	resp := &llm.ChatResponse{
		StopReason: normalizeStopReason(aResp.StopReason),
		Usage:      usage,
	}

	for _, block := range aResp.Content {
		switch block.Type {
		case "text":
			resp.Content = append(resp.Content, llm.ContentBlock{
				Type: llm.BlockText,
				Text: block.Text,
			})
		case "tool_use":
			resp.Content = append(resp.Content, llm.ContentBlock{
				Type: llm.BlockToolUse,
				ToolUse: &llm.ToolUse{
					ID:    block.ID,
					Name:  block.Name,
					Input: block.Input,
				},
			})
		}
	}

	return resp
}

// ---------- ChatStream ----------

func (c *client) ChatStream(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	aReq := c.buildChatRequest(req, true)

	resp, err := httpclient.DoSSERequest(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/messages", c.headers(), aReq)
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

		var inputTokens int
		var outputTokens int
		var cacheCreationTokens int
		var cacheReadTokens int

		for scanner.Scan() {
			data := scanner.Data()
			if data == "[DONE]" {
				break
			}

			var event struct {
				Type         string `json:"type"`
				Index        int    `json:"index"`
				ContentBlock *struct {
					Type string `json:"type"`
					ID   string `json:"id,omitempty"`
					Name string `json:"name,omitempty"`
				} `json:"content_block,omitempty"`
				Delta *struct {
					Type        string `json:"type,omitempty"`
					Text        string `json:"text,omitempty"`
					PartialJSON string `json:"partial_json,omitempty"`
					StopReason  string `json:"stop_reason,omitempty"`
				} `json:"delta,omitempty"`
				Message *struct {
					Usage *anthropicUsage `json:"usage,omitempty"`
				} `json:"message,omitempty"`
				Usage *struct {
					OutputTokens int `json:"output_tokens,omitempty"`
				} `json:"usage,omitempty"`
			}

			if err := json.Unmarshal([]byte(data), &event); err != nil {
				errCh <- fmt.Errorf("decode stream event: %w", err)
				return
			}

			switch event.Type {
			case "message_start":
				if event.Message != nil && event.Message.Usage != nil {
					inputTokens = event.Message.Usage.InputTokens
					cacheCreationTokens = event.Message.Usage.CacheCreationInputTokens
					cacheReadTokens = event.Message.Usage.CacheReadInputTokens
				}

			case "content_block_start":
				if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
					chunks <- llm.StreamChunk{
						Type: llm.ChunkToolUseStart,
						ToolUse: &llm.ToolUse{
							ID:   event.ContentBlock.ID,
							Name: event.ContentBlock.Name,
						},
					}
				}

			case "content_block_delta":
				if event.Delta != nil {
					switch event.Delta.Type {
					case "text_delta":
						if event.Delta.Text != "" {
							chunks <- llm.StreamChunk{
								Type: llm.ChunkText,
								Text: event.Delta.Text,
							}
						}
					case "input_json_delta":
						if event.Delta.PartialJSON != "" {
							chunks <- llm.StreamChunk{
								Type:    llm.ChunkToolUseDelta,
								Partial: event.Delta.PartialJSON,
							}
						}
					}
				}

			case "message_delta":
				if event.Usage != nil {
					outputTokens = event.Usage.OutputTokens
				}

			case "message_stop":
				totalTokens := inputTokens + outputTokens
				usage := &llm.TokenUsage{
					PromptTokens:             inputTokens,
					CompletionTokens:         outputTokens,
					TotalTokens:              totalTokens,
					CacheCreationInputTokens: cacheCreationTokens,
					CacheReadInputTokens:     cacheReadTokens,
				}
				c.addUsage(*usage)
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

func (c *client) Embed(_ context.Context, _ string) ([]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Anthropic does not support embeddings",
		Provider: providerName,
	}
}

func (c *client) EmbedBatch(_ context.Context, _ []string) ([][]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Anthropic does not support embeddings",
		Provider: providerName,
	}
}

// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Anthropic does not support reranking",
		Provider: "anthropic",
	}
}

// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Anthropic does not support multimodal embeddings",
		Provider: "anthropic",
	}
}

// ---------- ListModels ----------

type anthropicModelsResponse struct {
	Data []anthropicModelData `json:"data"`
}

type anthropicModelData struct {
	ID   string `json:"id"`
	Type string `json:"type"`
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

// ---------- ListModelsWithMetadata ----------

// anthropicModelCapabilities maps known model name PREFIXES to their
// capability sets. Lookup uses prefix matching so versioned model
// IDs (e.g. "claude-3-5-sonnet-20241022") resolve to the family entry.
var anthropicModelCapabilities = map[string][]llm.ModelCapability{
	"claude-3-5-sonnet": {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityStreaming},
	"claude-3-5-haiku":  {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityStreaming},
	"claude-3-opus":     {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityStreaming},
	"claude-3-sonnet":   {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityStreaming},
	"claude-3-haiku":    {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityStreaming},
}

// ListModelsWithMetadata returns the available Anthropic models with
// best-effort capability metadata. Unknown models get a default of
// [Chat, Streaming].
func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	var resp anthropicModelsResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodGet,
		c.baseURL+"/models", c.headers(), nil, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	var infos []llm.ModelInfo
	for _, m := range resp.Data {
		if m.Type == "model" {
			infos = append(infos, llm.ModelInfo{ID: m.ID, Capabilities: lookupAnthropicCapabilities(m.ID)})
		}
	}

	cfg := llm.ListModelsConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return llm.FilterModelInfos(infos, cfg), nil
}

func lookupAnthropicCapabilities(modelID string) []llm.ModelCapability {
	// Use longest-prefix matching so more-specific prefixes take priority.
	bestLen := -1
	var bestCaps []llm.ModelCapability
	for prefix, caps := range anthropicModelCapabilities {
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

type anthropicErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func mapError(status int, body []byte) error {
	var errResp anthropicErrorResponse
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

func normalizeStopReason(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopReasonEndTurn
	case "max_tokens":
		return llm.StopReasonMaxTokens
	case "stop_sequence":
		return llm.StopReasonStopSequence
	case "tool_use":
		return llm.StopReasonToolUse
	default:
		return llm.StopReasonEndTurn
	}
}
