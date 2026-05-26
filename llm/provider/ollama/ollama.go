//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package ollama implements the Ollama provider for the LLM client.
package ollama

import (
	"bytes"
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
	defaultBaseURL = "http://localhost:11434"
	providerName   = "ollama"
)

func init() {
	llm.RegisterProvider(providerName, func(opts llm.Options) (llm.Client, error) {
		return New(opts)
	})
}

// client implements llm.Client for Ollama.
type client struct {
	httpClient *http.Client
	model      string
	baseURL    string
	opts       llm.Options

	mu              sync.Mutex
	cumulativeUsage llm.TokenUsage
}

// New creates a new Ollama client.
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		var err error
		baseURL, err = httpclient.ValidateBaseURL(baseURL, "ollama")
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
	return map[string]string{
		"Content-Type": "application/json",
	}
}

// rejectDocumentBlocks returns ErrNotSupported if any message in the
// request carries a document content block. Ollama has no native
// document (PDF) input path; the caller must pre-extract text or use
// a provider with native document support (Anthropic, Gemini).
func rejectDocumentBlocks(req llm.ChatRequest) error {
	for _, m := range req.Messages {
		for _, b := range m.Content {
			if b.Type == llm.BlockDocument {
				return &llm.ProviderError{
					Err:      llm.ErrNotSupported,
					Message:  "Ollama does not support document content blocks; pre-extract text or use a provider with native document support (Anthropic, Gemini)",
					Provider: providerName,
				}
			}
		}
	}
	return nil
}

// ---------- Chat ----------

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Format   any             `json:"format,omitempty"` // "json" string or schema object
	Options  map[string]any  `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ollamaChatResponse struct {
	Message         ollamaRespMessage `json:"message"`
	Done            bool              `json:"done"`
	PromptEvalCount int               `json:"prompt_eval_count,omitempty"`
	EvalCount       int               `json:"eval_count,omitempty"`
}

type ollamaRespMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (c *client) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if err := rejectDocumentBlocks(req); err != nil {
		return nil, err
	}
	ollamaReq, err := c.buildChatRequest(req, false)
	if err != nil {
		return nil, err
	}

	var ollamaResp ollamaChatResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/api/chat", c.headers(), ollamaReq, &ollamaResp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	resp := c.parseChatResponse(&ollamaResp, len(req.Tools) > 0)
	c.addUsage(resp.Usage)
	return resp, nil
}

func (c *client) buildChatRequest(req llm.ChatRequest, stream bool) (ollamaChatRequest, error) {
	msgs, err := c.convertMessages(req)
	if err != nil {
		return ollamaChatRequest{}, err
	}
	ollamaReq := ollamaChatRequest{
		Model:    c.model,
		Messages: msgs,
		Stream:   stream,
	}
	// ToolChoice: Ollama has no native tool_choice wire field — its tool-call
	// support is implemented via prompt engineering (see buildToolInstructions).
	// ChatRequest.ToolChoice is intentionally ignored here.

	// ResponseFormat: Ollama supports native JSON output via the format field.
	// Recent versions accept "json" (free-form) or a schema object directly.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			ollamaReq.Format = "json"
		case llm.ResponseFormatJSONSchema:
			var schema any
			if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err == nil {
				ollamaReq.Format = schema
			}
		}
	}

	// StopSequences: Ollama accepts stop sequences in the options map.
	if len(req.StopSequences) > 0 {
		if ollamaReq.Options == nil {
			ollamaReq.Options = make(map[string]any)
		}
		ollamaReq.Options["stop"] = req.StopSequences
	}

	return ollamaReq, nil
}

// convertMessages flattens llm.Messages into Ollama's wire shape.
//
// Per-block handling:
//   - BlockText: appended to the message's content string.
//   - BlockImage: base64-encoded data is added to the message's
//     images array. URL-only images return an error since Ollama
//     does not support URL image input.
//   - BlockToolUse: serialised to JSON and concatenated into content,
//     matching the format Ollama models emit when calling tools.
//   - BlockToolResult: emitted as a role:"tool" message with the
//     result text as content.
func (c *client) convertMessages(req llm.ChatRequest) ([]ollamaMessage, error) {
	var msgs []ollamaMessage

	// Build system prompt: per-request only (no client-level default).
	sysPrompt := req.SystemPrompt

	// If tools are provided, inject tool instructions into the system prompt.
	if len(req.Tools) > 0 {
		toolInstructions := buildToolInstructions(req.Tools)
		if sysPrompt != "" {
			sysPrompt = sysPrompt + "\n\n" + toolInstructions
		} else {
			sysPrompt = toolInstructions
		}
	}

	if sysPrompt != "" {
		msgs = append(msgs, ollamaMessage{
			Role:    "system",
			Content: sysPrompt,
		})
	}

	for _, m := range req.Messages {
		converted, err := convertMessage(m)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, converted...)
	}
	return msgs, nil
}

// convertMessage maps a single llm.Message into one or more
// ollamaMessage values. Tool-result blocks become role:"tool"
// messages; everything else is collapsed onto the original role.
func convertMessage(m llm.Message) ([]ollamaMessage, error) {
	role := string(m.Role)

	var contentSB strings.Builder
	var images []string
	var toolResults []ollamaMessage

	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText:
			contentSB.WriteString(b.Text)

		case llm.BlockImage:
			if b.Image == nil {
				continue
			}
			if len(b.Image.Data) > 0 {
				images = append(images, base64.StdEncoding.EncodeToString(b.Image.Data))
				continue
			}
			if b.Image.URL != "" {
				return nil, fmt.Errorf("ollama: URL image input is not supported (provide base64 data instead)")
			}

		case llm.BlockToolUse:
			if b.ToolUse == nil {
				continue
			}
			// Serialise the tool call as JSON in the content stream
			// using the same shape we instruct the model to emit.
			payload := map[string]any{
				"tool":      b.ToolUse.Name,
				"arguments": b.ToolUse.Input,
			}
			data, _ := json.Marshal(payload)
			contentSB.Write(data)

		case llm.BlockToolResult:
			toolResults = append(toolResults, ollamaMessage{
				Role:    "tool",
				Content: b.Text,
			})
		}
	}

	var out []ollamaMessage

	// Emit the primary message only if it carries content or images.
	// Pure tool-result messages skip the empty "tool"-role primary
	// message and emit only their tool entries below.
	if contentSB.Len() > 0 || len(images) > 0 || (len(toolResults) == 0 && len(m.Content) == 0) {
		out = append(out, ollamaMessage{
			Role:    role,
			Content: contentSB.String(),
			Images:  images,
		})
	}

	out = append(out, toolResults...)
	return out, nil
}

func buildToolInstructions(tools []llm.Tool) string {
	var sb strings.Builder
	sb.WriteString("You have access to the following tools. When you need to call a tool, respond ONLY with a JSON object in this exact format:\n")
	sb.WriteString(`{"tool":"tool_name","arguments":{...}}`)
	sb.WriteString("\n\nAvailable tools:\n")

	for _, t := range tools {
		fmt.Fprintf(&sb, "- %s: %s", t.Name, t.Description)
		if len(t.InputSchema) > 0 {
			fmt.Fprintf(&sb, " (parameters: %s)", string(t.InputSchema))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

func (c *client) parseChatResponse(ollamaResp *ollamaChatResponse, hasTools bool) *llm.ChatResponse {
	usage := llm.TokenUsage{
		PromptTokens:     ollamaResp.PromptEvalCount,
		CompletionTokens: ollamaResp.EvalCount,
		TotalTokens:      ollamaResp.PromptEvalCount + ollamaResp.EvalCount,
	}
	resp := &llm.ChatResponse{
		Usage:      usage,
		StopReason: llm.StopReasonEndTurn,
	}

	// Strip <think>...</think> reasoning blocks. Reasoning models such as
	// deepseek-r1 emit these inline; if we leave them in place they
	// (a) break tool-call extraction when example JSON appears inside the
	// thinking block and (b) corrupt JSON-mode output.
	content := stripThinkTags(ollamaResp.Message.Content)
	if content == "" {
		return resp
	}

	// If tools were provided, try to parse content as a tool call.
	if hasTools {
		if toolCall := tryParseToolCall(content); toolCall != nil {
			resp.StopReason = llm.StopReasonToolUse
			resp.Content = []llm.ContentBlock{
				{
					Type:    llm.BlockToolUse,
					ToolUse: toolCall,
				},
			}
			return resp
		}
	}

	resp.Content = []llm.ContentBlock{
		{
			Type: llm.BlockText,
			Text: content,
		},
	}
	return resp
}

// stripThinkTags removes <think>...</think> blocks (case-insensitive,
// multi-line) from the response content. Reasoning models such as
// deepseek-r1 emit these blocks inline with their final answer.
// Downstream tool-call extraction and JSON parsing break on them, so
// the provider strips them at the boundary — analogous to the
// non-string-summary coercion that ai-dba-workbench needed when
// rendering reasoning-model output.
//
// Unterminated tags are tolerated: a leading "<think>" with no
// matching close consumes the remainder of the input. Whitespace
// immediately following the closing tag is also trimmed so the result
// looks like the model emitted only its final answer.
func stripThinkTags(s string) string {
	if s == "" {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	rest := s
	for {
		openIdx := indexFoldASCII(rest, "<think>")
		if openIdx < 0 {
			b.WriteString(rest)
			break
		}
		b.WriteString(rest[:openIdx])
		after := rest[openIdx+len("<think>"):]
		closeIdx := indexFoldASCII(after, "</think>")
		if closeIdx < 0 {
			// Unterminated — drop the rest entirely.
			break
		}
		rest = after[closeIdx+len("</think>"):]
		// Trim a single immediately-following newline/whitespace run so
		// the surviving answer is not preceded by blank lines.
		rest = strings.TrimLeft(rest, " \t\r\n")
	}
	return strings.TrimSpace(b.String())
}

// indexFoldASCII returns the index of the first ASCII-case-insensitive
// occurrence of needle in s, or -1 if not found.
func indexFoldASCII(s, needle string) int {
	if needle == "" {
		return 0
	}
	for i := 0; i+len(needle) <= len(s); i++ {
		if strings.EqualFold(s[i:i+len(needle)], needle) {
			return i
		}
	}
	return -1
}

// tryParseToolCall attempts to parse the content as a tool call.
// First tries direct JSON parse, then tries extracting JSON from text.
func tryParseToolCall(content string) *llm.ToolUse {
	// Try direct JSON parse.
	var toolCall struct {
		Tool      string          `json:"tool"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(content), &toolCall); err == nil && toolCall.Tool != "" {
		return &llm.ToolUse{
			ID:    "ollama-tool-1",
			Name:  toolCall.Tool,
			Input: toolCall.Arguments,
		}
	}

	// Try extracting JSON from surrounding text.
	extracted := extractJSONFromText(content)
	if extracted == "" {
		return nil
	}
	if err := json.Unmarshal([]byte(extracted), &toolCall); err == nil && toolCall.Tool != "" {
		return &llm.ToolUse{
			ID:    "ollama-tool-1",
			Name:  toolCall.Tool,
			Input: toolCall.Arguments,
		}
	}

	return nil
}

// extractJSONFromText uses brace matching to find the first complete
// JSON object in a string that may contain surrounding text.
func extractJSONFromText(text string) string {
	firstBrace := strings.Index(text, "{")
	if firstBrace == -1 {
		return ""
	}
	braceCount := 0
	lastBrace := -1
	for i := firstBrace; i < len(text); i++ {
		if text[i] == '{' {
			braceCount++
		} else if text[i] == '}' {
			braceCount--
			if braceCount == 0 {
				lastBrace = i
				break
			}
		}
	}
	if lastBrace == -1 {
		return ""
	}
	return text[firstBrace : lastBrace+1]
}

// ---------- ChatStream ----------

func (c *client) ChatStream(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	if err := rejectDocumentBlocks(req); err != nil {
		return nil, err
	}
	ollamaReq, err := c.buildChatRequest(req, true)
	if err != nil {
		return nil, err
	}

	resp, err := httpclient.DoSSERequest(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/api/chat", c.headers(), ollamaReq)
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
	hasTools := len(req.Tools) > 0

	go func() {
		defer resp.Body.Close()
		defer close(chunks)
		defer close(errCh)

		decoder := json.NewDecoder(resp.Body)
		var accumulated strings.Builder

		for decoder.More() {
			var chunk ollamaChatResponse
			if err := decoder.Decode(&chunk); err != nil {
				errCh <- fmt.Errorf("decode stream chunk: %w", err)
				return
			}

			if chunk.Message.Content != "" {
				accumulated.WriteString(chunk.Message.Content)

				// Only emit text chunks if we're not looking for tool calls,
				// or if we haven't detected one yet. We'll re-emit as tool
				// call at the end if needed.
				if !hasTools {
					chunks <- llm.StreamChunk{
						Type: llm.ChunkText,
						Text: chunk.Message.Content,
					}
				}
			}

			if chunk.Done {
				// Strip <think>...</think> reasoning blocks before
				// tool-call detection / final text emission so reasoning
				// models such as deepseek-r1 don't break tool extraction
				// or pollute buffered text.
				fullContent := stripThinkTags(accumulated.String())

				// If tools were provided, check if accumulated content is a tool call.
				if hasTools {
					if toolCall := tryParseToolCall(fullContent); toolCall != nil {
						chunks <- llm.StreamChunk{
							Type: llm.ChunkToolUseStart,
							ToolUse: &llm.ToolUse{
								ID:    toolCall.ID,
								Name:  toolCall.Name,
								Input: toolCall.Input,
							},
						}
					} else {
						// Not a tool call; emit accumulated text.
						if fullContent != "" {
							chunks <- llm.StreamChunk{
								Type: llm.ChunkText,
								Text: fullContent,
							}
						}
					}
				}

				usage := &llm.TokenUsage{
					PromptTokens:     chunk.PromptEvalCount,
					CompletionTokens: chunk.EvalCount,
					TotalTokens:      chunk.PromptEvalCount + chunk.EvalCount,
				}
				c.addUsage(*usage)
				chunks <- llm.StreamChunk{
					Type:  llm.ChunkDone,
					Usage: usage,
				}
				return
			}
		}
	}()

	return &llm.Stream{
		Chunks: chunks,
		Err:    errCh,
	}, nil
}

// ---------- Embed ----------

type ollamaEmbedRequest struct {
	Model   string         `json:"model"`
	Input   string         `json:"input"`
	Options map[string]any `json:"options,omitempty"`
}

// embedOptions returns Ollama's per-request "options" map populated
// from any ollama.Extension on the client. Returns nil when no
// extension is present or all knobs are at their zero value so the
// "options" object is omitted from the wire body entirely.
func (c *client) embedOptions() map[string]any {
	ext := findExtension(c.opts.Extensions)
	if ext == nil {
		return nil
	}
	var opts map[string]any
	if ext.EmbedContextLength > 0 {
		opts = map[string]any{"num_ctx": ext.EmbedContextLength}
	}
	return opts
}

type ollamaEmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (c *client) Embed(ctx context.Context, text string) ([]float64, error) {
	emb, status, body, err := c.embedOnce(ctx, text)
	if err == nil {
		return emb, nil
	}

	ext := findExtension(c.opts.Extensions)
	if ext == nil || !ext.EmbedTruncateOnOverflow {
		return nil, err
	}
	if !shouldRetryWithTruncation(status, body) {
		return nil, err
	}

	// Three retries at progressively smaller fractions of the
	// original input, cut at a word boundary. If all three fail the
	// original (full-text) error is returned per the spec — the
	// truncated errors are intentionally discarded since they are
	// less informative.
	for _, frac := range []float64{0.75, 0.50, 0.25} {
		truncated := truncateAtWordBoundary(text, frac)
		if truncated == "" || truncated == text {
			continue
		}
		if emb, _, _, retryErr := c.embedOnce(ctx, truncated); retryErr == nil {
			return emb, nil
		}
	}
	return nil, err
}

// embedOnce performs one /api/embed call without retry. status is 0
// when the request never reached the server (network error); body is
// nil in that case. On non-2xx responses body is the raw response body
// so the caller can inspect it for truncate-retry eligibility.
func (c *client) embedOnce(ctx context.Context, text string) ([]float64, int, []byte, error) {
	reqBody := ollamaEmbedRequest{
		Model:   c.model,
		Input:   text,
		Options: c.embedOptions(),
	}

	var resp ollamaEmbedResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/api/embed", c.headers(), reqBody, &resp)
	if err != nil && status == 0 {
		return nil, 0, nil, err
	}
	if status < 200 || status >= 300 {
		return nil, status, body, mapError(status, body)
	}

	if len(resp.Embeddings) == 0 {
		return nil, status, body, &llm.ProviderError{
			Err:      llm.ErrProviderError,
			Message:  "no embedding data returned",
			Provider: providerName,
		}
	}
	return resp.Embeddings[0], status, body, nil
}

// shouldRetryWithTruncation reports whether a failed /api/embed call
// is a candidate for truncation retry. The two known-deterministic
// failure modes are HTTP 500 (model-runner crash) and any status
// whose body contains Ollama's context-overflow message.
func shouldRetryWithTruncation(status int, body []byte) bool {
	if status == http.StatusInternalServerError {
		return true
	}
	return bytes.Contains(body, []byte("the input length exceeds the context length"))
}

// truncateAtWordBoundary returns text cut at a word boundary near
// fraction * len(text) bytes. It walks backwards from the target byte
// offset to the previous ASCII space and cuts there (excluding the
// space). If no space is found between the start and the target, it
// falls back to a hard byte cut at the target so each retry makes
// progress on inputs that contain no spaces.
//
// Returns "" when the target is at or below zero (text is too short
// for the fraction to leave any content), signalling the caller to
// skip this retry.
func truncateAtWordBoundary(text string, fraction float64) string {
	target := int(float64(len(text)) * fraction)
	if target <= 0 {
		return ""
	}
	if target >= len(text) {
		return text
	}
	for i := target; i > 0; i-- {
		if text[i] == ' ' {
			return text[:i]
		}
	}
	return text[:target]
}

func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	result := make([][]float64, len(texts))
	for i, text := range texts {
		embedding, err := c.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = embedding
	}
	return result, nil
}

// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Ollama does not support reranking",
		Provider: "ollama",
	}
}

// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Ollama does not support multimodal embeddings",
		Provider: "ollama",
	}
}

// ---------- ListModels ----------

type ollamaTagsResponse struct {
	Models []ollamaModelInfo `json:"models"`
}

type ollamaModelInfo struct {
	Name string `json:"name"`
}

// ollamaShowRequest is the request body for Ollama's /api/show endpoint.
type ollamaShowRequest struct {
	Name string `json:"name"`
}

// ollamaShowResponse is the response from Ollama's /api/show endpoint.
// Only the fields relevant to capability detection are included.
type ollamaShowResponse struct {
	// Capabilities is populated by Ollama ≥ 0.3; older versions omit it.
	// Example values: "completion", "embedding", "tools", "vision".
	Capabilities []string `json:"capabilities"`
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

func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	var resp ollamaTagsResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodGet,
		c.baseURL+"/api/tags", c.headers(), nil, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	infos := make([]llm.ModelInfo, len(resp.Models))
	for i, m := range resp.Models {
		caps := c.capabilitiesForModel(ctx, m.Name)
		infos[i] = llm.ModelInfo{
			ID:           m.Name,
			Capabilities: caps,
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

// capabilitiesForModel queries /api/show for the given model name and maps
// Ollama's capability strings to llm.ModelCapability values.
//
// If /api/show is unavailable or returns an empty capabilities list (older
// Ollama versions that predate the field), the function falls back to
// [Chat, Streaming] as a safe default for backward compatibility.
//
// For models that advertise "completion", Chat and Streaming are added.
// For models that advertise "embedding" without "completion", only
// ModelCapabilityEmbeddings is returned — chat capabilities are not
// appropriate for embedding-only models.
func (c *client) capabilitiesForModel(ctx context.Context, name string) []llm.ModelCapability {
	var showResp ollamaShowResponse
	status, _, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/api/show", c.headers(), ollamaShowRequest{Name: name}, &showResp)
	if err != nil || status < 200 || status >= 300 || len(showResp.Capabilities) == 0 {
		// Fall back to the safe default for older Ollama instances or errors.
		return []llm.ModelCapability{llm.ModelCapabilityChat, llm.ModelCapabilityStreaming}
	}

	var caps []llm.ModelCapability
	for _, cap := range showResp.Capabilities {
		switch cap {
		case "completion":
			caps = append(caps, llm.ModelCapabilityChat, llm.ModelCapabilityStreaming)
		case "embedding":
			caps = append(caps, llm.ModelCapabilityEmbeddings)
		case "tools":
			caps = append(caps, llm.ModelCapabilityTools)
		case "vision":
			caps = append(caps, llm.ModelCapabilityVision)
		}
	}

	// If no recognised capabilities were extracted but the response was non-empty,
	// fall back to Chat+Streaming to avoid returning an empty slice for unknown
	// future capability strings.
	if len(caps) == 0 {
		return []llm.ModelCapability{llm.ModelCapabilityChat, llm.ModelCapabilityStreaming}
	}

	// If the model only advertised "embedding" (no "completion"), do NOT add
	// chat/streaming defaults — embedding-only models should not appear when
	// callers filter for chat-capable models.
	return caps
}

// ---------- Error mapping ----------

type ollamaErrorResponse struct {
	Error string `json:"error"`
}

func mapError(status int, body []byte) error {
	var errResp ollamaErrorResponse
	_ = json.Unmarshal(body, &errResp) // best-effort; fall back to status-based message below

	msg := errResp.Error
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", status)
	}

	// Ollama is local, so all errors map to ErrProviderError.
	return &llm.ProviderError{
		Err:        llm.ErrProviderError,
		StatusCode: status,
		Message:    msg,
		Provider:   providerName,
	}
}
