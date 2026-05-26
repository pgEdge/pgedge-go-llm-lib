//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package gemini implements the Google Gemini provider for the LLM client.
package gemini

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
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	providerName   = "gemini"
)

func init() {
	llm.RegisterProvider(providerName, func(opts llm.Options) (llm.Client, error) {
		return New(opts)
	})
}

// client implements llm.Client for Gemini.
type client struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	opts       llm.Options

	mu              sync.Mutex
	cumulativeUsage llm.TokenUsage
}

// New creates a new Gemini client.
func New(opts llm.Options) (llm.Client, error) {
	opts = opts.WithDefaults()

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		var err error
		baseURL, err = httpclient.ValidateBaseURL(baseURL, "gemini")
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
		"Content-Type": "application/json",
	}
	// Pass the API key as a header rather than a query parameter so it
	// does not appear in HTTP intermediary access logs.
	if c.apiKey != "" {
		h["x-goog-api-key"] = c.apiKey
	}
	return h
}

// ---------- Gemini types ----------

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

// geminiInlineData carries inline base64-encoded media. Data is
// auto-base64-encoded by encoding/json when marshalled.
type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

// geminiFileData references media by URI (Gemini's File API or a
// public URL).
type geminiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// geminiToolConfig carries the tool-choice configuration for a Gemini request.
type geminiToolConfig struct {
	FunctionCallingConfig *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// geminiFunctionCallingConfig specifies how the model should call functions.
//
// Mode values: "AUTO" (model decides), "NONE" (no calls), "ANY" (must call one).
// AllowedFunctionNames restricts which functions the model may call (used for
// ToolChoiceSpecific — sets Mode "ANY" and lists the single allowed function).
type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent   `json:"contents"`
	Tools             []geminiTool      `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig `json:"toolConfig,omitempty"`
	SystemInstruction *geminiContent    `json:"systemInstruction,omitempty"`
	GenerationConfig  map[string]any    `json:"generationConfig,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate   `json:"candidates"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type geminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// ---------- Embed types ----------

type geminiEmbedRequest struct {
	Content geminiContent `json:"content"`
}

type geminiEmbedResponse struct {
	Embedding geminiEmbedding `json:"embedding"`
}

type geminiEmbedding struct {
	Values []float64 `json:"values"`
}

// ---------- ListModels types ----------

type geminiModelsResponse struct {
	Models []geminiModelData `json:"models"`
}

type geminiModelData struct {
	Name                       string   `json:"name"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
}

// ---------- Chat ----------

func (c *client) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	gReq := c.buildChatRequest(req)

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent", c.baseURL, c.model)

	var gResp geminiResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		url, c.headers(), gReq, &gResp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	resp := c.parseChatResponse(&gResp)
	c.addUsage(resp.Usage)
	return resp, nil
}

func (c *client) buildChatRequest(req llm.ChatRequest) geminiRequest {
	gReq := geminiRequest{
		Contents: c.convertMessages(req),
	}

	// System prompt via systemInstruction field: per-request only (no client-level default).
	if req.SystemPrompt != "" {
		gReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{
				{Text: req.SystemPrompt},
			},
		}
	}

	// Generation config.
	genConfig := make(map[string]any)

	// MaxTokens: per-request → client default → omit.
	var maxTokens *int
	if req.MaxTokens != nil {
		maxTokens = req.MaxTokens
	} else {
		maxTokens = c.opts.MaxTokens
	}
	if maxTokens != nil {
		genConfig["maxOutputTokens"] = *maxTokens
	}

	// Temperature: per-request → client default → omit (use provider default).
	var temp *float64
	if req.Temperature != nil {
		temp = req.Temperature
	} else {
		temp = c.opts.Temperature
	}
	if temp != nil {
		genConfig["temperature"] = *temp
	}

	// ResponseFormat: Gemini supports native JSON output via responseMimeType
	// and optional responseSchema in generationConfig.
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			genConfig["responseMimeType"] = "application/json"
		case llm.ResponseFormatJSONSchema:
			genConfig["responseMimeType"] = "application/json"
			var schema map[string]any
			if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err == nil {
				genConfig["responseSchema"] = schema
			}
		}
	}

	// StopSequences: passed via generationConfig.stopSequences.
	if len(req.StopSequences) > 0 {
		genConfig["stopSequences"] = req.StopSequences
	}

	if len(genConfig) > 0 {
		gReq.GenerationConfig = genConfig
	}

	// Tools.
	if len(req.Tools) > 0 {
		decls := make([]geminiFunctionDeclaration, len(req.Tools))
		for i, t := range req.Tools {
			decls[i] = geminiFunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			}
		}
		gReq.Tools = []geminiTool{
			{FunctionDeclarations: decls},
		}
	}

	// ToolChoice: map the normalised ToolChoice onto Gemini's toolConfig.
	// Gemini uses mode "AUTO", "NONE", "ANY"; for ToolChoiceSpecific we
	// use mode "ANY" and restrict via allowedFunctionNames.
	if req.ToolChoice != nil {
		cfg := &geminiFunctionCallingConfig{}
		switch req.ToolChoice.Mode {
		case llm.ToolChoiceAuto:
			cfg.Mode = "AUTO"
		case llm.ToolChoiceNone:
			cfg.Mode = "NONE"
		case llm.ToolChoiceRequired:
			cfg.Mode = "ANY"
		case llm.ToolChoiceSpecific:
			cfg.Mode = "ANY"
			cfg.AllowedFunctionNames = []string{req.ToolChoice.Name}
		}
		gReq.ToolConfig = &geminiToolConfig{FunctionCallingConfig: cfg}
	}

	return gReq
}

// convertMessages walks the request messages and emits Gemini's
// Contents array. As tool-use blocks appear, their (ID -> Name)
// mapping is recorded so subsequent tool-result blocks can be
// emitted with the correct functionResponse.name (Gemini requires
// the tool name on the response, but our llm.ContentBlock for a
// tool result carries only ToolUseID).
func (c *client) convertMessages(req llm.ChatRequest) []geminiContent {
	contents := make([]geminiContent, 0, len(req.Messages))

	// Maps tool-use ID -> tool name, populated as we traverse prior
	// assistant messages. Tool results in later messages look up the
	// name here when constructing functionResponse.
	toolNames := map[string]string{}

	for _, m := range req.Messages {
		contents = append(contents, convertMessage(m, toolNames)...)
	}
	return contents
}

// convertMessage maps an llm.Message to one or more geminiContent
// values. toolNames threads through call sites so tool-result blocks
// can recover the function name from a prior tool-use's ID.
//
// Role mapping: assistant -> "model"; user/system/tool -> "user".
// Gemini's wire format only accepts "user" and "model" roles for
// chat content; tool results are emitted as user-role messages
// containing functionResponse parts.
func convertMessage(m llm.Message, toolNames map[string]string) []geminiContent {
	role := mapRole(string(m.Role))

	var parts []geminiPart
	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText:
			parts = append(parts, geminiPart{Text: b.Text})

		case llm.BlockImage:
			if b.Image == nil {
				continue
			}
			if b.Image.URL != "" {
				parts = append(parts, geminiPart{
					FileData: &geminiFileData{
						MimeType: b.Image.MediaType,
						FileURI:  b.Image.URL,
					},
				})
			} else if len(b.Image.Data) > 0 {
				parts = append(parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: b.Image.MediaType,
						Data:     b.Image.Data,
					},
				})
			}

		case llm.BlockDocument:
			if b.Document == nil {
				continue
			}
			if b.Document.URL != "" {
				parts = append(parts, geminiPart{
					FileData: &geminiFileData{
						MimeType: b.Document.MediaType,
						FileURI:  b.Document.URL,
					},
				})
			} else if len(b.Document.Data) > 0 {
				parts = append(parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: b.Document.MediaType,
						Data:     b.Document.Data,
					},
				})
			}

		case llm.BlockToolUse:
			if b.ToolUse == nil {
				continue
			}
			var args map[string]any
			if len(b.ToolUse.Input) > 0 {
				_ = json.Unmarshal(b.ToolUse.Input, &args)
			}
			parts = append(parts, geminiPart{
				FunctionCall: &geminiFunctionCall{
					Name: b.ToolUse.Name,
					Args: args,
				},
			})
			// Remember (ID -> Name) for any later tool-result lookup.
			if b.ToolUse.ID != "" {
				toolNames[b.ToolUse.ID] = b.ToolUse.Name
			}

		case llm.BlockToolResult:
			parts = append(parts, geminiPart{
				FunctionResponse: buildFunctionResponse(b, toolNames),
			})
		}
	}

	if len(parts) == 0 {
		// Gemini requires a non-empty parts array; emit an empty text
		// part as a safe placeholder so we don't drop the message
		// entirely.
		parts = []geminiPart{{Text: ""}}
	}

	return []geminiContent{{Role: role, Parts: parts}}
}

// buildFunctionResponse converts a tool-result block into Gemini's
// functionResponse part. The function name is recovered from the
// toolNames map (populated from prior tool-use blocks); falling back
// to the legacy "gemini-tool-<name>" ID convention; falling back to
// the raw ToolUseID as a last resort.
func buildFunctionResponse(b llm.ContentBlock, toolNames map[string]string) *geminiFunctionResponse {
	name, ok := toolNames[b.ToolUseID]
	if !ok {
		name = strings.TrimPrefix(b.ToolUseID, "gemini-tool-")
	}

	// Parse the result text as JSON if possible; otherwise wrap.
	var responseMap map[string]any
	if err := json.Unmarshal([]byte(b.Text), &responseMap); err != nil {
		responseMap = map[string]any{"result": b.Text}
	}

	return &geminiFunctionResponse{
		Name:     name,
		Response: responseMap,
	}
}

// mapRole maps an llm.Role onto Gemini's wire-format role. Gemini's
// chat content only accepts "user" and "model"; system and tool
// inputs are folded onto "user". System prompts are normally passed
// through ChatRequest.SystemPrompt (emitted as systemInstruction),
// but a system Message in the conversation history is mapped to
// "user" so the upstream API does not reject the request.
func mapRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "tool", "system":
		return "user"
	default:
		return role
	}
}

func (c *client) parseChatResponse(gResp *geminiResponse) *llm.ChatResponse {
	resp := &llm.ChatResponse{
		Usage: llm.TokenUsage{
			PromptTokens:     gResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: gResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      gResp.UsageMetadata.TotalTokenCount,
		},
	}

	if len(gResp.Candidates) == 0 {
		return resp
	}

	candidate := gResp.Candidates[0]
	resp.StopReason = normalizeStopReason(candidate.FinishReason)

	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			resp.Content = append(resp.Content, llm.ContentBlock{
				Type: llm.BlockText,
				Text: part.Text,
			})
		}
		if part.FunctionCall != nil {
			argsJSON, _ := json.Marshal(part.FunctionCall.Args)
			resp.Content = append(resp.Content, llm.ContentBlock{
				Type: llm.BlockToolUse,
				ToolUse: &llm.ToolUse{
					ID:    fmt.Sprintf("gemini-tool-%s", part.FunctionCall.Name),
					Name:  part.FunctionCall.Name,
					Input: json.RawMessage(argsJSON),
				},
			})
		}
	}

	return resp
}

func normalizeStopReason(reason string) llm.StopReason {
	switch reason {
	case "STOP":
		return llm.StopReasonEndTurn
	case "MAX_TOKENS":
		return llm.StopReasonMaxTokens
	case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII":
		return llm.StopReasonContentFilter
	default:
		return llm.StopReasonEndTurn
	}
}

// ---------- ChatStream ----------

func (c *client) ChatStream(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	gReq := c.buildChatRequest(req)

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse",
		c.baseURL, c.model)

	resp, err := httpclient.DoSSERequest(ctx, c.httpClient, http.MethodPost,
		url, c.headers(), gReq)
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
		var lastUsage *geminiUsageMetadata

		for scanner.Scan() {
			data := scanner.Data()

			var chunk geminiResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				errCh <- fmt.Errorf("decode stream chunk: %w", err)
				return
			}

			// Track usage metadata.
			if chunk.UsageMetadata.TotalTokenCount > 0 {
				um := chunk.UsageMetadata
				lastUsage = &um
			}

			if len(chunk.Candidates) == 0 {
				continue
			}

			candidate := chunk.Candidates[0]

			// Emit text parts.
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					chunks <- llm.StreamChunk{
						Type: llm.ChunkText,
						Text: part.Text,
					}
				}
				if part.FunctionCall != nil {
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					chunks <- llm.StreamChunk{
						Type: llm.ChunkToolUseStart,
						ToolUse: &llm.ToolUse{
							ID:    fmt.Sprintf("gemini-tool-%s", part.FunctionCall.Name),
							Name:  part.FunctionCall.Name,
							Input: json.RawMessage(argsJSON),
						},
					}
				}
			}

			// Finish reason.
			if candidate.FinishReason != "" {
				usage := &llm.TokenUsage{}
				if lastUsage != nil {
					usage.PromptTokens = lastUsage.PromptTokenCount
					usage.CompletionTokens = lastUsage.CandidatesTokenCount
					usage.TotalTokens = lastUsage.TotalTokenCount
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

func (c *client) Embed(ctx context.Context, text string) ([]float64, error) {
	reqBody := geminiEmbedRequest{
		Content: geminiContent{
			Parts: []geminiPart{
				{Text: text},
			},
		},
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:embedContent", c.baseURL, c.model)

	var resp geminiEmbedResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		url, c.headers(), reqBody, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, body)
	}

	if len(resp.Embedding.Values) == 0 {
		return nil, &llm.ProviderError{
			Err:      llm.ErrProviderError,
			Message:  "no embedding data returned",
			Provider: providerName,
		}
	}
	return resp.Embedding.Values, nil
}

// EmbedBatch performs sequential Embed calls since Gemini doesn't
// have a native batch embedding endpoint.
func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	result := make([][]float64, len(texts))
	for i, text := range texts {
		embedding, err := c.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		result[i] = embedding
	}
	return result, nil
}

// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Gemini does not support reranking",
		Provider: "gemini",
	}
}

// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
	return nil, &llm.ProviderError{
		Err:      llm.ErrNotSupported,
		Message:  "Gemini does not support multimodal embeddings",
		Provider: "gemini",
	}
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

func supportsGenerateContent(methods []string) bool {
	for _, m := range methods {
		if m == "generateContent" {
			return true
		}
	}
	return false
}

// ---------- ListModelsWithMetadata ----------

var geminiModelCapabilities = map[string][]llm.ModelCapability{
	"gemini-1.5-pro":   {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gemini-1.5-flash": {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"gemini-2.0":       {llm.ModelCapabilityChat, llm.ModelCapabilityTools, llm.ModelCapabilityVision, llm.ModelCapabilityJSONMode, llm.ModelCapabilityStreaming},
	"text-embedding-":  {llm.ModelCapabilityEmbeddings},
	"embedding-":       {llm.ModelCapabilityEmbeddings},
}

func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
	url := fmt.Sprintf("%s/v1beta/models", c.baseURL)

	var resp geminiModelsResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodGet,
		url, c.headers(), nil, &resp)
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
	for _, m := range resp.Models {
		isEmbedding := geminiEmbeddingModel(m.SupportedGenerationMethods)
		if isEmbedding {
			// Embedding models are only included when the caller explicitly
			// requests ModelCapabilityEmbeddings; they are excluded from the
			// default (chat-focused) model list.
			if wantEmbeddings {
				name := strings.TrimPrefix(m.Name, "models/")
				infos = append(infos, llm.ModelInfo{ID: name, Capabilities: lookupGeminiCapabilities(name)})
			}
		} else if supportsGenerateContent(m.SupportedGenerationMethods) {
			// Strip "models/" prefix.
			name := strings.TrimPrefix(m.Name, "models/")
			infos = append(infos, llm.ModelInfo{ID: name, Capabilities: lookupGeminiCapabilities(name)})
		}
	}

	return llm.FilterModelInfos(infos, cfg), nil
}

// geminiEmbeddingModel reports whether the model's supported methods indicate
// it is an embedding model (not a chat/generation model).
func geminiEmbeddingModel(methods []string) bool {
	for _, m := range methods {
		if m == "embedContent" || m == "batchEmbedContents" {
			return true
		}
	}
	return false
}

func lookupGeminiCapabilities(modelID string) []llm.ModelCapability {
	// Gemini's ListModels returns "models/<id>" — strip if needed.
	id := strings.TrimPrefix(modelID, "models/")
	// Use longest-prefix matching so more-specific prefixes take priority.
	bestLen := -1
	var bestCaps []llm.ModelCapability
	for prefix, caps := range geminiModelCapabilities {
		if strings.HasPrefix(id, prefix) && len(prefix) > bestLen {
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

type geminiErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func mapError(status int, body []byte) error {
	var errResp geminiErrorResponse
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
