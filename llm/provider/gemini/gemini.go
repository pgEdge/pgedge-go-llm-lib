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
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/redact"
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

	// ThoughtSignature is an opaque token that Gemini's thinking
	// models attach to the part carrying a function call, so they can
	// resume their own reasoning on the following turn. It must be
	// echoed back verbatim, on the same part, whenever that function
	// call is replayed as conversation history; a Gemini 3 series
	// model refuses the whole request with a 400 rather than merely
	// degrading if it is missing. Where a response contains parallel
	// function calls only the first part carries a signature, so an
	// empty value on later parts is expected and not an error.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
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
	Embedding     geminiEmbedding     `json:"embedding"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
}

type geminiEmbedding struct {
	Values []float64 `json:"values"`
}

// geminiBatchEmbedRequest is the payload for models/{model}:batchEmbedContents.
// Each sub-request must repeat the fully-qualified "models/{model}" name —
// this is a quirk of the batch endpoint.
type geminiBatchEmbedRequest struct {
	Requests []geminiBatchEmbedSubRequest `json:"requests"`
}

type geminiBatchEmbedSubRequest struct {
	Model   string        `json:"model"`
	Content geminiContent `json:"content"`
}

type geminiBatchEmbedResponse struct {
	Embeddings    []geminiEmbedding   `json:"embeddings"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
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
	gReq, err := c.buildChatRequest(req)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent", c.baseURL, c.model)

	var gResp geminiResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		url, c.headers(), gReq, &gResp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, c.mapError(status, body)
	}

	resp := c.parseChatResponse(&gResp)
	c.addUsage(resp.Usage)
	return resp, nil
}

func (c *client) buildChatRequest(req llm.ChatRequest) (geminiRequest, error) {
	contents, err := c.convertMessages(req)
	if err != nil {
		return geminiRequest{}, err
	}
	gReq := geminiRequest{
		Contents: contents,
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
		useCompact := req.UseCompactDescriptions(c.baseURL)
		decls := make([]geminiFunctionDeclaration, len(req.Tools))
		for i, t := range req.Tools {
			decls[i] = geminiFunctionDeclaration{
				Name:        t.Name,
				Description: llm.EffectiveToolDescription(t, useCompact),
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

	return gReq, nil
}

// convertMessages walks the request messages and emits Gemini's
// Contents array. As tool-use blocks appear, their (ID -> Name)
// mapping is recorded so subsequent tool-result blocks can be
// emitted with the correct functionResponse.name (Gemini requires
// the tool name on the response, but our llm.ContentBlock for a
// tool result carries only ToolUseID).
//
// A message whose content yields no usable parts is dropped (see
// convertMessage). If that leaves no contents at all, the request is
// unsendable, so an error wrapping llm.ErrInvalidRequest is returned
// rather than letting Gemini reject an empty contents array with an
// opaque 400.
func (c *client) convertMessages(req llm.ChatRequest) ([]geminiContent, error) {
	contents := make([]geminiContent, 0, len(req.Messages))

	// Maps tool-use ID -> tool name, populated as we traverse prior
	// assistant messages. Tool results in later messages look up the
	// name here when constructing functionResponse.
	toolNames := map[string]string{}

	for _, m := range req.Messages {
		contents = append(contents, convertMessage(m, toolNames)...)
	}

	if len(contents) == 0 {
		msg := "no messages to send: the request carries no messages"
		if len(req.Messages) > 0 {
			msg = "no messages to send: every message had empty or " +
				"unrepresentable content"
		}
		return nil, &llm.ProviderError{
			Err:      llm.ErrInvalidRequest,
			Message:  msg,
			Provider: providerName,
		}
	}

	return contents, nil
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
			// An empty text block must never reach the wire.
			// geminiPart.Text is tagged omitempty, so a part carrying
			// only an empty string marshals to {} and Gemini rejects
			// the whole request with a 400 complaining that
			// contents[N].parts[M].data, a required oneof, has none
			// of its fields set. Empty text carries no information,
			// so drop it.
			if b.Text == "" {
				continue
			}
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
				// Echo back the thought signature captured when the
				// model made this call; without it a thinking model
				// rejects the request outright.
				ThoughtSignature: b.ToolUse.Signature,
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
		// The message yielded nothing representable: an empty Content
		// slice, only empty text, an image or document with neither a
		// URL nor inline data, or a nil ToolUse. Drop the message
		// rather than emitting a placeholder part.
		//
		// This used to emit geminiPart{Text: ""} on the reasoning that
		// a placeholder was safer than dropping the message, but that
		// was wrong: Text is tagged omitempty, so the placeholder
		// marshals to [{}] and Gemini rejects it, complaining that
		// parts[0].data, a required oneof, has none of its fields
		// set, which fails the entire request. A message
		// with no representable content carries no information, so
		// dropping it loses nothing, whereas the placeholder turned a
		// harmless gap into a guaranteed 400. convertMessages simply
		// appends our result, so returning an empty slice drops the
		// message cleanly; convertMessages then guards against every
		// message being dropped, which would leave an empty contents
		// array.
		return nil
	}

	return []geminiContent{{Role: role, Parts: parts}}
}

// unspecifiedToolError is the placeholder used when a tool result is
// flagged as an error but carries no text. An empty payload would be
// indistinguishable from a successful call that returned nothing, so
// something explicit is sent instead.
const unspecifiedToolError = "tool execution failed; no error message was provided"

// buildFunctionResponse converts a tool-result block into Gemini's
// functionResponse part. The function name is recovered from the
// toolNames map (populated from prior tool-use blocks); falling back
// to the legacy "gemini-tool-<name>" ID convention; falling back to
// the raw ToolUseID as a last resort.
//
// A failed tool result (IsError) is reported under an "error" key
// rather than the usual "result". Gemini's functionResponse.response
// is a free-form JSON object with no documented field for failures,
// so this is a convention rather than a protocol requirement, but it
// is the one the ecosystem has settled on and it makes the failure
// legible to the model. Without it an errored result looks like an
// ordinary success, and the model happily reissues the same call
// until the caller's agentic loop hits its cap.
func buildFunctionResponse(b llm.ContentBlock, toolNames map[string]string) *geminiFunctionResponse {
	name, ok := toolNames[b.ToolUseID]
	if !ok {
		name = strings.TrimPrefix(b.ToolUseID, "gemini-tool-")
	}

	key := "result"
	if b.IsError {
		key = "error"
	}

	// Parse the result text as JSON if possible; otherwise wrap. A
	// structured error payload is kept structured, nested under the
	// "error" key, rather than being flattened back to a string.
	var responseMap map[string]any
	if b.IsError && b.Text == "" {
		responseMap = map[string]any{key: unspecifiedToolError}
	} else if err := json.Unmarshal([]byte(b.Text), &responseMap); err != nil {
		responseMap = map[string]any{key: b.Text}
	} else if b.IsError {
		responseMap = map[string]any{key: responseMap}
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
					ID:        fmt.Sprintf("gemini-tool-%s", part.FunctionCall.Name),
					Name:      part.FunctionCall.Name,
					Input:     json.RawMessage(argsJSON),
					Signature: part.ThoughtSignature,
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
	gReq, err := c.buildChatRequest(req)
	if err != nil {
		return nil, err
	}

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
		return nil, c.mapError(resp.StatusCode, body[:n])
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
							ID:        fmt.Sprintf("gemini-tool-%s", part.FunctionCall.Name),
							Name:      part.FunctionCall.Name,
							Input:     json.RawMessage(argsJSON),
							Signature: part.ThoughtSignature,
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
		return nil, c.mapError(status, body)
	}

	if len(resp.Embedding.Values) == 0 {
		return nil, &llm.ProviderError{
			Err:      llm.ErrProviderError,
			Message:  "no embedding data returned",
			Provider: providerName,
		}
	}
	// Gemini's embed endpoints report only promptTokenCount; embeddings
	// have no completion tokens, so it doubles as the total.
	c.addUsage(llm.TokenUsage{
		PromptTokens: resp.UsageMetadata.PromptTokenCount,
		TotalTokens:  resp.UsageMetadata.PromptTokenCount,
	})
	return resp.Embedding.Values, nil
}

// EmbedBatch sends all texts in a single batchEmbedContents request.
// The response's embeddings array is returned in the same order as the
// input texts.
func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	// The batch endpoint requires each sub-request to carry the
	// fully-qualified "models/{model}" name, even though the same
	// model also appears in the URL path.
	qualifiedModel := "models/" + c.model
	subRequests := make([]geminiBatchEmbedSubRequest, len(texts))
	for i, text := range texts {
		subRequests[i] = geminiBatchEmbedSubRequest{
			Model: qualifiedModel,
			Content: geminiContent{
				Parts: []geminiPart{{Text: text}},
			},
		}
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents", c.baseURL, c.model)

	var resp geminiBatchEmbedResponse
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		url, c.headers(), geminiBatchEmbedRequest{Requests: subRequests}, &resp)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, c.mapError(status, body)
	}

	if len(resp.Embeddings) != len(texts) {
		return nil, &llm.ProviderError{
			Err: llm.ErrProviderError,
			Message: fmt.Sprintf("batch embed returned %d embeddings for %d inputs",
				len(resp.Embeddings), len(texts)),
			Provider: providerName,
		}
	}

	result := make([][]float64, len(texts))
	for i, emb := range resp.Embeddings {
		if len(emb.Values) == 0 {
			return nil, &llm.ProviderError{
				Err:      llm.ErrProviderError,
				Message:  fmt.Sprintf("no embedding data returned for input %d", i),
				Provider: providerName,
			}
		}
		result[i] = emb.Values
	}
	// Gemini's embed endpoints report only promptTokenCount; embeddings
	// have no completion tokens, so it doubles as the total.
	c.addUsage(llm.TokenUsage{
		PromptTokens: resp.UsageMetadata.PromptTokenCount,
		TotalTokens:  resp.UsageMetadata.PromptTokenCount,
	})
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
		return nil, c.mapError(status, body)
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
			//
			// Dimensions is intentionally left 0: Gemini embedding dimension
			// is model-specific and only known at runtime from the response
			// vector length, not from static metadata.
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

// mapError converts a non-2xx response into a *llm.ProviderError.
//
// The provider's own message is redacted before it is stored: an
// upstream API may quote part of the submitted credential back in the
// message on an authentication failure, and callers routinely surface
// Error() to untrusted readers. The unredacted text is never retained.
func (c *client) mapError(status int, body []byte) error {
	var errResp geminiErrorResponse
	_ = json.Unmarshal(body, &errResp) // best-effort; fall back to status-based message below

	msg := redact.Message(errResp.Error.Message, c.apiKey)
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
