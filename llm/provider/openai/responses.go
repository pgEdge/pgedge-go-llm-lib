//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

// useResponsesAPI reports whether a Chat / ChatStream call should be
// routed to /v1/responses instead of /v1/chat/completions. An explicit
// openai.Extension{ResponsesAPI: ...} override takes precedence; when
// unset, the decision is auto-detected from the model name.
func useResponsesAPI(model string, exts []llm.ProviderExtension) bool {
	if ext := findExtension(exts); ext != nil && ext.ResponsesAPI != nil {
		return *ext.ResponsesAPI
	}
	return modelRequiresResponsesAPI(model)
}

// modelRequiresResponsesAPI reports whether the named model rejects
// /v1/chat/completions and must be invoked via /v1/responses. Current
// members of the set: o1*, o3*, gpt-5*.
func modelRequiresResponsesAPI(model string) bool {
	return strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "gpt-5")
}

// ---------- Wire types: request ----------

// responsesRequest is the request body for POST /v1/responses.
type responsesRequest struct {
	Model           string                  `json:"model"`
	Input           []responsesInputItem    `json:"input"`
	Instructions    string                  `json:"instructions,omitempty"`
	MaxOutputTokens *int                    `json:"max_output_tokens,omitempty"`
	Temperature     *float64                `json:"temperature,omitempty"`
	Tools           []responsesTool         `json:"tools,omitempty"`
	ToolChoice      any                     `json:"tool_choice,omitempty"`
	Stream          bool                    `json:"stream,omitempty"`
	Text            *responsesTextFormatCfg `json:"text,omitempty"`
}

// responsesInputItem is one element of the Responses API "input" array.
// The Type field discriminates which other fields are populated:
//
//	"message"              -> Role, Content
//	"function_call"        -> CallID, Name, Arguments
//	"function_call_output" -> CallID, Output
//
// Some fields are pointers so they marshal to the wire only when set,
// which keeps the request body free of empty discriminator artefacts.
type responsesInputItem struct {
	Type      string                 `json:"type,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Content   []responsesContentPart `json:"content,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
	Output    string                 `json:"output,omitempty"`
}

// responsesContentPart is one element of an input message's content
// array. Use "input_text" / "input_image" for user/system messages and
// "output_text" when echoing prior assistant text back to the model.
type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type responsesTextFormatCfg struct {
	Format responsesTextFormat `json:"format"`
}

type responsesTextFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}

// ---------- Wire types: response (non-streaming) ----------

// responsesResponse is the response body from POST /v1/responses
// (non-streaming) and from the response.completed SSE event.
type responsesResponse struct {
	Output            []responsesOutputItem  `json:"output"`
	Usage             *responsesUsage        `json:"usage,omitempty"`
	IncompleteDetails *responsesIncompleteEx `json:"incomplete_details,omitempty"`
	Status            string                 `json:"status,omitempty"`
}

type responsesIncompleteEx struct {
	Reason string `json:"reason"`
}

// responsesOutputItem is one element of the "output" array. The Type
// field selects which fields are populated:
//
//	"message"       -> Role, Content (parts include output_text)
//	"function_call" -> CallID, Name, Arguments
//	"reasoning"     -> Summary (currently ignored — reasoning tokens
//	                   are counted in Usage; the summary text is not
//	                   surfaced through the unified ChatResponse)
type responsesOutputItem struct {
	Type      string                       `json:"type"`
	Role      string                       `json:"role,omitempty"`
	Content   []responsesOutputContentPart `json:"content,omitempty"`
	CallID    string                       `json:"call_id,omitempty"`
	Name      string                       `json:"name,omitempty"`
	Arguments string                       `json:"arguments,omitempty"`
}

type responsesOutputContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ---------- Wire types: response (streaming) ----------

// responsesStreamEvent captures the union of the SSE event payloads we
// consume. The Type field selects which other fields carry meaningful
// data; unknown event types are ignored.
type responsesStreamEvent struct {
	Type     string               `json:"type"`
	Delta    string               `json:"delta,omitempty"`
	Item     *responsesOutputItem `json:"item,omitempty"`
	Response *responsesResponse   `json:"response,omitempty"`
}

// ---------- Build request ----------

// buildResponsesRequest translates an llm.ChatRequest into the
// /v1/responses wire shape. StopSequences are rejected upstream because
// the Responses API has no equivalent parameter.
func (c *client) buildResponsesRequest(req llm.ChatRequest, stream bool) responsesRequest {
	out := responsesRequest{
		Model:        c.model,
		Instructions: req.SystemPrompt,
		Stream:       stream,
	}

	out.Input = buildResponsesInput(req.Messages)

	// Max tokens: per-request → client default → omit.
	if req.MaxTokens != nil {
		out.MaxOutputTokens = req.MaxTokens
	} else if c.opts.MaxTokens != nil {
		out.MaxOutputTokens = c.opts.MaxTokens
	}

	// Temperature: per-request → client default → omit. Reasoning
	// models (o1/o3/gpt-5) reject every value except their own default
	// (effectively 1); the library default of 0.7 would cause every
	// auto-routed call to fail. For those models we forward only an
	// explicitly-set per-request value and never the client default,
	// so omitting Temperature on the call lets the model use its own
	// default. Forced /v1/responses routing for non-reasoning models
	// (e.g. gpt-4o with Extension{ResponsesAPI: llm.Bool(true)}) still
	// honours the client default.
	if req.Temperature != nil {
		out.Temperature = req.Temperature
	} else if c.opts.Temperature != nil && !modelRequiresResponsesAPI(c.model) {
		out.Temperature = c.opts.Temperature
	}

	if len(req.Tools) > 0 {
		useCompact := false
		switch req.ToolDescriptions {
		case llm.ToolDescriptionCompact:
			useCompact = true
		case llm.ToolDescriptionFull:
			useCompact = false
		default: // "" (Default) or "auto"
			useCompact = httpclient.IsLocalBaseURL(c.baseURL)
		}
		out.Tools = make([]responsesTool, len(req.Tools))
		for i, t := range req.Tools {
			out.Tools[i] = responsesTool{
				Type:        "function",
				Name:        t.Name,
				Description: llm.EffectiveToolDescription(t, useCompact),
				Parameters:  t.InputSchema,
			}
		}
	}

	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case llm.ToolChoiceAuto:
			out.ToolChoice = "auto"
		case llm.ToolChoiceNone:
			out.ToolChoice = "none"
		case llm.ToolChoiceRequired:
			out.ToolChoice = "required"
		case llm.ToolChoiceSpecific:
			out.ToolChoice = map[string]any{
				"type": "function",
				"name": req.ToolChoice.Name,
			}
		}
	}

	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			out.Text = &responsesTextFormatCfg{
				Format: responsesTextFormat{Type: "json_object"},
			}
		case llm.ResponseFormatJSONSchema:
			out.Text = &responsesTextFormatCfg{
				Format: responsesTextFormat{
					Type:   "json_schema",
					Name:   "response",
					Schema: req.ResponseFormat.JSONSchema,
					Strict: true,
				},
			}
		}
	}

	return out
}

// buildResponsesInput converts the unified llm.Message slice into the
// Responses API "input" array. Assistant tool-use blocks and tool-role
// messages are emitted as top-level function_call / function_call_output
// items, not nested under an assistant message — this matches the
// Responses wire format.
func buildResponsesInput(msgs []llm.Message) []responsesInputItem {
	var out []responsesInputItem
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleAssistant:
			out = append(out, convertResponsesAssistantMessage(m)...)
		case llm.RoleTool:
			out = append(out, convertResponsesToolMessage(m)...)
		default:
			out = append(out, convertResponsesUserMessage(m))
		}
	}
	return out
}

func convertResponsesUserMessage(m llm.Message) responsesInputItem {
	item := responsesInputItem{
		Type: "message",
		Role: string(m.Role),
	}
	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText:
			item.Content = append(item.Content, responsesContentPart{
				Type: "input_text",
				Text: b.Text,
			})
		case llm.BlockImage:
			if b.Image == nil {
				continue
			}
			url := b.Image.URL
			if url == "" && len(b.Image.Data) > 0 {
				mediaType := b.Image.MediaType
				if mediaType == "" {
					mediaType = "application/octet-stream"
				}
				url = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(b.Image.Data)
			}
			if url == "" {
				continue
			}
			item.Content = append(item.Content, responsesContentPart{
				Type:     "input_image",
				ImageURL: url,
			})
		}
	}
	return item
}

func convertResponsesAssistantMessage(m llm.Message) []responsesInputItem {
	items := make([]responsesInputItem, 0, len(m.Content))

	var textParts []string
	for _, b := range m.Content {
		if b.Type == llm.BlockText {
			textParts = append(textParts, b.Text)
		}
	}
	if len(textParts) > 0 {
		items = append(items, responsesInputItem{
			Type: "message",
			Role: "assistant",
			Content: []responsesContentPart{{
				Type: "output_text",
				Text: strings.Join(textParts, ""),
			}},
		})
	}

	for _, b := range m.Content {
		if b.Type != llm.BlockToolUse || b.ToolUse == nil {
			continue
		}
		items = append(items, responsesInputItem{
			Type:      "function_call",
			CallID:    b.ToolUse.ID,
			Name:      b.ToolUse.Name,
			Arguments: string(b.ToolUse.Input),
		})
	}

	return items
}

func convertResponsesToolMessage(m llm.Message) []responsesInputItem {
	out := make([]responsesInputItem, 0, len(m.Content))
	for _, b := range m.Content {
		if b.Type != llm.BlockToolResult {
			continue
		}
		out = append(out, responsesInputItem{
			Type:   "function_call_output",
			CallID: b.ToolUseID,
			Output: b.Text,
		})
	}
	return out
}

// ---------- Chat (non-streaming) ----------

func (c *client) chatResponses(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if err := rejectResponsesUnsupported(req); err != nil {
		return nil, err
	}
	body := c.buildResponsesRequest(req, false)

	var raw responsesResponse
	status, respBody, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/responses", c.headers(), body, &raw)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, mapError(status, respBody)
	}

	resp := parseResponsesResponse(&raw)
	c.addUsage(resp.Usage)
	return resp, nil
}

func parseResponsesResponse(raw *responsesResponse) *llm.ChatResponse {
	resp := &llm.ChatResponse{}
	if raw.Usage != nil {
		resp.Usage = llm.TokenUsage{
			PromptTokens:     raw.Usage.InputTokens,
			CompletionTokens: raw.Usage.OutputTokens,
			TotalTokens:      raw.Usage.TotalTokens,
		}
	}

	var sawToolCall bool
	for _, item := range raw.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					resp.Content = append(resp.Content, llm.ContentBlock{
						Type: llm.BlockText,
						Text: part.Text,
					})
				}
			}
		case "function_call":
			sawToolCall = true
			resp.Content = append(resp.Content, llm.ContentBlock{
				Type: llm.BlockToolUse,
				ToolUse: &llm.ToolUse{
					ID:    item.CallID,
					Name:  item.Name,
					Input: json.RawMessage(item.Arguments),
				},
			})
		}
	}

	resp.StopReason = responsesStopReason(raw, sawToolCall)
	return resp
}

// responsesStopReason maps the Responses API status / incomplete_details
// onto the unified StopReason enum. tool_calls in the output trump the
// status so callers can tell tool-use turns apart from plain end-of-turn.
func responsesStopReason(raw *responsesResponse, sawToolCall bool) llm.StopReason {
	if sawToolCall {
		return llm.StopReasonToolUse
	}
	if raw.IncompleteDetails != nil && raw.IncompleteDetails.Reason == "max_output_tokens" {
		return llm.StopReasonMaxTokens
	}
	if raw.Status == "incomplete" {
		return llm.StopReasonMaxTokens
	}
	return llm.StopReasonEndTurn
}

// ---------- ChatStream ----------

func (c *client) chatStreamResponses(ctx context.Context, req llm.ChatRequest) (*llm.Stream, error) {
	if err := rejectResponsesUnsupported(req); err != nil {
		return nil, err
	}
	body := c.buildResponsesRequest(req, true)

	resp, err := httpclient.DoSSERequest(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/responses", c.headers(), body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		buf := make([]byte, 4096)
		n, _ := resp.Body.Read(buf)
		return nil, mapError(resp.StatusCode, buf[:n])
	}

	chunks := make(chan llm.StreamChunk, 64)
	errCh := make(chan error, 1)

	go func() {
		defer resp.Body.Close()
		defer close(chunks)
		defer close(errCh)

		scanner := httpclient.NewSSEScanner(resp.Body)

		var finalResponse *responsesResponse

		for scanner.Scan() {
			data := scanner.Data()
			if data == "" || data == "[DONE]" {
				continue
			}

			var ev responsesStreamEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				errCh <- fmt.Errorf("decode stream chunk: %w", err)
				return
			}

			switch ev.Type {
			case "response.output_item.added":
				if ev.Item == nil || ev.Item.Type != "function_call" {
					continue
				}
				chunks <- llm.StreamChunk{
					Type: llm.ChunkToolUseStart,
					ToolUse: &llm.ToolUse{
						ID:   ev.Item.CallID,
						Name: ev.Item.Name,
					},
				}

			case "response.output_text.delta":
				if ev.Delta == "" {
					continue
				}
				chunks <- llm.StreamChunk{
					Type: llm.ChunkText,
					Text: ev.Delta,
				}

			case "response.function_call_arguments.delta":
				if ev.Delta == "" {
					continue
				}
				chunks <- llm.StreamChunk{
					Type:    llm.ChunkToolUseDelta,
					Partial: ev.Delta,
				}

			case "response.completed":
				if ev.Response != nil {
					finalResponse = ev.Response
				}

			case "response.failed":
				errCh <- &llm.ProviderError{
					Err:      llm.ErrProviderError,
					Message:  "responses stream reported failure",
					Provider: providerName,
				}
				return
			}
		}

		if err := scanner.Err(); err != nil {
			errCh <- fmt.Errorf("read responses stream: %w", err)
			return
		}

		usage := &llm.TokenUsage{}
		if finalResponse != nil && finalResponse.Usage != nil {
			usage.PromptTokens = finalResponse.Usage.InputTokens
			usage.CompletionTokens = finalResponse.Usage.OutputTokens
			usage.TotalTokens = finalResponse.Usage.TotalTokens
			c.addUsage(*usage)
		}
		chunks <- llm.StreamChunk{
			Type:  llm.ChunkDone,
			Usage: usage,
		}
	}()

	return &llm.Stream{
		Chunks: chunks,
		Err:    errCh,
	}, nil
}

// rejectResponsesUnsupported returns ErrNotSupported for ChatRequest
// fields that the Responses API does not accept. StopSequences is the
// only such field at present — pass-through would silently lose them.
func rejectResponsesUnsupported(req llm.ChatRequest) error {
	if err := rejectDocumentBlocks(req); err != nil {
		return err
	}
	if len(req.StopSequences) > 0 {
		return &llm.ProviderError{
			Err:      llm.ErrNotSupported,
			Message:  "OpenAI Responses API does not accept stop sequences; omit StopSequences when targeting o1/o3/gpt-5 models or set openai.Extension{ResponsesAPI: llm.Bool(false)} to force the Chat Completions API",
			Provider: providerName,
		}
	}
	return nil
}
