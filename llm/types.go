//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm

import (
	"encoding/json"
	"net/http"
	"time"
)

// Message represents a chat message sent to or received from an LLM.
//
// Content always carries a list of typed blocks. Use the convenience
// constructors UserText, AssistantText, ToolResultMessage, etc. for
// common cases (added in Task 6). Content blocks must NOT mix
// incompatible types within a single message — e.g., a tool-result
// message contains only BlockToolResult blocks.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock is one element of a message's content. The Type field
// selects which payload field is populated:
//
//	BlockText       -> Text
//	BlockImage      -> Image
//	BlockDocument   -> Document
//	BlockToolUse    -> ToolUse
//	BlockToolResult -> ToolUseID, Text, optional IsError
//
// CacheControl is an Anthropic-specific marker for prompt caching;
// other providers ignore it.
type ContentBlock struct {
	Type ContentBlockType `json:"type"`

	// Text — populated for BlockText and (simple) BlockToolResult.
	Text string `json:"text,omitempty"`

	// Image — populated for BlockImage.
	Image *ImageContent `json:"image,omitempty"`

	// Document — populated for BlockDocument.
	Document *DocumentContent `json:"document,omitempty"`

	// ToolUse — populated for BlockToolUse.
	ToolUse *ToolUse `json:"tool_use,omitempty"`

	// ToolUseID — populated for BlockToolResult, identifying the
	// ToolUse this result responds to.
	ToolUseID string `json:"tool_use_id,omitempty"`

	// IsError — populated for BlockToolResult when the tool errored.
	IsError bool `json:"is_error,omitempty"`

	// CacheControl — Anthropic-specific marker for prompt caching.
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ImageContent carries an image in either URL form (when the provider
// supports URL input) or as inline base64-encoded data. Exactly one
// of URL or Data should be set. MediaType (e.g., "image/png") is
// required when Data is set.
type ImageContent struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

// DocumentContent carries a document (e.g. PDF) in either URL form
// (when the provider supports URL input) or as inline base64-encoded
// data. Exactly one of URL or Data should be set. MediaType (e.g.,
// "application/pdf") is required when Data is set. Filename is an
// optional human-readable label that some providers surface to the
// model (e.g., as the document's title).
//
// Provider support is uneven: Anthropic and Gemini accept documents
// natively; OpenAI and Ollama return llm.ErrNotSupported when a
// document block is included in the request.
type DocumentContent struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Filename  string `json:"filename,omitempty"`
}

// ToolUse represents a tool/function call made by the LLM.
type ToolUse struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// Tool defines a tool/function available for the LLM to call.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// CacheControl specifies caching behaviour (Anthropic only).
type CacheControl struct {
	Type string `json:"type"`
}

// ProviderExtension is a provider-specific request modifier. Each
// provider's package defines its own extension type implementing
// this interface. Providers receiving an extension whose ProviderName
// does not match are expected to ignore it silently — this keeps
// requests forward-compatible across providers.
type ProviderExtension interface {
	ProviderName() string
}

// ResponseFormatType identifies the structured-output mode.
type ResponseFormatType string

const (
	// ResponseFormatText is plain text output (the default; equivalent to omitting ResponseFormat).
	ResponseFormatText ResponseFormatType = "text"

	// ResponseFormatJSON requests free-form JSON output. The model
	// is instructed to produce valid JSON; structure is not enforced.
	ResponseFormatJSON ResponseFormatType = "json_object"

	// ResponseFormatJSONSchema requests JSON output conforming to
	// the schema in ResponseFormat.JSONSchema. Providers that don't
	// support strict schema validation fall back to ResponseFormatJSON
	// behaviour with a system-prompt instruction; check the provider's
	// docs.
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// ResponseFormat constrains the model's output format. JSONSchema is
// required when Type is ResponseFormatJSONSchema; ignored otherwise.
// Providers that don't support a given format return ErrNotSupported.
type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema json.RawMessage    `json:"json_schema,omitempty"`
}

// ToolChoiceMode identifies the high-level tool-choice strategy.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	// This is the default when ToolChoice is nil.
	ToolChoiceAuto ToolChoiceMode = "auto"
	// ToolChoiceNone forbids tool calls.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces the model to call any tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceSpecific forces the model to call a named tool.
	// ToolChoice.Name must be set.
	ToolChoiceSpecific ToolChoiceMode = "specific"
)

// ToolChoice constrains the model's tool-use behaviour. Providers
// that don't support tool-choice (e.g., Ollama with prompt-based
// tool-call workaround) ignore this field.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string // required when Mode == ToolChoiceSpecific
}

// ChatRequest contains the parameters for a chat completion request.
type ChatRequest struct {
	Messages       []Message
	Tools          []Tool
	SystemPrompt   string
	MaxTokens      *int
	Temperature    *float64
	Extensions     []ProviderExtension
	ResponseFormat *ResponseFormat
	ToolChoice     *ToolChoice
	// StopSequences are strings that, when encountered in the model's
	// output, terminate generation. Most providers cap the count at 4.
	StopSequences []string
}

// FindExtension returns the first extension matching providerName,
// type-asserted to *T. Returns nil if no match.
//
// Providers should call this from their request builder:
//
//	if ext := llm.FindExtension[anthropic.Extension](req, "anthropic"); ext != nil {
//	    // use ext.ExtendedThinking, ext.BudgetTokens, etc.
//	}
func FindExtension[T any](req ChatRequest, providerName string) *T {
	for _, e := range req.Extensions {
		if e.ProviderName() != providerName {
			continue
		}
		if t, ok := e.(T); ok {
			return &t
		}
	}
	return nil
}

// ChatResponse contains the result of a chat completion request.
type ChatResponse struct {
	Content    []ContentBlock
	StopReason StopReason
	Usage      TokenUsage
}

// TokenUsage tracks token consumption for a request or cumulatively.
type TokenUsage struct {
	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// Add accumulates token usage from another TokenUsage into this one.
func (u *TokenUsage) Add(other TokenUsage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
	u.CacheCreationInputTokens += other.CacheCreationInputTokens
	u.CacheReadInputTokens += other.CacheReadInputTokens
}

// RetryEvent describes a single retry attempt that just failed.
// Supplied to Options.OnRetry just before the retry layer sleeps
// and retries.
type RetryEvent struct {
	// Attempt is the 1-indexed attempt number that just failed.
	Attempt int

	// StatusCode is the HTTP status of the failed attempt. Zero if
	// the attempt failed with a network error before a response.
	StatusCode int

	// Err is the network error from the failed attempt, or nil if
	// the failure was a retryable HTTP status.
	Err error

	// Wait is the duration the retry layer will sleep before the
	// next attempt.
	Wait time.Duration
}

// RetryConfig configures HTTP retry behaviour for transient failures
// (network errors, 429, 500/502/503/504, and Anthropic's 529
// "overloaded" status). Set Disabled to true to opt out entirely.
type RetryConfig struct {
	// MaxRetries is the maximum number of retries AFTER the initial
	// attempt. Default 5.
	MaxRetries int

	// InitialBackoff is the wait before the first retry. The backoff
	// doubles after each attempt, capped at MaxBackoff. A 429 with a
	// Retry-After header overrides the computed backoff for that
	// attempt. Default 2s.
	InitialBackoff time.Duration

	// MaxBackoff caps individual backoff durations. Default 60s.
	MaxBackoff time.Duration

	// Disabled turns off retries entirely. When true, every request
	// is sent exactly once.
	Disabled bool
}

// Options configures an LLM client.
//
// Precedence rule: fields that appear on BOTH Options and ChatRequest
// (Temperature, MaxTokens) are CLIENT-LEVEL DEFAULTS on Options.
// When the same field is set on a per-request ChatRequest, the
// per-request value takes precedence. A nil pointer on a ChatRequest
// field falls through to the Options default.
//
// Fields that appear ONLY on Options (APIKey, Model, BaseURL,
// CustomHeaders, Retry, RequestTimeout, HTTPClient, OnRetry)
// configure the client itself and cannot be overridden per-request.
//
// Fields that appear ONLY on ChatRequest (Tools, SystemPrompt,
// Extensions, and others added in later tasks) are per-request —
// they have no client-level default.
type Options struct {
	APIKey        string
	Model         string
	BaseURL       string
	CustomHeaders map[string]string

	// HTTPClient is the http.Client used for all upstream requests. If
	// nil, the library builds one with the configured retry middleware
	// and custom headers. Supply this to integrate with corporate
	// proxies, mTLS, OpenTelemetry round-trippers, or custom timeouts.
	//
	// When HTTPClient is set, the library still wraps its Transport with
	// the retry middleware (unless Retry.Disabled) and the header
	// middleware (when CustomHeaders is non-empty).
	HTTPClient *http.Client

	// MaxTokens caps the model's response length. Use llm.Int(n) to
	// set; nil means "use the library default of 4096" after
	// WithDefaults runs. A pointer to 0 means "no client-side cap"
	// and is sent to the provider as-is (most providers reject 0;
	// Anthropic substitutes its own default).
	MaxTokens *int

	// Temperature controls sampling randomness. Use llm.Float(t) to
	// set; nil means "use the library default of 0.7" after
	// WithDefaults runs. A pointer to 0 means deterministic sampling
	// and is sent to the provider as `temperature: 0` on the wire.
	Temperature *float64

	// RequestTimeout caps the wall-clock time of a single HTTP request,
	// including all retries. Zero (default) uses the library's
	// 120-second default. For streaming requests, this caps the time to
	// receive the response headers and start of stream — the stream
	// itself can run longer if the upstream keeps sending events.
	RequestTimeout time.Duration

	Retry RetryConfig

	// OnRetry is invoked once per retry attempt that just failed,
	// before the retry layer sleeps. Useful for circuit breakers,
	// dashboards, or rate-limit tracking. Hooks run synchronously on
	// the request goroutine; keep them fast.
	OnRetry func(RetryEvent)
}

// WithDefaults returns a copy of Options with default values applied
// for any unset fields. Pointer fields are populated only when nil,
// so explicit zero values (Temperature=0 for deterministic sampling,
// MaxTokens=0 for "no client cap") are preserved.
func (o Options) WithDefaults() Options {
	if o.Temperature == nil {
		def := 0.7
		o.Temperature = &def
	}
	if o.MaxTokens == nil {
		def := 4096
		o.MaxTokens = &def
	}
	if !o.Retry.Disabled {
		if o.Retry.MaxRetries == 0 {
			o.Retry.MaxRetries = 5
		}
		if o.Retry.InitialBackoff == 0 {
			o.Retry.InitialBackoff = 2 * time.Second
		}
		if o.Retry.MaxBackoff == 0 {
			o.Retry.MaxBackoff = 60 * time.Second
		}
	}
	return o
}

// Float returns a pointer to f. Use for setting *float64 option fields.
func Float(f float64) *float64 { return &f }

// Int returns a pointer to i. Use for setting *int option fields.
func Int(i int) *int { return &i }

// UserText returns a user Message with a single text block.
func UserText(text string) Message {
	return Message{
		Role:    RoleUser,
		Content: []ContentBlock{{Type: BlockText, Text: text}},
	}
}

// AssistantText returns an assistant Message with a single text block.
func AssistantText(text string) Message {
	return Message{
		Role:    RoleAssistant,
		Content: []ContentBlock{{Type: BlockText, Text: text}},
	}
}

// SystemText returns a system Message with a single text block.
func SystemText(text string) Message {
	return Message{
		Role:    RoleSystem,
		Content: []ContentBlock{{Type: BlockText, Text: text}},
	}
}

// ToolResultMessage returns a tool-role Message with a single
// tool-result block referencing toolUseID. If the tool errored,
// pass isError=true.
func ToolResultMessage(toolUseID, text string, isError bool) Message {
	return Message{
		Role: RoleTool,
		Content: []ContentBlock{{
			Type:      BlockToolResult,
			ToolUseID: toolUseID,
			Text:      text,
			IsError:   isError,
		}},
	}
}

// UserBlocks returns a user Message with the given blocks (e.g.,
// text + image for multimodal input).
func UserBlocks(blocks ...ContentBlock) Message {
	return Message{Role: RoleUser, Content: blocks}
}

// AssistantBlocks returns an assistant Message with the given blocks.
func AssistantBlocks(blocks ...ContentBlock) Message {
	return Message{Role: RoleAssistant, Content: blocks}
}

// TextBlock is a shorthand for ContentBlock{Type: BlockText, Text: t}.
func TextBlock(t string) ContentBlock {
	return ContentBlock{Type: BlockText, Text: t}
}

// ImageBlock returns a ContentBlock for an image with inline base64
// data. mediaType should be e.g. "image/png".
func ImageBlock(data []byte, mediaType string) ContentBlock {
	return ContentBlock{
		Type:  BlockImage,
		Image: &ImageContent{Data: data, MediaType: mediaType},
	}
}

// ImageURLBlock returns a ContentBlock for an image referenced by URL.
// Note: not all providers support URL image input — Anthropic and
// OpenAI do; Gemini accepts file URIs; Ollama rejects URL-only images.
func ImageURLBlock(url string) ContentBlock {
	return ContentBlock{
		Type:  BlockImage,
		Image: &ImageContent{URL: url},
	}
}

// DocumentBlock returns a ContentBlock for a document (e.g. PDF) with
// inline base64-encoded data. mediaType should be e.g. "application/pdf".
// filename is an optional label; pass "" to omit it.
//
// Anthropic and Gemini support documents natively; OpenAI and Ollama
// return llm.ErrNotSupported when a document block is in the request.
func DocumentBlock(data []byte, mediaType, filename string) ContentBlock {
	return ContentBlock{
		Type: BlockDocument,
		Document: &DocumentContent{
			Data:      data,
			MediaType: mediaType,
			Filename:  filename,
		},
	}
}

// DocumentURLBlock returns a ContentBlock for a document referenced by
// URL. mediaType (e.g. "application/pdf") and filename are optional —
// pass "" to omit either. Provider support for URL documents varies:
// Anthropic accepts URL sources for PDFs; Gemini accepts file URIs.
func DocumentURLBlock(url, mediaType, filename string) ContentBlock {
	return ContentBlock{
		Type: BlockDocument,
		Document: &DocumentContent{
			URL:       url,
			MediaType: mediaType,
			Filename:  filename,
		},
	}
}

// ToolResultBlock is a shorthand for a BlockToolResult ContentBlock.
func ToolResultBlock(toolUseID, text string, isError bool) ContentBlock {
	return ContentBlock{
		Type:      BlockToolResult,
		ToolUseID: toolUseID,
		Text:      text,
		IsError:   isError,
	}
}

// Role identifies the role of a Message. Use RoleUser/RoleAssistant/
// RoleSystem/RoleTool — providers compare against these constants
// when building wire messages.
type Role string

// Role values recognised by all providers.
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleTool      Role = "tool"
)

// StopReason identifies why the model terminated generation.
// Providers map their native stop reasons onto this normalised set.
// Unrecognised native values fall back to StopReasonEndTurn.
type StopReason string

// Normalised StopReason values; providers map their native reasons
// onto this set.
const (
	StopReasonEndTurn       StopReason = "end_turn"
	StopReasonMaxTokens     StopReason = "max_tokens"
	StopReasonStopSequence  StopReason = "stop_sequence"
	StopReasonToolUse       StopReason = "tool_use"
	StopReasonContentFilter StopReason = "content_filter"
	StopReasonError         StopReason = "error"
)

// ContentBlockType identifies the kind of payload carried by a
// ContentBlock. The block's Type field selects which other fields
// are populated; see the ContentBlock godoc.
type ContentBlockType string

// ContentBlockType values recognised by ContentBlock.Type.
const (
	BlockText       ContentBlockType = "text"
	BlockImage      ContentBlockType = "image"
	BlockDocument   ContentBlockType = "document"
	BlockToolUse    ContentBlockType = "tool_use"
	BlockToolResult ContentBlockType = "tool_result"
)

// CacheControlType identifies the cache scope for prompt-caching.
// Currently only "ephemeral" is defined (Anthropic-specific).
type CacheControlType string

// CacheControlEphemeral marks a content block as a cache prefix
// boundary using Anthropic's ephemeral cache scope.
const CacheControlEphemeral CacheControlType = "ephemeral"

// ModelInfo describes a model's capabilities and limits. Fields are
// best-effort: providers populate what they know; unknown values are
// zero/empty.
type ModelInfo struct {
	ID            string            `json:"id"`
	ContextWindow int               `json:"context_window,omitempty"`
	MaxOutput     int               `json:"max_output,omitempty"`
	Capabilities  []ModelCapability `json:"capabilities,omitempty"`
	Deprecated    bool              `json:"deprecated,omitempty"`
}

// ModelCapability is a coarse-grained feature flag describing what a
// model can do. The set is intentionally small — providers map their
// native capability descriptions onto this set.
type ModelCapability string

// ModelCapability values reported by ListModelsWithMetadata.
const (
	ModelCapabilityChat                 ModelCapability = "chat"
	ModelCapabilityTools                ModelCapability = "tools"
	ModelCapabilityVision               ModelCapability = "vision"
	ModelCapabilityEmbeddings           ModelCapability = "embeddings"
	ModelCapabilityJSONMode             ModelCapability = "json_mode"
	ModelCapabilityStreaming            ModelCapability = "streaming"
	ModelCapabilityMultimodalEmbeddings ModelCapability = "multimodal_embeddings"
	ModelCapabilityReranking            ModelCapability = "reranking"
)
