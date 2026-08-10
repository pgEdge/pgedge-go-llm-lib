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
	"os"
	"strings"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
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

	// Signature is an opaque, provider-specific token that some
	// providers attach to a tool call and then require to be sent
	// back, unchanged, on any later request that replays the call as
	// conversation history. Gemini's thinking models use it to resume
	// their own reasoning across turns, and reject a request whose
	// history omits it; other providers currently ignore the field.
	//
	// Callers should treat this as opaque and simply keep assistant
	// messages intact when appending a tool result, which is what the
	// ordinary tool-calling loop does anyway. Never synthesise or
	// modify a value here.
	Signature string `json:"signature,omitempty"`
}

// Tool defines a tool/function available for the LLM to call.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// CompactDescription is an optional shorter description used when a
	// caller (or the auto policy) selects compact tool descriptions —
	// see ToolDescriptionMode and EffectiveToolDescription. When empty,
	// the full Description is always used.
	CompactDescription string          `json:"compact_description,omitempty"`
	InputSchema        json.RawMessage `json:"input_schema"`
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
	Mode ToolChoiceMode `json:"mode"`
	Name string         `json:"name,omitempty"` // required when Mode == ToolChoiceSpecific
}

// ToolDescriptionMode selects which tool description providers send.
type ToolDescriptionMode string

// ToolDescriptionMode values recognised by ChatRequest.ToolDescriptions.
const (
	ToolDescriptionDefault ToolDescriptionMode = ""        // provider default: Auto
	ToolDescriptionFull    ToolDescriptionMode = "full"    // always full Description
	ToolDescriptionCompact ToolDescriptionMode = "compact" // use CompactDescription when present
	ToolDescriptionAuto    ToolDescriptionMode = "auto"    // compact when talking to a local base URL
)

// EffectiveToolDescription returns the compact description when useCompact
// is true and a CompactDescription is set, otherwise the full Description.
func EffectiveToolDescription(t Tool, useCompact bool) string {
	if useCompact && t.CompactDescription != "" {
		return t.CompactDescription
	}
	return t.Description
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
	// ToolDescriptions selects which tool description text providers send
	// on the wire. The zero value (ToolDescriptionDefault) and
	// ToolDescriptionAuto both auto-select compact descriptions when the
	// provider's base URL is local; see ToolDescriptionMode.
	ToolDescriptions ToolDescriptionMode
}

// UseCompactDescriptions reports whether tool descriptions should be sent in
// their compact form for a request bound for baseURL. Compact and Full force
// the choice; Default and Auto select compact when baseURL is local.
func (r ChatRequest) UseCompactDescriptions(baseURL string) bool {
	switch r.ToolDescriptions {
	case ToolDescriptionCompact:
		return true
	case ToolDescriptionFull:
		return false
	default: // ToolDescriptionDefault ("") or ToolDescriptionAuto
		return httpclient.IsLocalBaseURL(baseURL)
	}
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

// CacheSavingsPercent returns the percentage of input tokens that were
// served from the prompt cache: CacheReadInputTokens relative to the total
// input tokens (PromptTokens + CacheReadInputTokens +
// CacheCreationInputTokens). It returns 0 when there is no input or no
// cache read. Only Anthropic currently populates the cache token fields;
// for other providers this is always 0.
func (u TokenUsage) CacheSavingsPercent() float64 {
	totalInput := u.PromptTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	if totalInput == 0 || u.CacheReadInputTokens == 0 {
		return 0
	}
	return float64(u.CacheReadInputTokens) / float64(totalInput) * 100
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
// CustomHeaders, Retry, RequestTimeout, HTTPClient, OnRetry, Extensions)
// configure the client itself and cannot be overridden per-request.
//
// Fields that appear ONLY on ChatRequest (Tools, SystemPrompt,
// Extensions, and others added in later tasks) are per-request —
// they have no client-level default. Note that Extensions exists on
// both Options (client-level) and the various request types
// (per-request): the Options form is used by methods that take plain
// arguments (Embed, EmbedBatch) rather than a request struct.
type Options struct {
	APIKey        string
	APIKeyFile    string // optional: path to a file containing the API key; used only when APIKey is empty
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
	// set; nil means omit the field entirely and let the provider use
	// its own default — WithDefaults leaves it untouched. A pointer to
	// 0 means deterministic sampling and is sent to the provider as
	// `temperature: 0` on the wire.
	//
	// WithDefaults previously filled a nil Temperature with 0.7, which
	// made it impossible for a caller to omit the field: some models
	// (e.g. newer Claude models) reject any temperature value outright.
	// Leaving it nil now genuinely omits it.
	Temperature *float64

	// RequestTimeout caps the wall-clock time of a single HTTP request,
	// including all retries. Zero (default) uses the library's
	// 120-second default. For streaming requests, this caps the time to
	// receive the response headers and start of stream — the stream
	// itself can run longer if the upstream keeps sending events.
	//
	// Because this budget spans every retry, a single attempt slow
	// enough to exhaust it leaves no room to retry — the request fails
	// with a timeout that cannot be retried. To make slow attempts
	// retryable, set PerAttemptTimeout below RequestTimeout.
	RequestTimeout time.Duration

	// PerAttemptTimeout, when > 0, bounds each individual HTTP attempt.
	// An attempt that stalls past it is abandoned and retried (subject
	// to Retry), so a slow upstream — e.g. a heavy embedding batch that
	// would otherwise burn the entire RequestTimeout in one attempt —
	// becomes retryable instead of failing outright. It is derived from
	// the request context, so a per-attempt timeout never cancels the
	// caller's context. For streaming requests it bounds only the time
	// to receive response headers; the stream body is not interrupted.
	//
	// Zero (default) disables per-attempt timeouts, preserving the
	// historical behaviour where only RequestTimeout applies. Set it
	// smaller than RequestTimeout to leave room for retries.
	PerAttemptTimeout time.Duration

	Retry RetryConfig

	// OnRetry is invoked once per retry attempt that just failed,
	// before the retry layer sleeps. Useful for circuit breakers,
	// dashboards, or rate-limit tracking. Hooks run synchronously on
	// the request goroutine; keep them fast.
	OnRetry func(RetryEvent)

	// Extensions carries client-level provider-specific options. Use
	// this for tunables that the unified Client API doesn't surface
	// and that can't be attached to a per-request struct — most
	// notably Embed and EmbedBatch, which take plain arguments rather
	// than a request type. Providers read only extensions whose
	// ProviderName matches their own and ignore the rest, so an
	// extension intended for one provider is safe to pass alongside
	// others.
	Extensions []ProviderExtension
}

// WithDefaults returns a copy of Options with default values applied
// for any unset fields. Pointer fields are populated only when nil,
// so an explicit zero value (MaxTokens=0 for "no client cap") is
// preserved. Temperature is deliberately NOT defaulted here — see its
// field doc — so it stays nil (and is omitted from the wire) unless a
// caller explicitly sets it, at either the Options or ChatRequest
// level.
func (o Options) WithDefaults() Options {
	if o.APIKey == "" && o.APIKeyFile != "" {
		if b, err := os.ReadFile(o.APIKeyFile); err == nil { //nolint:gosec // path is operator-supplied config
			o.APIKey = strings.TrimSpace(string(b))
		}
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

// Bool returns a pointer to b. Use for setting *bool option fields.
func Bool(b bool) *bool { return &b }

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
	Dimensions    int               `json:"dimensions,omitempty"` // embedding vector size; 0 if unknown/not an embedding model
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

// MultimodalContentType identifies the kind of content in a
// MultimodalContent value. The discriminator selects which of the
// Text / ImageURL / ImageData fields is read; other fields are
// ignored for that content item.
type MultimodalContentType string

const (
	// MultimodalContentText is a UTF-8 text fragment in MultimodalContent.Text.
	MultimodalContentText MultimodalContentType = "text"
	// MultimodalContentImageURL is a remote image fetched from MultimodalContent.ImageURL.
	MultimodalContentImageURL MultimodalContentType = "image_url"
	// MultimodalContentImageData is an inline image in MultimodalContent.ImageData with MIME type in MultimodalContent.MIMEType.
	MultimodalContentImageData MultimodalContentType = "image_base64"
)

// MultimodalContent is a single piece of content in a multimodal
// embedding input. The Type field selects which of Text / ImageURL /
// ImageData is read.
type MultimodalContent struct {
	Type      MultimodalContentType
	Text      string
	ImageURL  string
	ImageData []byte
	MIMEType  string
}

// MultimodalInput is one input to EmbedMultimodal. Each input
// produces exactly one embedding vector; the order in
// MultimodalEmbedRequest.Inputs is preserved in the returned slice.
type MultimodalInput struct {
	Content []MultimodalContent
}

// MultimodalEmbedRequest is the request body for Client.EmbedMultimodal.
// Providers that do not support multimodal embeddings return ErrNotSupported.
type MultimodalEmbedRequest struct {
	Inputs     []MultimodalInput
	Extensions []ProviderExtension
}

// RerankRequest is the request body for Client.Rerank. TopK, when
// non-nil, asks the provider to return at most the top-K most-relevant
// documents. Providers that do not support reranking return
// ErrNotSupported.
type RerankRequest struct {
	Query      string
	Documents  []string
	TopK       *int
	Extensions []ProviderExtension
}

// RerankResult is one row of a rerank response. Index is the position
// in the original RerankRequest.Documents slice. RelevanceScore is the
// provider's relevance value (typically [0,1] but not strictly bounded).
// Document is non-empty only when the provider returns documents in
// its response (e.g. when ReturnDocuments was requested via a provider
// extension).
type RerankResult struct {
	Index          int
	RelevanceScore float64
	Document       string
}

// RerankResponse is the body returned by Client.Rerank. Results are
// ordered by descending RelevanceScore. Usage carries token accounting
// where the provider reports it; PromptTokens / CompletionTokens are
// usually zero for rerank.
type RerankResponse struct {
	Results []RerankResult
	Usage   TokenUsage
}

// ListModelsConfig is the configuration accumulated from ListModelsOption
// values passed to Client.ListModels and Client.ListModelsWithMetadata.
// Callers don't construct this directly; use options like WithCapabilities.
type ListModelsConfig struct {
	// Capabilities, when non-empty, restricts results to models whose
	// ModelInfo.Capabilities contains EVERY listed capability. An
	// empty Capabilities slice means "no filter".
	Capabilities []ModelCapability
}

// ListModelsOption configures a single ListModels call. Pass values
// returned by WithCapabilities (and future option constructors) as
// the variadic argument to Client.ListModels.
type ListModelsOption func(*ListModelsConfig)

// WithCapabilities filters ListModels to models whose Capabilities
// contain every listed value. Calls accumulate: passing two
// WithCapabilities options is equivalent to one call with all
// capabilities concatenated.
func WithCapabilities(caps ...ModelCapability) ListModelsOption {
	return func(c *ListModelsConfig) {
		c.Capabilities = append(c.Capabilities, caps...)
	}
}
