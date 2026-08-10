# API Reference

This document provides a complete reference for all exported
types, interfaces, and functions in pgEdge Go LLM Library.

For full godoc, run `go doc github.com/pgEdge/pgedge-go-llm-lib/llm`
or browse [pkg.go.dev](https://pkg.go.dev/github.com/pgEdge/pgedge-go-llm-lib).

---

## Client Interface

The `Client` interface is the primary API for interacting with
LLM providers. All providers implement this interface.

```go
type Client interface {
    Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
    ChatStream(ctx context.Context, req ChatRequest) (*Stream, error)

    Embed(ctx context.Context, text string) ([]float64, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)

    Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error)
    EmbedMultimodal(ctx context.Context, req MultimodalEmbedRequest) ([][]float64, error)

    ListModels(ctx context.Context, opts ...ListModelsOption) ([]string, error)
    ListModelsWithMetadata(ctx context.Context, opts ...ListModelsOption) ([]ModelInfo, error)

    Ping(ctx context.Context) error
    Provider() string
    Model() string
    Usage() TokenUsage
    ResetUsage()
}
```

| Method                   | Description |
|--------------------------|-------------|
| `Chat`                   | Send a chat request and get a response. |
| `ChatStream`             | Send a chat request and stream chunks. |
| `Embed`                  | Generate an embedding for a text string. |
| `EmbedBatch`             | Generate embeddings for multiple texts. |
| `Rerank`                 | Reorder documents by relevance to a query. |
| `EmbedMultimodal`        | Generate embeddings for multimodal (text + images) inputs. |
| `ListModels`             | List available model names from the provider. |
| `ListModelsWithMetadata` | List models with capability and limit metadata. |
| `Ping`                   | Check provider connectivity (lightweight HEAD/GET). |
| `Provider`               | Return the provider name. |
| `Model`                  | Return the model name. |
| `Usage`                  | Return cumulative token usage since creation (or last `ResetUsage`). |
| `ResetUsage`             | Zero the cumulative token usage counter. |

---

## NewClient

```go
func NewClient(provider string, opts Options) (Client, error)
```

Creates a new LLM client for the named provider. Applies
`opts.WithDefaults()` before constructing. Returns a
`*ProviderError` wrapping `ErrInvalidRequest` if the provider
name is empty or unregistered.

## RegisterProvider

```go
func RegisterProvider(name string, constructor ProviderConstructor)
```

Registers a provider constructor under the given name. Called
by provider `init()` functions. You do not need to call this
unless implementing a custom provider.

## RegisteredProviders

```go
func RegisteredProviders() []string
```

Returns the names of all currently-registered providers,
sorted alphabetically. Safe to call without importing any
provider package (returns empty slice if none are registered).

## ProviderConstructor

```go
type ProviderConstructor func(opts Options) (Client, error)
```

---

## Options

`Options` configures an LLM client. Fields that overlap with
`ChatRequest` (`Temperature`, `MaxTokens`) are client-level
defaults; `ChatRequest` fields override them per-request.

```go
type Options struct {
    APIKey        string
    Model         string
    BaseURL       string
    CustomHeaders map[string]string
    HTTPClient    *http.Client
    MaxTokens     *int
    Temperature   *float64
    RequestTimeout time.Duration
    PerAttemptTimeout time.Duration
    Retry         RetryConfig
    OnRetry       func(RetryEvent)
}
```

| Field            | Default     | Description |
|------------------|-------------|-------------|
| `APIKey`         | —           | Provider API key. |
| `Model`          | —           | Model name or ID. |
| `BaseURL`        | per-provider| Override the API endpoint. Validated at construction. |
| `CustomHeaders`  | —           | Headers injected into every request. |
| `HTTPClient`     | library-built | Custom `*http.Client` (mTLS, round-trippers). |
| `MaxTokens`      | `4096`      | Default response length cap. Use `llm.Int(n)`. |
| `Temperature`    | omitted     | Default sampling temperature. Unset means the field is left out of the upstream request, so the provider's own default applies; some newer models reject any temperature value. Use `llm.Float(t)`. |
| `RequestTimeout` | `120s`      | Wall-clock cap per request, spanning all retries. |
| `PerAttemptTimeout` | `0` (off) | Wall-clock cap per attempt; makes slow attempts retryable. Set below `RequestTimeout`. |
| `Retry`          | 5 retries   | Retry policy. |
| `OnRetry`        | —           | Observability hook fired before each retry sleep. |

`WithDefaults()` returns a copy of `Options` with library
defaults applied (preserving explicit zero values).

Helper constructors: `llm.Int(n int) *int` and
`llm.Float(f float64) *float64`.

---

## RetryConfig

```go
type RetryConfig struct {
    MaxRetries     int
    InitialBackoff time.Duration
    MaxBackoff     time.Duration
    Disabled       bool
}
```

| Field            | Default | Description |
|------------------|---------|-------------|
| `MaxRetries`     | 5       | Maximum retry attempts after the initial try. |
| `InitialBackoff` | 2s      | Wait before first retry; doubles each attempt. |
| `MaxBackoff`     | 60s     | Cap on individual backoff duration. |
| `Disabled`       | false   | Set to true to send every request exactly once. |

Retryable conditions: network errors, HTTP 429 (honours
`Retry-After`), 500/502/503/504, Anthropic 529.

---

## RetryEvent

Supplied to `Options.OnRetry` before each retry sleep.

```go
type RetryEvent struct {
    Attempt    int
    StatusCode int
    Err        error
    Wait       time.Duration
}
```

---

## Message

```go
type Message struct {
    Role    Role           `json:"role"`
    Content []ContentBlock `json:"content"`
}
```

`Content` is always `[]ContentBlock`. Use the convenience
constructors to build common message shapes:

| Constructor                                    | Description |
|------------------------------------------------|-------------|
| `UserText(text string) Message`                | User message with one text block. |
| `AssistantText(text string) Message`           | Assistant message with one text block. |
| `SystemText(text string) Message`              | System message with one text block. |
| `ToolResultMessage(id, text string, isError bool) Message` | Tool-role message with one tool-result block. |
| `UserBlocks(blocks ...ContentBlock) Message`   | User message with arbitrary blocks. |
| `AssistantBlocks(blocks ...ContentBlock) Message` | Assistant message with arbitrary blocks. |
| `DocumentBlock(data []byte, mediaType, filename string) ContentBlock` | Document block with inline base64 data. |
| `DocumentURLBlock(url, mediaType, filename string) ContentBlock` | Document block referenced by URL. |

---

## Role

```go
type Role string

const (
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleSystem    Role = "system"
    RoleTool      Role = "tool"
)
```

---

## ContentBlock

```go
type ContentBlock struct {
    Type         ContentBlockType `json:"type"`
    Text         string           `json:"text,omitempty"`
    Image        *ImageContent    `json:"image,omitempty"`
    Document     *DocumentContent `json:"document,omitempty"`
    ToolUse      *ToolUse         `json:"tool_use,omitempty"`
    ToolUseID    string           `json:"tool_use_id,omitempty"`
    IsError      bool             `json:"is_error,omitempty"`
    CacheControl *CacheControl    `json:"cache_control,omitempty"`
}
```

`Type` selects which payload fields are populated:

| Constant               | String value    | Populated fields |
|------------------------|-----------------|------------------|
| `llm.BlockText`        | `"text"`        | `Text` |
| `llm.BlockImage`       | `"image"`       | `Image` |
| `llm.BlockDocument`    | `"document"`    | `Document` |
| `llm.BlockToolUse`     | `"tool_use"`    | `ToolUse` |
| `llm.BlockToolResult`  | `"tool_result"` | `ToolUseID`, `Text`, `IsError` |

`CacheControl` is Anthropic-specific; other providers ignore it.

Block shorthand constructors:

```go
TextBlock(t string) ContentBlock
ImageBlock(data []byte, mediaType string) ContentBlock
ImageURLBlock(url string) ContentBlock
DocumentBlock(data []byte, mediaType, filename string) ContentBlock
DocumentURLBlock(url, mediaType, filename string) ContentBlock
ToolResultBlock(toolUseID, text string, isError bool) ContentBlock
```

---

## ImageContent

```go
type ImageContent struct {
    URL       string `json:"url,omitempty"`
    Data      []byte `json:"data,omitempty"`
    MediaType string `json:"media_type,omitempty"`
}
```

Set either `URL` or `Data`+`MediaType` (e.g. `"image/png"`).
Anthropic and OpenAI support URL images; Gemini accepts file
URIs; Ollama rejects URL-only images.

---

## DocumentContent

```go
type DocumentContent struct {
    URL       string `json:"url,omitempty"`
    Data      []byte `json:"data,omitempty"`
    MediaType string `json:"media_type,omitempty"`
    Filename  string `json:"filename,omitempty"`
}
```

Set either `URL` or `Data`+`MediaType` (e.g. `"application/pdf"`).
`Filename` is an optional label that some providers surface to the
model (for Anthropic, this becomes the document's `title`).

Provider support:

- **Anthropic** — native PDF support, base64 or URL source.
- **Gemini** — native inline document or file-URI document input.
- **OpenAI** and **Ollama** — `Chat`/`ChatStream` return
  `llm.ErrNotSupported` when a document block is in the request.

---

## ChatRequest

```go
type ChatRequest struct {
    Messages       []Message
    Tools          []Tool
    SystemPrompt   string
    MaxTokens      *int
    Temperature    *float64
    Extensions     []ProviderExtension
    ResponseFormat *ResponseFormat
    ToolChoice     *ToolChoice
    StopSequences  []string
}
```

| Field            | Description |
|------------------|-------------|
| `Messages`       | Conversation history. |
| `Tools`          | Tool definitions available to the model. |
| `SystemPrompt`   | Per-request system instruction. |
| `MaxTokens`      | Override client default (`nil` → use `Options.MaxTokens`). |
| `Temperature`    | Override client default (`nil` → use `Options.Temperature`). |
| `Extensions`     | Provider-specific options (ignored by non-matching providers). |
| `ResponseFormat` | Constrain output format; see `ResponseFormatType`. |
| `ToolChoice`     | Control tool-selection behaviour; see `ToolChoiceMode`. |
| `StopSequences`  | Strings that terminate generation (most providers cap at 4). |

---

## ChatResponse

```go
type ChatResponse struct {
    Content    []ContentBlock
    StopReason StopReason
    Usage      TokenUsage
}
```

---

## StopReason

```go
type StopReason string

const (
    StopReasonEndTurn       StopReason = "end_turn"
    StopReasonMaxTokens     StopReason = "max_tokens"
    StopReasonStopSequence  StopReason = "stop_sequence"
    StopReasonToolUse       StopReason = "tool_use"
    StopReasonContentFilter StopReason = "content_filter"
    StopReasonError         StopReason = "error"
)
```

---

## ResponseFormat

```go
type ResponseFormat struct {
    Type       ResponseFormatType `json:"type"`
    JSONSchema json.RawMessage    `json:"json_schema,omitempty"`
}

type ResponseFormatType string

const (
    ResponseFormatText       ResponseFormatType = "text"
    ResponseFormatJSON       ResponseFormatType = "json_object"
    ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)
```

`JSONSchema` is required when `Type` is `ResponseFormatJSONSchema`.

---

## ToolChoice

```go
type ToolChoice struct {
    Mode ToolChoiceMode
    Name string // required when Mode == ToolChoiceSpecific
}

type ToolChoiceMode string

const (
    ToolChoiceAuto     ToolChoiceMode = "auto"
    ToolChoiceNone     ToolChoiceMode = "none"
    ToolChoiceRequired ToolChoiceMode = "required"
    ToolChoiceSpecific ToolChoiceMode = "specific"
)
```

---

## Tool

```go
type Tool struct {
    Name        string          `json:"name"`
    Description string          `json:"description"`
    InputSchema json.RawMessage `json:"input_schema"`
}
```

---

## ToolUse

```go
type ToolUse struct {
    ID        string          `json:"id"`
    Name      string          `json:"name"`
    Input     json.RawMessage `json:"input"`
    Signature string          `json:"signature,omitempty"`
}
```

`Signature` is an opaque, provider-specific token that some
providers attach to a tool call and require to be sent back
unchanged whenever that call is replayed as conversation
history. Gemini's thinking models populate the field and reject
a request whose history omits the value; other providers ignore
the field. Treat the value as opaque, and preserve assistant
messages intact rather than rebuilding them, which the loop
shown in the Tool Calling document already does.

---

## ProviderExtension

```go
type ProviderExtension interface {
    ProviderName() string
}
```

Providers silently ignore extensions whose `ProviderName()`
does not match theirs, keeping requests forward-compatible.

Helper: `FindExtension[T any](req ChatRequest, providerName string) *T`

See `llm/provider/anthropic` for `Extension`, `WithToolCaching`,
`WithSystemCaching`, and `WithExtendedThinking`.

---

## CacheControl

```go
type CacheControl struct {
    Type string `json:"type"`
}

type CacheControlType string

const CacheControlEphemeral CacheControlType = "ephemeral"
```

Anthropic-specific. Set `Type: "ephemeral"` on a `ContentBlock`
to mark it as a prompt-cache prefix boundary. In practice use
`anthropic.WithToolCaching` or `anthropic.WithSystemCaching`
rather than setting this directly.

---

## TokenUsage

```go
type TokenUsage struct {
    PromptTokens             int `json:"prompt_tokens"`
    CompletionTokens         int `json:"completion_tokens"`
    TotalTokens              int `json:"total_tokens"`
    CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
    CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}
```

`Add(other TokenUsage)` accumulates token usage into the receiver.
Cache fields are populated only by the Anthropic provider.

---

## ModelInfo

```go
type ModelInfo struct {
    ID            string            `json:"id"`
    ContextWindow int               `json:"context_window,omitempty"`
    MaxOutput     int               `json:"max_output,omitempty"`
    Capabilities  []ModelCapability `json:"capabilities,omitempty"`
    Deprecated    bool              `json:"deprecated,omitempty"`
}

type ModelCapability string

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
```

Returned by `Client.ListModelsWithMetadata`. Fields are
best-effort; providers populate what they know.

---

## Stream

```go
type Stream struct {
    Chunks <-chan StreamChunk
    Err    <-chan error
}
```

Prefer `Stream.Recv()` over reading channels directly.
`Stream.Collect(ctx context.Context) (*ChatResponse, error)`
drains the stream and assembles the full response.

---

## StreamChunk

```go
type StreamChunk struct {
    Type    StreamChunkType `json:"type"`
    Text    string          `json:"text,omitempty"`
    ToolUse *ToolUse        `json:"tool_use,omitempty"`
    Partial string          `json:"partial,omitempty"`
    Usage   *TokenUsage     `json:"usage,omitempty"`
}
```

`Type` selects populated fields:

| Constant                | String value      | Populated fields |
|-------------------------|-------------------|------------------|
| `llm.ChunkText`         | `"text"`          | `Text` |
| `llm.ChunkToolUseStart` | `"tool_use_start"`| `ToolUse` (`ID`, `Name`) |
| `llm.ChunkToolUseDelta` | `"tool_use_delta"`| `Partial` (JSON fragment) |
| `llm.ChunkDone`         | `"done"`          | `Usage` (always non-nil) |

---

## Sentinel Errors

| Error               | Sentinel value |
|---------------------|----------------|
| `ErrNotSupported`   | `"operation not supported by provider"` |
| `ErrAuthentication` | `"authentication failed"` |
| `ErrRateLimit`      | `"rate limit exceeded"` |
| `ErrInvalidRequest` | `"invalid request"` |
| `ErrProviderError`  | `"provider error"` |

---

## ProviderError

```go
type ProviderError struct {
    Err        error
    StatusCode int
    Message    string
    Provider   string
}
```

`Error()` formats as `"provider (status): message"`.
`Unwrap()` returns the sentinel error for `errors.Is`.

---

## HTTP Proxy Config (llm/proxy)

```go
type Config struct {
    DefaultProvider string
    Providers       map[string]llm.Options
    OnRequest       func(r *http.Request, info RequestInfo)
    OnResponse      func(r *http.Request, info ResponseInfo)
    OnError         func(r *http.Request, info ErrorInfo)
    Authorize       func(*http.Request) error
    RequestIDHeader string
}
```

`RequestIDFromContext(ctx context.Context) string` retrieves
the request ID attached by the proxy.

`AuthError{Err, Status}` can be returned from `Authorize` to
set a custom HTTP status code (default 401).

### RequestInfo and ResponseInfo

```go
type RequestInfo struct {
    Provider  string
    Model     string
    Stream    bool
    RequestID string
    Request   *llm.ChatRequest
}

type ResponseInfo struct {
    Provider   string
    Model      string
    Stream     bool
    Usage      llm.TokenUsage
    StatusCode int
    RequestID  string
    Response   *llm.ChatResponse
}
```

`RequestInfo.Request` is the fully-resolved `llm.ChatRequest` —
messages, tools, system prompt, tool-choice, response format, and
stop sequences. Use it to log prompts, audit tool definitions, or
correlate trace IDs with request content.

`ResponseInfo.Response` is the `llm.ChatResponse` the proxy returned —
content blocks (text and tool-use), stop reason, and token usage.
For streaming requests it is assembled from the SSE chunks, so hooks
see the same shape regardless of whether the call was streaming.

Both pointers are owned by the proxy for the request lifetime; do
not mutate them from inside a hook.

See the README and [proxy godoc](https://pkg.go.dev/github.com/pgEdge/pgedge-go-llm-lib/llm/proxy)
for `ErrorInfo`, `HealthResponse`, and the SSE wire format.

---

## Reranking and Multimodal Embeddings (v0.2+)

### `Client.Rerank(ctx, RerankRequest) (*RerankResponse, error)`

Reorders `Documents` by relevance to `Query`. Returns `ErrNotSupported`
on providers that don't support reranking.

### `Client.EmbedMultimodal(ctx, MultimodalEmbedRequest) ([][]float64, error)`

Embeds inputs containing interleaved text and image content. Returns
`ErrNotSupported` on providers that don't support multimodal embeddings.

### `ListModels` capability filter

`Client.ListModels` and `Client.ListModelsWithMetadata` accept zero or
more `ListModelsOption` values. The only built-in option is
`WithCapabilities(caps ...ModelCapability)`, which filters results to
models whose `Capabilities` field contains every listed capability.

### New `ModelCapability` constants

- `ModelCapabilityMultimodalEmbeddings`
- `ModelCapabilityReranking`

### New types

- `MultimodalContent`, `MultimodalContentType`, `MultimodalInput`,
  `MultimodalEmbedRequest`
- `RerankRequest`, `RerankResult`, `RerankResponse`
- `ListModelsConfig`, `ListModelsOption`, `WithCapabilities`,
  `FilterModelInfos`
