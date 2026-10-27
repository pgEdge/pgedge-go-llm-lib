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
    Decide(ctx context.Context, req DecisionRequest) (*DecisionResponse, error)
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
| `Decide`                 | Ask a decision model questions and get probabilities back. |
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
`ChatRequest` (`Temperature`, `MaxTokens`, `ReasoningEffort`,
`PromptCaching`) are client-level defaults; `ChatRequest` fields
override them per-request.

```go
type Options struct {
    APIKey        string
    Model         string
    BaseURL       string
    CustomHeaders map[string]string
    HTTPClient    *http.Client
    MaxTokens     *int
    Temperature   *float64
    ReasoningEffort ReasoningEffort
    PromptCaching PromptCaching
    ExtraBody     map[string]any
    RequestTimeout time.Duration
    PerAttemptTimeout time.Duration
    MaxResponseBytes int64
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
| `HTTPClient`     | library-built | Custom `*http.Client` (mTLS, round-trippers). Redirects are refused unless the client sets its own `CheckRedirect`. |
| `MaxTokens`      | `4096`      | Default response length cap. Use `llm.Int(n)`. |
| `Temperature`    | omitted     | Default sampling temperature. Unset means the field is left out of the upstream request, so the provider's own default applies; some newer models reject any temperature value. Use `llm.Float(t)`. |
| `ReasoningEffort` | omitted    | Default reasoning effort; see [Reasoning effort](#reasoning-effort). |
| `PromptCaching` | off         | Default automatic prompt caching; see [Prompt caching](#prompt-caching). |
| `ExtraBody`     | —           | Fields the OpenAI provider adds to every chat request body; see [Extra body fields](#extra-body-fields). |
| `RequestTimeout` | `120s`      | Wall-clock cap per request, spanning all retries. |
| `PerAttemptTimeout` | `0` (off) | Wall-clock cap per attempt; makes slow attempts retryable. Set below `RequestTimeout`. |
| `MaxResponseBytes` | `256 MiB` | Cap on a single response body, or a whole stream; a longer response fails with an error. |
| `Retry`          | 5 retries   | Retry policy. |
| `OnRetry`        | —           | Observability hook fired before each retry sleep. |
| `Logger`         | — (silent)  | `*slog.Logger` for diagnostics; records at Debug level each adjustment a provider makes after the API rejects a parameter or endpoint. |

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
    ReasoningEffort ReasoningEffort
    PromptCaching  PromptCaching
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
| `ReasoningEffort` | Override client default (`""` → use `Options.ReasoningEffort`); see [Reasoning effort](#reasoning-effort). |
| `PromptCaching` | Override client default (`""` → use `Options.PromptCaching`); see [Prompt caching](#prompt-caching). |

### Reasoning effort

`ReasoningEffort` sets how much the model reasons before it answers,
with one of `llm.ReasoningEffortNone`, `Minimal`, `Low`, `Medium`,
`High`, `XHigh` or `Max`. Leaving it unset on both `ChatRequest` and
`Options` leaves the field out of the request, so the model's own
default applies. On the providers that send the setting (OpenAI,
Anthropic and Gemini), an unknown value fails with
`ErrInvalidRequest` before any request is sent; the others ignore
it. Each provider translates the level:

| Provider  | Sent as |
|-----------|---------|
| OpenAI    | `reasoning_effort` on `/v1/chat/completions` and `reasoning: {effort}` on `/v1/responses`, with the level unchanged. |
| Anthropic | `thinking: {type: "adaptive"}` with `output_config: {effort}`; `Minimal` is sent as `low`, and `None` sends `thinking: {type: "disabled"}` with no effort. Thinking set through `anthropic.WithExtendedThinking` takes precedence over the adaptive setting, but the effort is still sent. |
| Gemini    | `generationConfig.thinkingConfig.thinkingLevel`; `None` is sent as `minimal`, and `XHigh` and `Max` as `high`. |
| Others    | Left out. |

Models differ in which levels they accept. When a model rejects a
level, the OpenAI, Anthropic and Gemini providers resend the request
without it, remember the rejection for the rest of the client's life,
and log the adjustment to `Options.Logger` at Debug level. Anthropic
drops the adaptive or disabled thinking setting and the effort level
separately, so a model that accepts an effort but not adaptive
thinking still gets the effort. On OpenAI, a rejected level is
replaced by the nearest level towards `medium` that the model has
not rejected (so `none` becomes `minimal`, and `max` becomes
`xhigh`), since omitting it would apply the model's default, which
is usually further from the level asked for. A call steps only
once: if the replacement is rejected too, or the rejected level is
`medium`, the rest of the call omits the effort, whilst later calls
start from the nearest level not yet rejected. A model that rejects
the parameter itself, rather than one level, has it omitted at
every level. An OpenAI rejection that
names the effort only in its message, of a request that carried
tools, may be about tools and effort together, so the effort is
omitted for that call only.

Anthropic does not allow a temperature other than 1 alongside
thinking, so a request that carries adaptive thinking leaves any
other temperature out.

### Prompt caching

`PromptCaching` is `llm.PromptCachingOff` (the default when unset on
both `ChatRequest` and `Options`) or `llm.PromptCachingAuto`. On
Anthropic, `Auto` sends the top-level `cache_control: {type:
"ephemeral"}`, which places a cache breakpoint on the last cacheable
block and moves it forward as the conversation grows, so an agent
loop's resent history is read from the cache. Anthropic bills a cache
write at 1.25 times the input rate and a read at 0.1 times or less,
so one reuse of the prefix pays for the write.

The markers from `anthropic.WithToolCaching`, `WithSystemCaching` and
`ContentBlock.CacheControl` still apply alongside it. The automatic
marker counts towards Anthropic's limit of four breakpoints per
request, so it is left out of a request whose blocks already carry
four. An endpoint that rejects the top-level marker, as some
Anthropic-compatible ones do, has it dropped and remembered in the
same way as a rejected temperature. When the request also carries
block markers, which the rejection could be about, the call is resent
once without the top-level marker, and the drop is remembered only if
that resend succeeds; if it fails the same way, the error is
returned.

OpenAI and Gemini cache automatically, so `Auto` changes nothing
there, and other providers ignore it. An unknown value fails with
`ErrInvalidRequest` before sending on Anthropic.

### Extra body fields

The OpenAI provider adds the fields in `ExtraBody` to the JSON body
of every `Chat` and `ChatStream` request, on `/v1/chat/completions`
and `/v1/responses` alike, for fields the library does not model, such
as an OpenAI-compatible gateway's own options:

```go
req := llm.ChatRequest{
    Messages: msgs,
    Extensions: []llm.ProviderExtension{openai.Extension{
        ExtraBody: map[string]any{
            "providerOptions": map[string]any{
                "gateway": map[string]any{"caching": "auto"},
            },
        },
    }},
}
```

`llm.Options.ExtraBody` sets the client default, an `openai.Extension`
on `Options.Extensions` is merged over it, and one on the request is
merged over both, key by key. Each value is encoded with
`encoding/json`, and a value that cannot be encoded fails with
`ErrInvalidRequest` before sending. A `nil` value is sent as JSON
`null` rather than removing the field. A `stream` field also fails
with `ErrInvalidRequest` before sending, since `Chat` and `ChatStream`
choose streaming and a field that disagreed would have the response
read in the wrong format. Other providers ignore `ExtraBody`.

The fields are set at the top level after the library builds the
body, so a key the library also sets is replaced whole by the extra
value. This is deliberate, as an escape hatch, and it applies to the
fields that `ReasoningEffort` produces too: an extra
`reasoning_effort` replaces the level on `/v1/chat/completions`, and
an extra `reasoning` replaces the whole `reasoning` object on
`/v1/responses`, not only its `effort`. Since every field goes to
whichever endpoint the request is routed to, pin the route before
setting a field that only one endpoint accepts. The route is read only
from `Options.Extensions`, so put `openai.Extension{ResponsesAPI: ...}`
there; `ResponsesAPI` on a request's extension has no effect.

An error whose `param` names an extra field, or whose message names
one in single quotes when `param` is empty, is returned to the caller without the client adjusting
and resending the request, because the caller asked for that field.
That includes a library field the extra value replaced, so a
`reasoning_effort` set through `ExtraBody` is never dropped by the
reasoning effort adjustment.

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
`PromptCaching: llm.PromptCachingAuto` for the conversation, with
`anthropic.WithToolCaching` or `anthropic.WithSystemCaching` for the
tools and system prompt, rather than setting this directly.

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

## Decisions

### `Client.Decide(ctx, DecisionRequest) (*DecisionResponse, error)`

Asks a decision model one or more questions about the same input and returns
one answer per question, in request order. Returns `ErrNotSupported` on
providers that don't support decision models. Supported by the `typesafe` and
`openai` providers; see [Decisions](decisions.md).

### `DecisionKind`

A string type naming the shape of a question.

- `DecisionYesNo` (`"yes_no"`): the answer is the probability that the
  statement is true.
- `DecisionChoice` (`"choice"`): the model picks one of the options.
- `DecisionScore` (`"score"`): the model rates on an ordered scale of levels,
  lowest first.

`MaxDecisionChoices` (255) is the largest number of options a choice question
may carry.

### `DecisionOption`

| Field | Type | Description |
|---|---|---|
| `Value` | `string` | The option returned when chosen (choice), or the level's label (score). Required and unique within the question. |
| `Description` | `string` | Optionally tells the model what the option means. |

### `DecisionQuestion`

| Field | Type | Description |
|---|---|---|
| `Name` | `string` | Identifies the question and its answer. Required and unique within the request. |
| `Kind` | `DecisionKind` | Selects the question type. |
| `Instructions` | `string` | The question put to the model. Required. |
| `Options` | `[]DecisionOption` | Options of a choice question (2 to `MaxDecisionChoices`) or levels of a score question (at least 2, lowest first). Must be empty for a yes/no question. |

### `DecisionRequest`

| Field | Type | Description |
|---|---|---|
| `Model` | `string` | Overrides the default decision model for this request. |
| `Input` | `string` | The text the questions are about. Required unless `Images` is set. |
| `Images` | `[]ImageContent` | Optional images. Providers that don't accept images return `ErrNotSupported`. |
| `Questions` | `[]DecisionQuestion` | At least one question. |
| `Extensions` | `[]ProviderExtension` | Provider-specific options. |

### `DecisionRequest.Validate() error`

Checks the request for errors that every provider would reject, so that they
fail before any network call. The error is a `*ProviderError` wrapping
`ErrInvalidRequest`, with `Provider` unset; providers fill it in.

### `OptionProbability`

| Field | Type | Description |
|---|---|---|
| `Value` | `string` | The option's `Value` (choice) or the level's label (score). |
| `Probability` | `float64` | The probability the model assigned to the option. |

### `DecisionAnswer`

| Field | Type | Description |
|---|---|---|
| `Name` | `string` | The question's `Name`. |
| `Kind` | `DecisionKind` | The question's `Kind`. |
| `Refused` | `bool` | True when the provider declined to answer; every field except `Name`, `Kind` and `Refused` is then zero. |
| `Probability` | `float64` | Yes/no: the probability that the answer is yes. |
| `Choice` | `string` | Choice: the `Value` of the selected option. |
| `Score` | `float64` | Score: the probability-weighted level index, where 0 is the first (lowest) level. |
| `Confidence` | `float64` | The provider's confidence in `Choice` or `Score`. |
| `Probabilities` | `[]OptionProbability` | Choice and score: the probability of each option, in request order. |

### `DecisionResponse`

| Field | Type | Description |
|---|---|---|
| `Answers` | `[]DecisionAnswer` | One answer per question, in request order. |
| `Model` | `string` | The model the provider reports having used. |
| `Usage` | `TokenUsage` | Token usage for this request. |

### `DecisionResponse.Answer(name string) (DecisionAnswer, bool)`

Returns the answer to the named question, and false when there is none. Safe
to call on a nil response.

### `ModelCapabilityDecisions`

The `ModelCapability` (`"decisions"`) reported for models that support
`Decide`, for use with `WithCapabilities`.

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

Both methods return a chat-focused list rather than everything the
provider advertises, so a caller can present the result as a choice of
conversational models without inspecting it further. Embedding models
are excluded unless `WithCapabilities(ModelCapabilityEmbeddings)` asks
for them. Models that cannot hold a text conversation at all are always
excluded, because several providers list them alongside the
conversational ones with nothing in the metadata to tell them apart:
OpenAI reports its speech, transcription and image models, and Gemini
reports text-to-speech, image, music and agent models that all claim
`generateContent`. The exclusions are matched on the model name, so a
newly released family of that kind may appear until the provider is
updated. Reaching such a model is a matter for the provider's own API
rather than this library's chat methods.

### New `ModelCapability` constants

- `ModelCapabilityMultimodalEmbeddings`
- `ModelCapabilityReranking`

### New types

- `MultimodalContent`, `MultimodalContentType`, `MultimodalInput`,
  `MultimodalEmbedRequest`
- `RerankRequest`, `RerankResult`, `RerankResponse`
- `ListModelsConfig`, `ListModelsOption`, `WithCapabilities`,
  `FilterModelInfos`
