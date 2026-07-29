# Changelog

All notable changes to pgEdge Go LLM Library will be
documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this
project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Security
- Provider error messages no longer relay credentials to the caller. Every provider previously lifted the upstream API's own `error.message` string verbatim into `ProviderError.Message`, and the Voyage provider formatted the entire raw response body into it. Providers characteristically quote a partially masked form of the submitted key in that message on an authentication failure, so a service returning `err.Error()` to an HTTP client leaked a fragment of the operator's configured API key. All five providers now redact the configured key, any identifiable fragment of it, and anything matching a known credential format (the OpenAI, Anthropic, Google, and Voyage key prefixes, an `Authorization` value, and labelled values such as `api_key=`) before the message is stored, replacing each with `[REDACTED]`. The Voyage provider parses the error body rather than quoting it, and degrades to the status code alone when the body cannot be parsed. The unredacted text is never retained, and there is no option to disable redaction
- The proxy redacts error text on its way to the wire, covering the JSON error responses written by `writeError`, the server-sent `event: error` payloads, and the per-provider strings in the `GET /v1/health` response. This is a second layer for provider errors, which are already redacted at source, and the first line of defence for errors generated locally, most notably an `AuthError` whose message comes from a caller-supplied `Authorize` hook. The `OnError` hook continues to receive the unmodified `ErrorInfo`, since it is in-process server-side logging rather than a trust boundary

### Fixed
- The Gemini provider no longer emits a content part that serialises to `{}`, which Gemini rejects outright with `400: GenerateContentRequest.contents[N].parts[M].data: required oneof field 'data' must have one initialised field`. `geminiPart.Text` is tagged `omitempty`, so a part carrying only an empty string marshals to an empty object; two places on the request path constructed exactly that. `convertMessage` appended a part for every text block without checking whether the text was empty, so an empty text block arriving from any source (another provider's replayed history, history compaction producing an empty assistant turn, or a client bug) failed the whole request; empty text blocks are now skipped. The `len(parts) == 0` fallback deliberately emitted `geminiPart{Text: ""}` as a supposedly safe placeholder, but `[{}]` is rejected just as firmly as an omitted `parts` array, so a message whose content was entirely unrepresentable (an empty `Content` slice, an image or document with neither a URL nor inline data, or a nil `ToolUse`) guaranteed a 400; such a message is now dropped, since it carries no information. Should every message be dropped, `convertMessages` returns a `ProviderError` wrapping `ErrInvalidRequest` rather than sending an empty `contents` array, so the caller gets a clear client-side error instead of an opaque upstream one
- `Client.Usage()` now accumulates embedding token usage for the OpenAI, Gemini, and Ollama providers. Previously only Voyage tracked embedding usage; the other three parsed the embedding vectors but discarded the provider's token counts, so a caller reading `Usage()` after `Embed`/`EmbedBatch` under-reported by the full embedding cost. The embed response types now capture the provider's usage payload (`usage.{prompt_tokens,total_tokens}` for OpenAI, `usageMetadata.promptTokenCount` for Gemini, and `prompt_eval_count` for Ollama) and feed it into the cumulative counter. Embeddings carry no completion tokens, so only `PromptTokens`/`TotalTokens` are populated
- `Options.WithDefaults()` no longer fills an unset `Temperature` with `0.7`. Previously this made it impossible for a caller to omit `temperature` from the wire: some models (e.g. newer Claude models) reject the field outright with `400: 'temperature' is deprecated for this model`. An unset `Temperature` (on both `Options` and `ChatRequest`) is now genuinely omitted; an explicitly-set value (including `0`) is unaffected
- A request whose body is large enough to exceed OS socket buffers (e.g. a large `EmbedBatch` call), sent to a peer that never reads it, no longer leaves its connection open indefinitely after the caller's context expires. Context cancellation alone does not reliably interrupt a body write blocked at the OS level; the underlying connection is now force-closed in that case, for both the overall request context and `PerAttemptTimeout`

## [0.1.1] - 2026-07-16

### Added
- `ResponseInfo.Duration` and `ErrorInfo.Duration`: the wall-clock time spent on the upstream provider call, populated on the proxy `OnResponse` and `OnError` hook payloads so consumers can record how long a provider request took without instrumenting it themselves. The chat and streaming-chat handlers set it on both the success `ResponseInfo` and the upstream-call `ErrorInfo` (streaming covers the full stream to completion); the embed, rerank, and multimodal-embed handlers set it on their upstream-call `ErrorInfo`. Errors raised before any upstream call (authorization, request parsing, transform, client construction) leave `Duration` at zero. Both fields are additive on server-side hook structs that consumers only read, so this is backward compatible with 0.1.0

## [0.1.0] - 2026-06-12

### Added
- `Options.PerAttemptTimeout`: an optional per-attempt wall-clock cap that makes a slow individual attempt retryable, instead of letting it consume the whole `RequestTimeout` budget with no room to retry. Derived from the request context so it never cancels the caller's context, and detached on success so it does not interrupt a streaming response body
- `anthropic.WithSystemCaching` helper that marks the system prompt as a cacheable prefix on Anthropic requests
- Voyage AI as a fifth supported provider (embeddings, multimodal embeddings, reranking)
- `Client.Rerank` method for reordering documents by query relevance
- `Client.EmbedMultimodal` method for multimodal (text + images) embeddings
- `WithCapabilities` option on `ListModels` / `ListModelsWithMetadata` to filter models by capability
- New `ModelCapability` constants: `ModelCapabilityReranking`, `ModelCapabilityMultimodalEmbeddings`
- Proxy HTTP routes: `POST /v1/embed`, `POST /v1/embed/multimodal`, `POST /v1/rerank`
- `?capability=` query parameter on `GET /v1/models` (repeatable; AND semantics)
- OpenAI provider auto-routes `o1`, `o3`, and `gpt-5` model families to `/v1/responses` (transparently translating the request/response wire shape); `openai.Extension.ResponsesAPI` overrides the auto-detection
- `llm.Bool` helper for setting `*bool` option fields
- `proxy.Config.TransformRequest`: a hook invoked after a chat request is parsed and before it is dispatched, permitted to mutate `SystemPrompt`, `Messages`, and `Tools`. Returning an error rejects the request (default 400, overridable via an `HTTPStatus() int` method). It is the sanctioned request-rewrite point; `OnRequest` observes the post-transform request
- `proxy.Config.PathPrefix`: configurable base path for the gateway routes (defaults to `/v1`)
- `proxy.Config.MaxBodyBytes`: optional request-body size limit for `/chat`, `/chat/stream`, `/embed`, and `/rerank` (0 means unlimited)
- `ProviderInfo.DisplayName`: a human-readable provider label surfaced by `GET /v1/providers`, with sensible defaults and a raw-name fallback
- `ModelInfo.Dimensions`: embedding vector dimension, populated where statically known (OpenAI and Voyage embedding models)
- `TokenUsage.CacheSavingsPercent`: derived percentage of input tokens served from the prompt cache
- `Tool.CompactDescription` and `ChatRequest.ToolDescriptions` (`ToolDescriptionMode`): an optional shorter tool description plus a selection policy (explicit `full`/`compact`, or an `auto` policy that uses the compact form for local/loopback endpoints). Exposed on the proxy via the `tool_descriptions` request field
- `Options.APIKeyFile`: resolve the API key from a file path at client construction (used only when `APIKey` is empty)
- `llm/vec` package: pure embedding-vector helpers (`Float64ToFloat32`, `Normalize`, `Resize`, and `Float32ToFloat16` for pgvector `halfvec` storage)
- Unified `Client` interface for interacting with multiple
  LLM providers through a single API.
- Anthropic provider with chat completions, streaming, tool
  calling, and prompt caching support.
- OpenAI provider with chat completions, streaming,
  embeddings, and tool calling support.
- Gemini provider with chat completions, streaming,
  embeddings, and tool calling support.
- Ollama provider with chat completions, streaming,
  embeddings, and text-based tool calling support.
- `NewClient` factory function with automatic provider
  registration via `init()` functions.
- Convenience `llm/all` package that imports all built-in
  providers.
- `ChatRequest` and `ChatResponse` types with unified message
  format across providers.
- `Stream` and `StreamChunk` types for real-time streaming
  responses via server-sent events.
- `Stream.Recv` for ergonomic chunk iteration and `Stream.Collect`
  for assembling a full `*ChatResponse` from a stream.
- `Embed` and `EmbedBatch` methods for text embedding
  generation.
- `ListModels` and `ListModelsWithMetadata` methods for
  retrieving available models with capability metadata.
- `Ping` method for lightweight provider connectivity checks.
- `Provider`, `Model`, `Usage`, and `ResetUsage` accessors
  on the `Client` interface.
- `Tool` and `ToolUse` types for function calling workflows.
- `ToolChoice` field on `ChatRequest` with auto, none,
  required, and specific modes for controlling tool selection.
- `ResponseFormat` field on `ChatRequest` for free-form JSON
  and JSON-schema constrained output.
- `StopSequences` field on `ChatRequest` to terminate
  generation on matching strings.
- `BlockText`, `BlockImage`, `BlockDocument`, `BlockToolUse`,
  and `BlockToolResult` content block types for unified message
  content.
- Multimodal image input via `ImageBlock` (inline base64) and
  `ImageURLBlock` (URL reference).
- Document content blocks (`BlockDocument`, `DocumentContent`,
  `DocumentBlock`, `DocumentURLBlock`) for passing PDFs and other
  document formats directly to the model without Go-side text
  extraction. Anthropic and Gemini support documents natively;
  OpenAI and Ollama reject document blocks with
  `llm.ErrNotSupported`. The Anthropic provider auto-enables the
  PDF beta header.
- Convenience constructors `UserText`, `AssistantText`,
  `SystemText`, `ToolResultMessage`, `UserBlocks`,
  `AssistantBlocks`, `TextBlock`, and `ToolResultBlock`.
- `CacheControl` type for Anthropic prompt caching on content
  blocks; `anthropic.WithToolCaching` helper for tool caching.
- `anthropic.WithExtendedThinking` helper that enables
  Anthropic extended-thinking mode with a configurable token
  budget.
- `ProviderExtension` interface and generic `FindExtension[T]`
  helper for forward-compatible provider-specific options.
- `TokenUsage` type with cumulative tracking across requests
  and Anthropic cache-token fields
  (`CacheCreationInputTokens`, `CacheReadInputTokens`).
- `ProviderError` wrapper with sentinel errors
  (`ErrNotSupported`, `ErrAuthentication`, `ErrRateLimit`,
  `ErrInvalidRequest`, `ErrProviderError`) for structured
  error handling.
- Production-grade retry with exponential back-off, capped
  backoff, `Retry-After` honouring, and configurable
  `RetryConfig`. Retries fire on network errors, 429,
  500/502/503/504, and Anthropic 529.
- `OnRetry` observability hook fired before each retry sleep,
  with attempt number, status code, error, and wait duration.
- `RequestTimeout` option for per-attempt wall-clock cap and
  `HTTPClient` injection for mTLS, OpenTelemetry round-trippers,
  and corporate proxies.
- Custom HTTP header injection via `Options.CustomHeaders`.
- Configurable base URL override via `Options.BaseURL`, with
  validation at client construction.
- Per-request overrides for system prompt, temperature, and
  max tokens.
- `llm.Int` and `llm.Float` helpers for setting pointer-typed
  option fields.
- Automatic stop reason normalization across all providers.
- Internal HTTP client with SSE scanner for streaming support.
- HTTP proxy package (`llm/proxy`) exposing the provider
  abstraction over an HTTP API with SSE streaming, including
  `/v1/health`, `/v1/providers`, `/v1/models`, `/v1/chat`,
  and `/v1/chat/stream` endpoints.
- Proxy `Authorize` hook with custom HTTP status via
  `proxy.AuthError`.
- Proxy `OnRequest`, `OnResponse`, and `OnError` telemetry
  hooks; request-ID propagation via configurable header
  (`Config.RequestIDHeader`) and `RequestIDFromContext`
  accessor.
- Proxy `RequestInfo.Request` and `ResponseInfo.Response` fields
  expose the full `*llm.ChatRequest` and `*llm.ChatResponse` to
  `OnRequest` and `OnResponse` hooks, so consumers can log
  prompts, tool calls (with parameters and results), and the
  stop reason without re-implementing SSE accumulation. For
  streaming requests the response is assembled from the chunks
  using the same logic as `Stream.Collect`.
- Defensive `<think>...</think>` stripping in the Ollama provider
  so reasoning models such as `deepseek-r1` do not break tool-call
  extraction or JSON-mode output.
- JSON mode and JSON-schema output for the Ollama provider via
  Ollama's `format` field (Ollama 0.5.0 or later).
- Comprehensive test suite with unit and integration tests.

### Changed
- **Breaking (for external implementers of `llm.Client`):** `ListModels` and `ListModelsWithMetadata` are now variadic, accepting `...ListModelsOption`. Source-compatible for callers; interface change for external implementers.
- **Breaking (for external implementers of `llm.Client`):** `Rerank` and `EmbedMultimodal` added to the interface. External implementers must add these methods (returning `ErrNotSupported` if not supported).
- **Breaking (proxy wire format):** `ToolChoice` now serialises with snake_case JSON tags (`{"mode":...,"name":...}`) instead of the previous Go default (`{"Mode":...,"Name":...}`). Clients sending `tool_choice` through the proxy must use the snake_case keys.
