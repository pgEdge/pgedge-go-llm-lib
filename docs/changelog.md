# Changelog

All notable changes to pgEdge Go LLM Library will be
documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this
project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Voyage AI as a fifth supported provider (embeddings, multimodal embeddings, reranking)
- `Client.Rerank` method for reordering documents by query relevance
- `Client.EmbedMultimodal` method for multimodal (text + images) embeddings
- `WithCapabilities` option on `ListModels` / `ListModelsWithMetadata` to filter models by capability
- New `ModelCapability` constants: `ModelCapabilityReranking`, `ModelCapabilityMultimodalEmbeddings`
- Proxy HTTP routes: `POST /v1/embed`, `POST /v1/embed/multimodal`, `POST /v1/rerank`
- `?capability=` query parameter on `GET /v1/models` (repeatable; AND semantics)

### Changed
- **Breaking (for external implementers of `llm.Client`):** `ListModels` and `ListModelsWithMetadata` are now variadic, accepting `...ListModelsOption`. Source-compatible for callers; interface change for external implementers.
- **Breaking (for external implementers of `llm.Client`):** `Rerank` and `EmbedMultimodal` added to the interface. External implementers must add these methods (returning `ErrNotSupported` if not supported).

---

## [Unreleased] - Alpha 1

### Added

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
