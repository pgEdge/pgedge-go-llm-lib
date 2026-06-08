<div class="banner" markdown>
![pgEdge Labs](img/pgedge-labs-light.svg#only-light){ width="320" }
![pgEdge Labs](img/pgedge-labs-dark.svg#only-dark){ width="320" }
</div>

# pgEdge Go LLM Library

pgEdge Go LLM Library is a unified Go library for interacting
with multiple large language model providers through a single,
provider-agnostic interface.

The library abstracts away provider-specific API differences so
that you can write code once and switch between LLM providers
with minimal changes. pgEdge Go LLM Library has zero external
dependencies; the library uses only the Go standard library.

pgEdge Go LLM Library supports the following features:

- chat completions through a unified request and response format.
- streaming responses via server-sent events for real-time output.
- text embeddings for supported providers.
- tool and function calling with JSON schema definitions.
- multimodal image input (inline base64 and URL).
- document input (PDFs and other formats supported by the provider) inline or by URL.
- JSON mode and JSON schema output constraints.
- cumulative token usage tracking across requests; reset with `ResetUsage()`.
- provider discovery: `RegisteredProviders()`, `Ping()`, `ListModelsWithMetadata()`.
- production-grade retry with exponential back-off and `OnRetry` observability hook.
- per-request timeouts and custom `HTTPClient` injection.
- prompt caching for the Anthropic provider.
- extended thinking mode for the Anthropic provider.
- custom HTTP headers and base URL overrides.
- HTTP proxy with SSE streaming, Authorize hook, and request-ID propagation.

## Supported Providers

The following table describes the providers that pgEdge Go LLM
Library supports and the features available for each provider:

| Feature             | Anthropic | OpenAI | Gemini | Ollama |
|---------------------|-----------|--------|--------|--------|
| Chat                | Yes       | Yes    | Yes    | Yes    |
| Streaming           | Yes       | Yes    | Yes    | Yes    |
| Embeddings          | No        | Yes    | Yes    | Yes    |
| Tool Calling        | Yes       | Yes    | Yes    | Yes*   |
| Multimodal Images   | Yes       | Yes    | Yes    | No†    |
| Documents (PDFs)    | Yes       | No‡    | Yes    | No‡    |
| JSON Mode           | Yes       | Yes    | Yes    | Yes    |
| Prompt Caching      | Yes       | No     | No     | No     |
| Token Tracking      | Yes       | Yes    | Yes    | Yes    |

\* Ollama tool calling is implemented via text-based parsing; results vary by model.  
† Ollama rejects URL images; inline base64 may work with vision-capable models.  
‡ Request is rejected with `llm.ErrNotSupported`. Extract document text yourself before calling.

## How the Library Works

pgEdge Go LLM Library uses a provider registration pattern.
Each provider package registers itself at import time using
Go's `init()` function. The `llm.NewClient` factory function
looks up the registered provider by name and returns a `Client`
interface that works identically across all providers.

You can import individual providers or use the convenience
package `llm/all` to register all built-in providers at once.

## Next Steps

The following documents provide additional information about
pgEdge Go LLM Library:

- The [Getting Started](getting_started.md) document explains
  how to install the library and make your first API call.
- The [Providers](providers.md) document describes how to
  configure each supported provider.
- The [API Reference](api_reference.md) document provides
  complete details on all types, interfaces, and functions.
