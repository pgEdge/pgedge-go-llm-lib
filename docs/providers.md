# Providers

pgEdge Go LLM Library supports five LLM providers (Anthropic,
OpenAI, Gemini, Ollama, and Voyage). Each provider registers
itself automatically when you import the provider package.

## Anthropic

The Anthropic provider connects to the Anthropic Messages API
for Claude models. You can create an Anthropic client with the
following code:

```go
client, err := llm.NewClient("anthropic", llm.Options{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Model:  "claude-sonnet-4-20250514",
})
```

The following table describes the Anthropic provider defaults:

| Option   | Default Value                      |
|----------|------------------------------------|
| BaseURL  | https://api.anthropic.com/v1       |
| API Version | 2023-06-01                      |

The Anthropic provider supports the following features:

- chat completions and streaming.
- tool and function calling.
- prompt caching via the `anthropic.WithToolCaching` helper and
  `CacheControl` markers on `ContentBlock` values.
- extended thinking mode via `anthropic.WithExtendedThinking`.
- cumulative token usage tracking with cache metrics
  (`CacheCreationInputTokens`, `CacheReadInputTokens`).

The Anthropic provider does not support embeddings. Calling
`Embed` or `EmbedBatch` returns an `ErrNotSupported` error.

The `ListModels` method returns only models with `type` set
to `"model"`.

The provider sends the `anthropic-beta: prompt-caching-2024-07-31`
header with every request to enable prompt caching support.

## OpenAI

The OpenAI provider connects to the OpenAI Chat Completions
API and Embeddings API. You can create an OpenAI client with
the following code:

```go
client, err := llm.NewClient("openai", llm.Options{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o",
})
```

The following table describes the OpenAI provider defaults:

| Option  | Default Value                   |
|---------|---------------------------------|
| BaseURL | https://api.openai.com/v1       |

The OpenAI provider supports the following features:

- chat completions and streaming.
- text embeddings with single and batch operations.
- tool and function calling.
- cumulative token usage tracking.

The provider automatically uses `max_completion_tokens` instead
of `max_tokens` for models that require the newer parameter.
These models include those with `o1`, `o3`, or `gpt-5`
prefixes.

The `ListModels` method filters out non-chat models. The
filter removes models with the following name prefixes:
`text-embedding`, `embedding`, `tts`, `whisper`, `dall-e`,
`davinci`, `babbage`, `text-moderation`, `text-search`,
`text-similarity`, and `code-search`.

## Gemini

The Gemini provider connects to the Google Generative Language
API. You can create a Gemini client with the following code:

```go
client, err := llm.NewClient("gemini", llm.Options{
    APIKey: os.Getenv("GEMINI_API_KEY"),
    Model:  "gemini-2.0-flash",
})
```

The following table describes the Gemini provider defaults:

| Option  | Default Value                                    |
|---------|--------------------------------------------------|
| BaseURL | https://generativelanguage.googleapis.com         |

The Gemini provider supports the following features:

- chat completions and streaming.
- text embeddings with single and batch operations.
- tool and function calling via function declarations.
- cumulative token usage tracking.

The Gemini provider passes the API key in the `x-goog-api-key`
header rather than as a `?key=...` query parameter, so the key
does not appear in HTTP intermediary access logs. The
`EmbedBatch` method makes sequential calls to the single-embed
endpoint because Gemini does not provide a native batch
embedding API.

The `ListModels` method returns only models that support the
`generateContent` generation method.

Tool call IDs use the format `gemini-tool-{name}` because
Gemini does not assign unique tool call identifiers. When
sending tool results back, the provider extracts the tool
name from the tool use ID.

## Ollama

The Ollama provider connects to a local Ollama instance. You
can create an Ollama client with the following code:

```go
client, err := llm.NewClient("ollama", llm.Options{
    Model: "llama3",
})
```

The following table describes the Ollama provider defaults:

| Option  | Default Value              |
|---------|----------------------------|
| BaseURL | http://localhost:11434      |

The Ollama provider supports the following features:

- chat completions and streaming.
- text embeddings with single and batch operations.
- tool calling via text-based parsing.
- JSON mode via Ollama's `format` field. `ResponseFormatJSON`
  passes `format: "json"`; `ResponseFormatJSONSchema` passes the
  schema object directly (Ollama 0.5.0 or later).
- cumulative token usage tracking populated from Ollama's
  `prompt_eval_count` and `eval_count` fields.

The Ollama provider does not require an API key because Ollama
runs locally.

### Reasoning Models

Reasoning models served by Ollama (for example `deepseek-r1`)
emit `<think>...</think>` blocks inline with their final answer.
The Ollama provider strips these blocks before returning the
response so they do not pollute text content, break tool-call
extraction (when an example JSON snippet appears inside a
thinking block), or corrupt JSON-mode output. The behaviour is
case-insensitive and tolerates an unterminated open tag by
dropping the remainder.

This is applied unconditionally and does not need to be
configured. If you want to inspect the raw thinking output,
read it directly from Ollama's API rather than through this
library.

The `ListModels` method returns all models listed by the
`/api/tags` endpoint.

Tool calling works differently in the Ollama provider. The
provider injects tool definitions into the system prompt and
parses the model's text response to extract JSON tool calls.
The provider looks for JSON objects matching the format
`{"tool":"tool_name","arguments":{...}}` in the response text.

## Voyage AI

Voyage AI provides text embeddings, multimodal embeddings, and rerankers.
It does not provide chat completions; `Chat` and `ChatStream` return
`ErrNotSupported`.

### Authentication

Set `Options.APIKey` or the `VOYAGE_API_KEY` environment variable. Voyage
uses bearer-token auth.

### Models

| Model | Capabilities |
|---|---|
| `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large` | embeddings |
| `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` | embeddings |
| `voyage-multimodal-3` | embeddings, multimodal_embeddings |
| `rerank-2.5`, `rerank-2.5-lite` | reranking |

Use `client.ListModelsWithMetadata(ctx, llm.WithCapabilities(llm.ModelCapabilityReranking))`
to discover rerank-capable models programmatically.

### Per-call options

Voyage exposes per-call knobs through `voyage.Extension`:

| Field | Purpose |
|---|---|
| `InputType` | `voyage.InputTypeQuery` or `voyage.InputTypeDocument` — affects retrieval quality. |
| `OutputDimension` | 256 / 512 / 1024 / 2048 (model-dependent). |
| `Truncation` | Pointer to bool; provider default when nil. |
| `OutputDtype` | `float` / `int8` / `uint8` / `binary` / `ubinary`. |
| `ReturnDocuments` | Rerank only; include documents in the response. |

Example:

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)

vecs, err := client.EmbedMultimodal(ctx, llm.MultimodalEmbedRequest{
    Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
        {Type: llm.MultimodalContentText, Text: "kittens"},
    }}},
    Extensions: []llm.ProviderExtension{voyage.Extension{
        InputType:       voyage.InputTypeQuery,
        OutputDimension: 1024,
    }},
})
```

## Custom Base URLs

All providers accept a `BaseURL` option that overrides the
default API endpoint. You can use custom base URLs to connect
to proxy servers, self-hosted instances, or API-compatible
services:

```go
client, err := llm.NewClient("openai", llm.Options{
    APIKey:  os.Getenv("API_KEY"),
    Model:   "my-model",
    BaseURL: "https://my-proxy.example.com/v1",
})
```

## Custom Headers

All providers support injecting custom HTTP headers into every
request. You can use custom headers for authentication proxies,
request tracing, or other middleware:

```go
client, err := llm.NewClient("openai", llm.Options{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o",
    CustomHeaders: map[string]string{
        "X-Request-ID": "my-trace-id",
    },
})
```

Custom headers do not override headers that the provider sets
explicitly, such as `Content-Type` or authentication headers.
