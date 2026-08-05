<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/pgedge-labs-dark.svg">
    <img alt="pgEdge Labs" src="docs/img/pgedge-labs-light.svg" width="320">
  </picture>
</div>

# pgEdge Go LLM Library

[![CI - Build](https://github.com/pgEdge/pgedge-go-llm-lib/actions/workflows/ci-build.yml/badge.svg)](https://github.com/pgEdge/pgedge-go-llm-lib/actions/workflows/ci-build.yml)
[![CI - Docs](https://github.com/pgEdge/pgedge-go-llm-lib/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/pgEdge/pgedge-go-llm-lib/actions/workflows/ci-docs.yml)

pgEdge Go LLM Library is a unified Go interface for Anthropic, OpenAI,
Gemini, and Ollama. It includes an HTTP proxy package, production-grade
retry with exponential back-off, per-request timeouts, observability
hooks, and zero external dependencies.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Content Blocks](#content-blocks)
- [Streaming](#streaming)
- [Tools](#tools)
- [Multimodal Images](#multimodal-images)
- [Provider Extensions](#provider-extensions)
- [Errors and Retries](#errors-and-retries)
- [Discovery](#discovery)
- [HTTP Proxy](#http-proxy)
- [Supported Providers](#supported-providers)
- [Documentation](#documentation)
- [Development](#development)

## Quick Start

```bash
go get github.com/pgEdge/pgedge-go-llm-lib
```

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
)

func main() {
    client, err := llm.NewClient("openai", llm.Options{
        APIKey: os.Getenv("OPENAI_API_KEY"),
        Model:  "gpt-4o-mini",
    })
    if err != nil {
        log.Fatal(err)
    }

    resp, err := client.Chat(context.Background(), llm.ChatRequest{
        Messages: []llm.Message{llm.UserText("Hello!")},
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, block := range resp.Content {
        if block.Type == llm.BlockText {
            fmt.Println(block.Text)
        }
    }
}
```

Import all five providers at once with `_ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"`,
or import individual provider packages to keep binary size small:

```go
import (
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)
```

## Configuration

`llm.Options` configures the client:

| Field            | Type                | Description |
|------------------|---------------------|-------------|
| `APIKey`         | `string`            | Provider API key. |
| `Model`          | `string`            | Model name or ID. |
| `BaseURL`        | `string`            | Override the API endpoint (proxy gateway, LM Studio, etc.). Validated at construction. |
| `CustomHeaders`  | `map[string]string` | Headers injected into every upstream request (tracing IDs, gateway tokens). |
| `HTTPClient`     | `*http.Client`      | Bring your own HTTP client (mTLS, OpenTelemetry round-tripper, corporate proxy). |
| `MaxTokens`      | `*int`              | Default response length cap. `nil` → library default of 4096. Use `llm.Int(n)`. |
| `Temperature`    | `*float64`          | Default sampling temperature. `nil` → omitted from the request, leaving the provider's own default. Use `llm.Float(t)`. |
| `RequestTimeout` | `time.Duration`     | Wall-clock cap for a single HTTP attempt (0 → 120 s default). |
| `Retry`          | `RetryConfig`       | Retry policy (see below). |
| `OnRetry`        | `func(RetryEvent)`  | Observability hook fired before each retry sleep. |

**Precedence rule:** `Temperature` and `MaxTokens` on `ChatRequest` override the
`Options` defaults for that request. A `nil` pointer on `ChatRequest` falls
through to the `Options` value.

Leaving `Temperature` unset in both places omits the field from the upstream
request entirely, which matters because some newer models reject any
temperature at all; an explicitly-set `0` is preserved and sent as
`temperature: 0`.

### Retry

All providers retry transient failures by default: network errors, HTTP 429
(honouring `Retry-After`), 500/502/503/504, and Anthropic's 529 "overloaded"
status. Backoff is exponential, capped at 60 s, with up to 5 retries:

```go
client, _ := llm.NewClient("openai", llm.Options{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o",
    Retry: llm.RetryConfig{
        MaxRetries:     3,
        InitialBackoff: 1 * time.Second,
        MaxBackoff:     30 * time.Second,
    },
    OnRetry: func(e llm.RetryEvent) {
        log.Printf("attempt %d failed (status %d); retrying in %s",
            e.Attempt, e.StatusCode, e.Wait)
    },
})
```

To disable retries: `Retry: llm.RetryConfig{Disabled: true}`.

## Content Blocks

Messages carry a `[]ContentBlock` slice, not a plain string. Each block has
a `Type` constant that selects which payload field is populated:

| Type constant        | String value   | Payload field(s)                    |
|----------------------|----------------|-------------------------------------|
| `llm.BlockText`      | `"text"`       | `Text string`                       |
| `llm.BlockImage`     | `"image"`      | `Image *ImageContent`               |
| `llm.BlockDocument`  | `"document"`   | `Document *DocumentContent`         |
| `llm.BlockToolUse`   | `"tool_use"`   | `ToolUse *ToolUse`                  |
| `llm.BlockToolResult`| `"tool_result"`| `ToolUseID`, `Text`, `IsError`      |

**Convenience constructors** cover the common cases so you rarely build blocks
by hand:

```go
// Simple messages
llm.UserText("Hello!")                          // user role, one text block
llm.AssistantText("Hi!")                        // assistant role
llm.SystemText("You are a helpful assistant.")  // system role

// Tool results
llm.ToolResultMessage(toolUseID, resultText, false) // tool role

// Mixed content (e.g. text + image)
llm.UserBlocks(
    llm.TextBlock("Describe this image:"),
    llm.ImageBlock(pngBytes, "image/png"),
)

// Block builders
llm.TextBlock("plain text")
llm.ImageBlock(data, "image/jpeg")
llm.ImageURLBlock("https://example.com/photo.jpg")
llm.DocumentBlock(pdfBytes, "application/pdf", "report.pdf")
llm.DocumentURLBlock("https://example.com/report.pdf", "application/pdf", "report.pdf")
llm.ToolResultBlock(toolUseID, result, isError)
```

## Streaming

Call `ChatStream` instead of `Chat` and drive the result with `stream.Recv()`:

```go
stream, err := client.ChatStream(context.Background(), llm.ChatRequest{
    Messages: []llm.Message{llm.UserText("Tell me a story.")},
})
if err != nil {
    log.Fatal(err)
}

for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    if chunk.Type == llm.ChunkText {
        fmt.Print(chunk.Text)
    }
}
```

**`Stream.Collect`** is a one-liner that drains the stream and assembles a
`*ChatResponse` — useful when you want the full response but need the
streaming path (e.g. to avoid provider timeouts on long responses):

```go
resp, err := stream.Collect(context.Background())
```

`Recv` returns `io.EOF` at end of stream. The final `ChunkDone` chunk always
carries a non-nil `*TokenUsage`. The underlying `Chunks` and `Err` channels
are exported for advanced callers that need `select`-driven cancellation.

See [docs/streaming.md](docs/streaming.md) for full details.

## Tools

Define tools, pass them on `ChatRequest`, detect a `tool_use` stop reason,
execute the tool, and send results back:

```go
tools := []llm.Tool{{
    Name:        "get_weather",
    Description: "Get current weather for a city.",
    InputSchema: json.RawMessage(`{
        "type":"object",
        "properties":{"city":{"type":"string"}},
        "required":["city"]
    }`),
}}

resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: []llm.Message{llm.UserText("Weather in London?")},
    Tools:    tools,
})

// Detect tool call
if resp.StopReason == llm.StopReasonToolUse {
    for _, block := range resp.Content {
        if block.Type == llm.BlockToolUse {
            tu := block.ToolUse // *llm.ToolUse{ID, Name, Input json.RawMessage}
            result := executeTool(tu.Name, tu.Input)

            followUp, err := client.Chat(ctx, llm.ChatRequest{
                Messages: []llm.Message{
                    llm.UserText("Weather in London?"),
                    llm.AssistantBlocks(resp.Content...),
                    llm.ToolResultMessage(tu.ID, result, false),
                },
                Tools: tools,
            })
            _ = followUp
        }
    }
}
```

**ToolChoice** controls how the model selects tools:

```go
// Force the model to call a specific tool
req.ToolChoice = &llm.ToolChoice{
    Mode: llm.ToolChoiceSpecific,
    Name: "get_weather",
}
// Or: llm.ToolChoiceAuto (default), llm.ToolChoiceNone, llm.ToolChoiceRequired
```

**StopSequences** halt generation when a string is encountered:

```go
req.StopSequences = []string{"</answer>", "STOP"}
```

**JSON mode** constrains the model's output format:

```go
// Free-form JSON
req.ResponseFormat = &llm.ResponseFormat{Type: llm.ResponseFormatJSON}

// JSON matching a schema (providers that don't support strict schemas
// fall back to free-form JSON with a system-prompt instruction)
req.ResponseFormat = &llm.ResponseFormat{
    Type:       llm.ResponseFormatJSONSchema,
    JSONSchema: json.RawMessage(`{"type":"object","properties":{...}}`),
}
```

See [docs/tool_calling.md](docs/tool_calling.md) for complete examples.

## Multimodal Images

Send images inline (base64) or by URL alongside text:

```go
imgData, _ := os.ReadFile("photo.png")

resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: []llm.Message{
        llm.UserBlocks(
            llm.TextBlock("What is in this image?"),
            llm.ImageBlock(imgData, "image/png"),   // inline base64
        ),
    },
})
```

For URL images use `llm.ImageURLBlock("https://...")`. Note: Anthropic and
OpenAI support URL images; Gemini accepts file URIs; Ollama rejects
URL-only images.

## Documents (PDFs)

Send a PDF (or other document type the provider supports) inline alongside
text, without extracting the text yourself:

```go
pdf, _ := os.ReadFile("itinerary.pdf")

resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: []llm.Message{
        llm.UserBlocks(
            llm.TextBlock("Extract the flight legs from this confirmation:"),
            llm.DocumentBlock(pdf, "application/pdf", "itinerary.pdf"),
        ),
    },
})
```

For documents referenced by URL use `llm.DocumentURLBlock(url, mediaType,
filename)`. Provider support:

- **Anthropic** — native PDF support (base64 inline up to 32 MB / 100 pages,
  or URL source). The PDF-beta header is enabled automatically.
- **Gemini** — native, via inline base64 or file URI.
- **OpenAI** — not supported by the Chat Completions API; the request is
  rejected with `llm.ErrNotSupported` before any upstream call.
- **Ollama** — not supported; the request is rejected with
  `llm.ErrNotSupported`.

For non-supporting providers, extract the text upstream of the call and pass
it as a regular text block.

## Provider Extensions

`ProviderExtension` is a typed escape hatch for provider-specific options.
Providers ignore extensions whose `ProviderName()` does not match theirs,
so extensions are forward-compatible across providers.

### Anthropic: tool caching

```go
import "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"

req = anthropic.WithToolCaching(req) // marks entire tools block as cacheable prefix
```

### Anthropic: extended thinking

```go
req = anthropic.WithExtendedThinking(req, 8000) // enable thinking with 8 000-token budget
```

### OpenAI: embedding dimensions

`text-embedding-3-small` and `text-embedding-3-large` support requesting a
lower-dimensional vector via the `dimensions` parameter. Attach an
`openai.Extension` to `llm.Options.Extensions` (client-level, because
`Embed` / `EmbedBatch` take plain arguments rather than a request struct):

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
)

client, _ := llm.NewClient("openai", llm.Options{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "text-embedding-3-small",
    Extensions: []llm.ProviderExtension{
        openai.Extension{EmbeddingDimensions: 512},
    },
})
vec, _ := client.Embed(ctx, "hello") // 512-dim vector
```

`EmbeddingDimensions: 0` (the zero value) omits the parameter entirely
and the model returns its native vector size.

### Ollama: embedding context length

Ollama's `/api/embed` endpoint accepts an `options.num_ctx` field that
overrides the model's compiled context window. Raise it when chunks
routinely exceed the default (e.g. `nomic-embed-text` ships with 2048
and rejects longer inputs with "the input length exceeds the context
length"). Attach an `ollama.Extension` to `llm.Options.Extensions`:

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
)

client, _ := llm.NewClient("ollama", llm.Options{
    BaseURL: "http://localhost:11434",
    Model:   "nomic-embed-text",
    Extensions: []llm.ProviderExtension{
        ollama.Extension{EmbedContextLength: 8192},
    },
})
vec, _ := client.Embed(ctx, longText) // num_ctx=8192 on the wire
```

`EmbedContextLength: 0` (the zero value) omits `num_ctx` from the
request body and Ollama uses the model's compiled default.

## Errors and Retries

```go
resp, err := client.Chat(ctx, req)
if err != nil {
    switch {
    case errors.Is(err, llm.ErrAuthentication):
        log.Println("check your API key")
    case errors.Is(err, llm.ErrRateLimit):
        log.Println("rate limit — back off")
    case errors.Is(err, llm.ErrInvalidRequest):
        log.Println("bad request")
    case errors.Is(err, llm.ErrNotSupported):
        log.Println("unsupported operation")
    default:
        log.Println("provider error:", err)
    }

    // Get provider-specific details
    var pe *llm.ProviderError
    if errors.As(err, &pe) {
        log.Printf("%s (%d): %s", pe.Provider, pe.StatusCode, pe.Message)
    }
}
```

Sentinel errors: `ErrNotSupported`, `ErrAuthentication`, `ErrRateLimit`,
`ErrInvalidRequest`, `ErrProviderError`. All are wrapped in `*ProviderError`.

The `OnRetry` hook on `Options` receives a `RetryEvent` before each sleep,
with `Attempt`, `StatusCode`, `Err`, and `Wait` fields — useful for
dashboards and circuit breakers.

## Discovery

```go
// List all registered providers (sorted)
names := llm.RegisteredProviders() // []string{"anthropic","gemini","ollama","openai","voyage"}

// Check connectivity
if err := client.Ping(ctx); err != nil {
    log.Println("provider unreachable:", err)
}

// Cumulative token usage across all requests
total := client.Usage()

// Reset the cumulative counter
client.ResetUsage()

// Rich model listing with capabilities and context window
models, _ := client.ListModelsWithMetadata(ctx) // []llm.ModelInfo
for _, m := range models {
    fmt.Printf("%s context=%d\n", m.ID, m.ContextWindow)
}
```

`ModelInfo` fields: `ID`, `ContextWindow`, `MaxOutput`, `Capabilities`
(`"chat"`, `"tools"`, `"vision"`, `"embeddings"`, `"json_mode"`, `"streaming"`),
`Deprecated`.

## HTTP Proxy

The `llm/proxy` package exposes the library's provider abstraction as an
`http.Handler` with SSE streaming. Any HTTP client — browser, curl, a
separate service — can drive LLM requests without linking against the Go
library.

### Endpoints

| Method | Path                               | Description |
|--------|------------------------------------|-------------|
| `GET`  | `/v1/providers`                    | List configured providers and the default. |
| `GET`  | `/v1/models?provider=X`            | List model names for provider X. |
| `GET`  | `/v1/models?provider=X&metadata=true` | List models with `ModelInfo` metadata. |
| `POST` | `/v1/chat`                         | Non-streaming chat (JSON response). |
| `POST` | `/v1/chat/stream`                  | Streaming chat (SSE response). |
| `GET`  | `/v1/health`                       | Ping all configured providers; 200 if all OK, 503 if any are down. |
| `POST` | `/v1/embed`                        | Generate text embeddings for one or more inputs. |
| `POST` | `/v1/embed/multimodal`             | Generate multimodal embeddings (text + images). |
| `POST` | `/v1/rerank`                       | Rerank documents by relevance to a query. |
| `GET`  | `/v1/models?...&capability=X`      | Filter models by capability (repeatable; AND of values). |

### Quick Start

```go
import (
    "net/http"
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/proxy"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
)

p := proxy.New(proxy.Config{
    DefaultProvider: "anthropic",
    Providers: map[string]llm.Options{
        "anthropic": {APIKey: os.Getenv("ANTHROPIC_API_KEY"), Model: "claude-sonnet-4-20250514"},
        "openai":    {APIKey: os.Getenv("OPENAI_API_KEY"),    Model: "gpt-4o"},
    },
})
http.Handle("/api/", http.StripPrefix("/api", p.Handler()))
```

### Authorize Hook

For multi-tenant gateways, supply an `Authorize` function. It fires before
every request (including GET endpoints). Return a non-nil error to reject:

```go
proxy.Config{
    Authorize: func(r *http.Request) error {
        token := r.Header.Get("Authorization")
        if !isValidToken(token) {
            return &proxy.AuthError{Err: errors.New("invalid token"), Status: 401}
        }
        return nil
    },
}
```

### Request-ID Propagation

The proxy reads an incoming `X-Request-ID` header (configurable via
`Config.RequestIDHeader`), generates one if absent, and echoes it back on
the response. It is also attached to the request context so telemetry hooks
can correlate events:

```go
proxy.Config{
    OnRequest: func(r *http.Request, info proxy.RequestInfo) {
        reqID := proxy.RequestIDFromContext(r.Context())
        log.Printf("[%s] %s/%s stream=%v", reqID, info.Provider, info.Model, info.Stream)
    },
}
```

Set `RequestIDHeader: "-"` to disable propagation entirely.

### SSE Wire Format

Streaming responses (`POST /v1/chat/stream`) use Server-Sent Events:

```
data: {"type":"text","text":"Hello"}\n\n
data: {"type":"text","text":" world"}\n\n
event: done\ndata: {"type":"done","usage":{"total_tokens":42}}\n\n
```

On error: `event: error\ndata: {"error":"..."}\n\n`

### Telemetry Hooks

`Config` accepts `OnRequest`, `OnResponse`, and `OnError` callbacks. They
fire synchronously on the request goroutine; keep them fast:

```go
proxy.Config{
    OnResponse: func(r *http.Request, info proxy.ResponseInfo) {
        metrics.RecordTokens(info.Provider, info.Usage.TotalTokens)
    },
    OnError: func(r *http.Request, info proxy.ErrorInfo) {
        log.Printf("error %d: %v", info.StatusCode, info.Err)
    },
}
```

`RequestInfo.Request` carries the resolved `*llm.ChatRequest` the proxy
is about to send (messages, tools, system prompt, tool-choice, response
format, stop sequences). `ResponseInfo.Response` carries the
`*llm.ChatResponse` the proxy returned — including text, tool calls
with parameters, stop reason, and the same `TokenUsage` exposed on
`ResponseInfo.Usage`. For streaming requests the `Response` is
assembled from the SSE chunks (text deltas concatenated, tool-use
deltas folded into their start block), so hooks see one shape regardless
of whether the call streamed:

```go
proxy.Config{
    OnRequest: func(r *http.Request, info proxy.RequestInfo) {
        if info.Request == nil { return }
        for _, m := range info.Request.Messages {
            for _, b := range m.Content {
                if b.Type == llm.BlockText {
                    auditLog.Prompt(info.RequestID, m.Role, b.Text)
                }
            }
        }
    },
    OnResponse: func(r *http.Request, info proxy.ResponseInfo) {
        if info.Response == nil { return }
        for _, b := range info.Response.Content {
            switch b.Type {
            case llm.BlockText:
                auditLog.Response(info.RequestID, b.Text)
            case llm.BlockToolUse:
                auditLog.ToolCall(info.RequestID, b.ToolUse.Name, b.ToolUse.Input)
            }
        }
    },
}
```

## Supported Providers

| Feature               | Anthropic | OpenAI | Gemini | Ollama | Voyage |
|-----------------------|-----------|--------|--------|--------|--------|
| Chat                  | Yes       | Yes    | Yes    | Yes    | No     |
| Streaming             | Yes       | Yes    | Yes    | Yes    | No     |
| Embeddings            | No        | Yes    | Yes    | Yes    | Yes    |
| Multimodal Embeddings | No        | No     | No     | No     | Yes    |
| Reranking             | No        | No     | No     | No     | Yes    |
| Tool Calling          | Yes       | Yes    | Yes    | Yes*   | No     |
| Multimodal Images     | Yes       | Yes    | Yes    | No†    | (via multimodal embed) |
| JSON Mode             | Yes       | Yes    | Yes    | Yes    | No     |
| Prompt Caching        | Yes       | No     | No     | No     | No     |
| Token Tracking        | Yes       | Yes    | Yes    | Yes    | Yes    |

\* Ollama tool calling is implemented via text-based parsing; results vary by model.  
† Ollama rejects URL images; inline base64 may work with vision-capable models.  
‡ Request is rejected with `llm.ErrNotSupported`. Extract document text yourself before calling.

## Documentation

Full reference documentation is in the [`docs/`](docs/) directory:

- [Getting Started](docs/getting_started.md)
- [Providers](docs/providers.md)
- [Chat Completions](docs/chat_completions.md)
- [Streaming](docs/streaming.md)
- [Tool Calling](docs/tool_calling.md)
- [Embeddings](docs/embeddings.md)
- [Error Handling](docs/error_handling.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

For godoc, run `go doc github.com/pgEdge/pgedge-go-llm-lib/llm` or browse
[pkg.go.dev](https://pkg.go.dev/github.com/pgEdge/pgedge-go-llm-lib).

The project documentation is also built with MkDocs Material:

```bash
make docs        # serve locally
make docs-build  # build (strict) into site/
```

## Development

The repository ships with a `Makefile` covering the common loops:

```bash
make build         # compile all packages
make test          # go test -race ./...
make coverage      # writes coverage.out + coverage.html
make fmt           # gofmt -s -w
make fmt-check     # fail on unformatted files (CI-friendly)
make vet           # go vet ./...
make lint          # gofmt-check, go vet, and golangci-lint
make lint-install  # install the pinned golangci-lint version
make tidy          # go mod tidy
make help          # list every target
```

`make lint` runs `golangci-lint` against the configuration in
[`.golangci.yml`](.golangci.yml). Run `make lint-install` once to drop the
pinned binary into `$GOBIN`. The coverage target is **90%+ per file**, not
just in aggregate; the current suite holds every source file at or above
that threshold.

## Support & Resources

For more information, visit [docs.pgedge.com](https://docs.pgedge.com).

To report an issue with the software, visit the
[Issues page](https://github.com/pgEdge/pgedge-go-llm-lib/issues).

## License

This project is licensed under the [PostgreSQL License](LICENSE.md).
