# Troubleshooting

This document describes common issues you may encounter when
using pgEdge Go LLM Library and how to resolve them.

## Authentication Issues

### Invalid API Key

The provider returns an `ErrAuthentication` error when the API
key is missing, invalid, or expired. You can verify the error
type with the following code:

```go
if errors.Is(err, llm.ErrAuthentication) {
    log.Println("Check your API key")
}
```

You should verify that you are using the correct API key for
the provider and that the key has not been revoked or expired.

### Wrong API Key for Provider

Each provider requires an API key from the correct service.
An OpenAI key does not work with the Anthropic provider, and
an Anthropic key does not work with the OpenAI provider. Make
sure you are passing the correct key for the provider you are
using.

## Connection Issues

### Provider Unreachable

A network error or DNS resolution failure indicates that the
client cannot reach the provider's API endpoint. You should
verify that you have network connectivity and that any custom
`BaseURL` value is correct.

### Ollama Not Running

The Ollama provider connects to `http://localhost:11434` by
default. If Ollama is not running, the client returns a
connection refused error. You can start Ollama by running the
following command:

```bash
ollama serve
```

### Custom Base URL Issues

When using a custom `BaseURL`, verify that the URL includes
the correct path prefix. For example, the OpenAI provider
appends `/chat/completions` to the base URL, so the base URL
should end with `/v1` rather than `/v1/chat/completions`.

## Request Issues

### Unknown Provider

The `NewClient` function returns an error when you specify a
provider name that has not been registered. You should verify
that you have imported the provider package:

```go
import (
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
)
```

If you import individual providers, verify that the provider
name matches the import. The valid provider names are
`"anthropic"`, `"openai"`, `"gemini"`, and `"ollama"`.

### Embeddings Not Supported

The Anthropic provider does not support embeddings. Calling
`Embed` or `EmbedBatch` on an Anthropic client returns an
`ErrNotSupported` error. You should use the OpenAI, Gemini,
or Ollama provider for embedding operations.

### Rate Limiting

The provider returns an `ErrRateLimit` error when you exceed
the API rate limit. The library retries 429 responses
automatically (up to 5 times by default, honouring
`Retry-After` if present). If you still receive `ErrRateLimit`
after retries are exhausted, consider increasing
`RetryConfig.MaxRetries` or reducing your request rate.
Use `Options.OnRetry` to observe retry events.

## Tool Calling Issues

### Ollama Tool Call Parsing

The Ollama provider parses tool calls from the model's text
response. If the model does not respond with a valid JSON
tool call, the provider returns the response as plain text
instead of a tool call. You should use a model that follows
instructions well for tool calling with Ollama.

### Gemini Tool Call IDs

The Gemini provider generates tool call IDs in the format
`gemini-tool-{name}`. When sending tool results back to
Gemini, the provider extracts the tool name from this ID. You
should use the exact tool call ID from the `ToolUse` struct
when constructing `ToolResult` messages.

### Ollama Reasoning Models Wrapping JSON

Reasoning models served by Ollama (for example `deepseek-r1`)
emit `<think>...</think>` blocks before their final answer.
The Ollama provider strips these blocks from response content
and from the buffer used for tool-call detection, so reasoning
output should not break tool calling or JSON parsing
downstream. If you observe a tool call being missed, verify
the model emits a valid JSON object matching the format
`{"tool":"...","arguments":{...}}` *outside* its thinking
block, and consider using a tool-tuned model.

## Streaming Issues

### Stream Errors After Chunks

Errors that occur during stream processing are surfaced by
`stream.Recv()` as Go errors alongside `io.EOF`. Stop
iterating immediately when `Recv()` returns a non-nil,
non-EOF error. If you are reading the `Chunks` channel
directly, check `<-stream.Err` after the channel closes.

### Missing Token Usage in Streams

The final `ChunkDone` chunk always includes a non-nil `Usage`
pointer, but if the upstream provider does not report token
counts the `TokenUsage` is the zero value (all fields 0).
Check whether `TotalTokens` is non-zero before relying on
the counts.

## Still Have Questions?

For additional help with pgEdge Go LLM Library:

- Visit the
  [GitHub repository](https://github.com/pgEdge/pgedge-go-llm-lib)
  for source code and examples.
- To report an issue, visit the
  [Issues page](https://github.com/pgEdge/pgedge-go-llm-lib/issues).
- For more information about pgEdge, visit
  [docs.pgedge.com](https://docs.pgedge.com).
