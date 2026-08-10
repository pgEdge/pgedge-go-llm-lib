# Streaming

The `ChatStream` method sends a chat request and returns a
stream of response chunks in real time. Streaming allows your
application to display partial results as the model generates
the response.

## Basic Streaming

You can start a streaming request by calling `ChatStream`
with the same `ChatRequest` used for non-streaming calls.
Use `stream.Recv()` to read chunks one at a time; it returns
`io.EOF` when the stream is complete and a Go error on failure:

```go
import (
    "errors"
    "io"
)

stream, err := client.ChatStream(
    context.Background(),
    llm.ChatRequest{
        Messages: []llm.Message{llm.UserText("Tell me a story.")},
    },
)
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

    switch chunk.Type {
    case llm.ChunkText:
        fmt.Print(chunk.Text)
    case llm.ChunkDone:
        fmt.Println()
    }
}
```

## Stream Structure

The `ChatStream` method returns a `Stream` struct. The
recommended API is `Stream.Recv()`, which handles
channel-close coordination and surfaces stream errors as Go
errors.

`Stream.Collect` is a one-liner alternative that drains the
entire stream and assembles a `*ChatResponse`. It is useful
when you need the streaming path (e.g. to avoid upstream
timeouts) but ultimately want the complete response:

```go
resp, err := stream.Collect(context.Background())
if err != nil {
    log.Fatal(err)
}
for _, block := range resp.Content {
    if block.Type == llm.BlockText {
        fmt.Println(block.Text)
    }
}
```

`Collect` respects context cancellation and returns the
partial response along with the error when cancelled
mid-drain.

For advanced use cases that require `select`-driven
cancellation, the struct also exposes two channels directly:

- `Chunks` is a receive-only channel of `StreamChunk` values
  that delivers response data as the model generates output.
- `Err` is a receive-only channel that delivers an error if
  the stream encounters a problem during processing.

Both channels are closed when the stream completes.

## Chunk Types

Each `StreamChunk` has a `Type` field that indicates the kind
of data the chunk contains. Use the typed constants rather
than string literals:

| Constant              | String value    | Description                            |
|-----------------------|-----------------|----------------------------------------|
| `llm.ChunkText`       | `text`          | A text content delta.                  |
| `llm.ChunkToolUseStart` | `tool_use_start` | A tool call has started.             |
| `llm.ChunkToolUseDelta` | `tool_use_delta` | A partial tool call argument fragment. |
| `llm.ChunkDone`       | `done`          | The stream is complete.                |

## Processing Text Chunks

Text chunks carry partial response text in the `Text` field.
You can concatenate text chunks to build the complete response,
or use `stream.Collect` to assemble them automatically:

```go
var fullResponse strings.Builder
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    if chunk.Type == llm.ChunkText {
        fullResponse.WriteString(chunk.Text)
    }
}
fmt.Println(fullResponse.String())
```

## Processing Tool Call Chunks

When the model makes a tool call during streaming, the stream
emits a `llm.ChunkToolUseStart` chunk followed by one or more
`llm.ChunkToolUseDelta` chunks. The start chunk contains the
tool name, the ID, and any provider `Signature` in a
`*llm.ToolUse` value. Each delta chunk carries a partial JSON
argument fragment in `chunk.Partial`
(not `chunk.Text`). Concatenating all `Partial` values gives
you the complete `json.RawMessage` for `ToolUse.Input`.

The following example collects tool call arguments from a
stream and then sends the result back to the model:

```go
var toolArgs strings.Builder
var currentTool *llm.ToolUse

for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }

    switch chunk.Type {
    case llm.ChunkToolUseStart:
        // *llm.ToolUse{ID, Name, Input, Signature}
        currentTool = chunk.ToolUse
        toolArgs.Reset()
        // A provider may deliver complete arguments up front.
        toolArgs.Write(currentTool.Input)
    case llm.ChunkToolUseDelta:
        toolArgs.WriteString(chunk.Partial) // partial JSON fragment
    case llm.ChunkDone:
        if currentTool != nil {
            // toolArgs.String() is the complete JSON arguments
            currentTool.Input = json.RawMessage(toolArgs.String())
            result := executeTool(
                currentTool.Name, currentTool.Input,
            )

            // Replay the model's own tool call, then the result.
            followUp, _ := client.Chat(ctx, llm.ChatRequest{
                Messages: []llm.Message{
                    llm.UserText("original question"),
                    llm.AssistantBlocks(llm.ContentBlock{
                        Type:    llm.BlockToolUse,
                        ToolUse: currentTool,
                    }),
                    llm.ToolResultMessage(
                        currentTool.ID, result, false,
                    ),
                },
            })
            _ = followUp
        }
    }
}
```

The assistant turn must carry the `*llm.ToolUse` value that the
stream delivered, as shown above. Building a replacement from the
name and arguments alone discards `Signature`, and Gemini's
thinking models then reject the follow-up request outright.

Alternatively, use `stream.Collect` which handles this buffering
automatically and returns a `*ChatResponse` with complete
`BlockToolUse` content blocks, `Signature` included.

## Token Usage in Streams

The final `llm.ChunkDone` chunk always includes a non-nil
`Usage` field with token consumption data. If the upstream
provider does not report token counts, `Usage` is the zero
`TokenUsage` rather than nil.

The following example retrieves token usage from a stream:

```go
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    if chunk.Type == llm.ChunkDone {
        fmt.Printf("Tokens used: %d\n", chunk.Usage.TotalTokens)
    }
}
```

The client accumulates streaming token usage into cumulative
totals, which you can retrieve with `client.Usage()`.

## Error Handling

`stream.Recv()` returns a non-nil error (other than `io.EOF`)
when the stream encounters a problem. Stop iterating
immediately and handle the error:

```go
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Printf("Stream error: %v", err)
        return
    }
    // process chunk
}
```

## Low-Level Channel API

Advanced callers that need `select`-driven cancellation can
read the `Chunks` and `Err` channels directly. This is the
low-level alternative to `Recv()` and is not recommended for
typical use:

```go
for chunk := range stream.Chunks {
    switch chunk.Type {
    case llm.ChunkText:
        fmt.Print(chunk.Text)
    case llm.ChunkDone:
        fmt.Println()
    }
}

if err := <-stream.Err; err != nil {
    log.Printf("Stream error: %v", err)
}
```

## Provider Differences

The streaming implementation varies slightly between providers.

Anthropic, OpenAI, and Gemini use server-sent events for
streaming. Ollama uses newline-delimited JSON.

The Ollama provider buffers the entire streamed response when
tools are defined and checks the complete text for tool calls
after the stream finishes. When tools are not defined, the
Ollama provider emits text chunks incrementally.

## Next Steps

The following documents provide additional information:

- The [Tool Calling](tool_calling.md) document explains how
  to define and handle tool calls.
- The [Chat Completions](chat_completions.md) document
  covers non-streaming requests.
