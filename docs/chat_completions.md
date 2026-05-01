# Chat Completions

The `Chat` method sends a request to the LLM provider and
returns a complete response. This page explains how to
construct chat requests and process responses.

## Sending a Basic Request

You can send a chat request by calling `Chat` with a
`ChatRequest` containing one or more messages. The convenience
constructors `llm.UserText`, `llm.AssistantText`, and
`llm.SystemText` create the most common message shapes:

```go
resp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        Messages: []llm.Message{
            llm.UserText("What is Go?"),
        },
    },
)
if err != nil {
    log.Fatal(err)
}

for _, block := range resp.Content {
    if block.Type == llm.BlockText {
        fmt.Println(block.Text)
    }
}
```

## Message Roles

Each message has a `Role` field that identifies the sender.
Use the typed constants rather than string literals:

| Constant          | String value  | Description |
|-------------------|---------------|-------------|
| `llm.RoleUser`      | `"user"`      | User turn.            |
| `llm.RoleAssistant` | `"assistant"` | Model response.       |
| `llm.RoleSystem`    | `"system"`    | System-level instruction (OpenAI and Ollama). |
| `llm.RoleTool`      | `"tool"`      | Tool result.          |

## Message Content

`Message.Content` is always `[]ContentBlock`. Each block has
a `Type` constant:

| Constant               | Payload          |
|------------------------|------------------|
| `llm.BlockText`        | `Text string`    |
| `llm.BlockImage`       | `Image *ImageContent` |
| `llm.BlockToolUse`     | `ToolUse *ToolUse` |
| `llm.BlockToolResult`  | `ToolUseID`, `Text`, `IsError` |

Use the convenience constructors to avoid building blocks by
hand:

```go
llm.UserText("Hello!")                          // user + text block
llm.AssistantText("Hi there!")                  // assistant + text block
llm.SystemText("You are a helpful assistant.")  // system + text block
llm.ToolResultMessage(id, result, false)        // tool + tool-result block

// Multi-block message (text + image)
llm.UserBlocks(
    llm.TextBlock("What is in this image?"),
    llm.ImageBlock(pngBytes, "image/png"),
)
```

## Multi-Turn Conversations

You can send multi-turn conversations by including multiple
messages in the request. The model uses the full conversation
history as context:

```go
resp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        Messages: []llm.Message{
            llm.UserText("My name is Alice."),
            llm.AssistantText("Hello Alice! How can I help?"),
            llm.UserText("What is my name?"),
        },
    },
)
```

## System Prompts

Set a system prompt per-request using the `SystemPrompt` field
on `ChatRequest`:

```go
resp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        SystemPrompt: "You are a pirate.",
        Messages: []llm.Message{
            llm.UserText("Hello!"),
        },
    },
)
```

Alternatively, include a system message as the first element
of `Messages` using `llm.SystemText`:

```go
llm.ChatRequest{
    Messages: []llm.Message{
        llm.SystemText("You are a helpful assistant."),
        llm.UserText("Hello!"),
    },
}
```

## Temperature and Max Tokens

You can set default values for temperature and max tokens in
`Options` when creating the client. Use `llm.Float` and
`llm.Int` to set the pointer fields:

```go
client, err := llm.NewClient("openai", llm.Options{
    APIKey:      os.Getenv("OPENAI_API_KEY"),
    Model:       "gpt-4o",
    Temperature: llm.Float(0.5),
    MaxTokens:   llm.Int(2048),
})
```

You can also override these values on a per-request basis:

```go
resp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        Messages:    messages,
        Temperature: llm.Float(0.0), // deterministic
        MaxTokens:   llm.Int(512),
    },
)
```

A `nil` pointer on `ChatRequest` falls through to the `Options`
default. The following table describes the library defaults
when no value is specified:

| Option      | Default |
|-------------|---------|
| Temperature | 0.7     |
| MaxTokens   | 4096    |

## Response Format (JSON Mode)

The `ResponseFormat` field constrains the model's output
format. Three modes are supported:

```go
// Free-form JSON (model is instructed to produce valid JSON)
req.ResponseFormat = &llm.ResponseFormat{
    Type: llm.ResponseFormatJSON,
}

// JSON matching a schema (strict validation where supported)
req.ResponseFormat = &llm.ResponseFormat{
    Type:       llm.ResponseFormatJSONSchema,
    JSONSchema: json.RawMessage(`{"type":"object","properties":{...}}`),
}
```

Providers that don't support a given format return
`llm.ErrNotSupported`.

## Response Structure

The `ChatResponse` contains three fields:

- `Content` is a slice of `ContentBlock` values containing
  the model's response.
- `StopReason` indicates why the model stopped generating.
- `Usage` contains token consumption data for the request.

The following table describes the normalised `StopReason`
values:

| Constant                       | String value      | Description |
|--------------------------------|-------------------|-------------|
| `llm.StopReasonEndTurn`        | `"end_turn"`      | Model finished its response. |
| `llm.StopReasonToolUse`        | `"tool_use"`      | Model wants to call a tool.  |
| `llm.StopReasonMaxTokens`      | `"max_tokens"`    | Token limit reached.         |
| `llm.StopReasonStopSequence`   | `"stop_sequence"` | A stop sequence was matched. |
| `llm.StopReasonContentFilter`  | `"content_filter"`| Response filtered.           |
| `llm.StopReasonError`          | `"error"`         | Provider-level error.        |

All providers normalise their native stop reasons onto this
set; unrecognised values fall back to `StopReasonEndTurn`.

## Token Usage

Each response includes a `Usage` field that tracks token
consumption:

```go
fmt.Printf("Prompt: %d tokens\n",
    resp.Usage.PromptTokens)
fmt.Printf("Completion: %d tokens\n",
    resp.Usage.CompletionTokens)
fmt.Printf("Total: %d tokens\n",
    resp.Usage.TotalTokens)
```

The client also tracks cumulative usage across all requests.
Retrieve it with `client.Usage()` and reset it with
`client.ResetUsage()`:

```go
total := client.Usage()
fmt.Printf("Total tokens used this session: %d\n",
    total.TotalTokens)
client.ResetUsage() // reset the running counter
```

## Listing Available Models

`ListModels` returns model names; `ListModelsWithMetadata`
returns `[]llm.ModelInfo` with richer details:

```go
models, err := client.ListModels(context.Background())
for _, model := range models {
    fmt.Println(model)
}

infos, err := client.ListModelsWithMetadata(context.Background())
for _, m := range infos {
    fmt.Printf("%s: context=%d capabilities=%v\n",
        m.ID, m.ContextWindow, m.Capabilities)
}
```

Each provider filters the model list to return only
chat-capable models.

## Next Steps

The following documents provide additional information:

- The [Streaming](streaming.md) document explains how to
  receive responses in real time.
- The [Tool Calling](tool_calling.md) document describes how
  to use function calling with the chat API.
- The [Error Handling](error_handling.md) document covers how
  to handle errors from chat requests.
