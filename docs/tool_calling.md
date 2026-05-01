# Tool Calling

Tool calling allows the model to request execution of
functions that you define. The model receives tool definitions,
decides when to call them, and returns structured arguments
that your code can execute.

## Defining Tools

You define tools using the `Tool` struct with a name,
description, and a JSON Schema for the input parameters:

```go
tools := []llm.Tool{
    {
        Name:        "get_weather",
        Description: "Get the current weather for a city.",
        InputSchema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name."
                }
            },
            "required": ["city"]
        }`),
    },
}
```

## Sending a Request with Tools

You can include tools in a `ChatRequest` by setting the
`Tools` field:

```go
resp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        Messages: []llm.Message{
            llm.UserText("What is the weather in London?"),
        },
        Tools: tools,
    },
)
```

## Detecting Tool Calls

When the model decides to call a tool, the response has a
`StopReason` of `llm.StopReasonToolUse` and the `Content`
slice contains one or more `ContentBlock` values with `Type`
set to `llm.BlockToolUse`:

```go
if resp.StopReason == llm.StopReasonToolUse {
    for _, block := range resp.Content {
        if block.Type == llm.BlockToolUse {
            fmt.Printf("Tool: %s\n", block.ToolUse.Name)
            fmt.Printf("ID:   %s\n", block.ToolUse.ID)
            fmt.Printf("Args: %s\n", block.ToolUse.Input)
        }
    }
}
```

The `ToolUse` struct contains the following fields:

- `ID` is a unique identifier for the tool call.
- `Name` is the name of the tool the model wants to call.
- `Input` is a `json.RawMessage` containing the tool
  arguments as JSON.

## Executing Tools and Sending Results

After executing a tool, send the result back by appending
the assistant's response and a tool-result message to the
conversation. Use `llm.ToolResultMessage` to build the
tool-role message:

```go
// Execute the tool.
result := getWeather("London")

// Build the follow-up conversation.
messages := []llm.Message{
    llm.UserText("What is the weather in London?"),
    llm.AssistantBlocks(resp.Content...),           // include assistant's tool-use turn
    llm.ToolResultMessage(block.ToolUse.ID, result, false),
}

// Send the follow-up request.
followUp, err := client.Chat(
    context.Background(),
    llm.ChatRequest{
        Messages: messages,
        Tools:    tools,
    },
)
```

`llm.ToolResultMessage(toolUseID, text, isError)` is shorthand
for a tool-role message containing a single `BlockToolResult`
content block. For multiple tool results in one turn, use
`llm.UserBlocks` (or `llm.AssistantBlocks`) with explicit
`llm.ToolResultBlock` calls:

```go
messages = append(messages, llm.Message{
    Role: llm.RoleTool,
    Content: []llm.ContentBlock{
        llm.ToolResultBlock("tool-id-1", "Result 1", false),
        llm.ToolResultBlock("tool-id-2", "Result 2", false),
    },
})
```

## Tool Call Loop

A common pattern is to loop until the model stops requesting
tools:

```go
messages := []llm.Message{
    llm.UserText("What is the weather in London?"),
}

for {
    resp, err := client.Chat(ctx, llm.ChatRequest{
        Messages: messages,
        Tools:    tools,
    })
    if err != nil {
        log.Fatal(err)
    }

    // Append the assistant's response.
    messages = append(messages, llm.AssistantBlocks(resp.Content...))

    if resp.StopReason != llm.StopReasonToolUse {
        // Model finished; print the final response.
        for _, block := range resp.Content {
            if block.Type == llm.BlockText {
                fmt.Println(block.Text)
            }
        }
        break
    }

    // Execute each tool call and append results.
    for _, block := range resp.Content {
        if block.Type == llm.BlockToolUse {
            result := executeTool(
                block.ToolUse.Name,
                block.ToolUse.Input,
            )
            messages = append(messages,
                llm.ToolResultMessage(block.ToolUse.ID, result, false),
            )
        }
    }
}
```

## Controlling Tool Choice

The `ToolChoice` field on `ChatRequest` constrains how the
model selects tools:

```go
// Force the model to call a specific tool
resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: messages,
    Tools:    tools,
    ToolChoice: &llm.ToolChoice{
        Mode: llm.ToolChoiceSpecific,
        Name: "get_weather",
    },
})
```

| Mode constant              | Description |
|----------------------------|-------------|
| `llm.ToolChoiceAuto`       | Model decides whether to call a tool (default). |
| `llm.ToolChoiceNone`       | Forbid tool calls.                              |
| `llm.ToolChoiceRequired`   | Force the model to call any tool.               |
| `llm.ToolChoiceSpecific`   | Force the model to call a named tool.           |

Providers that do not support `ToolChoice` (e.g. Ollama with
text-based tool parsing) ignore the field.

## Prompt Caching for Tools (Anthropic)

The Anthropic provider supports prompt caching on tool
definitions. Use the `anthropic.WithToolCaching` helper to
mark the entire tools block as a cacheable prefix:

```go
import "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"

req = anthropic.WithToolCaching(llm.ChatRequest{
    Messages: messages,
    Tools:    tools,
})
resp, err := client.Chat(ctx, req)
```

`WithToolCaching` is safe to call unconditionally — providers
other than Anthropic ignore the extension.

## Streaming Tool Calls

Tool calls also work with streaming responses. The stream
emits a `llm.ChunkToolUseStart` chunk (with `*ToolUse`
containing `ID` and `Name`) followed by `llm.ChunkToolUseDelta`
chunks carrying partial `json.RawMessage` fragments in
`chunk.Partial`. Concatenate the fragments to reconstruct the
full `Input`.

Alternatively, use `stream.Collect(ctx)` which buffers the
entire stream and returns a `*ChatResponse` with complete
`BlockToolUse` content blocks ready to pass back to the model.

See the [Streaming](streaming.md) document for a worked
example.

## Provider Differences

The Anthropic, OpenAI, and Gemini providers support native
tool calling through their respective APIs.

The Ollama provider implements tool calling through text-based
parsing. The provider injects tool definitions into the system
prompt and parses the model's response for JSON objects
matching the format
`{"tool":"tool_name","arguments":{...}}`. Tool call IDs from
Ollama always use the value `"ollama-tool-1"`.

## Next Steps

The following documents provide additional information:

- The [Streaming](streaming.md) document covers streaming
  tool call chunks.
- The [API Reference](api_reference.md) document provides
  complete type definitions for tool-related structures.
