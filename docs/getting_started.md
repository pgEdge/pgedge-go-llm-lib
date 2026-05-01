# Getting Started

This guide explains how to install pgEdge Go LLM Library and
make your first API call.

## Prerequisites

pgEdge Go LLM Library requires the following software:

- Go 1.26 or later.

## Installation

You can install pgEdge Go LLM Library with `go get`:

```bash
go get github.com/pgEdge/pgedge-go-llm-lib
```

## Quick Start

The following example demonstrates how to create a client and
send a chat completion request using the OpenAI provider:

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
        Model:  "gpt-4o",
    })
    if err != nil {
        log.Fatal(err)
    }

    resp, err := client.Chat(
        context.Background(),
        llm.ChatRequest{
            Messages: []llm.Message{
                llm.UserText("Hello!"),
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
}
```

## Importing Providers

pgEdge Go LLM Library offers two ways to register providers.

You can import all providers at once using a blank import of
the `llm/all` package:

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
)
```

You can also import only the providers you need to reduce the
compiled binary size:

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
)
```

## Switching Providers

Because all providers implement the same `Client` interface,
you can switch providers by changing the provider name and
options. The following example uses Anthropic instead of
OpenAI:

```go
client, err := llm.NewClient("anthropic", llm.Options{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Model:  "claude-sonnet-4-20250514",
})
```

The rest of your code remains the same regardless of the
provider you choose.

## Next Steps

The following documents provide additional information:

- The [Providers](providers.md) document describes how to
  configure each supported provider.
- The [Chat Completions](chat_completions.md) document
  explains how to use the chat API in detail.
- The [Streaming](streaming.md) document covers real-time
  streaming responses.
