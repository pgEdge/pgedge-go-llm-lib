# Error Handling

pgEdge Go LLM Library provides structured error handling with
sentinel errors and a `ProviderError` wrapper that includes
provider-specific details.

## Sentinel Errors

The library defines five sentinel errors that categorize
common failure modes. The following table describes each
sentinel error:

| Error             | Description                          |
|-------------------|--------------------------------------|
| ErrNotSupported   | The operation is not supported.      |
| ErrAuthentication | Authentication failed (401 or 403).  |
| ErrRateLimit      | Rate limit exceeded (429).           |
| ErrInvalidRequest | The request is invalid (400).        |
| ErrProviderError  | A general provider error occurred.   |

## Checking Error Types

You can check for specific error types using `errors.Is`
from the standard library:

```go
resp, err := client.Chat(ctx, req)
if err != nil {
    switch {
    case errors.Is(err, llm.ErrAuthentication):
        log.Println("Check your API key")
    case errors.Is(err, llm.ErrRateLimit):
        log.Println("Too many requests; retry later")
    case errors.Is(err, llm.ErrInvalidRequest):
        log.Println("Invalid request parameters")
    case errors.Is(err, llm.ErrNotSupported):
        log.Println("Operation not supported")
    default:
        log.Printf("Provider error: %v", err)
    }
}
```

## ProviderError Details

All provider errors are wrapped in a `ProviderError` struct
that contains additional context. You can extract the
`ProviderError` using `errors.As`:

```go
var provErr *llm.ProviderError
if errors.As(err, &provErr) {
    fmt.Printf("Provider: %s\n", provErr.Provider)
    fmt.Printf("Status: %d\n", provErr.StatusCode)
    fmt.Printf("Message: %s\n", provErr.Message)
}
```

The `ProviderError` struct contains the following fields:

- `Err` is the sentinel error that categorizes the failure.
- `StatusCode` is the HTTP status code from the provider.
- `Message` is the error message from the provider.
- `Provider` is the name of the provider that returned the
  error.

The `Error()` method formats the error as
`"provider (status): message"`.

The `Unwrap()` method returns the sentinel error so that
`errors.Is` works correctly with the wrapped error.

## Credential Redaction

The library redacts credentials from every provider error
message before storing the message on `ProviderError`. This
matters because an upstream API commonly quotes part of the
credential you submitted back in the body of an authentication
failure; OpenAI, for example, includes a partially masked form
of the submitted key in the message of a 401 response. A
service that returns `err.Error()` to a caller would otherwise
hand a fragment of the operator's API key to that caller.

Redaction replaces each credential it finds with the literal
string `[REDACTED]`. The library looks for:

- the configured API key, wherever the key appears verbatim.
- any fragment of the configured key long enough to identify
  the key, which covers a provider that echoes a truncated or
  partially masked form.
- text matching a known credential format, including the
  OpenAI, Anthropic, Google, and Voyage key prefixes, an HTTP
  `Authorization` value, and a labelled value such as
  `api_key=`.

The following example shows a redacted authentication failure
from the OpenAI provider:

```text
openai (401): Incorrect API key provided: [REDACTED]. You can
find your API key at https://platform.openai.com/account/api-keys.
```

Redaction always applies, and no option disables it. The
library never retains the unredacted text, so no caller can
log the credential by accident. The surrounding message
survives intact, which keeps the error useful for diagnosis.

The proxy applies the same redaction to the JSON error
responses and server-sent error events that it writes to HTTP
clients. The `OnError` hook still receives the unmodified
error, because that hook runs in your own process and serves
your server-side logging.

Redaction operates on the text of an error message, and it
cannot protect a credential that your own code places
somewhere else. Continue to keep keys out of log lines,
request bodies, and base URLs.

## HTTP Status Code Mapping

The Anthropic, OpenAI, and Gemini providers map HTTP status
codes to sentinel errors using the following rules:

| Status Code | Sentinel Error    |
|-------------|-------------------|
| 400         | ErrInvalidRequest |
| 401         | ErrAuthentication |
| 403         | ErrAuthentication |
| 429         | ErrRateLimit      |
| Other       | ErrProviderError  |

The Ollama provider maps all errors to `ErrProviderError`
because Ollama runs locally and does not use HTTP
authentication or rate limiting.

## Streaming Errors

Streaming responses can encounter errors during processing.
Use `stream.Recv()` which surfaces stream errors as Go errors
alongside `io.EOF` at end of stream:

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

For advanced callers that read channels directly, check the
`Err` channel after `Chunks` closes:

```go
for chunk := range stream.Chunks {
    // process chunks
}
if err := <-stream.Err; err != nil {
    log.Printf("Stream error: %v", err)
}
```

Errors from the initial HTTP request (such as authentication
failures) are returned directly from `ChatStream`. Errors that
occur during stream processing are surfaced through `Recv()`
or the `Err` channel.

## Client Creation Errors

The `NewClient` function returns a `ProviderError` with
`ErrInvalidRequest` when you provide an empty provider name
or an unregistered provider name:

```go
client, err := llm.NewClient("", llm.Options{})
// Error: " (0): provider name is required"

client, err = llm.NewClient("unknown", llm.Options{})
// Error: "unknown (0): unknown provider: unknown"
```
