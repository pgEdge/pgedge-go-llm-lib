# Embeddings

Embeddings convert text into numerical vectors that capture
semantic meaning. You can use embeddings for similarity search,
clustering, classification, and retrieval-augmented generation.

## Provider Support

The following table describes embedding support across
providers:

| Provider  | Embed | EmbedBatch | Batch Method      |
|-----------|-------|------------|-------------------|
| OpenAI    | Yes   | Yes        | Native batch API  |
| Gemini    | Yes   | Yes        | Sequential calls  |
| Ollama    | Yes   | Yes        | Sequential calls  |
| Anthropic | No    | No         | Not supported     |

Calling `Embed` or `EmbedBatch` on the Anthropic provider
returns an `ErrNotSupported` error.

## Single Embedding

The `Embed` method generates an embedding vector for a single
text string:

```go
client, err := llm.NewClient("openai", llm.Options{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "text-embedding-3-small",
})
if err != nil {
    log.Fatal(err)
}

vector, err := client.Embed(
    context.Background(),
    "The quick brown fox jumps over the lazy dog.",
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Dimensions: %d\n", len(vector))
```

The method returns a `[]float64` slice containing the
embedding vector.

## Batch Embeddings

The `EmbedBatch` method generates embeddings for multiple
text strings in a single call:

```go
texts := []string{
    "First document to embed.",
    "Second document to embed.",
    "Third document to embed.",
}

vectors, err := client.EmbedBatch(
    context.Background(),
    texts,
)
if err != nil {
    log.Fatal(err)
}

for i, vec := range vectors {
    fmt.Printf("Text %d: %d dimensions\n",
        i, len(vec))
}
```

The method returns a `[][]float64` slice where each element
corresponds to the input text at the same index.

## Batch Behavior by Provider

The OpenAI provider sends all texts in a single API request
and sorts the results by index to maintain order.

The Gemini and Ollama providers do not have native batch
embedding endpoints. The `EmbedBatch` method makes sequential
calls to the single-embed endpoint for each text in the input
slice.

## Embedding Models

You should use an embedding-specific model when generating
embeddings. The following table describes common embedding
models for each provider:

| Provider | Model                    |
|----------|--------------------------|
| OpenAI   | text-embedding-3-small   |
| OpenAI   | text-embedding-3-large   |
| Gemini   | text-embedding-004       |
| Ollama   | nomic-embed-text         |

## Error Handling

Embedding calls return a `ProviderError` when the provider
returns an error response. You can check for specific error
types using `errors.Is`:

```go
vector, err := client.Embed(ctx, text)
if err != nil {
    if errors.Is(err, llm.ErrNotSupported) {
        log.Println("Provider does not support embeddings")
    } else {
        log.Fatal(err)
    }
}
```

## Next Steps

The following documents provide additional information:

- The [Providers](providers.md) document describes
  provider-specific configuration options.
- The [Error Handling](error_handling.md) document covers
  error types and handling patterns.
