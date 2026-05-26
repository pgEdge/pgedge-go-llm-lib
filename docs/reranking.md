# Reranking

`Client.Rerank` reorders a slice of documents by relevance to a query.
Today only Voyage implements this; other providers return
`ErrNotSupported`.

```go
import "github.com/pgEdge/pgedge-go-llm-lib/llm"

client, _ := llm.NewClient("voyage", llm.Options{
    APIKey: os.Getenv("VOYAGE_API_KEY"),
    Model:  "rerank-2.5",
})

resp, err := client.Rerank(ctx, llm.RerankRequest{
    Query: "what is a kitten",
    Documents: []string{
        "The Eiffel Tower is in Paris.",
        "A kitten is a juvenile cat.",
        "Cats are small mammals.",
    },
})
```

The response's `Results` slice is ordered by descending
`RelevanceScore`. Each result's `Index` is the position in the original
`Documents` slice.

## Top-K

Pass `RerankRequest.TopK` to ask the provider to return only the top K
most-relevant documents:

```go
k := 3
resp, err := client.Rerank(ctx, llm.RerankRequest{
    Query: q, Documents: docs, TopK: &k,
})
```

## Returning documents inline

By default, `RerankResult.Document` is empty (the response only carries
indexes). For Voyage, set `ReturnDocuments` via `voyage.Extension`:

```go
import "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"

tru := true
resp, err := client.Rerank(ctx, llm.RerankRequest{
    Query: q, Documents: docs,
    Extensions: []llm.ProviderExtension{voyage.Extension{
        ReturnDocuments: &tru,
    }},
})
```

## Discovering rerank-capable models

```go
infos, err := client.ListModelsWithMetadata(ctx,
    llm.WithCapabilities(llm.ModelCapabilityReranking))
```

The library's `Client.Rerank` returns `ErrNotSupported` for any provider
that doesn't support reranking, so always check the error.
