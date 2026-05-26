# Voyage AI Provider — Design

**Status:** approved (brainstorming)
**Date:** 2026-05-26
**Author:** Dave Page (with Claude)

## Summary

Add Voyage AI as a first-class provider in `pgedge-go-llm-lib`. Voyage is an
embeddings-and-rerankers provider — it does not offer chat completions — so
this change extends the unified `llm.Client` interface with two new
capabilities (reranking and multimodal embeddings), generalises `ListModels`
with an optional capability filter, and wires Voyage in as the inaugural
provider for the new methods. Chat-first providers (Anthropic, OpenAI,
Gemini, Ollama) gain `ErrNotSupported` stubs for the new methods, mirroring
the way Anthropic returns `ErrNotSupported` for `Embed` today.

The motivation is coverage: the library aims to be a comprehensive interface
over the major LLM-adjacent providers, and Voyage is a notable gap.

## Goals

- Voyage as a registered provider under the name `"voyage"`, importable
  individually or via `llm/all`.
- Full text-embeddings support (single and batch) against Voyage's models.
- Multimodal-embeddings support against `voyage-multimodal-3`.
- Reranking support against Voyage's rerank models.
- A unified `Client` interface that surfaces reranking and multimodal
  embeddings on every provider (with `ErrNotSupported` where unsupported).
- An optional capability filter on `ListModels` / `ListModelsWithMetadata`
  so callers can discover, e.g., "models that support reranking" in a
  provider-agnostic way.
- Proxy HTTP routes for embed, multimodal embed, and rerank.
- Documentation, unit tests, and an opt-in integration test against the live
  Voyage API gated by `VOYAGE_API_KEY`.

## Non-goals

- No signature change to existing `Embed` / `EmbedBatch` methods. They stay
  text-only with no per-call options. Voyage's per-call knobs (`input_type`,
  `output_dimension`, `output_dtype`, `truncation`) reach text embeddings
  only via `EmbedMultimodal` with a single text input, or via a follow-up
  spec.
- No streaming rerank (Voyage's API doesn't stream it).
- No batch rerank across multiple queries — one query, N documents per call.
- No reranking-as-a-tool or fused chat+rerank flows.
- No support for Voyage's `contextualized-embeddings` beta endpoint.
- No structured-output / function-calling capabilities on Voyage (it doesn't
  offer them; the relevant capability strings are simply absent from
  Voyage models' capability lists).
- No `pgvector` integration helpers — out of scope for this library.
- CI secret provisioning (`VOYAGE_API_KEY` in GitHub Actions) is a manual
  prerequisite, not part of this code change.

## Interface changes

### New types in `llm/types.go`

Multimodal embedding inputs:

```go
type MultimodalContentType string

const (
    MultimodalContentText      MultimodalContentType = "text"
    MultimodalContentImageURL  MultimodalContentType = "image_url"
    MultimodalContentImageData MultimodalContentType = "image_base64"
)

type MultimodalContent struct {
    Type      MultimodalContentType
    Text      string
    ImageURL  string
    ImageData []byte
    MIMEType  string
}

type MultimodalInput struct {
    Content []MultimodalContent
}

type MultimodalEmbedRequest struct {
    Inputs     []MultimodalInput
    Extensions []ProviderExtension
}
```

Reranking:

```go
type RerankRequest struct {
    Query      string
    Documents  []string
    TopK       *int
    Extensions []ProviderExtension
}

type RerankResult struct {
    Index          int
    RelevanceScore float64
    Document       string // populated iff the provider returned documents
}

type RerankResponse struct {
    Results []RerankResult
    Usage   TokenUsage
}
```

Capability constants — the typed `ModelCapability` enum already exists in
`llm/types.go`. Two new values are added to the existing const block:

```go
const (
    // existing
    ModelCapabilityChat       ModelCapability = "chat"
    ModelCapabilityTools      ModelCapability = "tools"
    ModelCapabilityVision     ModelCapability = "vision"
    ModelCapabilityEmbeddings ModelCapability = "embeddings"
    ModelCapabilityJSONMode   ModelCapability = "json_mode"
    ModelCapabilityStreaming  ModelCapability = "streaming"
    // NEW
    ModelCapabilityMultimodalEmbeddings ModelCapability = "multimodal_embeddings"
    ModelCapabilityReranking            ModelCapability = "reranking"
)
```

`ListModels` configuration uses the same typed enum:

```go
type ListModelsConfig struct {
    Capabilities []ModelCapability
}

type ListModelsOption func(*ListModelsConfig)

func WithCapabilities(caps ...ModelCapability) ListModelsOption {
    return func(c *ListModelsConfig) {
        c.Capabilities = append(c.Capabilities, caps...)
    }
}
```

`ModelCapability` is stringly-backed, so providers may still advertise
capability values the library doesn't yet have a constant for — forward
compatibility is preserved.

### Updated `llm.Client` interface

Two new methods and two variadic signatures:

```go
type Client interface {
    Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
    ChatStream(ctx context.Context, req ChatRequest) (*Stream, error)

    Embed(ctx context.Context, text string) ([]float64, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)

    // NEW
    EmbedMultimodal(ctx context.Context, req MultimodalEmbedRequest) ([][]float64, error)
    // NEW
    Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error)

    // SIGNATURE CHANGE — was (ctx) ([]string, error)
    ListModels(ctx context.Context, opts ...ListModelsOption) ([]string, error)
    // SIGNATURE CHANGE — was (ctx) ([]ModelInfo, error)
    ListModelsWithMetadata(ctx context.Context, opts ...ListModelsOption) ([]ModelInfo, error)

    Ping(ctx context.Context) error
    Provider() string
    Model() string
    Usage() TokenUsage
    ResetUsage()
}
```

The variadic change is **source-compatible** for callers: existing call
sites that pass no options continue to compile unchanged and observe
identical behaviour. It is, however, a **breaking change for any external
code that implements `Client`** (custom providers, hand-rolled test
doubles). This is an accepted trade-off — the library is pre-1.0, and
adding `Rerank` / `EmbedMultimodal` is already a breaking interface
change in the same release.

### Capability-filter semantics

- **No options** → providers return their current list (chat-first
  providers return chat models; Voyage returns embedding + rerank models).
  The old docstring "chat-capable models only" is replaced with "models
  the provider considers user-facing".
- **`WithCapabilities(a, b)`** → AND: only models whose
  `ModelInfo.Capabilities` contain *all* listed capabilities.
- **Unknown capability string** → empty result, not an error.
- Filtering is implemented once as a helper in `llm/llm.go`
  (`func FilterModelInfos([]ModelInfo, ListModelsConfig) []ModelInfo`).
  Providers fetch their full catalogue and call the helper; only Voyage
  needs to know about the new capability constants.

## Voyage provider package

### Layout

```
llm/provider/voyage/
├── voyage.go        # client, ProviderConstructor, init() registration
├── voyage_test.go   # unit tests with httptest
└── extension.go     # voyage.Extension and per-call option constants
```

Registers under `"voyage"` via `llm.RegisterProvider("voyage", New)` in
`init()`.

### Auth & transport

- API key from `Options.APIKey`, falling back to `VOYAGE_API_KEY`.
- `Authorization: Bearer <key>` header.
- Default base URL `https://api.voyageai.com/v1/`, overridable via
  `Options.BaseURL` for tests and self-hosted proxies.
- HTTP client uses `llm/internal/httpclient`, inheriting retry/backoff
  on 429 / 5xx.

### Endpoints

| Method | Path                       | Maps to                       |
|--------|----------------------------|-------------------------------|
| POST   | `/v1/embeddings`           | `Embed`, `EmbedBatch`         |
| POST   | `/v1/multimodalembeddings` | `EmbedMultimodal`             |
| POST   | `/v1/rerank`               | `Rerank`                      |

`Chat` and `ChatStream` return
`*ProviderError{Err: ErrNotSupported, Message: "Voyage does not support chat completions"}`.

### `Ping` strategy

Voyage has no dedicated ping endpoint. The implementer chooses during
implementation between:

1. A one-token `POST /v1/embeddings` against the configured model.
2. `HEAD /v1/embeddings` (likely 405, treated as reachable).
3. Returning `ErrNotSupported` from `Ping`.

The choice is recorded in the implementation plan once verified against
the live API.

### Extension type

```go
package voyage

type InputType string

const (
    InputTypeQuery    InputType = "query"
    InputTypeDocument InputType = "document"
)

type OutputDtype string

const (
    OutputDtypeFloat   OutputDtype = "float"
    OutputDtypeInt8    OutputDtype = "int8"
    OutputDtypeUint8   OutputDtype = "uint8"
    OutputDtypeBinary  OutputDtype = "binary"
    OutputDtypeUbinary OutputDtype = "ubinary"
)

type Extension struct {
    InputType       InputType
    OutputDimension int   // 256 / 512 / 1024 / 2048; model-dependent
    Truncation      *bool // pointer so unset != false
    OutputDtype     OutputDtype
    ReturnDocuments *bool // rerank only; pointer so unset != false
}

func (Extension) ProviderName() string { return "voyage" }
```

Attached to `MultimodalEmbedRequest.Extensions` or
`RerankRequest.Extensions`. Other providers ignore it per the existing
forward-compatibility contract.

### Model catalogue

Voyage exposes no `/models` endpoint, so the list is hard-coded
(same approach as Anthropic):

| Model                                           | Capabilities                                |
|-------------------------------------------------|---------------------------------------------|
| `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large`    | `embeddings`                              |
| `voyage-code-3`, `voyage-finance-2`, `voyage-law-2`  | `embeddings`                              |
| `voyage-multimodal-3`                                | `embeddings`, `multimodal_embeddings`     |
| `rerank-2.5`, `rerank-2.5-lite`                      | `reranking`                               |

(Capability values shown here are the underlying `ModelCapability`
string values; in code they're `ModelCapabilityEmbeddings`,
`ModelCapabilityMultimodalEmbeddings`, `ModelCapabilityReranking`.)

`ListModels(ctx, WithCapabilities(ModelCapabilityReranking))` returns
just the two rerank models.

### Token tracking

Voyage's response envelope includes `usage.total_tokens`. The provider
maps that to `TokenUsage.TotalTokens`; `PromptTokens` /
`CompletionTokens` stay zero (not meaningful for embeddings/rerank).

## Existing provider updates

### Stubs

Each of Anthropic, OpenAI, Gemini, and Ollama gains identical stubs:

```go
func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "<Provider> does not support multimodal embeddings",
        Provider: "<provider>",
    }
}

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "<Provider> does not support reranking",
        Provider: "<provider>",
    }
}
```

### `ListModels` signature update

Every provider's existing `ListModels` / `ListModelsWithMetadata` becomes
variadic. With zero options, behaviour is byte-identical to today.
Providers call the shared `llm.FilterModelInfos` helper.

### Capability data updates

| Provider  | Models gaining capability                                                                            |
|-----------|------------------------------------------------------------------------------------------------------|
| OpenAI    | `text-embedding-3-*`, `text-embedding-ada-002` → `embeddings`                                        |
| Gemini    | `text-embedding-004`, `embedding-001` → `embeddings`                                                 |
| Ollama    | Models returning embeddings → `embeddings`. Use `/api/show` `details.capabilities` where available; otherwise infer from model family name. |
| Anthropic | No change (chat capabilities only).                                                                  |

### Test doubles

`llm/proxy/fake_provider_test.go` gains `Rerank` and `EmbedMultimodal`
stubs returning the configured error so the proxy tests continue to
compile and exercise the new endpoints.

## Proxy HTTP surface

Four new routes wired in `proxy.go` alongside the existing chat handlers,
with handlers in `handlers.go` and request/response types in `types.go`:

| Method | Path                                          | Maps to                                                                       |
|--------|-----------------------------------------------|-------------------------------------------------------------------------------|
| POST   | `/v1/embed`                                   | `Embed` / `EmbedBatch` (one or many inputs in the JSON body)                  |
| POST   | `/v1/embed/multimodal`                        | `EmbedMultimodal`                                                             |
| POST   | `/v1/rerank`                                  | `Rerank`                                                                      |
| GET    | `/v1/models?provider=X&capability=embeddings` | `ListModels` / `ListModelsWithMetadata` with the capability filter (repeatable) |

`ErrNotSupported` from the underlying client maps to HTTP 501. Auth and
hook semantics match the existing endpoints. `/v1/health` is unchanged —
`Ping` covers connectivity regardless of which capabilities a provider
supports.

## `llm/all` registration

```go
import (
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/gemini"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)
```

## Documentation

- `README.md`: Supported-Providers table gains a `Voyage` column and two
  new rows (`Reranking`, `Multimodal Embeddings`); the proxy endpoint
  table gains the four new routes; the package overview lists Voyage.
- `docs/providers.md`: new Voyage section covering auth, models,
  capabilities, and the `voyage.Extension` type.
- `docs/embeddings.md`: new "Multimodal Embeddings" subsection with a
  `voyage-multimodal-3` example.
- `docs/reranking.md`: new file covering `RerankRequest` /
  `RerankResponse`, capability discovery, and Voyage examples.
- `mkdocs.yml`: add `reranking.md` to navigation.
- `docs/api_reference.md`: new types and methods.
- Package godoc (`llm/llm.go`): update to list five providers and the
  reranking / multimodal-embeddings capabilities.

## Tests

- `llm/provider/voyage/voyage_test.go` — unit tests using `httptest`
  + `Options.BaseURL` override. Coverage: text embed (single, batch);
  multimodal embed (text-only, image-URL, image-data); rerank (with and
  without returned documents); `Chat`/`ChatStream` `ErrNotSupported`;
  `ListModels` capability filtering; 401 / 429 / 5xx error mapping;
  `voyage.Extension` round-trip into the request body.
- `llm/integration_test.go` — Voyage block guarded by `VOYAGE_API_KEY`,
  performing a small embed + rerank against the live API. Skipped when
  the key is absent.
- `llm/proxy/proxy_test.go` — coverage for the four new routes: happy
  path, `ErrNotSupported` → 501, malformed JSON → 400, auth-hook
  rejection.
- `llm/llm_test.go` — small test for `FilterModelInfos` /
  `WithCapabilities` against synthetic `ModelInfo` slices.
- Each existing provider's `*_test.go` — two short tests asserting
  `EmbedMultimodal` and `Rerank` return `ErrNotSupported`, plus a
  `ListModels(ctx, WithCapabilities(ModelCapabilityChat))` assertion
  that the existing list is returned and
  `WithCapabilities(ModelCapabilityReranking)` returns empty.

## CI / operational prerequisites

- Add `VOYAGE_API_KEY` to GitHub Actions repository secrets so the
  optional integration tests run on PRs from trusted contexts. This is
  a manual step performed by a maintainer; the implementation plan lists
  it as a prerequisite rather than a code task.
- No changes to existing CI workflow files are required — the
  integration-test guard pattern already in place handles absence of
  the secret gracefully.

## Risk & rollout notes

- This release breaks the `llm.Client` interface for any external
  implementer (custom providers, hand-rolled mocks). The project is
  pre-1.0, so this is acceptable, but the changelog and release notes
  must call it out explicitly.
- All in-repo implementations (four providers, fake proxy provider) are
  updated in the same PR, so the internal build stays green.
- The `ListModels` signature change is source-compatible for callers
  (variadic), so library consumers who don't implement `Client`
  themselves are unaffected.
