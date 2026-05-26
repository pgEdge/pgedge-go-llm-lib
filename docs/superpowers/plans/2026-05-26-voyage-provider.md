# Voyage AI Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Voyage AI as a first-class provider in `pgedge-go-llm-lib`, supporting text embeddings, multimodal embeddings, and reranking. Extends the unified `llm.Client` interface accordingly.

**Architecture:** Add new request/response types and capability constants to `llm/types.go`. Extend `llm.Client` with `Rerank` and `EmbedMultimodal`, and change `ListModels`/`ListModelsWithMetadata` to variadic-options form. Implement Voyage in `llm/provider/voyage/` against `https://api.voyageai.com/v1/`. Existing chat-first providers (Anthropic, OpenAI, Gemini, Ollama) gain `ErrNotSupported` stubs for the new methods, mirroring Anthropic's existing `Embed` stub. Proxy gains routes for embeddings, multimodal embeddings, and reranking.

**Tech Stack:** Go 1.22+, `net/http`, `httptest`, existing `llm/internal/httpclient`. Voyage REST API. No new external dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-26-voyage-provider-design.md`

---

## Conventions

- Every Go file in this repo begins with the standard header block (see existing files). New files must include it.
- Run `go build ./...` and `go test ./...` from the repo root.
- Commit after every task. Use conventional-commit style: `feat:`, `refactor:`, `test:`, `docs:`. Multi-line commit bodies are encouraged.
- For provider stub tasks that touch all four existing providers, treat it as one atomic refactor: edit all files, run `go build ./...` once at the end, then commit.
- Voyage unit tests use `httptest.Server` with `Options.BaseURL` override, mirroring `llm/provider/openai/openai_test.go`. Look there for the pattern if unsure.
- The shared header block looks like this (use it verbatim on every new `.go` file):

```go
//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------
```

---

## Phase 1 — New types and helpers

The first four tasks add new type definitions and helpers without changing any existing interface signature. The build stays green throughout.

---

### Task 1: Add `ModelCapability` constants for reranking and multimodal embeddings

**Files:**
- Modify: `llm/types.go` (around line 497, the existing `const` block of `ModelCapability` values)

- [ ] **Step 1: Add two constants to the existing ModelCapability const block**

In `llm/types.go`, find the existing block:

```go
const (
    ModelCapabilityChat       ModelCapability = "chat"
    ModelCapabilityTools      ModelCapability = "tools"
    ModelCapabilityVision     ModelCapability = "vision"
    ModelCapabilityEmbeddings ModelCapability = "embeddings"
    ModelCapabilityJSONMode   ModelCapability = "json_mode"
    ModelCapabilityStreaming  ModelCapability = "streaming"
)
```

Replace with:

```go
const (
    ModelCapabilityChat                 ModelCapability = "chat"
    ModelCapabilityTools                ModelCapability = "tools"
    ModelCapabilityVision               ModelCapability = "vision"
    ModelCapabilityEmbeddings           ModelCapability = "embeddings"
    ModelCapabilityJSONMode             ModelCapability = "json_mode"
    ModelCapabilityStreaming            ModelCapability = "streaming"
    ModelCapabilityMultimodalEmbeddings ModelCapability = "multimodal_embeddings"
    ModelCapabilityReranking            ModelCapability = "reranking"
)
```

- [ ] **Step 2: Verify the build still passes**

Run: `go build ./...`
Expected: exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add llm/types.go
git commit -m "feat: add Reranking and MultimodalEmbeddings model capabilities"
```

---

### Task 2: Add multimodal-embedding request types

**Files:**
- Modify: `llm/types.go` (append near the bottom, after the existing `ModelCapability` block)

- [ ] **Step 1: Add types to `llm/types.go`**

Append at the bottom of `llm/types.go`:

```go
// MultimodalContentType identifies the kind of content in a
// MultimodalContent value. The discriminator selects which of the
// Text / ImageURL / ImageData fields is read; other fields are
// ignored for that content item.
type MultimodalContentType string

const (
    // MultimodalContentText is a UTF-8 text fragment in MultimodalContent.Text.
    MultimodalContentText MultimodalContentType = "text"
    // MultimodalContentImageURL is a remote image fetched from MultimodalContent.ImageURL.
    MultimodalContentImageURL MultimodalContentType = "image_url"
    // MultimodalContentImageData is an inline image in MultimodalContent.ImageData with MIME type in MultimodalContent.MIMEType.
    MultimodalContentImageData MultimodalContentType = "image_base64"
)

// MultimodalContent is a single piece of content in a multimodal
// embedding input. The Type field selects which of Text / ImageURL /
// ImageData is read.
type MultimodalContent struct {
    Type      MultimodalContentType
    Text      string
    ImageURL  string
    ImageData []byte
    MIMEType  string
}

// MultimodalInput is one input to EmbedMultimodal. Each input
// produces exactly one embedding vector; the order in
// MultimodalEmbedRequest.Inputs is preserved in the returned slice.
type MultimodalInput struct {
    Content []MultimodalContent
}

// MultimodalEmbedRequest is the request body for Client.EmbedMultimodal.
// Providers that do not support multimodal embeddings return ErrNotSupported.
type MultimodalEmbedRequest struct {
    Inputs     []MultimodalInput
    Extensions []ProviderExtension
}
```

- [ ] **Step 2: Verify the build still passes**

Run: `go build ./...`
Expected: exit code 0.

- [ ] **Step 3: Commit**

```bash
git add llm/types.go
git commit -m "feat: add multimodal embedding request types"
```

---

### Task 3: Add reranking request and response types

**Files:**
- Modify: `llm/types.go` (append after the multimodal types added in Task 2)

- [ ] **Step 1: Add types to `llm/types.go`**

Append at the bottom of `llm/types.go`:

```go
// RerankRequest is the request body for Client.Rerank. TopK, when
// non-nil, asks the provider to return at most the top-K most-relevant
// documents. Providers that do not support reranking return
// ErrNotSupported.
type RerankRequest struct {
    Query      string
    Documents  []string
    TopK       *int
    Extensions []ProviderExtension
}

// RerankResult is one row of a rerank response. Index is the position
// in the original RerankRequest.Documents slice. RelevanceScore is the
// provider's relevance value (typically [0,1] but not strictly bounded).
// Document is non-empty only when the provider returns documents in
// its response (e.g. when ReturnDocuments was requested via a provider
// extension).
type RerankResult struct {
    Index          int
    RelevanceScore float64
    Document       string
}

// RerankResponse is the body returned by Client.Rerank. Results are
// ordered by descending RelevanceScore. Usage carries token accounting
// where the provider reports it; PromptTokens / CompletionTokens are
// usually zero for rerank.
type RerankResponse struct {
    Results []RerankResult
    Usage   TokenUsage
}
```

- [ ] **Step 2: Verify the build still passes**

Run: `go build ./...`
Expected: exit code 0.

- [ ] **Step 3: Commit**

```bash
git add llm/types.go
git commit -m "feat: add reranking request and response types"
```

---

### Task 4: Add `ListModelsConfig`, `WithCapabilities`, and `FilterModelInfos` helper with tests

**Files:**
- Modify: `llm/types.go` (append ListModelsConfig / ListModelsOption / WithCapabilities)
- Modify: `llm/llm.go` (add `FilterModelInfos` helper)
- Modify: `llm/llm_test.go` (add tests)

- [ ] **Step 1: Write the failing tests in `llm/llm_test.go`**

`llm/llm_test.go` is in `package llm` (internal test), so references the package's exported symbols without any prefix. Append at the bottom:

```go
func TestWithCapabilitiesAccumulates(t *testing.T) {
    cfg := ListModelsConfig{}
    WithCapabilities(ModelCapabilityChat)(&cfg)
    WithCapabilities(ModelCapabilityReranking, ModelCapabilityEmbeddings)(&cfg)
    want := []ModelCapability{
        ModelCapabilityChat,
        ModelCapabilityReranking,
        ModelCapabilityEmbeddings,
    }
    if !reflect.DeepEqual(cfg.Capabilities, want) {
        t.Fatalf("got %v, want %v", cfg.Capabilities, want)
    }
}

func TestFilterModelInfosNoOptionsReturnsAll(t *testing.T) {
    infos := []ModelInfo{
        {ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat}},
        {ID: "b", Capabilities: []ModelCapability{ModelCapabilityEmbeddings}},
    }
    got := FilterModelInfos(infos, ListModelsConfig{})
    if !reflect.DeepEqual(got, infos) {
        t.Fatalf("expected all infos returned unchanged, got %v", got)
    }
}

func TestFilterModelInfosCapabilityAND(t *testing.T) {
    infos := []ModelInfo{
        {ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat, ModelCapabilityTools}},
        {ID: "b", Capabilities: []ModelCapability{ModelCapabilityChat}},
        {ID: "c", Capabilities: []ModelCapability{ModelCapabilityReranking}},
    }
    cfg := ListModelsConfig{Capabilities: []ModelCapability{
        ModelCapabilityChat, ModelCapabilityTools,
    }}
    got := FilterModelInfos(infos, cfg)
    if len(got) != 1 || got[0].ID != "a" {
        t.Fatalf("expected only 'a', got %v", got)
    }
}

func TestFilterModelInfosUnknownCapabilityIsEmpty(t *testing.T) {
    infos := []ModelInfo{
        {ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat}},
    }
    cfg := ListModelsConfig{Capabilities: []ModelCapability{"never-heard-of-it"}}
    got := FilterModelInfos(infos, cfg)
    if len(got) != 0 {
        t.Fatalf("expected empty result, got %v", got)
    }
}
```

Add `"reflect"` to the import block (the current imports are `"errors"`, `"testing"`).

- [ ] **Step 2: Run tests, verify they fail with undefined symbols**

Run: `go test ./llm/ -run TestWithCapabilities -run TestFilterModelInfos`
Expected: compile error referencing `ListModelsConfig`, `WithCapabilities`, `FilterModelInfos`.

- [ ] **Step 3: Add `ListModelsConfig` and `WithCapabilities` to `llm/types.go`**

Append at the bottom of `llm/types.go`:

```go
// ListModelsConfig is the configuration accumulated from ListModelsOption
// values passed to Client.ListModels and Client.ListModelsWithMetadata.
// Callers don't construct this directly; use options like WithCapabilities.
type ListModelsConfig struct {
    // Capabilities, when non-empty, restricts results to models whose
    // ModelInfo.Capabilities contains EVERY listed capability. An
    // empty Capabilities slice means "no filter".
    Capabilities []ModelCapability
}

// ListModelsOption configures a single ListModels call. Pass values
// returned by WithCapabilities (and future option constructors) as
// the variadic argument to Client.ListModels.
type ListModelsOption func(*ListModelsConfig)

// WithCapabilities filters ListModels to models whose Capabilities
// contain every listed value. Calls accumulate: passing two
// WithCapabilities options is equivalent to one call with all
// capabilities concatenated.
func WithCapabilities(caps ...ModelCapability) ListModelsOption {
    return func(c *ListModelsConfig) {
        c.Capabilities = append(c.Capabilities, caps...)
    }
}
```

- [ ] **Step 4: Add `FilterModelInfos` helper to `llm/llm.go`**

Append at the bottom of `llm/llm.go`:

```go
// FilterModelInfos applies a ListModelsConfig to a slice of ModelInfo.
// It is a building block for provider implementations of ListModels /
// ListModelsWithMetadata: providers fetch their full catalogue, then
// call FilterModelInfos to apply caller-supplied options.
//
// Filtering is AND-of-capabilities: a model is kept only if its
// Capabilities slice contains every value in cfg.Capabilities.
// An empty cfg.Capabilities slice keeps every input.
func FilterModelInfos(infos []ModelInfo, cfg ListModelsConfig) []ModelInfo {
    if len(cfg.Capabilities) == 0 {
        return infos
    }
    out := make([]ModelInfo, 0, len(infos))
    for _, info := range infos {
        if hasAllCapabilities(info.Capabilities, cfg.Capabilities) {
            out = append(out, info)
        }
    }
    return out
}

func hasAllCapabilities(have, want []ModelCapability) bool {
    for _, w := range want {
        found := false
        for _, h := range have {
            if h == w {
                found = true
                break
            }
        }
        if !found {
            return false
        }
    }
    return true
}
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `go test ./llm/ -run TestWithCapabilities -run TestFilterModelInfos -v`
Expected: all four tests PASS.

- [ ] **Step 6: Commit**

```bash
git add llm/types.go llm/llm.go llm/llm_test.go
git commit -m "feat: add ListModels capability filter helpers"
```

---

## Phase 2 — Interface signature changes (atomic refactors)

These three tasks each change the `Client` interface and update every implementer in the same task. The build is broken between edits inside a task but green at task completion.

---

### Task 5: Make `ListModels` and `ListModelsWithMetadata` variadic across the interface and all providers

**Files:**
- Modify: `llm/llm.go` (Client interface, lines ~59–68)
- Modify: `llm/provider/anthropic/anthropic.go` (lines 639, 675)
- Modify: `llm/provider/openai/openai.go` (lines 789, 831)
- Modify: `llm/provider/gemini/gemini.go` (lines 699, 742)
- Modify: `llm/provider/ollama/ollama.go` (lines 655, 675)
- Modify: `llm/proxy/fake_provider_test.go` (the two ListModels methods on `fakeProvider`)

- [ ] **Step 1: Update the `Client` interface in `llm/llm.go`**

Find:

```go
ListModels(ctx context.Context) ([]string, error)
```

Replace with:

```go
ListModels(ctx context.Context, opts ...ListModelsOption) ([]string, error)
```

Find:

```go
ListModelsWithMetadata(ctx context.Context) ([]ModelInfo, error)
```

Replace with:

```go
ListModelsWithMetadata(ctx context.Context, opts ...ListModelsOption) ([]ModelInfo, error)
```

Also update the docstring on `ListModels` — change "chat-capable" to "user-facing":

Find:

```go
// ListModels returns the names of chat-capable models available
// from the provider. Each provider filters its list to relevant
// models only.
```

Replace with:

```go
// ListModels returns the names of user-facing models available from
// the provider. With no options, providers return their default
// catalogue (typically chat models for chat-first providers, and
// embedding/rerank models for embedding-first providers). Pass
// WithCapabilities to filter by ModelCapability.
```

- [ ] **Step 2: Update Anthropic's two methods**

In `llm/provider/anthropic/anthropic.go`, change the signatures and apply the filter helper.

Find:

```go
func (c *client) ListModels(ctx context.Context) ([]string, error) {
```

Replace with:

```go
func (c *client) ListModels(ctx context.Context, opts ...llm.ListModelsOption) ([]string, error) {
```

At the top of the function body, inside this method, add (before any existing code that builds the string slice):

```go
infos, err := c.ListModelsWithMetadata(ctx, opts...)
if err != nil {
    return nil, err
}
names := make([]string, len(infos))
for i, info := range infos {
    names[i] = info.ID
}
return names, nil
```

Replace the entire body of the existing `ListModels` with the four lines above so it now delegates to `ListModelsWithMetadata`.

Find:

```go
func (c *client) ListModelsWithMetadata(ctx context.Context) ([]llm.ModelInfo, error) {
```

Replace with:

```go
func (c *client) ListModelsWithMetadata(ctx context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
```

At the end of `ListModelsWithMetadata` — just before the function returns the populated `[]llm.ModelInfo` — apply the filter. Find the `return` statement that hands back the populated slice (e.g. `return infos, nil`) and replace it with:

```go
cfg := llm.ListModelsConfig{}
for _, opt := range opts {
    opt(&cfg)
}
return llm.FilterModelInfos(infos, cfg), nil
```

(If the existing variable is not named `infos`, substitute the actual name.)

- [ ] **Step 3: Update OpenAI's two methods**

In `llm/provider/openai/openai.go`, apply exactly the same transformation as in Step 2:
- Change `ListModels` signature to `(ctx context.Context, opts ...llm.ListModelsOption)`, body delegates to `ListModelsWithMetadata` and projects to names.
- Change `ListModelsWithMetadata` signature to `(ctx context.Context, opts ...llm.ListModelsOption)`, apply `FilterModelInfos` at the return.

The delegation body for `ListModels`:

```go
infos, err := c.ListModelsWithMetadata(ctx, opts...)
if err != nil {
    return nil, err
}
names := make([]string, len(infos))
for i, info := range infos {
    names[i] = info.ID
}
return names, nil
```

The filter at the end of `ListModelsWithMetadata`:

```go
cfg := llm.ListModelsConfig{}
for _, opt := range opts {
    opt(&cfg)
}
return llm.FilterModelInfos(infos, cfg), nil
```

(Substitute the actual local-variable name if it isn't `infos`.)

- [ ] **Step 4: Update Gemini's two methods**

Same transformation in `llm/provider/gemini/gemini.go`. Same delegation body and same filter block.

- [ ] **Step 5: Update Ollama's two methods**

Same transformation in `llm/provider/ollama/ollama.go`. Same delegation body and same filter block.

- [ ] **Step 6: Update `fakeProvider` in `llm/proxy/fake_provider_test.go`**

Find the two methods on `fakeProvider`:

```go
func (f *fakeProvider) ListModels(ctx context.Context) ([]string, error) { ... }
func (f *fakeProvider) ListModelsWithMetadata(ctx context.Context) ([]llm.ModelInfo, error) { ... }
```

Update both signatures to accept `, opts ...llm.ListModelsOption`. Inside each method, ignore the new `opts` argument (the fake doesn't need to filter — tests don't depend on filtering through the fake). Function bodies otherwise unchanged.

- [ ] **Step 7: Build and test**

Run: `go build ./...`
Expected: exit code 0.

Run: `go test ./...`
Expected: all existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add llm/llm.go llm/provider/ llm/proxy/fake_provider_test.go
git commit -m "refactor: make ListModels and ListModelsWithMetadata variadic

Adds an opts ...ListModelsOption parameter to both methods on the
Client interface, with each provider's implementation delegating
filtering to the shared llm.FilterModelInfos helper. Source-compatible
for callers; interface change for external implementers."
```

---

### Task 6: Add `Rerank` to the `Client` interface and stub it on every provider

**Files:**
- Modify: `llm/llm.go` (Client interface)
- Modify: `llm/provider/anthropic/anthropic.go` (new method)
- Modify: `llm/provider/openai/openai.go` (new method)
- Modify: `llm/provider/gemini/gemini.go` (new method)
- Modify: `llm/provider/ollama/ollama.go` (new method)
- Modify: `llm/proxy/fake_provider_test.go` (new method on `fakeProvider`)
- Modify: each provider's `_test.go` — add a one-liner asserting the stub returns `ErrNotSupported`

- [ ] **Step 1: Add `Rerank` to the `Client` interface in `llm/llm.go`**

Inside the `Client` interface, after the `EmbedBatch` method and before `ListModels`, add:

```go
// Rerank reorders a slice of documents by relevance to a query.
// Results are returned in descending RelevanceScore order. Returns
// ErrNotSupported for providers that do not support reranking
// (currently every provider except Voyage).
Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error)
```

- [ ] **Step 2: Add the stub method to Anthropic**

Append after the existing `EmbedBatch` method in `llm/provider/anthropic/anthropic.go`:

```go
// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Anthropic does not support reranking",
        Provider: "anthropic",
    }
}
```

- [ ] **Step 3: Add the stub method to OpenAI**

Append after the existing `EmbedBatch` method in `llm/provider/openai/openai.go`:

```go
// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "OpenAI does not support reranking",
        Provider: "openai",
    }
}
```

- [ ] **Step 4: Add the stub method to Gemini**

Append after the existing `EmbedBatch` method in `llm/provider/gemini/gemini.go`:

```go
// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Gemini does not support reranking",
        Provider: "gemini",
    }
}
```

- [ ] **Step 5: Add the stub method to Ollama**

Append after the existing `EmbedBatch` method in `llm/provider/ollama/ollama.go`:

```go
// ---------- Rerank ----------

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Ollama does not support reranking",
        Provider: "ollama",
    }
}
```

- [ ] **Step 6: Add the stub method to `fakeProvider` in `llm/proxy/fake_provider_test.go`**

Append on `fakeProvider`:

```go
func (f *fakeProvider) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "fake does not support reranking",
        Provider: "fake",
    }
}
```

- [ ] **Step 7: Add a `Rerank` ErrNotSupported test to each provider's test file**

In `llm/provider/anthropic/anthropic_test.go`, append:

```go
func TestRerankUnsupported(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
    defer srv.Close()
    c := newTestClient(t, srv.URL)
    _, err := c.Rerank(context.Background(), llm.RerankRequest{Query: "q", Documents: []string{"a"}})
    if !errors.Is(err, llm.ErrNotSupported) {
        t.Fatalf("expected ErrNotSupported, got %v", err)
    }
}
```

Ensure `"errors"`, `"net/http"`, and `"net/http/httptest"` are in the file's import block (most of these test files already import them).

Repeat in `openai_test.go`, `gemini_test.go`, `ollama_test.go` with the same body. All four providers use the same `newTestClient(t, url)` helper signature.

- [ ] **Step 8: Build and test**

Run: `go build ./...`
Expected: exit code 0.

Run: `go test ./...`
Expected: all tests pass, including the four new TestRerankUnsupported tests.

- [ ] **Step 9: Commit**

```bash
git add llm/llm.go llm/provider/ llm/proxy/fake_provider_test.go
git commit -m "feat: add Rerank to Client interface; stub on existing providers

Rerank returns ErrNotSupported on Anthropic, OpenAI, Gemini, and
Ollama. Voyage will implement it in a later task."
```

---

### Task 7: Add `EmbedMultimodal` to the `Client` interface and stub it on every provider

**Files:** same set as Task 6.

- [ ] **Step 1: Add `EmbedMultimodal` to the `Client` interface in `llm/llm.go`**

Inside the `Client` interface, immediately after the `Rerank` method, add:

```go
// EmbedMultimodal generates embedding vectors for multimodal inputs
// (text and/or images). Returns ErrNotSupported for providers that
// do not support multimodal embeddings.
EmbedMultimodal(ctx context.Context, req MultimodalEmbedRequest) ([][]float64, error)
```

- [ ] **Step 2: Add the stub method to Anthropic**

Append in `llm/provider/anthropic/anthropic.go`:

```go
// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Anthropic does not support multimodal embeddings",
        Provider: "anthropic",
    }
}
```

- [ ] **Step 3: Add the stub method to OpenAI**

Append in `llm/provider/openai/openai.go`:

```go
// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "OpenAI does not support multimodal embeddings",
        Provider: "openai",
    }
}
```

- [ ] **Step 4: Add the stub method to Gemini**

Append in `llm/provider/gemini/gemini.go`:

```go
// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Gemini does not support multimodal embeddings",
        Provider: "gemini",
    }
}
```

- [ ] **Step 5: Add the stub method to Ollama**

Append in `llm/provider/ollama/ollama.go`:

```go
// ---------- EmbedMultimodal ----------

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Ollama does not support multimodal embeddings",
        Provider: "ollama",
    }
}
```

- [ ] **Step 6: Add the stub method to `fakeProvider`**

Append on `fakeProvider`:

```go
func (f *fakeProvider) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "fake does not support multimodal embeddings",
        Provider: "fake",
    }
}
```

- [ ] **Step 7: Add an `EmbedMultimodal` ErrNotSupported test to each provider's test file**

In each of `anthropic_test.go`, `openai_test.go`, `gemini_test.go`, `ollama_test.go`, append:

```go
func TestEmbedMultimodalUnsupported(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
    defer srv.Close()
    c := newTestClient(t, srv.URL)
    _, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
        Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
            {Type: llm.MultimodalContentText, Text: "hi"},
        }}},
    })
    if !errors.Is(err, llm.ErrNotSupported) {
        t.Fatalf("expected ErrNotSupported, got %v", err)
    }
}
```

- [ ] **Step 8: Build and test**

Run: `go build ./...`
Expected: exit code 0.

Run: `go test ./...`
Expected: all tests pass, including the four new TestEmbedMultimodalUnsupported tests.

- [ ] **Step 9: Commit**

```bash
git add llm/llm.go llm/provider/ llm/proxy/fake_provider_test.go
git commit -m "feat: add EmbedMultimodal to Client interface; stub on existing providers

EmbedMultimodal returns ErrNotSupported on Anthropic, OpenAI, Gemini,
and Ollama. Voyage will implement it in a later task."
```

---

## Phase 3 — Capability data on existing providers

These three tasks add `ModelCapabilityEmbeddings` to the `ModelInfo` entries that providers already emit for their embedding models. No interface changes; just data updates and assertion tests.

---

### Task 8: Tag OpenAI's embedding models with `ModelCapabilityEmbeddings`

**Files:**
- Modify: `llm/provider/openai/openai.go` (`ListModelsWithMetadata` and any model-info table it consults)
- Modify: `llm/provider/openai/openai_test.go`

- [ ] **Step 1: Write a failing test in `openai_test.go`**

Look at how the existing `openai_test.go` mocks `/v1/models` (search for the test that calls `c.ListModels` or `c.ListModelsWithMetadata`). Re-use that fixture pattern. The new test:

```go
func TestListModelsCapabilityFilterEmbeddings(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"data":[
            {"id":"gpt-4o","object":"model"},
            {"id":"text-embedding-3-small","object":"model"}
        ]}`))
    }))
    defer srv.Close()
    c := newTestClient(t, srv.URL)
    infos, err := c.ListModelsWithMetadata(context.Background(),
        llm.WithCapabilities(llm.ModelCapabilityEmbeddings))
    if err != nil {
        t.Fatal(err)
    }
    if len(infos) == 0 {
        t.Fatalf("expected at least one embedding model")
    }
    for _, info := range infos {
        found := false
        for _, cap := range info.Capabilities {
            if cap == llm.ModelCapabilityEmbeddings {
                found = true
                break
            }
        }
        if !found {
            t.Fatalf("model %s missing embeddings capability", info.ID)
        }
    }
}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `go test ./llm/provider/openai/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: FAIL — the embedding capability isn't yet emitted.

- [ ] **Step 3: Update `ListModelsWithMetadata` in `openai.go`**

Inside `ListModelsWithMetadata`, when building each `llm.ModelInfo`, add `ModelCapabilityEmbeddings` to `Capabilities` for any model ID matching the embedding-model prefixes. A small helper makes this tidy:

```go
func openaiEmbeddingModel(id string) bool {
    return strings.HasPrefix(id, "text-embedding-") ||
        id == "text-embedding-ada-002"
}
```

Then, where the `ModelInfo` is constructed for each model:

```go
caps := []llm.ModelCapability{}
// existing capability detection logic continues...
if openaiEmbeddingModel(model.ID) {
    caps = append(caps, llm.ModelCapabilityEmbeddings)
} else {
    caps = append(caps, llm.ModelCapabilityChat, llm.ModelCapabilityStreaming)
    // existing per-model logic for tools/vision/json_mode/etc.
}
```

Adjust to match whatever construction pattern the existing code uses — the key requirement is that for embedding models, `Capabilities` includes `ModelCapabilityEmbeddings`, and for chat models the existing capability set is preserved unchanged.

Ensure `"strings"` is imported.

- [ ] **Step 4: Re-run the test**

Run: `go test ./llm/provider/openai/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: PASS.

- [ ] **Step 5: Run all tests in the package**

Run: `go test ./llm/provider/openai/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add llm/provider/openai/
git commit -m "feat(openai): tag embedding models with ModelCapabilityEmbeddings"
```

---

### Task 9: Tag Gemini's embedding models with `ModelCapabilityEmbeddings`

**Files:**
- Modify: `llm/provider/gemini/gemini.go`
- Modify: `llm/provider/gemini/gemini_test.go`

- [ ] **Step 1: Write a failing test in `gemini_test.go`**

Look at the existing test that calls Gemini's models endpoint (search for `ListModels` / `ListModelsWithMetadata`). Re-use that fixture. The new test:

```go
func TestListModelsCapabilityFilterEmbeddings(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"models":[
            {"name":"models/gemini-pro","supportedGenerationMethods":["generateContent"]},
            {"name":"models/text-embedding-004","supportedGenerationMethods":["embedContent"]}
        ]}`))
    }))
    defer srv.Close()
    c := newTestClient(t, srv.URL)
    infos, err := c.ListModelsWithMetadata(context.Background(),
        llm.WithCapabilities(llm.ModelCapabilityEmbeddings))
    if err != nil {
        t.Fatal(err)
    }
    if len(infos) == 0 {
        t.Fatalf("expected at least one embedding model")
    }
    for _, info := range infos {
        found := false
        for _, cap := range info.Capabilities {
            if cap == llm.ModelCapabilityEmbeddings {
                found = true
                break
            }
        }
        if !found {
            t.Fatalf("model %s missing embeddings capability", info.ID)
        }
    }
}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `go test ./llm/provider/gemini/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: FAIL.

- [ ] **Step 3: Update `ListModelsWithMetadata` in `gemini.go`**

Add (or extend) the per-model capability-construction logic to set `ModelCapabilityEmbeddings` whenever the model is an embedding model. Gemini's `/models` response includes a `supportedGenerationMethods` array; treat the presence of `"embedContent"` (or absence of `"generateContent"`, depending on existing code's approach) as the embedding marker:

```go
isEmbedding := false
for _, m := range model.SupportedGenerationMethods {
    if m == "embedContent" || m == "batchEmbedContents" {
        isEmbedding = true
        break
    }
}
if isEmbedding {
    caps = append(caps, llm.ModelCapabilityEmbeddings)
}
```

Adjust the field name to match the existing Go struct that maps the Gemini models response. The key requirement: embedding models gain `ModelCapabilityEmbeddings`.

- [ ] **Step 4: Re-run the test**

Run: `go test ./llm/provider/gemini/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: PASS.

- [ ] **Step 5: Run all tests in the package**

Run: `go test ./llm/provider/gemini/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add llm/provider/gemini/
git commit -m "feat(gemini): tag embedding models with ModelCapabilityEmbeddings"
```

---

### Task 10: Tag Ollama's embedding-capable models with `ModelCapabilityEmbeddings`

**Files:**
- Modify: `llm/provider/ollama/ollama.go`
- Modify: `llm/provider/ollama/ollama_test.go`

Ollama's `/api/show` endpoint (per-model) exposes a `capabilities` array in recent versions, e.g. `["completion", "embedding"]`. Use it where available; fall back to family-name inference where it isn't.

- [ ] **Step 1: Write a failing test in `ollama_test.go`**

Look at the existing tests that exercise Ollama's `/api/tags` and `/api/show` endpoints. The new test needs an httptest server that returns at least one tag entry, plus a `/api/show` response that includes `"capabilities":["embedding"]` for that entry:

```go
func TestListModelsCapabilityFilterEmbeddings(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/tags":
            w.Header().Set("Content-Type", "application/json")
            w.Write([]byte(`{"models":[
                {"name":"llama3:latest"},
                {"name":"nomic-embed-text:latest"}
            ]}`))
        case "/api/show":
            w.Header().Set("Content-Type", "application/json")
            // Return embedding capability only for the embed model.
            // Inspect the request body to choose; here we keep it simple
            // and return embedding for any /api/show — adjust if the
            // existing test fixtures key on body content.
            w.Write([]byte(`{"capabilities":["embedding"]}`))
        default:
            w.WriteHeader(http.StatusNotFound)
        }
    }))
    defer srv.Close()
    c := newTestClient(t, srv.URL)
    infos, err := c.ListModelsWithMetadata(context.Background(),
        llm.WithCapabilities(llm.ModelCapabilityEmbeddings))
    if err != nil {
        t.Fatal(err)
    }
    if len(infos) == 0 {
        t.Fatalf("expected at least one embedding model")
    }
    for _, info := range infos {
        found := false
        for _, cap := range info.Capabilities {
            if cap == llm.ModelCapabilityEmbeddings {
                found = true
                break
            }
        }
        if !found {
            t.Fatalf("model %s missing embeddings capability", info.ID)
        }
    }
}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `go test ./llm/provider/ollama/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: FAIL.

- [ ] **Step 3: Update Ollama capability detection**

In `ollama.go`, where each model's `ModelInfo` is constructed (likely inside `ListModelsWithMetadata` or a helper), add to the capabilities slice when the `/api/show` response says so:

```go
for _, cap := range showResp.Capabilities {
    switch cap {
    case "embedding":
        caps = append(caps, llm.ModelCapabilityEmbeddings)
    case "completion":
        // already covered by existing chat/streaming logic; skip
    case "tools":
        caps = append(caps, llm.ModelCapabilityTools)
    case "vision":
        caps = append(caps, llm.ModelCapabilityVision)
    }
}
```

If the existing struct doesn't include the `Capabilities` field on the `/api/show` response, add it: `Capabilities []string `json:"capabilities"``.

- [ ] **Step 4: Re-run the test**

Run: `go test ./llm/provider/ollama/ -run TestListModelsCapabilityFilterEmbeddings -v`
Expected: PASS.

- [ ] **Step 5: Run all tests in the package**

Run: `go test ./llm/provider/ollama/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add llm/provider/ollama/
git commit -m "feat(ollama): tag embedding-capable models from /api/show capabilities"
```

---

## Phase 4 — Voyage provider

---

### Task 11: Create Voyage package skeleton (extension type, client struct, no-chat, Ping, ListModels)

**Files:**
- Create: `llm/provider/voyage/extension.go`
- Create: `llm/provider/voyage/voyage.go`
- Create: `llm/provider/voyage/voyage_test.go`

- [ ] **Step 1: Create `llm/provider/voyage/extension.go`**

```go
//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package voyage implements the llm.Client interface for Voyage AI.
// Voyage offers text embeddings, multimodal embeddings, and rerankers;
// it does not offer chat completions, so Chat and ChatStream return
// llm.ErrNotSupported.
//
// Per-call options specific to Voyage (input_type, output_dimension,
// truncation, output_dtype, return_documents) are passed via the
// Extension type as a ProviderExtension.
package voyage

// InputType is Voyage's hint for whether an embedding text is a search
// query or a stored document. It affects retrieval quality and is the
// most-impactful Voyage-specific tuning knob.
type InputType string

const (
    InputTypeQuery    InputType = "query"
    InputTypeDocument InputType = "document"
)

// OutputDtype is the numeric encoding for embedding vector components.
type OutputDtype string

const (
    OutputDtypeFloat   OutputDtype = "float"
    OutputDtypeInt8    OutputDtype = "int8"
    OutputDtypeUint8   OutputDtype = "uint8"
    OutputDtypeBinary  OutputDtype = "binary"
    OutputDtypeUbinary OutputDtype = "ubinary"
)

// Extension is a Voyage-specific per-call extension attached to
// MultimodalEmbedRequest.Extensions or RerankRequest.Extensions.
// Providers other than Voyage ignore it.
type Extension struct {
    InputType       InputType
    OutputDimension int   // 256 / 512 / 1024 / 2048, model-dependent; 0 = provider default
    Truncation      *bool // nil = provider default
    OutputDtype     OutputDtype
    ReturnDocuments *bool // rerank only; nil = provider default
}

// ProviderName returns "voyage" so llm.FindExtension can locate this
// extension in a request's Extensions slice.
func (Extension) ProviderName() string { return "voyage" }
```

- [ ] **Step 2: Create `llm/provider/voyage/voyage.go` with the skeleton**

```go
//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package voyage

import (
    "context"
    "errors"
    "net/http"
    "os"
    "sync"

    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

const (
    defaultBaseURL = "https://api.voyageai.com/v1"
    providerName   = "voyage"
)

// New constructs a Voyage client. Registered with llm.RegisterProvider
// at package init time, so callers normally invoke
// llm.NewClient("voyage", opts).
func New(opts llm.Options) (llm.Client, error) {
    opts = opts.WithDefaults()

    if opts.APIKey == "" {
        opts.APIKey = os.Getenv("VOYAGE_API_KEY")
    }
    if opts.APIKey == "" {
        return nil, &llm.ProviderError{
            Err:      llm.ErrInvalidRequest,
            Message:  "Voyage API key is required (Options.APIKey or VOYAGE_API_KEY)",
            Provider: providerName,
        }
    }

    baseURL := opts.BaseURL
    if baseURL == "" {
        baseURL = defaultBaseURL
    } else {
        var err error
        baseURL, err = httpclient.ValidateBaseURL(baseURL, providerName)
        if err != nil {
            return nil, err
        }
    }

    retryCfg := httpclient.RetryConfig{
        MaxRetries:     opts.Retry.MaxRetries,
        InitialBackoff: opts.Retry.InitialBackoff,
        MaxBackoff:     opts.Retry.MaxBackoff,
        Disabled:       opts.Retry.Disabled,
    }
    if opts.OnRetry != nil {
        hook := opts.OnRetry
        retryCfg.OnRetry = func(e httpclient.RetryEvent) {
            hook(llm.RetryEvent{
                Attempt:    e.Attempt,
                StatusCode: e.StatusCode,
                Err:        e.Err,
                Wait:       e.Wait,
            })
        }
    }

    return &client{
        httpClient: httpclient.New(opts.HTTPClient, opts.CustomHeaders, retryCfg, opts.RequestTimeout),
        apiKey:     opts.APIKey,
        baseURL:    baseURL,
        model:      opts.Model,
        opts:       opts,
    }, nil
}

func init() {
    llm.RegisterProvider(providerName, New)
}

type client struct {
    httpClient *http.Client
    apiKey     string
    baseURL    string
    model      string
    opts       llm.Options

    usageMu sync.Mutex
    usage   llm.TokenUsage
}

func (c *client) Provider() string { return providerName }
func (c *client) Model() string    { return c.model }

func (c *client) Usage() llm.TokenUsage {
    c.usageMu.Lock()
    defer c.usageMu.Unlock()
    return c.usage
}

func (c *client) ResetUsage() {
    c.usageMu.Lock()
    defer c.usageMu.Unlock()
    c.usage = llm.TokenUsage{}
}

func (c *client) addUsage(u llm.TokenUsage) {
    c.usageMu.Lock()
    defer c.usageMu.Unlock()
    c.usage.PromptTokens += u.PromptTokens
    c.usage.CompletionTokens += u.CompletionTokens
    c.usage.TotalTokens += u.TotalTokens
}

// ---------- Chat (not supported) ----------

func (c *client) Chat(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Voyage does not support chat completions",
        Provider: providerName,
    }
}

func (c *client) ChatStream(_ context.Context, _ llm.ChatRequest) (*llm.Stream, error) {
    return nil, &llm.ProviderError{
        Err:      llm.ErrNotSupported,
        Message:  "Voyage does not support chat completions",
        Provider: providerName,
    }
}

// ---------- Ping ----------
//
// Voyage exposes no dedicated ping endpoint. We send a minimal one-token
// embeddings request to the configured model (or voyage-3.5-lite if no
// model is configured). A 2xx response is success; an auth failure
// (401 / 403) propagates as an error; any other 4xx is treated as
// reachable (the API answered, just rejected this particular request).
func (c *client) Ping(ctx context.Context) error {
    model := c.model
    if model == "" {
        model = "voyage-3.5-lite"
    }
    _, err := c.embed(ctx, []string{"."}, model, nil)
    if err == nil {
        return nil
    }
    var pe *llm.ProviderError
    if errors.As(err, &pe) {
        if pe.StatusCode == http.StatusUnauthorized || pe.StatusCode == http.StatusForbidden {
            return err
        }
        if pe.StatusCode >= 400 && pe.StatusCode < 500 {
            return nil
        }
    }
    return err
}

// ---------- ListModels ----------

func (c *client) ListModels(ctx context.Context, opts ...llm.ListModelsOption) ([]string, error) {
    infos, err := c.ListModelsWithMetadata(ctx, opts...)
    if err != nil {
        return nil, err
    }
    names := make([]string, len(infos))
    for i, info := range infos {
        names[i] = info.ID
    }
    return names, nil
}

func (c *client) ListModelsWithMetadata(_ context.Context, opts ...llm.ListModelsOption) ([]llm.ModelInfo, error) {
    cfg := llm.ListModelsConfig{}
    for _, opt := range opts {
        opt(&cfg)
    }
    return llm.FilterModelInfos(voyageCatalog(), cfg), nil
}

// voyageCatalog returns the hard-coded list of Voyage models. Voyage
// exposes no /models endpoint, so we maintain this list manually.
func voyageCatalog() []llm.ModelInfo {
    embed := []llm.ModelCapability{llm.ModelCapabilityEmbeddings}
    multimodal := []llm.ModelCapability{llm.ModelCapabilityEmbeddings, llm.ModelCapabilityMultimodalEmbeddings}
    rerank := []llm.ModelCapability{llm.ModelCapabilityReranking}
    return []llm.ModelInfo{
        {ID: "voyage-3.5", Capabilities: embed, ContextWindow: 32000},
        {ID: "voyage-3.5-lite", Capabilities: embed, ContextWindow: 32000},
        {ID: "voyage-3-large", Capabilities: embed, ContextWindow: 32000},
        {ID: "voyage-code-3", Capabilities: embed, ContextWindow: 32000},
        {ID: "voyage-finance-2", Capabilities: embed, ContextWindow: 32000},
        {ID: "voyage-law-2", Capabilities: embed, ContextWindow: 16000},
        {ID: "voyage-multimodal-3", Capabilities: multimodal, ContextWindow: 32000},
        {ID: "rerank-2.5", Capabilities: rerank, ContextWindow: 8000},
        {ID: "rerank-2.5-lite", Capabilities: rerank, ContextWindow: 8000},
    }
}

// embed, Embed, EmbedBatch, EmbedMultimodal, Rerank are filled in by
// later tasks. The stubs below keep the package compiling so the
// skeleton's tests can run.

func (c *client) embed(_ context.Context, _ []string, _ string, _ *Extension) ([][]float64, error) {
    return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: embed not implemented yet", Provider: providerName}
}

func (c *client) Embed(_ context.Context, _ string) ([]float64, error) {
    return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: Embed not implemented yet", Provider: providerName}
}

func (c *client) EmbedBatch(_ context.Context, _ []string) ([][]float64, error) {
    return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: EmbedBatch not implemented yet", Provider: providerName}
}

func (c *client) EmbedMultimodal(_ context.Context, _ llm.MultimodalEmbedRequest) ([][]float64, error) {
    return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: EmbedMultimodal not implemented yet", Provider: providerName}
}

func (c *client) Rerank(_ context.Context, _ llm.RerankRequest) (*llm.RerankResponse, error) {
    return nil, &llm.ProviderError{Err: llm.ErrNotSupported, Message: "voyage: Rerank not implemented yet", Provider: providerName}
}

// findExtension locates a voyage.Extension in a generic
// []llm.ProviderExtension. (llm.FindExtension only operates on
// ChatRequest; we have multiple non-chat request types that need this.)
func findExtension(exts []llm.ProviderExtension) *Extension {
    for _, e := range exts {
        if e.ProviderName() == providerName {
            if ext, ok := e.(Extension); ok {
                return &ext
            }
            if ext, ok := e.(*Extension); ok {
                return ext
            }
        }
    }
    return nil
}
```

- [ ] **Step 3: Create `llm/provider/voyage/voyage_test.go` with the skeleton tests**

```go
//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package voyage_test

import (
    "context"
    "errors"
    "net/http"
    "net/http/httptest"
    "testing"

    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, llm.Client) {
    t.Helper()
    srv := httptest.NewServer(handler)
    t.Cleanup(srv.Close)
    c, err := llm.NewClient("voyage", llm.Options{
        APIKey:  "test-key",
        Model:   "voyage-3.5-lite",
        BaseURL: srv.URL, // ValidateBaseURL trims any trailing slash
    })
    if err != nil {
        t.Fatal(err)
    }
    return srv, c
}

func TestVoyageRegistered(t *testing.T) {
    names := llm.RegisteredProviders()
    found := false
    for _, n := range names {
        if n == "voyage" {
            found = true
            break
        }
    }
    if !found {
        t.Fatalf("voyage not in registered providers: %v", names)
    }
}

func TestChatUnsupported(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
    _, err := c.Chat(context.Background(), llm.ChatRequest{})
    if !errors.Is(err, llm.ErrNotSupported) {
        t.Fatalf("expected ErrNotSupported, got %v", err)
    }
}

func TestChatStreamUnsupported(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
    _, err := c.ChatStream(context.Background(), llm.ChatRequest{})
    if !errors.Is(err, llm.ErrNotSupported) {
        t.Fatalf("expected ErrNotSupported, got %v", err)
    }
}

func TestListModelsContainsExpectedFamilies(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
    names, err := c.ListModels(context.Background())
    if err != nil {
        t.Fatal(err)
    }
    want := []string{"voyage-3.5", "voyage-multimodal-3", "rerank-2.5"}
    for _, w := range want {
        found := false
        for _, n := range names {
            if n == w {
                found = true
                break
            }
        }
        if !found {
            t.Errorf("expected %q in catalogue, got %v", w, names)
        }
    }
}

func TestListModelsCapabilityFilterReranking(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
    infos, err := c.ListModelsWithMetadata(context.Background(),
        llm.WithCapabilities(llm.ModelCapabilityReranking))
    if err != nil {
        t.Fatal(err)
    }
    if len(infos) != 2 {
        t.Fatalf("expected 2 rerank models, got %d (%v)", len(infos), infos)
    }
}

func TestNewWithoutAPIKey(t *testing.T) {
    t.Setenv("VOYAGE_API_KEY", "")
    _, err := llm.NewClient("voyage", llm.Options{})
    if err == nil {
        t.Fatal("expected error when no API key supplied")
    }
}
```

- [ ] **Step 4: Build and run the voyage package tests**

Run: `go build ./...`
Expected: exit code 0.

Run: `go test ./llm/provider/voyage/ -v`
Expected: all six tests in this file pass.

- [ ] **Step 5: Commit**

```bash
git add llm/provider/voyage/
git commit -m "feat(voyage): scaffold provider package with model catalogue

Adds Voyage provider with Extension type, client constructor, chat
stubs returning ErrNotSupported, hard-coded model catalogue, and
capability-filterable ListModels. Embed / Rerank / EmbedMultimodal
are stubs that the next tasks fill in."
```

---

### Task 12: Implement Voyage text embeddings (`Embed` and `EmbedBatch`)

**Files:**
- Modify: `llm/provider/voyage/voyage.go`
- Modify: `llm/provider/voyage/voyage_test.go`

- [ ] **Step 1: Write the failing tests**

Append to `voyage_test.go`:

```go
func TestEmbedHappyPath(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path != "/embeddings" {
            t.Errorf("unexpected path %q", r.URL.Path)
        }
        if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
            t.Errorf("auth header %q", got)
        }
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[0.1,0.2,0.3],"index":0}],
            "model":"voyage-3.5-lite",
            "usage":{"total_tokens":3}
        }`))
    })
    v, err := c.Embed(context.Background(), "hello")
    if err != nil {
        t.Fatal(err)
    }
    if len(v) != 3 || v[0] != 0.1 {
        t.Fatalf("unexpected vector %v", v)
    }
    if got := c.Usage().TotalTokens; got != 3 {
        t.Errorf("usage TotalTokens = %d, want 3", got)
    }
}

func TestEmbedBatchPreservesOrder(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[
                {"embedding":[1],"index":1},
                {"embedding":[0],"index":0},
                {"embedding":[2],"index":2}
            ],
            "model":"voyage-3.5-lite",
            "usage":{"total_tokens":9}
        }`))
    })
    vs, err := c.EmbedBatch(context.Background(), []string{"a", "b", "c"})
    if err != nil {
        t.Fatal(err)
    }
    if len(vs) != 3 || vs[0][0] != 0 || vs[1][0] != 1 || vs[2][0] != 2 {
        t.Fatalf("EmbedBatch did not sort by index: %v", vs)
    }
}

func TestEmbedAuthError(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusUnauthorized)
        w.Write([]byte(`{"detail":"invalid key"}`))
    })
    _, err := c.Embed(context.Background(), "x")
    if err == nil {
        t.Fatal("expected error")
    }
    if !errors.Is(err, llm.ErrAuthentication) {
        t.Fatalf("expected ErrAuthentication, got %v", err)
    }
}
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `go test ./llm/provider/voyage/ -run TestEmbed -v`
Expected: FAIL with the current "not implemented yet" stubs.

- [ ] **Step 3: Implement `Embed` and `EmbedBatch` in `voyage.go`**

Replace the stub `Embed`, `EmbedBatch`, and `embed` methods in `voyage.go` with:

```go
type embeddingsRequest struct {
    Input           []string `json:"input"`
    Model           string   `json:"model"`
    InputType       string   `json:"input_type,omitempty"`
    Truncation      *bool    `json:"truncation,omitempty"`
    OutputDimension int      `json:"output_dimension,omitempty"`
    OutputDtype     string   `json:"output_dtype,omitempty"`
}

type embeddingsResponseDatum struct {
    Embedding []float64 `json:"embedding"`
    Index     int       `json:"index"`
}

type embeddingsResponse struct {
    Data  []embeddingsResponseDatum `json:"data"`
    Model string                    `json:"model"`
    Usage struct {
        TotalTokens int `json:"total_tokens"`
    } `json:"usage"`
}

func (c *client) Embed(ctx context.Context, text string) ([]float64, error) {
    vecs, err := c.embed(ctx, []string{text}, c.model, nil)
    if err != nil {
        return nil, err
    }
    if len(vecs) == 0 {
        return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "empty embedding response", Provider: providerName}
    }
    return vecs[0], nil
}

func (c *client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
    return c.embed(ctx, texts, c.model, nil)
}

func (c *client) embed(ctx context.Context, texts []string, model string, ext *Extension) ([][]float64, error) {
    if model == "" {
        return nil, &llm.ProviderError{
            Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName,
        }
    }
    req := embeddingsRequest{Input: texts, Model: model}
    if ext != nil {
        req.InputType = string(ext.InputType)
        req.Truncation = ext.Truncation
        req.OutputDimension = ext.OutputDimension
        req.OutputDtype = string(ext.OutputDtype)
    }
    var resp embeddingsResponse
    if err := c.postJSON(ctx, "/embeddings", req, &resp); err != nil {
        return nil, err
    }
    out := make([][]float64, len(texts))
    for _, d := range resp.Data {
        if d.Index < 0 || d.Index >= len(out) {
            return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "embedding index out of range", Provider: providerName}
        }
        out[d.Index] = d.Embedding
    }
    c.addUsage(llm.TokenUsage{TotalTokens: resp.Usage.TotalTokens})
    return out, nil
}
```

And add a JSON-POST helper at the bottom of `voyage.go`. It delegates to the existing `httpclient.DoJSON` (see `openai.go` for the established pattern):

```go
func (c *client) postJSON(ctx context.Context, path string, body, out any) error {
    headers := map[string]string{
        "Authorization": "Bearer " + c.apiKey,
        "Content-Type":  "application/json",
    }
    status, respBody, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost, c.baseURL+path, headers, body, out)
    if err != nil {
        return &llm.ProviderError{Err: llm.ErrProviderError, Message: err.Error(), Provider: providerName, StatusCode: status}
    }
    if status < 200 || status >= 300 {
        return &llm.ProviderError{
            Err:        statusToErr(status),
            Message:    fmt.Sprintf("voyage: HTTP %d: %s", status, string(respBody)),
            Provider:   providerName,
            StatusCode: status,
        }
    }
    return nil
}

func statusToErr(status int) error {
    switch {
    case status == http.StatusUnauthorized || status == http.StatusForbidden:
        return llm.ErrAuthentication
    case status == http.StatusTooManyRequests:
        return llm.ErrRateLimit
    case status == http.StatusBadRequest:
        return llm.ErrInvalidRequest
    default:
        return llm.ErrProviderError
    }
}
```

Add `"fmt"` to the import block of `voyage.go` for the `Sprintf` call above. (No `bytes` / `io` / `encoding/json` needed — `httpclient.DoJSON` handles all of that internally.)

- [ ] **Step 4: Run all voyage tests**

Run: `go test ./llm/provider/voyage/ -v`
Expected: all tests pass, including the three new TestEmbed* tests.

- [ ] **Step 5: Commit**

```bash
git add llm/provider/voyage/
git commit -m "feat(voyage): implement text embeddings (Embed and EmbedBatch)"
```

---

### Task 13: Implement Voyage multimodal embeddings (`EmbedMultimodal`)

**Files:**
- Modify: `llm/provider/voyage/voyage.go`
- Modify: `llm/provider/voyage/voyage_test.go`

- [ ] **Step 1: Write failing tests**

Append to `voyage_test.go`:

```go
func TestEmbedMultimodalTextOnly(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path != "/multimodalembeddings" {
            t.Errorf("path = %q", r.URL.Path)
        }
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[0.5,0.6],"index":0}],
            "model":"voyage-multimodal-3",
            "usage":{"total_tokens":4}
        }`))
    })
    vs, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
        Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
            {Type: llm.MultimodalContentText, Text: "a kitten"},
        }}},
    })
    if err != nil {
        t.Fatal(err)
    }
    if len(vs) != 1 || vs[0][0] != 0.5 {
        t.Fatalf("unexpected result %v", vs)
    }
}

func TestEmbedMultimodalImageURL(t *testing.T) {
    var captured map[string]any
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&captured)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[1],"index":0}],
            "model":"voyage-multimodal-3",
            "usage":{"total_tokens":2}
        }`))
    })
    _, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
        Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
            {Type: llm.MultimodalContentImageURL, ImageURL: "https://example.com/x.jpg"},
        }}},
    })
    if err != nil {
        t.Fatal(err)
    }
    inputs := captured["inputs"].([]any)
    first := inputs[0].(map[string]any)
    content := first["content"].([]any)[0].(map[string]any)
    if content["type"] != "image_url" {
        t.Errorf("wire type = %v", content["type"])
    }
    if content["image_url"] != "https://example.com/x.jpg" {
        t.Errorf("wire image_url = %v", content["image_url"])
    }
}

func TestEmbedMultimodalImageData(t *testing.T) {
    var captured map[string]any
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&captured)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[1],"index":0}],
            "model":"voyage-multimodal-3",
            "usage":{"total_tokens":2}
        }`))
    })
    _, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
        Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
            {Type: llm.MultimodalContentImageData, ImageData: []byte{0xff, 0xd8, 0xff}, MIMEType: "image/jpeg"},
        }}},
    })
    if err != nil {
        t.Fatal(err)
    }
    content := captured["inputs"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
    if content["type"] != "image_base64" {
        t.Errorf("wire type = %v", content["type"])
    }
    if content["image_base64"] == nil {
        t.Errorf("expected image_base64 in payload, got %v", content)
    }
}

func TestEmbedMultimodalExtensionRoundtrip(t *testing.T) {
    var captured map[string]any
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&captured)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[1],"index":0}],
            "model":"voyage-multimodal-3",
            "usage":{"total_tokens":1}
        }`))
    })
    _, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
        Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{{Type: llm.MultimodalContentText, Text: "x"}}}},
        Extensions: []llm.ProviderExtension{voyage.Extension{
            InputType:       voyage.InputTypeQuery,
            OutputDimension: 512,
        }},
    })
    if err != nil {
        t.Fatal(err)
    }
    if captured["input_type"] != "query" {
        t.Errorf("input_type = %v", captured["input_type"])
    }
    if captured["output_dimension"].(float64) != 512 {
        t.Errorf("output_dimension = %v", captured["output_dimension"])
    }
}
```

Add `"encoding/json"` and the import `voyage "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"` to `voyage_test.go` (replacing the existing blank import). Use `voyage.` to reference the package's exported types so the extension test compiles.

(The current `voyage_test.go` is in package `voyage_test`, so it needs the named import to call `voyage.Extension`.)

- [ ] **Step 2: Run the tests, verify they fail**

Run: `go test ./llm/provider/voyage/ -run TestEmbedMultimodal -v`
Expected: FAIL.

- [ ] **Step 3: Implement `EmbedMultimodal` in `voyage.go`**

Replace the stub `EmbedMultimodal` with:

```go
type multimodalContentWire struct {
    Type        string `json:"type"`
    Text        string `json:"text,omitempty"`
    ImageURL    string `json:"image_url,omitempty"`
    ImageBase64 string `json:"image_base64,omitempty"`
}

type multimodalInputWire struct {
    Content []multimodalContentWire `json:"content"`
}

type multimodalEmbeddingsRequest struct {
    Inputs          []multimodalInputWire `json:"inputs"`
    Model           string                `json:"model"`
    InputType       string                `json:"input_type,omitempty"`
    Truncation      *bool                 `json:"truncation,omitempty"`
    OutputDimension int                   `json:"output_dimension,omitempty"`
    OutputDtype     string                `json:"output_dtype,omitempty"`
}

func (c *client) EmbedMultimodal(ctx context.Context, req llm.MultimodalEmbedRequest) ([][]float64, error) {
    if c.model == "" {
        return nil, &llm.ProviderError{Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName}
    }
    wireInputs := make([]multimodalInputWire, len(req.Inputs))
    for i, in := range req.Inputs {
        wireContent := make([]multimodalContentWire, len(in.Content))
        for j, c := range in.Content {
            wireContent[j] = contentToWire(c)
        }
        wireInputs[i] = multimodalInputWire{Content: wireContent}
    }
    wire := multimodalEmbeddingsRequest{Inputs: wireInputs, Model: c.model}
    if ext := findExtension(req.Extensions); ext != nil {
        wire.InputType = string(ext.InputType)
        wire.Truncation = ext.Truncation
        wire.OutputDimension = ext.OutputDimension
        wire.OutputDtype = string(ext.OutputDtype)
    }
    var resp embeddingsResponse
    if err := c.postJSON(ctx, "/multimodalembeddings", wire, &resp); err != nil {
        return nil, err
    }
    out := make([][]float64, len(req.Inputs))
    for _, d := range resp.Data {
        if d.Index < 0 || d.Index >= len(out) {
            return nil, &llm.ProviderError{Err: llm.ErrProviderError, Message: "embedding index out of range", Provider: providerName}
        }
        out[d.Index] = d.Embedding
    }
    c.addUsage(llm.TokenUsage{TotalTokens: resp.Usage.TotalTokens})
    return out, nil
}

func contentToWire(c llm.MultimodalContent) multimodalContentWire {
    switch c.Type {
    case llm.MultimodalContentText:
        return multimodalContentWire{Type: "text", Text: c.Text}
    case llm.MultimodalContentImageURL:
        return multimodalContentWire{Type: "image_url", ImageURL: c.ImageURL}
    case llm.MultimodalContentImageData:
        return multimodalContentWire{Type: "image_base64", ImageBase64: base64.StdEncoding.EncodeToString(c.ImageData)}
    default:
        return multimodalContentWire{}
    }
}
```

Add `"encoding/base64"` to the imports.

The `findExtension(...)` helper is defined in the voyage skeleton (Task 11). It walks the provided `[]llm.ProviderExtension`, returns a `*Extension` if one is present, else nil. `llm.FindExtension` can't be used here because it's bound to `ChatRequest`.

- [ ] **Step 4: Run the multimodal tests**

Run: `go test ./llm/provider/voyage/ -run TestEmbedMultimodal -v`
Expected: PASS.

- [ ] **Step 5: Run the whole voyage package**

Run: `go test ./llm/provider/voyage/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm/provider/voyage/
git commit -m "feat(voyage): implement EmbedMultimodal with text/url/base64 content"
```

---

### Task 14: Implement Voyage reranking (`Rerank`)

**Files:**
- Modify: `llm/provider/voyage/voyage.go`
- Modify: `llm/provider/voyage/voyage_test.go`

- [ ] **Step 1: Write failing tests**

Append to `voyage_test.go`:

```go
func TestRerankHappyPath(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path != "/rerank" {
            t.Errorf("path = %q", r.URL.Path)
        }
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[
                {"index":2,"relevance_score":0.9},
                {"index":0,"relevance_score":0.7},
                {"index":1,"relevance_score":0.3}
            ],
            "model":"rerank-2.5",
            "usage":{"total_tokens":42}
        }`))
    })
    // The fixture above doesn't enforce the model name, so we can
    // reuse the default-model test client for rerank too.
    res, err := c.Rerank(context.Background(), llm.RerankRequest{
        Query:     "kittens",
        Documents: []string{"alpha", "beta", "gamma"},
    })
    if err != nil {
        t.Fatal(err)
    }
    if len(res.Results) != 3 {
        t.Fatalf("got %d results", len(res.Results))
    }
    if res.Results[0].Index != 2 || res.Results[0].RelevanceScore != 0.9 {
        t.Errorf("top result wrong: %+v", res.Results[0])
    }
    if res.Usage.TotalTokens != 42 {
        t.Errorf("usage TotalTokens = %d", res.Usage.TotalTokens)
    }
}

func TestRerankReturnDocuments(t *testing.T) {
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[
                {"index":0,"relevance_score":0.9,"document":"alpha"}
            ],
            "model":"rerank-2.5",
            "usage":{"total_tokens":5}
        }`))
    })
    tru := true
    res, err := c.Rerank(context.Background(), llm.RerankRequest{
        Query:     "q",
        Documents: []string{"alpha"},
        Extensions: []llm.ProviderExtension{voyage.Extension{
            ReturnDocuments: &tru,
        }},
    })
    if err != nil {
        t.Fatal(err)
    }
    if res.Results[0].Document != "alpha" {
        t.Errorf("expected document populated, got %q", res.Results[0].Document)
    }
}

func TestRerankTopK(t *testing.T) {
    var captured map[string]any
    _, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&captured)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"data":[],"model":"rerank-2.5","usage":{"total_tokens":0}}`))
    })
    k := 5
    _, err := c.Rerank(context.Background(), llm.RerankRequest{
        Query: "q", Documents: []string{"a"}, TopK: &k,
    })
    if err != nil {
        t.Fatal(err)
    }
    if captured["top_k"].(float64) != 5 {
        t.Errorf("top_k = %v, want 5", captured["top_k"])
    }
}
```

Drop the placeholder `c.(interface{ BaseURL() string })` block in `TestRerankHappyPath`; the simpler form just uses the existing `c`.

- [ ] **Step 2: Run, verify failures**

Run: `go test ./llm/provider/voyage/ -run TestRerank -v`
Expected: FAIL.

- [ ] **Step 3: Implement `Rerank` in `voyage.go`**

Replace the stub `Rerank` with:

```go
type rerankRequestWire struct {
    Query           string   `json:"query"`
    Documents       []string `json:"documents"`
    Model           string   `json:"model"`
    TopK            *int     `json:"top_k,omitempty"`
    ReturnDocuments *bool    `json:"return_documents,omitempty"`
    Truncation      *bool    `json:"truncation,omitempty"`
}

type rerankResponseWire struct {
    Data []struct {
        Index          int     `json:"index"`
        RelevanceScore float64 `json:"relevance_score"`
        Document       string  `json:"document,omitempty"`
    } `json:"data"`
    Model string `json:"model"`
    Usage struct {
        TotalTokens int `json:"total_tokens"`
    } `json:"usage"`
}

func (c *client) Rerank(ctx context.Context, req llm.RerankRequest) (*llm.RerankResponse, error) {
    if c.model == "" {
        return nil, &llm.ProviderError{Err: llm.ErrInvalidRequest, Message: "Voyage requires Options.Model", Provider: providerName}
    }
    wire := rerankRequestWire{Query: req.Query, Documents: req.Documents, Model: c.model, TopK: req.TopK}
    if ext := findExtension(req.Extensions); ext != nil {
        wire.ReturnDocuments = ext.ReturnDocuments
        wire.Truncation = ext.Truncation
    }
    var raw rerankResponseWire
    if err := c.postJSON(ctx, "/rerank", wire, &raw); err != nil {
        return nil, err
    }
    out := &llm.RerankResponse{
        Results: make([]llm.RerankResult, len(raw.Data)),
        Usage:   llm.TokenUsage{TotalTokens: raw.Usage.TotalTokens},
    }
    for i, d := range raw.Data {
        out.Results[i] = llm.RerankResult{Index: d.Index, RelevanceScore: d.RelevanceScore, Document: d.Document}
    }
    c.addUsage(out.Usage)
    return out, nil
}
```

Uses the same `findExtension` helper defined in the voyage skeleton (Task 11).

- [ ] **Step 4: Run rerank tests**

Run: `go test ./llm/provider/voyage/ -run TestRerank -v`
Expected: PASS.

- [ ] **Step 5: Run the whole package**

Run: `go test ./llm/provider/voyage/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm/provider/voyage/
git commit -m "feat(voyage): implement Rerank with top_k and return_documents support"
```

---

### Task 15: Register Voyage in `llm/all`

**Files:**
- Modify: `llm/all/all.go`

- [ ] **Step 1: Add the import line**

Edit `llm/all/all.go`:

```go
package all

import (
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/gemini"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)
```

- [ ] **Step 2: Verify the build and tests pass**

Run: `go build ./...`
Run: `go test ./llm/all/...`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add llm/all/all.go
git commit -m "feat(all): register voyage provider in convenience package"
```

---

### Task 16: Add a gated Voyage integration test

**Files:**
- Modify: `llm/integration_test.go`

Mirror whatever pattern existing providers use: a `testing.Short`-skip plus an env-var-presence check. If `VOYAGE_API_KEY` is absent, skip; otherwise hit the live API for a single embed + a single rerank.

- [ ] **Step 1: Inspect the existing pattern**

Read `llm/integration_test.go` and note how, for example, OpenAI integration is gated. Match the structure exactly (test-name prefix, skip message format, helper names).

- [ ] **Step 2: Append a `TestIntegrationVoyage` block**

Add a block like:

```go
func TestIntegrationVoyage(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test in -short mode")
    }
    key := os.Getenv("VOYAGE_API_KEY")
    if key == "" {
        t.Skip("VOYAGE_API_KEY not set; skipping Voyage integration test")
    }
    c, err := llm.NewClient("voyage", llm.Options{APIKey: key, Model: "voyage-3.5-lite"})
    if err != nil {
        t.Fatal(err)
    }
    vec, err := c.Embed(context.Background(), "the quick brown fox")
    if err != nil {
        t.Fatal(err)
    }
    if len(vec) == 0 {
        t.Fatal("expected non-empty embedding")
    }

    rerankClient, err := llm.NewClient("voyage", llm.Options{APIKey: key, Model: "rerank-2.5-lite"})
    if err != nil {
        t.Fatal(err)
    }
    res, err := rerankClient.Rerank(context.Background(), llm.RerankRequest{
        Query: "kittens",
        Documents: []string{
            "Cats are small carnivorous mammals.",
            "The Eiffel Tower is in Paris.",
            "A kitten is a juvenile cat.",
        },
    })
    if err != nil {
        t.Fatal(err)
    }
    if len(res.Results) == 0 || res.Results[0].Index != 2 && res.Results[0].Index != 0 {
        t.Errorf("expected the cat-related sentences to rank highest, got %+v", res.Results)
    }
}
```

Add `"os"` and `_ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"` to the imports if not already present.

- [ ] **Step 3: Run integration tests (will skip without the key)**

Run: `go test ./llm/ -run TestIntegrationVoyage -v`
Expected: SKIP message (since `VOYAGE_API_KEY` isn't set locally).

- [ ] **Step 4: Commit**

```bash
git add llm/integration_test.go
git commit -m "test(voyage): add gated live integration test for embed + rerank"
```

---

## Phase 5 — Proxy HTTP surface

---

### Task 17: Add `POST /v1/embed` route to the proxy

**Files:**
- Modify: `llm/proxy/types.go` (add request/response shapes)
- Modify: `llm/proxy/proxy.go` (route registration)
- Modify: `llm/proxy/handlers.go` (handler)
- Modify: `llm/proxy/proxy_test.go` (tests)

- [ ] **Step 1: Write failing tests in `proxy_test.go`**

Append a test that POSTs to `/v1/embed` against the proxy with a single input, asserts a 200 with a JSON body containing the vector, and a second test asserting 501 when the underlying provider returns `ErrNotSupported`. Mirror the existing chat tests for boilerplate (`p := newTestProxy(t, ...)`, `httptest.NewRecorder`, etc.).

```go
func TestEmbedHappyPath(t *testing.T) {
    setFake(func(p *fakeProvider) {
        // Configure the fake's embed behaviour. If fakeProvider's
        // existing surface doesn't support embed customisation,
        // extend it: add fields embedVec [][]float64 / embedErr error
        // and have Embed/EmbedBatch return them.
        p.embedVec = [][]float64{{0.1, 0.2}}
    })
    p := newTestProxy(t)
    body := bytes.NewReader([]byte(`{"provider":"fake","input":["hello"]}`))
    req := httptest.NewRequest(http.MethodPost, "/v1/embed", body)
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusOK {
        t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
    }
    var resp struct {
        Embeddings [][]float64 `json:"embeddings"`
    }
    if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
        t.Fatal(err)
    }
    if len(resp.Embeddings) != 1 || resp.Embeddings[0][0] != 0.1 {
        t.Fatalf("unexpected body: %s", rec.Body.String())
    }
}

func TestEmbedUnsupported(t *testing.T) {
    setFake(func(p *fakeProvider) {
        p.embedErr = &llm.ProviderError{Err: llm.ErrNotSupported, Message: "fake says no", Provider: "fake"}
    })
    p := newTestProxy(t)
    req := httptest.NewRequest(http.MethodPost, "/v1/embed",
        bytes.NewReader([]byte(`{"provider":"fake","input":["x"]}`)))
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusNotImplemented {
        t.Fatalf("expected 501, got %d", rec.Code)
    }
}
```

`fakeProvider` may need fields for embed behaviour. If so, extend `fake_provider_test.go` accordingly: add `embedVec [][]float64` and `embedErr error` fields and update `Embed` / `EmbedBatch` to return them.

- [ ] **Step 2: Run the tests, verify they fail**

Run: `go test ./llm/proxy/ -run TestEmbed -v`
Expected: FAIL.

- [ ] **Step 3: Add request/response types in `proxy/types.go`**

```go
// EmbedRequest is the JSON body of POST /v1/embed. Input may contain
// one or many strings; the proxy chooses Embed vs EmbedBatch
// accordingly.
type EmbedRequest struct {
    Provider string   `json:"provider,omitempty"`
    Model    string   `json:"model,omitempty"`
    Input    []string `json:"input"`
}

// EmbedResponse is the JSON body returned by POST /v1/embed.
type EmbedResponse struct {
    Embeddings [][]float64 `json:"embeddings"`
}
```

- [ ] **Step 4: Register the route**

In `proxy.go`, alongside the existing `mux.HandleFunc(...)` lines, add:

```go
mux.HandleFunc("POST /v1/embed", p.handleEmbed)
```

- [ ] **Step 5: Implement the handler in `handlers.go`**

Append:

```go
func (p *Proxy) handleEmbed(w http.ResponseWriter, r *http.Request) {
    r, reqID := p.ensureRequestID(r)
    if reqID != "" {
        w.Header().Set(p.requestIDHeaderName(), reqID)
    }
    if !p.authorize(w, r) {
        return
    }
    var req EmbedRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
        return
    }
    provider := req.Provider
    if provider == "" {
        provider = p.cfg.DefaultProvider
    }
    opts, ok := p.cfg.Providers[provider]
    if !ok {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
            Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
        return
    }
    if req.Model != "" {
        opts.Model = req.Model
    }
    client, err := llm.NewClient(provider, opts)
    if err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
        return
    }
    vecs, err := client.EmbedBatch(r.Context(), req.Input)
    if err != nil {
        status := http.StatusBadGateway
        if errors.Is(err, llm.ErrNotSupported) {
            status = http.StatusNotImplemented
        }
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID})
        return
    }
    w.Header().Set("Content-Type", "application/json")
    if err := encodeJSON(w, EmbedResponse{Embeddings: vecs}); err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
    }
}
```

Add `"errors"` to the import block of `handlers.go` if missing.

- [ ] **Step 6: Run tests**

Run: `go test ./llm/proxy/ -v`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add llm/proxy/
git commit -m "feat(proxy): add POST /v1/embed for text embeddings"
```

---

### Task 18: Add `POST /v1/embed/multimodal` route to the proxy

**Files:**
- Modify: `llm/proxy/types.go`
- Modify: `llm/proxy/proxy.go`
- Modify: `llm/proxy/handlers.go`
- Modify: `llm/proxy/proxy_test.go`

- [ ] **Step 1: Failing tests**

Append:

```go
func TestEmbedMultimodalHappyPath(t *testing.T) {
    setFake(func(p *fakeProvider) {
        p.multimodalVec = [][]float64{{0.5, 0.6}}
    })
    p := newTestProxy(t)
    body := bytes.NewReader([]byte(`{
        "provider":"fake",
        "inputs":[{"content":[{"type":"text","text":"hi"}]}]
    }`))
    req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal", body)
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusOK {
        t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
    }
}

func TestEmbedMultimodalUnsupportedReturns501(t *testing.T) {
    setFake(func(p *fakeProvider) {
        p.multimodalErr = &llm.ProviderError{Err: llm.ErrNotSupported, Message: "no", Provider: "fake"}
    })
    p := newTestProxy(t)
    req := httptest.NewRequest(http.MethodPost, "/v1/embed/multimodal",
        bytes.NewReader([]byte(`{"provider":"fake","inputs":[]}`)))
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusNotImplemented {
        t.Fatalf("expected 501, got %d", rec.Code)
    }
}
```

Extend `fakeProvider` with `multimodalVec`, `multimodalErr` fields and update its `EmbedMultimodal` to honour them.

- [ ] **Step 2: Verify failures**

Run: `go test ./llm/proxy/ -run TestEmbedMultimodal -v`
Expected: FAIL.

- [ ] **Step 3: Add request/response types**

In `proxy/types.go`:

```go
type MultimodalContentRequest struct {
    Type        string `json:"type"`
    Text        string `json:"text,omitempty"`
    ImageURL    string `json:"image_url,omitempty"`
    ImageBase64 string `json:"image_base64,omitempty"`
    MIMEType    string `json:"mime_type,omitempty"`
}

type MultimodalInputRequest struct {
    Content []MultimodalContentRequest `json:"content"`
}

type EmbedMultimodalRequest struct {
    Provider string                   `json:"provider,omitempty"`
    Model    string                   `json:"model,omitempty"`
    Inputs   []MultimodalInputRequest `json:"inputs"`
}

type EmbedMultimodalResponse struct {
    Embeddings [][]float64 `json:"embeddings"`
}
```

- [ ] **Step 4: Register the route**

In `proxy.go`:

```go
mux.HandleFunc("POST /v1/embed/multimodal", p.handleEmbedMultimodal)
```

- [ ] **Step 5: Implement the handler**

In `handlers.go`:

```go
func (p *Proxy) handleEmbedMultimodal(w http.ResponseWriter, r *http.Request) {
    r, reqID := p.ensureRequestID(r)
    if reqID != "" {
        w.Header().Set(p.requestIDHeaderName(), reqID)
    }
    if !p.authorize(w, r) {
        return
    }
    var req EmbedMultimodalRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
        return
    }
    provider := req.Provider
    if provider == "" {
        provider = p.cfg.DefaultProvider
    }
    opts, ok := p.cfg.Providers[provider]
    if !ok {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
            Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
        return
    }
    if req.Model != "" {
        opts.Model = req.Model
    }
    client, err := llm.NewClient(provider, opts)
    if err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
        return
    }
    libReq := llm.MultimodalEmbedRequest{
        Inputs: make([]llm.MultimodalInput, len(req.Inputs)),
    }
    for i, in := range req.Inputs {
        contents := make([]llm.MultimodalContent, len(in.Content))
        for j, c := range in.Content {
            mc := llm.MultimodalContent{
                Type:     llm.MultimodalContentType(c.Type),
                Text:     c.Text,
                ImageURL: c.ImageURL,
                MIMEType: c.MIMEType,
            }
            if c.ImageBase64 != "" {
                data, decErr := base64.StdEncoding.DecodeString(c.ImageBase64)
                if decErr != nil {
                    p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest, Err: decErr, RequestID: reqID})
                    return
                }
                mc.ImageData = data
            }
            contents[j] = mc
        }
        libReq.Inputs[i] = llm.MultimodalInput{Content: contents}
    }
    vecs, err := client.EmbedMultimodal(r.Context(), libReq)
    if err != nil {
        status := http.StatusBadGateway
        if errors.Is(err, llm.ErrNotSupported) {
            status = http.StatusNotImplemented
        }
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID})
        return
    }
    w.Header().Set("Content-Type", "application/json")
    if err := encodeJSON(w, EmbedMultimodalResponse{Embeddings: vecs}); err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
    }
}
```

Add `"encoding/base64"` to the imports if not already present.

- [ ] **Step 6: Run tests**

Run: `go test ./llm/proxy/ -v`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add llm/proxy/
git commit -m "feat(proxy): add POST /v1/embed/multimodal for multimodal embeddings"
```

---

### Task 19: Add `POST /v1/rerank` route to the proxy

**Files:**
- Modify: `llm/proxy/types.go`
- Modify: `llm/proxy/proxy.go`
- Modify: `llm/proxy/handlers.go`
- Modify: `llm/proxy/proxy_test.go`

- [ ] **Step 1: Failing tests**

```go
func TestRerankHappyPath(t *testing.T) {
    setFake(func(p *fakeProvider) {
        p.rerankResp = &llm.RerankResponse{
            Results: []llm.RerankResult{
                {Index: 1, RelevanceScore: 0.9},
                {Index: 0, RelevanceScore: 0.4},
            },
            Usage: llm.TokenUsage{TotalTokens: 7},
        }
    })
    p := newTestProxy(t)
    body := bytes.NewReader([]byte(`{
        "provider":"fake",
        "query":"q",
        "documents":["a","b"]
    }`))
    req := httptest.NewRequest(http.MethodPost, "/v1/rerank", body)
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusOK {
        t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
    }
    var resp struct {
        Results []struct {
            Index int     `json:"index"`
            Score float64 `json:"relevance_score"`
        } `json:"results"`
        Usage struct {
            TotalTokens int `json:"total_tokens"`
        } `json:"usage"`
    }
    if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
        t.Fatal(err)
    }
    if len(resp.Results) != 2 || resp.Results[0].Index != 1 {
        t.Fatalf("unexpected response %s", rec.Body.String())
    }
    if resp.Usage.TotalTokens != 7 {
        t.Errorf("usage TotalTokens = %d", resp.Usage.TotalTokens)
    }
}

func TestRerankUnsupportedReturns501(t *testing.T) {
    setFake(func(p *fakeProvider) {
        p.rerankErr = &llm.ProviderError{Err: llm.ErrNotSupported, Message: "no", Provider: "fake"}
    })
    p := newTestProxy(t)
    req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
        bytes.NewReader([]byte(`{"provider":"fake","query":"q","documents":["a"]}`)))
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusNotImplemented {
        t.Fatalf("expected 501, got %d", rec.Code)
    }
}
```

Extend `fakeProvider` with `rerankResp *llm.RerankResponse`, `rerankErr error` fields.

- [ ] **Step 2: Verify failure**

Run: `go test ./llm/proxy/ -run TestRerank -v`
Expected: FAIL.

- [ ] **Step 3: Add request/response types**

In `proxy/types.go`:

```go
type RerankRequest struct {
    Provider  string   `json:"provider,omitempty"`
    Model     string   `json:"model,omitempty"`
    Query     string   `json:"query"`
    Documents []string `json:"documents"`
    TopK      *int     `json:"top_k,omitempty"`
}

type RerankResult struct {
    Index          int     `json:"index"`
    RelevanceScore float64 `json:"relevance_score"`
    Document       string  `json:"document,omitempty"`
}

type RerankUsage struct {
    TotalTokens int `json:"total_tokens"`
}

type RerankResponse struct {
    Results []RerankResult `json:"results"`
    Usage   RerankUsage    `json:"usage"`
}
```

- [ ] **Step 4: Register the route**

In `proxy.go`:

```go
mux.HandleFunc("POST /v1/rerank", p.handleRerank)
```

- [ ] **Step 5: Implement the handler**

In `handlers.go`:

```go
func (p *Proxy) handleRerank(w http.ResponseWriter, r *http.Request) {
    r, reqID := p.ensureRequestID(r)
    if reqID != "" {
        w.Header().Set(p.requestIDHeaderName(), reqID)
    }
    if !p.authorize(w, r) {
        return
    }
    var req RerankRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        p.writeError(w, r, ErrorInfo{StatusCode: http.StatusBadRequest, Err: err, RequestID: reqID})
        return
    }
    provider := req.Provider
    if provider == "" {
        provider = p.cfg.DefaultProvider
    }
    opts, ok := p.cfg.Providers[provider]
    if !ok {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusBadRequest,
            Err: fmt.Errorf("provider %q is not configured", provider), RequestID: reqID})
        return
    }
    if req.Model != "" {
        opts.Model = req.Model
    }
    client, err := llm.NewClient(provider, opts)
    if err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
        return
    }
    libResp, err := client.Rerank(r.Context(), llm.RerankRequest{
        Query: req.Query, Documents: req.Documents, TopK: req.TopK,
    })
    if err != nil {
        status := http.StatusBadGateway
        if errors.Is(err, llm.ErrNotSupported) {
            status = http.StatusNotImplemented
        }
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: status, Err: err, RequestID: reqID})
        return
    }
    out := RerankResponse{Usage: RerankUsage{TotalTokens: libResp.Usage.TotalTokens}}
    out.Results = make([]RerankResult, len(libResp.Results))
    for i, res := range libResp.Results {
        out.Results[i] = RerankResult{Index: res.Index, RelevanceScore: res.RelevanceScore, Document: res.Document}
    }
    w.Header().Set("Content-Type", "application/json")
    if err := encodeJSON(w, out); err != nil {
        p.writeError(w, r, ErrorInfo{Provider: provider, StatusCode: http.StatusInternalServerError, Err: err, RequestID: reqID})
    }
}
```

- [ ] **Step 6: Run tests**

Run: `go test ./llm/proxy/ -v`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add llm/proxy/
git commit -m "feat(proxy): add POST /v1/rerank for reranking"
```

---

### Task 20: Wire capability filter into `GET /v1/models`

**Files:**
- Modify: `llm/proxy/handlers.go` (in `handleModels`)
- Modify: `llm/proxy/proxy_test.go`

- [ ] **Step 1: Failing test**

```go
func TestModelsCapabilityFilter(t *testing.T) {
    setFake(func(p *fakeProvider) {
        // fakeProvider returns ModelInfo with capability data; ensure
        // its ListModelsWithMetadata covers a chat model and an
        // embedding model so the filter is meaningful.
    })
    p := newTestProxy(t)
    req := httptest.NewRequest(http.MethodGet, "/v1/models?provider=fake&metadata=true&capability=embeddings", nil)
    rec := httptest.NewRecorder()
    p.Handler().ServeHTTP(rec, req)
    if rec.Code != http.StatusOK {
        t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
    }
    var resp ModelsMetadataResponse
    if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
        t.Fatal(err)
    }
    for _, m := range resp.Models {
        found := false
        for _, c := range m.Capabilities {
            if c == llm.ModelCapabilityEmbeddings {
                found = true
                break
            }
        }
        if !found {
            t.Errorf("model %s leaked through filter (caps=%v)", m.ID, m.Capabilities)
        }
    }
}
```

Update `fakeProvider`'s `ListModelsWithMetadata` so it returns at least two distinct entries, one tagged `ModelCapabilityChat` and another `ModelCapabilityEmbeddings`.

- [ ] **Step 2: Verify failure**

Run: `go test ./llm/proxy/ -run TestModelsCapabilityFilter -v`
Expected: FAIL.

- [ ] **Step 3: Update `handleModels` in `handlers.go`**

Inside the existing `handleModels`, before calling `client.ListModelsWithMetadata` / `client.ListModels`, gather capability filters from the query string:

```go
var listOpts []llm.ListModelsOption
if caps := r.URL.Query()["capability"]; len(caps) > 0 {
    typed := make([]llm.ModelCapability, 0, len(caps))
    for _, c := range caps {
        typed = append(typed, llm.ModelCapability(c))
    }
    listOpts = append(listOpts, llm.WithCapabilities(typed...))
}
```

Then change the two existing calls:

- `client.ListModelsWithMetadata(r.Context())` → `client.ListModelsWithMetadata(r.Context(), listOpts...)`
- `client.ListModels(r.Context())` → `client.ListModels(r.Context(), listOpts...)`

- [ ] **Step 4: Run tests**

Run: `go test ./llm/proxy/ -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add llm/proxy/
git commit -m "feat(proxy): accept ?capability= query param on /v1/models"
```

---

## Phase 6 — Documentation

These tasks are doc-only. Each is small and independent — execute in any order. Commit per task.

---

### Task 21: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Supported Providers table**

Find the existing table (around line 518):

```
| Feature               | Anthropic | OpenAI | Gemini | Ollama |
|-----------------------|-----------|--------|--------|--------|
| Chat                  | Yes       | Yes    | Yes    | Yes    |
...
```

Replace with:

```
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
```

- [ ] **Step 2: Update the proxy endpoints table**

Find the endpoints table (around line 388) and add:

```
| `POST` | `/v1/embed`                        | Generate text embeddings for one or more inputs. |
| `POST` | `/v1/embed/multimodal`             | Generate multimodal embeddings (text + images). |
| `POST` | `/v1/rerank`                       | Rerank documents by relevance to a query. |
| `GET`  | `/v1/models?...&capability=X`      | Filter models by capability (repeatable; AND of values). |
```

- [ ] **Step 3: Update the per-package description line**

Find:

```
or import individual provider packages to keep binary size small:
```

Update the example list to include voyage:

```go
_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
```

Also find the line `// []string{"anthropic","gemini","ollama","openai"}` near line 355 and add `"voyage"` to maintain alphabetical order:

```go
// []string{"anthropic","gemini","ollama","openai","voyage"}
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add Voyage to supported providers; list new proxy endpoints"
```

---

### Task 22: Update `docs/providers.md` with a Voyage section

**Files:**
- Modify: `docs/providers.md`

- [ ] **Step 1: Add a new section at the bottom**

Append:

```markdown
## Voyage AI

Voyage AI provides text embeddings, multimodal embeddings, and rerankers.
It does not provide chat completions; `Chat` and `ChatStream` return
`ErrNotSupported`.

### Authentication

Set `Options.APIKey` or the `VOYAGE_API_KEY` environment variable. Voyage
uses bearer-token auth.

### Models

| Model | Capabilities |
|---|---|
| `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large` | embeddings |
| `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` | embeddings |
| `voyage-multimodal-3` | embeddings, multimodal_embeddings |
| `rerank-2.5`, `rerank-2.5-lite` | reranking |

Use `client.ListModelsWithMetadata(ctx, llm.WithCapabilities(llm.ModelCapabilityReranking))`
to discover rerank-capable models programmatically.

### Per-call options

Voyage exposes per-call knobs through `voyage.Extension`:

| Field | Purpose |
|---|---|
| `InputType` | `voyage.InputTypeQuery` or `voyage.InputTypeDocument` — affects retrieval quality. |
| `OutputDimension` | 256 / 512 / 1024 / 2048 (model-dependent). |
| `Truncation` | Pointer to bool; provider default when nil. |
| `OutputDtype` | `float` / `int8` / `uint8` / `binary` / `ubinary`. |
| `ReturnDocuments` | Rerank only; include documents in the response. |

Example:

```go
import (
    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)

vecs, err := client.EmbedMultimodal(ctx, llm.MultimodalEmbedRequest{
    Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
        {Type: llm.MultimodalContentText, Text: "kittens"},
    }}},
    Extensions: []llm.ProviderExtension{voyage.Extension{
        InputType:       voyage.InputTypeQuery,
        OutputDimension: 1024,
    }},
})
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/providers.md
git commit -m "docs(providers): document Voyage AI provider"
```

---

### Task 23: Update `docs/embeddings.md` with a multimodal subsection

**Files:**
- Modify: `docs/embeddings.md`

- [ ] **Step 1: Append a Multimodal section**

Add at the bottom:

```markdown
## Multimodal Embeddings

`Client.EmbedMultimodal` returns an embedding vector for each input,
where each input may contain interleaved text and images. Today only
Voyage (`voyage-multimodal-3`) implements this; other providers return
`ErrNotSupported`.

```go
import "github.com/pgEdge/pgedge-go-llm-lib/llm"

vecs, err := client.EmbedMultimodal(ctx, llm.MultimodalEmbedRequest{
    Inputs: []llm.MultimodalInput{
        {Content: []llm.MultimodalContent{
            {Type: llm.MultimodalContentText, Text: "a photo of a kitten:"},
            {Type: llm.MultimodalContentImageURL, ImageURL: "https://example.com/kitten.jpg"},
        }},
    },
})
```

Use `client.ListModelsWithMetadata(ctx, llm.WithCapabilities(llm.ModelCapabilityMultimodalEmbeddings))`
to discover models that support this method.
```

- [ ] **Step 2: Commit**

```bash
git add docs/embeddings.md
git commit -m "docs(embeddings): document multimodal embeddings"
```

---

### Task 24: Create `docs/reranking.md` and update `mkdocs.yml`

**Files:**
- Create: `docs/reranking.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Create `docs/reranking.md`**

```markdown
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
```

- [ ] **Step 2: Add to `mkdocs.yml` navigation**

Open `mkdocs.yml` and add `- Reranking: reranking.md` to the `nav:` block, between Embeddings and Error Handling (alphabetical / logical grouping with the other capability docs).

- [ ] **Step 3: Commit**

```bash
git add docs/reranking.md mkdocs.yml
git commit -m "docs(reranking): add reranking guide and mkdocs entry"
```

---

### Task 25: Update `docs/api_reference.md`

**Files:**
- Modify: `docs/api_reference.md`

- [ ] **Step 1: Add entries for the new public API**

Append a section like:

```markdown
## Reranking and Multimodal Embeddings (v0.2+)

### `Client.Rerank(ctx, RerankRequest) (*RerankResponse, error)`

Reorders `Documents` by relevance to `Query`. Returns `ErrNotSupported`
on providers that don't support reranking.

### `Client.EmbedMultimodal(ctx, MultimodalEmbedRequest) ([][]float64, error)`

Embeds inputs containing interleaved text and image content. Returns
`ErrNotSupported` on providers that don't support multimodal embeddings.

### `ListModels` capability filter

`Client.ListModels` and `Client.ListModelsWithMetadata` accept zero or
more `ListModelsOption` values. The only built-in option is
`WithCapabilities(caps ...ModelCapability)`, which filters results to
models whose `Capabilities` field contains every listed capability.

### New `ModelCapability` constants

- `ModelCapabilityMultimodalEmbeddings`
- `ModelCapabilityReranking`

### New types

- `MultimodalContent`, `MultimodalContentType`, `MultimodalInput`,
  `MultimodalEmbedRequest`
- `RerankRequest`, `RerankResult`, `RerankResponse`
- `ListModelsConfig`, `ListModelsOption`, `WithCapabilities`,
  `FilterModelInfos`
```

- [ ] **Step 2: Commit**

```bash
git add docs/api_reference.md
git commit -m "docs(api): add Rerank, EmbedMultimodal, capability filter to API reference"
```

---

### Task 26: Update the package godoc in `llm/llm.go`

**Files:**
- Modify: `llm/llm.go` (the package comment block at lines 10–23)

- [ ] **Step 1: Replace the package doc comment**

Find:

```go
// Package llm provides a unified Go interface for chatting with
// multiple large-language-model providers (Anthropic, OpenAI, Gemini,
// and Ollama) through a single Client interface.
//
// Provider packages register themselves at import time via init().
// Import either the convenience package llm/all (for all four built-in
// providers) or individual provider packages, then call NewClient to
// construct a client.
//
// The package surface includes streaming via Stream, tool calling via
// Tool/ToolUse, multimodal images via ImageBlock/ImageURLBlock, JSON
// mode via ResponseFormat, retries via RetryConfig, and observability
// via OnRetry/Usage/Ping.
```

Replace with:

```go
// Package llm provides a unified Go interface to multiple
// large-language-model and embedding/rerank providers (Anthropic,
// OpenAI, Gemini, Ollama, Voyage) through a single Client interface.
//
// Provider packages register themselves at import time via init().
// Import the convenience package llm/all (for all five built-in
// providers) or individual provider packages, then call NewClient.
//
// The Client surface covers:
//   - Chat and streaming chat (Chat, ChatStream)
//   - Text embeddings (Embed, EmbedBatch)
//   - Multimodal embeddings (EmbedMultimodal)
//   - Reranking (Rerank)
//   - Model discovery with capability filtering (ListModels,
//     ListModelsWithMetadata, WithCapabilities)
//   - Connectivity check (Ping) and token-usage tracking (Usage)
//
// Methods unsupported by a given provider return ErrNotSupported
// (wrapped in *ProviderError) — e.g. Anthropic returns ErrNotSupported
// from Embed, and Voyage returns ErrNotSupported from Chat.
//
// Per-request provider-specific options are passed via
// ProviderExtension implementations (e.g. anthropic.Extension,
// voyage.Extension) in the request's Extensions slice.
```

- [ ] **Step 2: Commit**

```bash
git add llm/llm.go
git commit -m "docs(godoc): update package documentation for Voyage and new capabilities"
```

---

## Final verification

After all 26 tasks complete:

- [ ] **Run the full test suite**

```bash
go test ./... -v
```

Expected: all tests pass.

- [ ] **Run go vet and golangci-lint**

```bash
go vet ./...
golangci-lint run ./... # if installed locally; otherwise CI will catch
```

Expected: no warnings.

- [ ] **Build the docs locally (optional but recommended)**

```bash
pip install -r requirements.txt
mkdocs build --strict
```

Expected: clean build, no broken links.

- [ ] **Sanity-check the integration test against a live key (manual)**

If you have a Voyage API key locally:

```bash
VOYAGE_API_KEY=your-key go test ./llm/ -run TestIntegrationVoyage -v
```

Expected: PASS.

---

## CI / operational follow-up (not a code task)

A maintainer should add `VOYAGE_API_KEY` to the repository's GitHub Actions secrets so the integration test runs on PRs from trusted branches. The existing skip pattern means absence is graceful.
