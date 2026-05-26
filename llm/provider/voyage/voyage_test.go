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
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	voyage "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, llm.Client) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := llm.NewClient("voyage", llm.Options{
		APIKey:  "test-key",
		Model:   "voyage-3.5-lite",
		BaseURL: srv.URL, // ValidateBaseURL trims any trailing slash
		Retry:   llm.RetryConfig{Disabled: true},
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

func TestProviderAndModel(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
	if got := c.Provider(); got != "voyage" {
		t.Errorf("Provider() = %q, want %q", got, "voyage")
	}
	if got := c.Model(); got != "voyage-3.5-lite" {
		t.Errorf("Model() = %q, want %q", got, "voyage-3.5-lite")
	}
}

func TestResetUsage(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"embedding":[0.1],"index":0}],"model":"voyage-3.5-lite","usage":{"total_tokens":7}}`))
	})
	if _, err := c.Embed(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	if got := c.Usage().TotalTokens; got != 7 {
		t.Fatalf("usage before reset = %d, want 7", got)
	}
	c.ResetUsage()
	if got := c.Usage().TotalTokens; got != 0 {
		t.Errorf("usage after reset = %d, want 0", got)
	}
}

func TestPingSuccess(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"embedding":[0.1],"index":0}],"model":"voyage-3.5-lite","usage":{"total_tokens":1}}`))
	})
	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("Ping() = %v, want nil", err)
	}
}

func TestPingAuthError(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	})
	err := c.Ping(context.Background())
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("Ping() = %v, want ErrAuthentication", err)
	}
}

func TestPingOther4xxReachable(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("Ping() should treat non-auth 4xx as reachable, got %v", err)
	}
}

func TestEmbed429RateLimit(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	})
	_, err := c.Embed(context.Background(), "x")
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("expected ErrRateLimit, got %v", err)
	}
}

func TestEmbed5xxProviderError(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	_, err := c.Embed(context.Background(), "x")
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
}

func TestEmbedMissingModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(srv.Close)
	c, err := llm.NewClient("voyage", llm.Options{APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Embed(context.Background(), "x")
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest when Model is empty, got %v", err)
	}
}

func TestEmbedMissingIndex(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"embedding":[0.1],"index":0}],"model":"voyage-3.5-lite","usage":{"total_tokens":2}}`))
	})
	_, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError for missing embedding, got %v", err)
	}
}

func TestEmbedIndexOutOfRange(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"embedding":[0.1],"index":5}],"model":"voyage-3.5-lite","usage":{"total_tokens":1}}`))
	})
	_, err := c.Embed(context.Background(), "x")
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError for out-of-range index, got %v", err)
	}
}

func TestEmbedMultimodalUnknownContentType(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be called when content type is invalid")
	})
	_, err := c.EmbedMultimodal(context.Background(), llm.MultimodalEmbedRequest{
		Inputs: []llm.MultimodalInput{{Content: []llm.MultimodalContent{
			{Type: "bogus"},
		}}},
	})
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest for unknown content type, got %v", err)
	}
}

func TestNewWithEnvAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	c, err := llm.NewClient("voyage", llm.Options{Model: "voyage-3.5-lite"})
	if err != nil {
		t.Fatalf("expected env-var fallback to populate APIKey, got %v", err)
	}
	if c.Provider() != "voyage" {
		t.Errorf("Provider() = %q", c.Provider())
	}
}

func TestNewInvalidBaseURL(t *testing.T) {
	_, err := llm.NewClient("voyage", llm.Options{APIKey: "k", BaseURL: "::not-a-url::"})
	if err == nil {
		t.Fatal("expected error for malformed base URL")
	}
}

// otherProviderExt is a ProviderExtension whose ProviderName != "voyage".
type otherProviderExt struct{}

func (otherProviderExt) ProviderName() string { return "not-voyage" }

func TestRerankIgnoresForeignExtension(t *testing.T) {
	// Confirms findExtension's "wrong-provider" branch: passing an extension
	// for a different provider must not affect the wire request.
	var captured map[string]any
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[],"model":"rerank-2.5","usage":{"total_tokens":0}}`))
	})
	_, err := c.Rerank(context.Background(), llm.RerankRequest{
		Query: "q", Documents: []string{"a"},
		Extensions: []llm.ProviderExtension{otherProviderExt{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, present := captured["return_documents"]; present {
		t.Errorf("foreign extension should not have set return_documents; body=%v", captured)
	}
}

func TestRerankWithPointerExtension(t *testing.T) {
	// Exercises findExtension's *Extension type-assertion branch.
	var captured map[string]any
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[],"model":"rerank-2.5","usage":{"total_tokens":0}}`))
	})
	tru := true
	_, err := c.Rerank(context.Background(), llm.RerankRequest{
		Query: "q", Documents: []string{"a"},
		Extensions: []llm.ProviderExtension{&voyage.Extension{ReturnDocuments: &tru}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := captured["return_documents"].(bool); !ok || !got {
		t.Errorf("expected return_documents=true on wire, got %v", captured["return_documents"])
	}
}

func TestEmbedNetworkError(t *testing.T) {
	// Close the server immediately so the next request fails with a
	// transport-level error. This exercises postJSON's `err != nil`
	// branch (status == 0) and propagates up through embed and Embed.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()
	c, err := llm.NewClient("voyage", llm.Options{
		APIKey:  "test-key",
		Model:   "voyage-3.5-lite",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Embed(context.Background(), "x"); err == nil {
		t.Fatal("expected network error")
	}
}

func TestNewWithOnRetryHook(t *testing.T) {
	// Triggers the OnRetry-hook wrapping branch in New. The hook itself
	// is invoked when the retry layer fires (e.g., on 5xx). We don't
	// assert hook invocation — just that construction succeeds with the
	// hook wired up.
	c, err := llm.NewClient("voyage", llm.Options{
		APIKey:  "k",
		Model:   "voyage-3.5-lite",
		OnRetry: func(e llm.RetryEvent) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if c == nil {
		t.Fatal("expected non-nil client")
	}
}
