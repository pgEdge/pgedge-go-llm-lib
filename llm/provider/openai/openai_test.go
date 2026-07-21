//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// newTestClient creates a client pointed at the given test server.
func newTestClient(t *testing.T, url string) llm.Client {
	t.Helper()
	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-4o",
		BaseURL: url,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	return c
}

func TestChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify method, path, headers.
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected /chat/completions, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer test-key, got %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected application/json content type, got %s", r.Header.Get("Content-Type"))
		}

		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "gpt-4o" {
			t.Errorf("expected model gpt-4o, got %v", req["model"])
		}
		msgs := req["messages"].([]any)
		if len(msgs) != 1 {
			t.Errorf("expected 1 message, got %d", len(msgs))
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello!",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected stop reason end_turn, got %s", resp.StopReason)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != "text" {
		t.Errorf("expected text type, got %s", resp.Content[0].Type)
	}
	if resp.Content[0].Text != "Hello!" {
		t.Errorf("expected Hello!, got %s", resp.Content[0].Text)
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("expected 5 completion tokens, got %d", resp.Usage.CompletionTokens)
	}
}

func TestChatWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify tools are converted to OpenAI format.
		tools := req["tools"].([]any)
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		tool := tools[0].(map[string]any)
		if tool["type"] != "function" {
			t.Errorf("expected type function, got %v", tool["type"])
		}
		fn := tool["function"].(map[string]any)
		if fn["name"] != "get_weather" {
			t.Errorf("expected name get_weather, got %v", fn["name"])
		}
		if fn["description"] != "Get weather for a location" {
			t.Errorf("expected description, got %v", fn["description"])
		}

		// Return a tool call response.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []map[string]any{
							{
								"id":   "call_123",
								"type": "function",
								"function": map[string]any{
									"name":      "get_weather",
									"arguments": `{"location":"NYC"}`,
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 10,
				"total_tokens":      25,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "What's the weather in NYC?"}}},
		},
		Tools: []llm.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather for a location",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("expected stop reason tool_use, got %s", resp.StopReason)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != "tool_use" {
		t.Errorf("expected tool_use type, got %s", resp.Content[0].Type)
	}
	if resp.Content[0].ToolUse == nil {
		t.Fatal("expected tool use to be non-nil")
	}
	if resp.Content[0].ToolUse.ID != "call_123" {
		t.Errorf("expected call_123, got %s", resp.Content[0].ToolUse.ID)
	}
	if resp.Content[0].ToolUse.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", resp.Content[0].ToolUse.Name)
	}

	var input map[string]string
	json.Unmarshal(resp.Content[0].ToolUse.Input, &input)
	if input["location"] != "NYC" {
		t.Errorf("expected NYC, got %s", input["location"])
	}
}

func TestChatToolsCompactDescription(t *testing.T) {
	var gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages:         []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
		ToolDescriptions: llm.ToolDescriptionCompact,
		Tools: []llm.Tool{
			{
				Name:               "get_weather",
				Description:        "FULL DESCRIPTION should not appear",
				CompactDescription: "compact weather desc",
				InputSchema:        json.RawMessage(`{"type":"object"}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(gotBody, "compact weather desc") {
		t.Errorf("expected compact description in body, got %s", gotBody)
	}
	if strings.Contains(gotBody, "FULL DESCRIPTION") {
		t.Errorf("did not expect full description in body, got %s", gotBody)
	}
}

func TestChatWithSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		msgs := req["messages"].([]any)
		// Per-request system prompt should be first message.
		first := msgs[0].(map[string]any)
		if first["role"] != "system" {
			t.Errorf("expected system role, got %v", first["role"])
		}
		if first["content"] != "You are a pirate." {
			t.Errorf("expected per-request system prompt, got %v", first["content"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "Ahoy!",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-4o",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hello"}}},
		},
		SystemPrompt: "You are a pirate.",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content[0].Text != "Ahoy!" {
		t.Errorf("expected Ahoy!, got %s", resp.Content[0].Text)
	}
}

func TestChatRequestSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		msgs := req["messages"].([]any)
		first := msgs[0].(map[string]any)
		if first["role"] != "system" {
			t.Errorf("expected system role, got %v", first["role"])
		}
		if first["content"] != "You are helpful." {
			t.Errorf("expected system prompt from ChatRequest, got %v", first["content"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "I can help!",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-4o",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hello"}}},
		},
		SystemPrompt: "You are helpful.",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content[0].Text != "I can help!" {
		t.Errorf("expected 'I can help!', got %s", resp.Content[0].Text)
	}
}

func TestChatAuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Invalid API key",
				"type":    "invalid_request_error",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication, got %v", err)
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("expected ProviderError")
	}
	if pe.StatusCode != 401 {
		t.Errorf("expected status 401, got %d", pe.StatusCode)
	}
}

func TestChatRateLimitError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Rate limit exceeded",
				"type":    "rate_limit_error",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("expected ErrRateLimit, got %v", err)
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/models" {
			t.Errorf("expected /models, got %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "gpt-4o"},
				{"id": "gpt-4"},
				{"id": "text-embedding-ada-002"},
				{"id": "text-embedding-3-small"},
				{"id": "tts-1"},
				{"id": "whisper-1"},
				{"id": "dall-e-3"},
				{"id": "o1-preview"},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should filter out embedding, audio, and image models.
	expected := map[string]bool{
		"gpt-4o":     true,
		"gpt-4":      true,
		"o1-preview": true,
	}
	if len(models) != len(expected) {
		t.Errorf("expected %d models, got %d: %v", len(expected), len(models), models)
	}
	for _, m := range models {
		if !expected[m] {
			t.Errorf("unexpected model: %s", m)
		}
	}
}

func TestEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected /embeddings, got %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "gpt-4o" {
			t.Errorf("expected model gpt-4o, got %v", req["model"])
		}
		input := req["input"].(string)
		if input != "hello world" {
			t.Errorf("expected 'hello world', got %s", input)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{
					"embedding": []float64{0.1, 0.2, 0.3},
					"index":     0,
				},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	embedding, err := c.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(embedding) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(embedding))
	}
	if embedding[0] != 0.1 || embedding[1] != 0.2 || embedding[2] != 0.3 {
		t.Errorf("unexpected embedding values: %v", embedding)
	}
}

func TestEmbedBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Input should be an array of strings.
		inputs := req["input"].([]any)
		if len(inputs) != 2 {
			t.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{
					"embedding": []float64{0.1, 0.2, 0.3},
					"index":     0,
				},
				{
					"embedding": []float64{0.4, 0.5, 0.6},
					"index":     1,
				},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	embeddings, err := c.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(embeddings))
	}
	if embeddings[0][0] != 0.1 {
		t.Errorf("expected 0.1, got %f", embeddings[0][0])
	}
	if embeddings[1][0] != 0.4 {
		t.Errorf("expected 0.4, got %f", embeddings[1][0])
	}
}

func TestEmbedAccumulatesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2, 0.3}, "index": 0},
			},
			"usage": map[string]any{
				"prompt_tokens": 42,
				"total_tokens":  42,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	if _, err := c.Embed(context.Background(), "hello world"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	usage := c.Usage()
	if usage.PromptTokens != 42 {
		t.Errorf("expected PromptTokens 42, got %d", usage.PromptTokens)
	}
	if usage.TotalTokens != 42 {
		t.Errorf("expected TotalTokens 42, got %d", usage.TotalTokens)
	}
	if usage.CompletionTokens != 0 {
		t.Errorf("expected CompletionTokens 0, got %d", usage.CompletionTokens)
	}
}

func TestEmbedBatchAccumulatesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2, 0.3}, "index": 0},
				{"embedding": []float64{0.4, 0.5, 0.6}, "index": 1},
			},
			"usage": map[string]any{
				"prompt_tokens": 100,
				"total_tokens":  100,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	if _, err := c.EmbedBatch(context.Background(), []string{"hello", "world"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	usage := c.Usage()
	if usage.PromptTokens != 100 {
		t.Errorf("expected PromptTokens 100, got %d", usage.PromptTokens)
	}
	if usage.TotalTokens != 100 {
		t.Errorf("expected TotalTokens 100, got %d", usage.TotalTokens)
	}
}

// embedCaptureServer builds a /embeddings test server that decodes the
// request body into captured and returns a fixed one-vector response.
func embedCaptureServer(t *testing.T, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1}, "index": 0},
			},
		})
	}))
}

func newEmbedClient(t *testing.T, url string, exts ...llm.ProviderExtension) llm.Client {
	t.Helper()
	c, err := New(llm.Options{
		APIKey:     "test-key",
		Model:      "text-embedding-3-small",
		BaseURL:    url,
		Retry:      llm.RetryConfig{Disabled: true},
		Extensions: exts,
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

func TestEmbedDimensionsSet(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	c := newEmbedClient(t, srv.URL, Extension{EmbeddingDimensions: 256})
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	got, ok := captured["dimensions"]
	if !ok {
		t.Fatalf("dimensions missing from request body: %#v", captured)
	}
	if got.(float64) != 256 {
		t.Errorf("dimensions = %v, want 256", got)
	}
}

func TestEmbedDimensionsUnsetOmitted(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	// No extension passed at all.
	c := newEmbedClient(t, srv.URL)
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if _, present := captured["dimensions"]; present {
		t.Errorf("dimensions should be omitted when extension is absent, got %#v", captured)
	}
}

func TestEmbedDimensionsZeroValueOmitted(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	// Extension present but zero — should still be omitted on the wire.
	c := newEmbedClient(t, srv.URL, Extension{})
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if _, present := captured["dimensions"]; present {
		t.Errorf("dimensions should be omitted for zero EmbeddingDimensions, got %#v", captured)
	}
}

func TestEmbedBatchDimensionsSet(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1}, "index": 0},
				{"embedding": []float64{0.2}, "index": 1},
			},
		})
	}))
	defer srv.Close()

	c := newEmbedClient(t, srv.URL, Extension{EmbeddingDimensions: 1024})
	if _, err := c.EmbedBatch(context.Background(), []string{"a", "b"}); err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	got, ok := captured["dimensions"]
	if !ok {
		t.Fatalf("dimensions missing from request body: %#v", captured)
	}
	if got.(float64) != 1024 {
		t.Errorf("dimensions = %v, want 1024", got)
	}
}

// foreignExtension is a ProviderExtension whose ProviderName is not "openai".
type foreignExtension struct{}

func (foreignExtension) ProviderName() string { return "not-openai" }

func TestEmbedIgnoresForeignExtension(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	// A foreign extension alongside a valid OpenAI one — only the OpenAI
	// one should take effect; the foreign extension is ignored.
	c := newEmbedClient(t, srv.URL,
		foreignExtension{},
		Extension{EmbeddingDimensions: 512},
	)
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	got, ok := captured["dimensions"]
	if !ok {
		t.Fatalf("dimensions missing from request body: %#v", captured)
	}
	if got.(float64) != 512 {
		t.Errorf("dimensions = %v, want 512", got)
	}
}

func TestEmbedNilExtensionEntriesIgnored(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	// A nil interface entry and a typed-nil *Extension entry must not
	// panic; the valid Extension that follows them should still take
	// effect.
	var typedNil *Extension
	c := newEmbedClient(t, srv.URL,
		nil,
		typedNil,
		Extension{EmbeddingDimensions: 384},
	)
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	got, ok := captured["dimensions"]
	if !ok {
		t.Fatalf("dimensions missing from request body: %#v", captured)
	}
	if got.(float64) != 384 {
		t.Errorf("dimensions = %v, want 384", got)
	}
}

func TestEmbedPointerExtension(t *testing.T) {
	var captured map[string]any
	srv := embedCaptureServer(t, &captured)
	defer srv.Close()

	// findExtension must accept *Extension as well as Extension.
	ext := &Extension{EmbeddingDimensions: 768}
	c := newEmbedClient(t, srv.URL, ext)
	if _, err := c.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	got, ok := captured["dimensions"]
	if !ok {
		t.Fatalf("dimensions missing from request body: %#v", captured)
	}
	if got.(float64) != 768 {
		t.Errorf("dimensions = %v, want 768", got)
	}
}

func TestCumulativeUsage(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "response",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)

	// Initial usage should be zero.
	usage := c.Usage()
	if usage.TotalTokens != 0 {
		t.Errorf("expected 0 total tokens initially, got %d", usage.TotalTokens)
	}

	// Make first request.
	c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	usage = c.Usage()
	if usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 5 {
		t.Errorf("expected 5 completion tokens, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", usage.TotalTokens)
	}

	// Make second request - usage should accumulate.
	c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hello again"}}}},
	})
	usage = c.Usage()
	if usage.PromptTokens != 20 {
		t.Errorf("expected 20 prompt tokens, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 10 {
		t.Errorf("expected 10 completion tokens, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 30 {
		t.Errorf("expected 30 total tokens, got %d", usage.TotalTokens)
	}
}

func TestProviderAndModel(t *testing.T) {
	c, err := New(llm.Options{
		APIKey: "test-key",
		Model:  "gpt-4o",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	if c.Provider() != "openai" {
		t.Errorf("expected provider openai, got %s", c.Provider())
	}
	if c.Model() != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", c.Model())
	}
}

func TestChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["stream"] != true {
			t.Errorf("expected stream: true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)

		// Send text chunks.
		chunks := []string{
			`{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		}
		for _, chunk := range chunks {
			w.Write([]byte("data: " + chunk + "\n\n"))
			flusher.Flush()
		}
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var textParts []string
	var gotDone bool
	for chunk := range stream.Chunks {
		switch chunk.Type {
		case llm.ChunkText:
			textParts = append(textParts, chunk.Text)
		case llm.ChunkDone:
			gotDone = true
			if chunk.Usage == nil {
				t.Error("expected usage on done chunk")
			} else {
				if chunk.Usage.PromptTokens != 5 {
					t.Errorf("expected 5 prompt tokens, got %d", chunk.Usage.PromptTokens)
				}
			}
		}
	}

	// Check for errors.
	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	fullText := strings.Join(textParts, "")
	if fullText != "Hello world" {
		t.Errorf("expected 'Hello world', got %s", fullText)
	}
	if !gotDone {
		t.Error("did not receive done chunk")
	}
}

func TestChatStreamWithToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]},"finish_reason":null}]}`,
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}`,
		}
		for _, chunk := range chunks {
			w.Write([]byte("data: " + chunk + "\n\n"))
			flusher.Flush()
		}
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Weather in NYC?"}}}},
		Tools: []llm.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotToolStart bool
	var gotToolDelta bool
	var gotDone bool
	for chunk := range stream.Chunks {
		switch chunk.Type {
		case llm.ChunkToolUseStart:
			gotToolStart = true
			if chunk.ToolUse == nil {
				t.Error("expected tool use on tool_use_start")
			} else {
				if chunk.ToolUse.Name != "get_weather" {
					t.Errorf("expected get_weather, got %s", chunk.ToolUse.Name)
				}
				if chunk.ToolUse.ID != "call_abc" {
					t.Errorf("expected call_abc, got %s", chunk.ToolUse.ID)
				}
			}
		case llm.ChunkToolUseDelta:
			gotToolDelta = true
			if chunk.Partial == "" {
				t.Errorf("ChunkToolUseDelta: Partial must be non-empty")
			}
			if chunk.Text != "" {
				t.Errorf("ChunkToolUseDelta: Text must be empty (use Partial), got %q", chunk.Text)
			}
		case llm.ChunkDone:
			gotDone = true
		}
	}

	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	if !gotToolStart {
		t.Error("did not receive tool_use_start")
	}
	if !gotToolDelta {
		t.Error("did not receive tool_use_delta")
	}
	if !gotDone {
		t.Error("did not receive done chunk")
	}
}

func TestMaxCompletionTokensForNewModels(t *testing.T) {
	// When the ResponsesAPI override is set to false, o1/o3/gpt-5 models
	// stay on /v1/chat/completions and must use max_completion_tokens
	// instead of max_tokens.
	for _, model := range []string{"o1-preview", "o3-mini", "gpt-5"} {
		t.Run(model, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/chat/completions" {
					t.Errorf("expected /chat/completions, got %s", r.URL.Path)
				}
				body, _ := io.ReadAll(r.Body)
				var req map[string]any
				json.Unmarshal(body, &req)

				if _, ok := req["max_tokens"]; ok {
					t.Error("should not have max_tokens for this model")
				}
				if _, ok := req["max_completion_tokens"]; !ok {
					t.Error("expected max_completion_tokens for this model")
				}

				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{
						{
							"message": map[string]any{
								"role":    "assistant",
								"content": "ok",
							},
							"finish_reason": "stop",
						},
					},
					"usage": map[string]any{
						"prompt_tokens":     1,
						"completion_tokens": 1,
						"total_tokens":      2,
					},
				})
			}))
			defer srv.Close()

			c, _ := New(llm.Options{
				APIKey:     "test-key",
				Model:      model,
				BaseURL:    srv.URL,
				Extensions: []llm.ProviderExtension{Extension{ResponsesAPI: llm.Bool(false)}},
			})
			c.Chat(context.Background(), llm.ChatRequest{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
			})
		})
	}
}

// ---------- convertAssistantMessage ----------

func TestConvertAssistantMessage_ContentBlocks(t *testing.T) {
	input := json.RawMessage(`{"location":"NYC"}`)
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "Here is the result."},
			{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
				ID:    "call_1",
				Name:  "get_weather",
				Input: input,
			}},
		},
	}
	msg := convertAssistantMessage(m)

	if msg.Role != "assistant" {
		t.Errorf("expected role assistant, got %s", msg.Role)
	}
	if msg.Content != "Here is the result." {
		t.Errorf("expected content 'Here is the result.', got %v", msg.Content)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	tc := msg.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("expected ID call_1, got %s", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("expected type function, got %s", tc.Type)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected name get_weather, got %s", tc.Function.Name)
	}
	if tc.Function.Arguments != string(input) {
		t.Errorf("expected arguments %s, got %s", string(input), tc.Function.Arguments)
	}
}

func TestConvertAssistantMessage_TextBlocksConcatenated(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "Hello "},
			{Type: llm.BlockText, Text: "world"},
		},
	}
	msg := convertAssistantMessage(m)
	if msg.Content != "Hello world" {
		t.Errorf("unexpected content: %v", msg.Content)
	}
}

func TestConvertAssistantMessage_ToolUseOnly(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
				ID:    "tc_1",
				Name:  "do_thing",
				Input: json.RawMessage(`{"x":1}`),
			}},
		},
	}
	msg := convertAssistantMessage(m)
	// No text blocks: Content should be nil so the wire form sends
	// `"content": null`, which OpenAI accepts when tool_calls is set.
	if msg.Content != nil {
		t.Errorf("expected nil content, got %v", msg.Content)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
}

func TestConvertAssistantMessage_NilToolUseSkipped(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ToolUse: nil},
			{Type: llm.BlockText, Text: "only text"},
		},
	}
	msg := convertAssistantMessage(m)
	if msg.Content != "only text" {
		t.Errorf("expected 'only text', got %v", msg.Content)
	}
	if len(msg.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(msg.ToolCalls))
	}
}

// ---------- convertUserMessage ----------

func TestConvertUserMessage_TextOnlyUsesLegacyForm(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "What is the time?"},
		},
	}
	msg := convertUserMessage(m)
	// Text-only content uses the legacy "string" wire form for
	// maximum compatibility with older OpenAI-compatible servers.
	if s, ok := msg.Content.(string); !ok || s != "What is the time?" {
		t.Errorf("expected string content, got %T %v", msg.Content, msg.Content)
	}
}

func TestConvertUserMessage_TextAndImageUsesArrayForm(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "Describe this:"},
			{Type: llm.BlockImage, Image: &llm.ImageContent{
				Data:      []byte{0x89, 0x50},
				MediaType: "image/png",
			}},
		},
	}
	msg := convertUserMessage(m)
	parts, ok := msg.Content.([]openaiContentPart)
	if !ok {
		t.Fatalf("expected []openaiContentPart, got %T", msg.Content)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "Describe this:" {
		t.Errorf("text part: %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("image part: %+v", parts[1])
	}
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("expected data URL, got %q", parts[1].ImageURL.URL)
	}
}

func TestConvertUserMessage_ImageURLPassthrough(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockImage, Image: &llm.ImageContent{URL: "https://example.com/cat.png"}},
		},
	}
	msg := convertUserMessage(m)
	parts := msg.Content.([]openaiContentPart)
	if parts[0].ImageURL.URL != "https://example.com/cat.png" {
		t.Errorf("URL not passed through: %q", parts[0].ImageURL.URL)
	}
}

// ---------- convertToolMessage ----------

func TestConvertToolMessage_OneMessagePerResult(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "call_abc", Text: "result text"},
			{Type: llm.BlockToolResult, ToolUseID: "call_xyz", Text: "second result"},
		},
	}
	msgs := convertToolMessage(m)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 OpenAI messages, got %d", len(msgs))
	}
	for i, want := range []struct {
		id, content string
	}{
		{"call_abc", "result text"},
		{"call_xyz", "second result"},
	} {
		if msgs[i].Role != "tool" {
			t.Errorf("[%d] role = %q, want tool", i, msgs[i].Role)
		}
		if msgs[i].ToolCallID != want.id {
			t.Errorf("[%d] tool_call_id = %q, want %q", i, msgs[i].ToolCallID, want.id)
		}
		if c, ok := msgs[i].Content.(string); !ok || c != want.content {
			t.Errorf("[%d] content = %v, want %q", i, msgs[i].Content, want.content)
		}
	}
}

func TestConvertToolMessage_NoToolResultBlocksEmits(t *testing.T) {
	m := llm.Message{
		Role:    llm.RoleTool,
		Content: []llm.ContentBlock{},
	}
	msgs := convertToolMessage(m)
	if len(msgs) != 0 {
		t.Errorf("expected 0 OpenAI messages for empty tool message, got %d", len(msgs))
	}
}

// ---------- convertMessage ----------

func TestConvertMessage_RoleDispatch(t *testing.T) {
	cases := []struct {
		role    llm.Role
		want    string
		wantLen int
	}{
		{llm.RoleAssistant, "assistant", 1},
		{llm.RoleUser, "user", 1},
		{llm.RoleSystem, "system", 1},
		{llm.RoleTool, "tool", 0}, // empty content -> 0 messages
	}
	for _, c := range cases {
		msgs := convertMessage(llm.Message{Role: c.role})
		if len(msgs) != c.wantLen {
			t.Errorf("role %q -> %d messages, want %d", c.role, len(msgs), c.wantLen)
			continue
		}
		if c.wantLen == 0 {
			continue
		}
		if msgs[0].Role != c.want {
			t.Errorf("role %q -> wire role %q, want %q", c.role, msgs[0].Role, c.want)
		}
	}
}

// ---------- normalizeStopReason ----------

func TestNormalizeStopReason_Length(t *testing.T) {
	result := normalizeStopReason("length")
	if result != llm.StopReasonMaxTokens {
		t.Errorf("expected max_tokens, got %s", result)
	}
}

func TestNormalizeStopReason_Unknown(t *testing.T) {
	result := normalizeStopReason("some_unknown_reason")
	if result != llm.StopReasonEndTurn {
		t.Errorf("expected end_turn fallback for unknown reason, got %s", result)
	}
}

func TestNormalizeStopReason_Empty(t *testing.T) {
	result := normalizeStopReason("")
	if result != llm.StopReasonEndTurn {
		t.Errorf("expected end_turn fallback for empty reason, got %s", result)
	}
}

func TestChatStreamDoneAlwaysHasUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Stream a single text delta + a finish_reason, but DO NOT include a usage chunk.
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: server.URL,
		Model:   "test",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}

	var done llm.StreamChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if chunk.Type == llm.ChunkDone {
			done = chunk
		}
	}
	if done.Type != llm.ChunkDone {
		t.Fatal("never received done chunk")
	}
	if done.Usage == nil {
		t.Fatal("done chunk has nil Usage; expected zero TokenUsage")
	}
}

// ---------- mapError ----------

func TestMapError_400InvalidRequest(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": "bad request details",
			"type":    "invalid_request_error",
		},
	})
	err := mapError(400, body)

	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("expected ProviderError")
	}
	if pe.StatusCode != 400 {
		t.Errorf("expected status 400, got %d", pe.StatusCode)
	}
	if pe.Message != "bad request details" {
		t.Errorf("expected 'bad request details', got %s", pe.Message)
	}
	if pe.Provider != "openai" {
		t.Errorf("expected provider openai, got %s", pe.Provider)
	}
}

func TestMapError_500ProviderError(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": "internal server error",
			"type":    "server_error",
		},
	})
	err := mapError(500, body)

	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("expected ProviderError")
	}
	if pe.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", pe.StatusCode)
	}
}

func TestMapError_403Forbidden(t *testing.T) {
	err := mapError(403, []byte(`{"error":{"message":"forbidden"}}`))

	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication for 403, got %v", err)
	}
}

func TestMapError_EmptyBody(t *testing.T) {
	// When body cannot be parsed, message falls back to "HTTP <status>".
	err := mapError(503, []byte(""))

	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("expected ProviderError")
	}
	if pe.Message != "HTTP 503" {
		t.Errorf("expected 'HTTP 503', got %s", pe.Message)
	}
}

func TestExplicitZeroTemperatureReachesWire(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "test",
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	zero := 0.0
	_, err = c.Chat(context.Background(), llm.ChatRequest{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
		Temperature: &zero,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	temp, ok := captured["temperature"]
	if !ok {
		t.Fatalf("temperature missing from wire body: %+v", captured)
	}
	tempF, ok := temp.(float64)
	if !ok {
		t.Fatalf("temperature is %T, not float64: %v", temp, temp)
	}
	if tempF != 0.0 {
		t.Errorf("temperature on wire = %v, want 0", tempF)
	}
}

// TestUnsetTemperatureOmittedFromWire is a regression test: Options
// used to default Temperature to 0.7 when unset, which some models
// reject outright. When neither Options nor ChatRequest set
// Temperature, it must now be omitted from the wire entirely.
func TestUnsetTemperatureOmittedFromWire(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "test",
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if temp, present := captured["temperature"]; present {
		t.Errorf("temperature should be omitted from wire when unset, got %v", temp)
	}
}

func TestNewClientRejectsInvalidBaseURL(t *testing.T) {
	_, err := New(llm.Options{
		APIKey:  "k",
		Model:   "test",
		BaseURL: "not-a-valid-url",
	})
	if err == nil {
		t.Fatal("want error for invalid BaseURL")
	}
	if !strings.Contains(err.Error(), "openai") {
		t.Errorf("error should mention provider name: %v", err)
	}
}

func TestNewClientUsesInjectedHTTPClient(t *testing.T) {
	var hits int32
	custom := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			atomic.AddInt32(&hits, 1)
			return &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)),
				Header:     make(http.Header),
			}, nil
		}),
	}

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:     "k",
		Model:      "test",
		HTTPClient: custom,
		Retry:      llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if atomic.LoadInt32(&hits) != 1 {
		t.Errorf("custom transport not called: hits=%d", hits)
	}
}

func TestListModelsWithMetadata(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-4o-mini"},{"id":"gpt-4o"},{"id":"unknown-model-x"}]}`))
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "test",
		Retry:   llm.RetryConfig{Disabled: true},
	})
	models, err := c.ListModelsWithMetadata(context.Background())
	if err != nil {
		t.Fatalf("ListModelsWithMetadata: %v", err)
	}

	byID := map[string]llm.ModelInfo{}
	for _, m := range models {
		byID[m.ID] = m
	}

	// gpt-4o-mini should have Vision capability.
	if !hasCapability(byID["gpt-4o-mini"].Capabilities, llm.ModelCapabilityVision) {
		t.Errorf("gpt-4o-mini missing Vision: %v", byID["gpt-4o-mini"].Capabilities)
	}
	// Unknown model should fall back to [Chat, Streaming].
	if !hasCapability(byID["unknown-model-x"].Capabilities, llm.ModelCapabilityChat) {
		t.Errorf("unknown model missing fallback Chat: %v", byID["unknown-model-x"].Capabilities)
	}
}

func hasCapability(caps []llm.ModelCapability, want llm.ModelCapability) bool {
	for _, c := range caps {
		if c == want {
			return true
		}
	}
	return false
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestRequestTimeoutCancelsLongRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:         "k",
		BaseURL:        srv.URL,
		Model:          "test",
		RequestTimeout: 100 * time.Millisecond,
		Retry:          llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	// http.Client.Timeout produces a url.Error wrapping a "deadline exceeded"-ish
	// error from the underlying transport. The exact error type may vary, so
	// just assert that an error came back and the elapsed time is short.
}

func TestChatToolChoice(t *testing.T) {
	cases := []struct {
		name      string
		choice    llm.ToolChoice
		wantField any // nil means "check map shape manually"
	}{
		{"auto", llm.ToolChoice{Mode: llm.ToolChoiceAuto}, "auto"},
		{"none", llm.ToolChoice{Mode: llm.ToolChoiceNone}, "none"},
		{"required", llm.ToolChoice{Mode: llm.ToolChoiceRequired}, "required"},
		{"specific", llm.ToolChoice{Mode: llm.ToolChoiceSpecific, Name: "search"}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &captured)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
			}))
			defer srv.Close()

			c, err := New(llm.Options{
				APIKey:  "k",
				BaseURL: srv.URL,
				Model:   "test",
				Retry:   llm.RetryConfig{Disabled: true},
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			cp := tc.choice
			_, _ = c.Chat(context.Background(), llm.ChatRequest{
				Messages:   []llm.Message{llm.UserText("hi")},
				ToolChoice: &cp,
			})

			tcWire, ok := captured["tool_choice"]
			if !ok {
				t.Fatalf("tool_choice missing from wire body: %+v", captured)
			}
			if tc.name == "specific" {
				m, ok := tcWire.(map[string]any)
				if !ok {
					t.Fatalf("tool_choice should be map for specific, got %T: %v", tcWire, tcWire)
				}
				if m["type"] != "function" {
					t.Errorf("type = %v, want function", m["type"])
				}
				fn, ok := m["function"].(map[string]any)
				if !ok {
					t.Fatalf("function key missing or wrong type: %+v", m)
				}
				if fn["name"] != "search" {
					t.Errorf("name = %v, want search", fn["name"])
				}
			} else {
				if tcWire != tc.wantField {
					t.Errorf("tool_choice = %v, want %v", tcWire, tc.wantField)
				}
			}
		})
	}
}

func TestChatJSONResponseFormat(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{}","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "test",
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, err = c.Chat(context.Background(), llm.ChatRequest{
		Messages:       []llm.Message{llm.UserText("hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSON},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	rf, ok := captured["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format missing or wrong type: %+v", captured)
	}
	if rf["type"] != "json_object" {
		t.Errorf("response_format.type = %v, want json_object", rf["type"])
	}
}

func TestChatStopSequences(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		APIKey: "k", BaseURL: srv.URL, Model: "test", Retry: llm.RetryConfig{Disabled: true},
	})
	_, _ = c.Chat(context.Background(), llm.ChatRequest{
		Messages:      []llm.Message{llm.UserText("hi")},
		StopSequences: []string{"END", "STOP"},
	})
	stops, ok := captured["stop"].([]any)
	if !ok || len(stops) != 2 {
		t.Fatalf("stop missing or wrong length: %+v", captured["stop"])
	}
	if stops[0] != "END" || stops[1] != "STOP" {
		t.Errorf("stops = %v", stops)
	}
}

func TestPing(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "gpt-4o"}}})
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		if err := c.Ping(context.Background()); err != nil {
			t.Errorf("Ping returned %v, want nil", err)
		}
	})
	t.Run("propagates upstream error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		err := c.Ping(context.Background())
		if err == nil {
			t.Fatal("expected error from Ping, got nil")
		}
		if !errors.Is(err, llm.ErrAuthentication) {
			t.Errorf("err = %v, want ErrAuthentication", err)
		}
	})
}

func TestResetUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "hi"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	if _, err := c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := c.Usage(); got.TotalTokens != 15 {
		t.Fatalf("Usage().TotalTokens = %d, want 15 (test setup)", got.TotalTokens)
	}
	c.ResetUsage()
	if got := c.Usage(); got != (llm.TokenUsage{}) {
		t.Errorf("Usage() after ResetUsage = %+v, want zero value", got)
	}
}

// TestChatRejectsDocumentBlock verifies that a request carrying a
// BlockDocument is rejected with ErrNotSupported before the OpenAI
// upstream is contacted. The test server fails the test if it
// receives any request.
func TestChatRejectsDocumentBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Fatal("upstream must not be contacted when a document block is present")
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)

	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			llm.UserBlocks(
				llm.TextBlock("Summarise:"),
				llm.DocumentBlock([]byte("%PDF-1.4"), "application/pdf", "doc.pdf"),
			),
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", err)
	}
}

// TestChatStreamRejectsDocumentBlock mirrors TestChatRejectsDocumentBlock
// for the streaming path.
func TestChatStreamRejectsDocumentBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Fatal("upstream must not be contacted when a document block is present")
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)

	_, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			llm.UserBlocks(llm.DocumentBlock([]byte("%PDF"), "application/pdf", "")),
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", err)
	}
}

func TestRerankUnsupported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	_, err := c.Rerank(context.Background(), llm.RerankRequest{Query: "q", Documents: []string{"a"}})
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported, got %v", err)
	}
}

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

func TestListModelsWithMetadataEmbeddingDimensions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[
			{"id":"text-embedding-3-small"},
			{"id":"text-embedding-3-large"},
			{"id":"text-embedding-ada-002"},
			{"id":"gpt-4o"}
		]}`))
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	infos, err := c.ListModelsWithMetadata(context.Background(),
		llm.WithCapabilities(llm.ModelCapabilityEmbeddings))
	if err != nil {
		t.Fatal(err)
	}

	byID := make(map[string]llm.ModelInfo, len(infos))
	for _, info := range infos {
		byID[info.ID] = info
	}

	cases := []struct {
		id   string
		want int
	}{
		{"text-embedding-3-small", 1536},
		{"text-embedding-3-large", 3072},
		{"text-embedding-ada-002", 1536},
	}
	for _, tc := range cases {
		info, ok := byID[tc.id]
		if !ok {
			t.Errorf("model %q not found in results", tc.id)
			continue
		}
		if info.Dimensions != tc.want {
			t.Errorf("model %q: Dimensions = %d, want %d", tc.id, info.Dimensions, tc.want)
		}
	}
}
