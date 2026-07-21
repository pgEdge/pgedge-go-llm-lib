//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// newTestClient creates a client pointed at the given test server.
func newTestClient(t *testing.T, url string) llm.Client {
	t.Helper()
	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gemini-2.0-flash",
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
		// Verify method and path.
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		expectedPath := "/v1beta/models/gemini-2.0-flash:generateContent"
		if r.URL.Path != expectedPath {
			t.Errorf("expected %s, got %s", expectedPath, r.URL.Path)
		}

		// Verify API key passed as x-goog-api-key header (not in URL).
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("expected x-goog-api-key=test-key header, got %q", got)
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("expected no key in URL query, got %s", r.URL.Query().Get("key"))
		}
		if r.Header.Get("Authorization") != "" {
			t.Errorf("expected no Authorization header, got %s", r.Header.Get("Authorization"))
		}

		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		contents := req["contents"].([]any)
		if len(contents) != 1 {
			t.Errorf("expected 1 content, got %d", len(contents))
		}
		content := contents[0].(map[string]any)
		if content["role"] != "user" {
			t.Errorf("expected role user, got %v", content["role"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "Hello!"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     10,
				"candidatesTokenCount": 5,
				"totalTokenCount":      15,
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
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify tools are converted to Gemini format.
		tools := req["tools"].([]any)
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool group, got %d", len(tools))
		}
		tool := tools[0].(map[string]any)
		funcDecls := tool["functionDeclarations"].([]any)
		if len(funcDecls) != 1 {
			t.Fatalf("expected 1 function declaration, got %d", len(funcDecls))
		}
		fn := funcDecls[0].(map[string]any)
		if fn["name"] != "get_weather" {
			t.Errorf("expected name get_weather, got %v", fn["name"])
		}
		if fn["description"] != "Get weather for a location" {
			t.Errorf("expected description, got %v", fn["description"])
		}

		// Return a function call response.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{
								"functionCall": map[string]any{
									"name": "get_weather",
									"args": map[string]any{
										"location": "NYC",
									},
								},
							},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     15,
				"candidatesTokenCount": 10,
				"totalTokenCount":      25,
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
	if len(resp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != "tool_use" {
		t.Errorf("expected tool_use type, got %s", resp.Content[0].Type)
	}
	if resp.Content[0].ToolUse == nil {
		t.Fatal("expected tool use to be non-nil")
	}
	if resp.Content[0].ToolUse.ID != "gemini-tool-get_weather" {
		t.Errorf("expected gemini-tool-get_weather, got %s", resp.Content[0].ToolUse.ID)
	}
	if resp.Content[0].ToolUse.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", resp.Content[0].ToolUse.Name)
	}

	var input map[string]any
	json.Unmarshal(resp.Content[0].ToolUse.Input, &input)
	if input["location"] != "NYC" {
		t.Errorf("expected NYC, got %v", input["location"])
	}
}

func TestChatToolsCompactDescription(t *testing.T) {
	var gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"role": "model", "parts": []map[string]any{{"text": "ok"}}}, "finishReason": "STOP"},
			},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
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

func TestRoleMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		contents := req["contents"].([]any)
		// Should have 2 messages: user and model (mapped from assistant).
		if len(contents) != 2 {
			t.Fatalf("expected 2 contents, got %d", len(contents))
		}

		first := contents[0].(map[string]any)
		if first["role"] != "user" {
			t.Errorf("expected role user, got %v", first["role"])
		}

		second := contents[1].(map[string]any)
		if second["role"] != "model" {
			t.Errorf("expected role model (mapped from assistant), got %v", second["role"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "OK"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     5,
				"candidatesTokenCount": 1,
				"totalTokenCount":      6,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hello!"}}},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/v1beta/models" {
			t.Errorf("expected /v1beta/models, got %s", r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("expected x-goog-api-key=test-key header, got %q", got)
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("expected no key in URL query, got %s", r.URL.Query().Get("key"))
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{
					"name":                       "models/gemini-2.0-flash",
					"supportedGenerationMethods": []string{"generateContent", "countTokens"},
				},
				{
					"name":                       "models/gemini-1.5-pro",
					"supportedGenerationMethods": []string{"generateContent", "countTokens"},
				},
				{
					"name":                       "models/text-embedding-004",
					"supportedGenerationMethods": []string{"embedContent"},
				},
				{
					"name":                       "models/aqa",
					"supportedGenerationMethods": []string{"generateAnswer"},
				},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should only include models that support generateContent,
	// with "models/" prefix stripped.
	expected := map[string]bool{
		"gemini-2.0-flash": true,
		"gemini-1.5-pro":   true,
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
		expectedPath := "/v1beta/models/gemini-2.0-flash:embedContent"
		if r.URL.Path != expectedPath {
			t.Errorf("expected %s, got %s", expectedPath, r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("expected x-goog-api-key=test-key header, got %q", got)
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("expected no key in URL query, got %s", r.URL.Query().Get("key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		content := req["content"].(map[string]any)
		parts := content["parts"].([]any)
		part := parts[0].(map[string]any)
		if part["text"] != "hello world" {
			t.Errorf("expected 'hello world', got %v", part["text"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{
				"values": []float64{0.1, 0.2, 0.3},
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

func TestEmbedAccumulatesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{
				"values": []float64{0.1, 0.2, 0.3},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount": 11,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	if _, err := c.Embed(context.Background(), "hello world"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	usage := c.Usage()
	if usage.PromptTokens != 11 {
		t.Errorf("expected PromptTokens 11, got %d", usage.PromptTokens)
	}
	// Embeddings have no completion tokens, so promptTokenCount is the total.
	if usage.TotalTokens != 11 {
		t.Errorf("expected TotalTokens 11, got %d", usage.TotalTokens)
	}
	if usage.CompletionTokens != 0 {
		t.Errorf("expected CompletionTokens 0, got %d", usage.CompletionTokens)
	}
}

func TestEmbedBatchAccumulatesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2, 0.3}},
				{"values": []float64{0.4, 0.5, 0.6}},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount": 22,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	if _, err := c.EmbedBatch(context.Background(), []string{"hello", "world"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	usage := c.Usage()
	if usage.PromptTokens != 22 {
		t.Errorf("expected PromptTokens 22, got %d", usage.PromptTokens)
	}
	if usage.TotalTokens != 22 {
		t.Errorf("expected TotalTokens 22, got %d", usage.TotalTokens)
	}
}

func TestChatAuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "API key not valid",
				"status":  "PERMISSION_DENIED",
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
	if pe.StatusCode != 403 {
		t.Errorf("expected status 403, got %d", pe.StatusCode)
	}
}

func TestChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify streaming endpoint.
		expectedPath := "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
		if r.URL.Path != expectedPath {
			t.Errorf("expected %s, got %s", expectedPath, r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("expected alt=sse in query, got %s", r.URL.Query().Get("alt"))
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("expected x-goog-api-key=test-key header, got %q", got)
		}
		if r.URL.Query().Get("key") != "" {
			t.Errorf("expected no key in URL query, got %s", r.URL.Query().Get("key"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}],"usageMetadata":{}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}],"usageMetadata":{}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
		}
		for _, chunk := range chunks {
			w.Write([]byte("data: " + chunk + "\n\n"))
			flusher.Flush()
		}
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
				if chunk.Usage.CompletionTokens != 3 {
					t.Errorf("expected 3 completion tokens, got %d", chunk.Usage.CompletionTokens)
				}
			}
		}
	}

	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	fullText := strings.Join(textParts, "")
	if fullText != "Hello world!" {
		t.Errorf("expected 'Hello world!', got %s", fullText)
	}
	if !gotDone {
		t.Error("did not receive done chunk")
	}
}

func TestSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// System prompt should be in systemInstruction, not in contents.
		sysInst := req["systemInstruction"].(map[string]any)
		parts := sysInst["parts"].([]any)
		part := parts[0].(map[string]any)
		if part["text"] != "You are a pirate." {
			t.Errorf("expected per-request system prompt, got %v", part["text"])
		}

		// Contents should not contain a system message.
		contents := req["contents"].([]any)
		for _, c := range contents {
			content := c.(map[string]any)
			if content["role"] == "system" {
				t.Error("system role should not be in contents")
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "Ahoy!"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     10,
				"candidatesTokenCount": 5,
				"totalTokenCount":      15,
			},
		})
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gemini-2.0-flash",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Per-request SystemPrompt is the only source of system instructions.
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

func TestCumulativeUsage(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "response"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     10,
				"candidatesTokenCount": 5,
				"totalTokenCount":      15,
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
		Model:  "gemini-2.0-flash",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	if c.Provider() != "gemini" {
		t.Errorf("expected provider gemini, got %s", c.Provider())
	}
	if c.Model() != "gemini-2.0-flash" {
		t.Errorf("expected model gemini-2.0-flash, got %s", c.Model())
	}
}

func TestEmbedBatch(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		expectedPath := "/v1beta/models/gemini-2.0-flash:batchEmbedContents"
		if r.URL.Path != expectedPath {
			t.Errorf("expected %s, got %s", expectedPath, r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("expected x-goog-api-key=test-key header, got %q", got)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		requests, ok := req["requests"].([]any)
		if !ok {
			t.Fatalf("expected requests array in body, got %+v", req)
		}
		if len(requests) != 2 {
			t.Fatalf("expected 2 sub-requests, got %d", len(requests))
		}

		// Each sub-request must carry the fully-qualified model name.
		gotTexts := make([]string, len(requests))
		for i, r := range requests {
			sub := r.(map[string]any)
			if sub["model"] != "models/gemini-2.0-flash" {
				t.Errorf("sub-request %d: expected model models/gemini-2.0-flash, got %v",
					i, sub["model"])
			}
			content := sub["content"].(map[string]any)
			parts := content["parts"].([]any)
			part := parts[0].(map[string]any)
			gotTexts[i] = part["text"].(string)
		}
		if gotTexts[0] != "hello" || gotTexts[1] != "world" {
			t.Errorf("expected texts [hello world], got %v", gotTexts)
		}

		// Respond with embeddings in the same order as the request.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2, 0.3}},
				{"values": []float64{0.4, 0.5, 0.6}},
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

	// All N texts must be unpacked from a single batch request.
	if callCount != 1 {
		t.Errorf("expected 1 batch API call, got %d", callCount)
	}
}

func TestEmbedBatchEmpty(t *testing.T) {
	// An empty input slice must return an empty result without
	// issuing any HTTP request.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request to %s for empty input", r.URL.Path)
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	embeddings, err := c.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(embeddings) != 0 {
		t.Errorf("expected empty result, got %d embeddings", len(embeddings))
	}
}

func TestEmbedBatchLengthMismatch(t *testing.T) {
	// Provider responses returning fewer embeddings than inputs
	// must surface as a ProviderError, not silently truncate.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2}},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected error for length mismatch")
	}
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
}

func TestEmbedBatchEmptyValues(t *testing.T) {
	// An embedding entry with no values must surface as a
	// ProviderError rather than returning a zero-length vector.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2}},
				{"values": []float64{}},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected error for empty values in batch response")
	}
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
}

// ---------- Unit tests for internal helpers ----------

func TestConvertToolMessage_FunctionResponseLookup(t *testing.T) {
	// A tool-result message resolves the function name from the
	// (ID -> Name) map populated by the prior assistant tool_use.
	toolNames := map[string]string{
		"call_my_fn": "my_fn",
	}
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "call_my_fn", Text: `{"result":"ok"}`},
		},
	}
	contents := convertMessage(m, toolNames)
	if len(contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(contents))
	}
	if contents[0].Role != "user" {
		t.Errorf("expected user role, got %s", contents[0].Role)
	}
	if contents[0].Parts[0].FunctionResponse == nil {
		t.Fatal("expected FunctionResponse, got nil")
	}
	if contents[0].Parts[0].FunctionResponse.Name != "my_fn" {
		t.Errorf("expected name my_fn, got %s", contents[0].Parts[0].FunctionResponse.Name)
	}
	if contents[0].Parts[0].FunctionResponse.Response["result"] != "ok" {
		t.Errorf("expected result=ok, got %v", contents[0].Parts[0].FunctionResponse.Response)
	}
}

func TestConvertToolMessage_FallbackToLegacyIDPrefix(t *testing.T) {
	// When the (ID -> Name) map has no entry, the legacy
	// "gemini-tool-<name>" prefix convention is honoured for
	// backwards compatibility with older parseChatResponse output.
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "gemini-tool-weather", Text: `not json`},
		},
	}
	contents := convertMessage(m, map[string]string{})
	if contents[0].Parts[0].FunctionResponse.Name != "weather" {
		t.Errorf("name = %q, want weather", contents[0].Parts[0].FunctionResponse.Name)
	}
	// Non-JSON Text wraps in {"result": ...}.
	if contents[0].Parts[0].FunctionResponse.Response["result"] != "not json" {
		t.Errorf("expected result wrapping, got %v", contents[0].Parts[0].FunctionResponse.Response)
	}
}

func TestConvertToolMessage_FallbackToRawID(t *testing.T) {
	// When neither the toolNames map nor the legacy "gemini-tool-"
	// prefix matches, the raw ToolUseID is used as the function name.
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "weather", Text: "sunny"},
		},
	}
	contents := convertMessage(m, map[string]string{})
	if contents[0].Parts[0].FunctionResponse.Name != "weather" {
		t.Errorf("name = %q, want weather (raw ToolUseID fallback)",
			contents[0].Parts[0].FunctionResponse.Name)
	}
}

func TestConvertMessage_AssistantWithToolUseRecordsName(t *testing.T) {
	// An assistant message with a tool_use block populates the
	// toolNames map so subsequent tool-result messages can look up
	// the function name.
	toolNames := map[string]string{}
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
				ID:    "call_42",
				Name:  "do_thing",
				Input: json.RawMessage(`{"x":1}`),
			}},
		},
	}
	convertMessage(m, toolNames)
	if toolNames["call_42"] != "do_thing" {
		t.Errorf("toolNames missing entry: %+v", toolNames)
	}
}

func TestConvertMessage_BlockImageInline(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockImage, Image: &llm.ImageContent{
				Data:      []byte{0x89, 0x50},
				MediaType: "image/png",
			}},
		},
	}
	contents := convertMessage(m, nil)
	if len(contents) != 1 || len(contents[0].Parts) != 1 {
		t.Fatalf("unexpected output: %+v", contents)
	}
	id := contents[0].Parts[0].InlineData
	if id == nil || id.MimeType != "image/png" || len(id.Data) != 2 {
		t.Errorf("inlineData = %+v", id)
	}
}

func TestConvertMessage_BlockImageURL(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockImage, Image: &llm.ImageContent{
				URL:       "gs://my-bucket/cat.png",
				MediaType: "image/png",
			}},
		},
	}
	contents := convertMessage(m, nil)
	fd := contents[0].Parts[0].FileData
	if fd == nil || fd.FileURI != "gs://my-bucket/cat.png" || fd.MimeType != "image/png" {
		t.Errorf("fileData = %+v", fd)
	}
}

func TestConvertMessage_BlockDocumentInline(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockDocument, Document: &llm.DocumentContent{
				Data:      []byte{0x25, 0x50, 0x44, 0x46},
				MediaType: "application/pdf",
				Filename:  "doc.pdf",
			}},
		},
	}
	contents := convertMessage(m, nil)
	if len(contents) != 1 || len(contents[0].Parts) != 1 {
		t.Fatalf("unexpected output: %+v", contents)
	}
	id := contents[0].Parts[0].InlineData
	if id == nil || id.MimeType != "application/pdf" || len(id.Data) != 4 {
		t.Errorf("inlineData = %+v", id)
	}
}

func TestConvertMessage_BlockDocumentURL(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockDocument, Document: &llm.DocumentContent{
				URL:       "gs://my-bucket/doc.pdf",
				MediaType: "application/pdf",
			}},
		},
	}
	contents := convertMessage(m, nil)
	fd := contents[0].Parts[0].FileData
	if fd == nil || fd.FileURI != "gs://my-bucket/doc.pdf" || fd.MimeType != "application/pdf" {
		t.Errorf("fileData = %+v", fd)
	}
}

func TestConvertMessage_BlockDocumentNilSkipped(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "before"},
			{Type: llm.BlockDocument, Document: nil},
			{Type: llm.BlockText, Text: "after"},
		},
	}
	contents := convertMessage(m, nil)
	parts := contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts (nil document skipped), got %d", len(parts))
	}
	if parts[0].Text != "before" || parts[1].Text != "after" {
		t.Errorf("unexpected parts: %+v", parts)
	}
}

func TestConvertMessage_AssistantTextOnly(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "plain assistant text"},
		},
	}
	contents := convertMessage(m, nil)
	if contents[0].Role != "model" {
		t.Errorf("expected model role, got %s", contents[0].Role)
	}
	if contents[0].Parts[0].Text != "plain assistant text" {
		t.Errorf("text = %q", contents[0].Parts[0].Text)
	}
}

func TestConvertMessage_NilToolUseSkipped(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "before"},
			{Type: llm.BlockToolUse, ToolUse: nil},
			{Type: llm.BlockText, Text: "after"},
		},
	}
	contents := convertMessage(m, nil)
	parts := contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts (nil tool_use skipped), got %d", len(parts))
	}
	if parts[0].Text != "before" || parts[1].Text != "after" {
		t.Errorf("unexpected parts: %+v", parts)
	}
}

func TestNormalizeStopReason(t *testing.T) {
	tests := []struct {
		input string
		want  llm.StopReason
	}{
		{"STOP", llm.StopReasonEndTurn},
		{"MAX_TOKENS", llm.StopReasonMaxTokens},
		{"SAFETY", llm.StopReasonContentFilter},
		{"RECITATION", llm.StopReasonContentFilter},
		{"OTHER", llm.StopReasonEndTurn},
		{"", llm.StopReasonEndTurn},
	}
	for _, tt := range tests {
		got := normalizeStopReason(tt.input)
		if got != tt.want {
			t.Errorf("normalizeStopReason(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestMapRole(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"assistant", "model"},
		{"user", "user"},
		// "system" maps to "user" so a stray system-role Message in
		// the conversation history is not rejected by Gemini, which
		// only accepts "user" and "model" wire roles. System prompts
		// normally flow through ChatRequest.SystemPrompt instead.
		{"system", "user"},
		// "tool" maps to "user" because Gemini emits tool results
		// as user-role messages with functionResponse parts.
		{"tool", "user"},
		{"anything_else", "anything_else"},
	}
	for _, tt := range tests {
		got := mapRole(tt.input)
		if got != tt.want {
			t.Errorf("mapRole(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestMapError(t *testing.T) {
	t.Run("400 maps to ErrInvalidRequest", func(t *testing.T) {
		body := []byte(`{"error":{"message":"bad request","status":"INVALID_ARGUMENT"}}`)
		err := mapError(400, body)
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
		if pe.Message != "bad request" {
			t.Errorf("expected message 'bad request', got %s", pe.Message)
		}
	})

	t.Run("500 maps to ErrProviderError", func(t *testing.T) {
		body := []byte(`{"error":{"message":"internal error","status":"INTERNAL"}}`)
		err := mapError(500, body)
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
	})

	t.Run("empty body falls back to HTTP status message", func(t *testing.T) {
		err := mapError(503, []byte(``))
		var pe *llm.ProviderError
		if !errors.As(err, &pe) {
			t.Fatal("expected ProviderError")
		}
		if pe.Message != "HTTP 503" {
			t.Errorf("expected 'HTTP 503', got %s", pe.Message)
		}
	})
}

func TestBuildFunctionResponseValidJSON(t *testing.T) {
	// Exercise the successful json.Unmarshal branch (Text IS valid JSON).
	b := llm.ContentBlock{
		Type:      llm.BlockToolResult,
		ToolUseID: "gemini-tool-weather",
		Text:      `{"temp":72,"unit":"F"}`,
	}
	resp := buildFunctionResponse(b, map[string]string{})
	if resp == nil {
		t.Fatal("expected FunctionResponse")
	}
	if resp.Name != "weather" {
		t.Errorf("expected name 'weather', got %s", resp.Name)
	}
	if resp.Response["temp"] != float64(72) {
		t.Errorf("expected temp 72, got %v", resp.Response["temp"])
	}
}

func TestParseChatResponseNoCandidates(t *testing.T) {
	// Exercise the early-return when candidates is empty.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{},
			"usageMetadata": map[string]any{
				"promptTokenCount":     3,
				"candidatesTokenCount": 0,
				"totalTokenCount":      3,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Content) != 0 {
		t.Errorf("expected no content blocks, got %d", len(resp.Content))
	}
	if resp.Usage.PromptTokens != 3 {
		t.Errorf("expected 3 prompt tokens, got %d", resp.Usage.PromptTokens)
	}
}

func TestChatRateLimit(t *testing.T) {
	// Exercise the 429 -> ErrRateLimit branch in mapError.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "quota exceeded",
				"status":  "RESOURCE_EXHAUSTED",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("expected ErrRateLimit, got %v", err)
	}
}

func TestEmbedNoValues(t *testing.T) {
	// Exercise the "no embedding data returned" error path.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embedding": map[string]any{
				"values": []float64{},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for empty embedding")
	}
	if !errors.Is(err, llm.ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", err)
	}
}

func TestEmbedError(t *testing.T) {
	// Exercise the HTTP error path in Embed.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "API key not valid",
				"status":  "PERMISSION_DENIED",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication, got %v", err)
	}
}

func TestEmbedBatchError(t *testing.T) {
	// Exercise the error propagation in EmbedBatch.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"server error"}}`))
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err == nil {
		t.Fatal("expected error from EmbedBatch")
	}
}

func TestChatStreamDoneAlwaysHasUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		// Emit a text chunk and a done chunk with finishReason but NO usageMetadata.
		_, _ = w.Write([]byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}` + "\n\n"))
	}))
	defer server.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: server.URL,
		Model:   "gemini-2.0-flash",
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

func TestListModelsError(t *testing.T) {
	// Exercise the HTTP error path in ListModels.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "API key not valid",
				"status":  "PERMISSION_DENIED",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication, got %v", err)
	}
}

func TestChatStreamErrorStatus(t *testing.T) {
	// Exercise the non-2xx status path in ChatStream.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "API key not valid",
				"status":  "PERMISSION_DENIED",
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication, got %v", err)
	}
}

func TestChatStreamWithFunctionCall(t *testing.T) {
	// Exercise the FunctionCall path inside the streaming goroutine.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)

		chunk := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`
		w.Write([]byte("data: " + chunk + "\n\n"))
		flusher.Flush()
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Weather?"}}}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotToolUse bool
	for chunk := range stream.Chunks {
		if chunk.Type == llm.ChunkToolUseStart {
			gotToolUse = true
			if chunk.ToolUse == nil {
				t.Error("expected ToolUse to be non-nil")
			} else if chunk.ToolUse.Name != "get_weather" {
				t.Errorf("expected get_weather, got %s", chunk.ToolUse.Name)
			}
		}
	}
	if !gotToolUse {
		t.Error("expected tool_use_start chunk")
	}
	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}
}

func TestExplicitZeroTemperatureReachesWire(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "ok"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     1,
				"candidatesTokenCount": 1,
				"totalTokenCount":      2,
			},
		})
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "gemini-2.0-flash",
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

	genConfig, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or wrong type in wire body: %+v", captured)
	}
	temp, ok := genConfig["temperature"]
	if !ok {
		t.Fatalf("temperature missing from generationConfig: %+v", genConfig)
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
// Temperature, it must now be omitted from generationConfig entirely.
func TestUnsetTemperatureOmittedFromWire(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "ok"},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     1,
				"candidatesTokenCount": 1,
				"totalTokenCount":      2,
			},
		})
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "gemini-2.0-flash",
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

	if genConfig, ok := captured["generationConfig"].(map[string]any); ok {
		if temp, present := genConfig["temperature"]; present {
			t.Errorf("temperature should be omitted from wire when unset, got %v", temp)
		}
	}
}

func TestBuildChatRequestMaxTokensAndTemp(t *testing.T) {
	// Exercise the req.MaxTokens > 0 and req.Temperature != nil branches.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		genConfig := req["generationConfig"].(map[string]any)
		if genConfig["maxOutputTokens"] != float64(512) {
			t.Errorf("expected maxOutputTokens 512, got %v", genConfig["maxOutputTokens"])
		}
		if genConfig["temperature"] != float64(0.7) {
			t.Errorf("expected temperature 0.7, got %v", genConfig["temperature"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role":  "model",
						"parts": []map[string]any{{"text": "ok"}},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     1,
				"candidatesTokenCount": 1,
				"totalTokenCount":      2,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	temp := 0.7
	maxTok := 512
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
		MaxTokens:   &maxTok,
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToolResultMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		contents := req["contents"].([]any)
		// Should have 3 messages: user, model (with function call), user (with function response).
		if len(contents) != 3 {
			t.Fatalf("expected 3 contents, got %d", len(contents))
		}

		// Third message should be the tool result as functionResponse.
		third := contents[2].(map[string]any)
		if third["role"] != "user" {
			t.Errorf("expected role user for tool result, got %v", third["role"])
		}
		parts := third["parts"].([]any)
		part := parts[0].(map[string]any)
		funcResp := part["functionResponse"].(map[string]any)
		if funcResp["name"] != "get_weather" {
			t.Errorf("expected name get_weather, got %v", funcResp["name"])
		}
		response := funcResp["response"].(map[string]any)
		if response["result"] != "Sunny, 72F" {
			t.Errorf("expected result 'Sunny, 72F', got %v", response["result"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "The weather in NYC is sunny and 72F."},
						},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     20,
				"candidatesTokenCount": 10,
				"totalTokenCount":      30,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "What's the weather in NYC?"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{
					Type: "tool_use",
					ToolUse: &llm.ToolUse{
						ID:    "gemini-tool-get_weather",
						Name:  "get_weather",
						Input: json.RawMessage(`{"location":"NYC"}`),
					},
				},
			}},
			{Role: llm.RoleTool, Content: []llm.ContentBlock{{
				Type:      llm.BlockToolResult,
				ToolUseID: "gemini-tool-get_weather",
				Text:      `{"result":"Sunny, 72F"}`,
			}}},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content[0].Text != "The weather in NYC is sunny and 72F." {
		t.Errorf("expected weather response, got %s", resp.Content[0].Text)
	}
}

func TestChatJSONResponseFormat(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"role":  "model",
						"parts": []map[string]any{{"text": "{}"}},
					},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     1,
				"candidatesTokenCount": 1,
				"totalTokenCount":      2,
			},
		})
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "gemini-2.0-flash",
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

	genConfig, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or wrong type: %+v", captured)
	}
	if genConfig["responseMimeType"] != "application/json" {
		t.Errorf("generationConfig.responseMimeType = %v, want application/json", genConfig["responseMimeType"])
	}
}

func TestChatToolChoice(t *testing.T) {
	cases := []struct {
		name      string
		choice    llm.ToolChoice
		wantMode  string
		wantNames []string // nil means no allowedFunctionNames expected
	}{
		{"auto", llm.ToolChoice{Mode: llm.ToolChoiceAuto}, "AUTO", nil},
		{"none", llm.ToolChoice{Mode: llm.ToolChoiceNone}, "NONE", nil},
		{"required", llm.ToolChoice{Mode: llm.ToolChoiceRequired}, "ANY", nil},
		{"specific", llm.ToolChoice{Mode: llm.ToolChoiceSpecific, Name: "search"}, "ANY", []string{"search"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &captured)
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"candidates": []map[string]any{
						{
							"content": map[string]any{
								"role":  "model",
								"parts": []map[string]any{{"text": "ok"}},
							},
							"finishReason": "STOP",
						},
					},
					"usageMetadata": map[string]any{
						"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2,
					},
				})
			}))
			defer srv.Close()

			c, err := New(llm.Options{
				APIKey:  "k",
				BaseURL: srv.URL,
				Model:   "gemini-2.0-flash",
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

			tcRaw, ok := captured["toolConfig"]
			if !ok {
				t.Fatalf("toolConfig missing from wire body: %+v", captured)
			}
			toolCfg, ok := tcRaw.(map[string]any)
			if !ok {
				t.Fatalf("toolConfig should be a map, got %T: %v", tcRaw, tcRaw)
			}
			fcc, ok := toolCfg["functionCallingConfig"].(map[string]any)
			if !ok {
				t.Fatalf("functionCallingConfig missing or wrong type: %+v", toolCfg)
			}
			if fcc["mode"] != tc.wantMode {
				t.Errorf("mode = %v, want %v", fcc["mode"], tc.wantMode)
			}
			if tc.wantNames != nil {
				allowedRaw, ok := fcc["allowedFunctionNames"]
				if !ok {
					t.Fatalf("allowedFunctionNames missing for specific mode: %+v", fcc)
				}
				allowed, ok := allowedRaw.([]any)
				if !ok {
					t.Fatalf("allowedFunctionNames is %T, want []any: %v", allowedRaw, allowedRaw)
				}
				if len(allowed) != len(tc.wantNames) {
					t.Fatalf("allowedFunctionNames len = %d, want %d", len(allowed), len(tc.wantNames))
				}
				for i, want := range tc.wantNames {
					if allowed[i] != want {
						t.Errorf("allowedFunctionNames[%d] = %v, want %v", i, allowed[i], want)
					}
				}
			} else {
				if _, has := fcc["allowedFunctionNames"]; has {
					t.Errorf("unexpected allowedFunctionNames for mode %q: %v", tc.wantMode, fcc["allowedFunctionNames"])
				}
			}
		})
	}
}

func TestChatStopSequences(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content":      map[string]any{"role": "model", "parts": []map[string]any{{"text": "ok"}}},
					"finishReason": "STOP",
				},
			},
			"usageMetadata": map[string]any{
				"promptTokenCount":     1,
				"candidatesTokenCount": 1,
				"totalTokenCount":      2,
			},
		})
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		APIKey: "k", BaseURL: srv.URL, Model: "test", Retry: llm.RetryConfig{Disabled: true},
	})
	_, _ = c.Chat(context.Background(), llm.ChatRequest{
		Messages:      []llm.Message{llm.UserText("hi")},
		StopSequences: []string{"END", "STOP"},
	})
	genCfg, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or wrong type: %+v", captured["generationConfig"])
	}
	stops, ok := genCfg["stopSequences"].([]any)
	if !ok || len(stops) != 2 {
		t.Fatalf("stopSequences missing or wrong length: %+v", genCfg["stopSequences"])
	}
	if stops[0] != "END" || stops[1] != "STOP" {
		t.Errorf("stopSequences = %v", stops)
	}
}

func TestPing(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"models": []map[string]any{
					{"name": "models/gemini-2.0-flash", "supportedGenerationMethods": []string{"generateContent"}},
				},
			})
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		if err := c.Ping(context.Background()); err != nil {
			t.Errorf("Ping returned %v, want nil", err)
		}
	})
	t.Run("propagates upstream error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"error":{"message":"key disabled"}}`))
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
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{{"text": "hi"}}}, "finishReason": "STOP"},
			},
			"usageMetadata": map[string]any{"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
		})
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	if _, err := c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := c.Usage(); got.TotalTokens == 0 {
		t.Fatal("Usage() reported zero tokens; test setup is wrong")
	}
	c.ResetUsage()
	if got := c.Usage(); got != (llm.TokenUsage{}) {
		t.Errorf("Usage() after ResetUsage = %+v, want zero value", got)
	}
}

func TestListModelsWithMetadata(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "models/gemini-1.5-pro", "supportedGenerationMethods": []string{"generateContent"}},
				{"name": "models/text-embedding-004", "supportedGenerationMethods": []string{"embedContent"}},
				{"name": "models/future-model-z", "supportedGenerationMethods": []string{"generateContent"}},
			},
		})
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	infos, err := c.ListModelsWithMetadata(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// text-embedding-004 only supports embedContent so it's filtered
	// out by ListModels. The other two survive.
	if len(infos) != 2 {
		t.Fatalf("got %d infos, want 2: %+v", len(infos), infos)
	}
	byID := map[string]llm.ModelInfo{}
	for _, info := range infos {
		byID[info.ID] = info
	}
	if _, ok := byID["gemini-1.5-pro"]; !ok {
		t.Errorf("missing gemini-1.5-pro: %+v", byID)
	}
	// Unknown model falls back to [Chat, Streaming].
	unknown := byID["future-model-z"]
	if len(unknown.Capabilities) != 2 {
		t.Errorf("unknown fallback caps = %v, want 2 entries", unknown.Capabilities)
	}
}

func TestNewClientRejectsInvalidBaseURL(t *testing.T) {
	_, err := New(llm.Options{APIKey: "k", Model: "test", BaseURL: "not-a-valid-url"})
	if err == nil {
		t.Fatal("want error for invalid BaseURL")
	}
	if !strings.Contains(err.Error(), "gemini") {
		t.Errorf("error should mention provider name: %v", err)
	}
}

func TestNewClientPlumbsOnRetryHook(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{{"text": "ok"}}}, "finishReason": "STOP"},
			},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
		})
	}))
	defer srv.Close()

	var got llm.RetryEvent
	c, err := New(llm.Options{
		APIKey:  "k",
		Model:   "m",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{MaxRetries: 2, InitialBackoff: time.Millisecond, MaxBackoff: 5 * time.Millisecond},
		OnRetry: func(e llm.RetryEvent) { got = e },
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("chat failed: %v", err)
	}
	if got.Attempt == 0 {
		t.Errorf("OnRetry never fired; got = %+v", got)
	}
	if got.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("OnRetry StatusCode = %d, want %d", got.StatusCode, http.StatusServiceUnavailable)
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
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{
					"name":                       "models/gemini-pro",
					"supportedGenerationMethods": []string{"generateContent"},
				},
				{
					"name":                       "models/text-embedding-004",
					"supportedGenerationMethods": []string{"embedContent"},
				},
			},
		})
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
