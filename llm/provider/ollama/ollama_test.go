//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
		Model:   "llama3",
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
		if r.URL.Path != "/api/chat" {
			t.Errorf("expected /api/chat, got %s", r.URL.Path)
		}

		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "llama3" {
			t.Errorf("expected model llama3, got %v", req["model"])
		}
		if req["stream"] != false {
			t.Errorf("expected stream: false, got %v", req["stream"])
		}

		msgs := req["messages"].([]any)
		if len(msgs) != 1 {
			t.Errorf("expected 1 message, got %d", len(msgs))
		}
		msg := msgs[0].(map[string]any)
		if msg["role"] != "user" {
			t.Errorf("expected role user, got %v", msg["role"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"role":    "assistant",
				"content": "Hello from Ollama!",
			},
			"done": true,
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
	if resp.Content[0].Text != "Hello from Ollama!" {
		t.Errorf("expected 'Hello from Ollama!', got %s", resp.Content[0].Text)
	}
	// Fixture omits token counts; expect zero.
	if resp.Usage.TotalTokens != 0 {
		t.Errorf("expected 0 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatWithToolsInjectsSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		msgs := req["messages"].([]any)
		// Should have system message injected + user message.
		if len(msgs) < 2 {
			t.Fatalf("expected at least 2 messages, got %d", len(msgs))
		}

		sysMsg := msgs[0].(map[string]any)
		if sysMsg["role"] != "system" {
			t.Errorf("expected first message to be system, got %v", sysMsg["role"])
		}
		content := sysMsg["content"].(string)
		// System prompt should contain tool names.
		if !strings.Contains(content, "get_weather") {
			t.Errorf("system prompt should contain tool name get_weather, got: %s", content)
		}
		if !strings.Contains(content, "Get weather for a location") {
			t.Errorf("system prompt should contain tool description, got: %s", content)
		}
		// Should contain the JSON format instruction.
		if !strings.Contains(content, `"tool"`) {
			t.Errorf("system prompt should contain tool JSON format instruction, got: %s", content)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"role":    "assistant",
				"content": "The weather in NYC is sunny.",
			},
			"done": true,
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
	if resp.StopReason != "end_turn" {
		t.Errorf("expected end_turn, got %s", resp.StopReason)
	}
}

func TestChatToolCallParsing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"role":    "assistant",
				"content": `{"tool":"get_weather","arguments":{"location":"NYC"}}`,
			},
			"done": true,
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "What's the weather?"}}},
		},
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
	if resp.Content[0].ToolUse.ID != "ollama-tool-1" {
		t.Errorf("expected ollama-tool-1, got %s", resp.Content[0].ToolUse.ID)
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

func TestChatToolCallExtractionFromText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"role":    "assistant",
				"content": `Sure, let me check the weather for you. {"tool":"get_weather","arguments":{"location":"NYC"}} I'll get that information.`,
			},
			"done": true,
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "What's the weather?"}}},
		},
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
	if resp.Content[0].ToolUse.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", resp.Content[0].ToolUse.Name)
	}

	var input map[string]string
	json.Unmarshal(resp.Content[0].ToolUse.Input, &input)
	if input["location"] != "NYC" {
		t.Errorf("expected NYC, got %s", input["location"])
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/api/tags" {
			t.Errorf("expected /api/tags, got %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "llama3:latest"},
				{"name": "mistral:latest"},
				{"name": "codellama:7b"},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d", len(models))
	}
	expected := map[string]bool{
		"llama3:latest":  true,
		"mistral:latest": true,
		"codellama:7b":   true,
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
		if r.URL.Path != "/api/embed" {
			t.Errorf("expected /api/embed, got %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "llama3" {
			t.Errorf("expected model llama3, got %v", req["model"])
		}
		input := req["input"].(string)
		if input != "hello world" {
			t.Errorf("expected 'hello world', got %s", input)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float64{
				{0.1, 0.2, 0.3},
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
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		input := req["input"].(string)

		w.Header().Set("Content-Type", "application/json")
		if input == "hello" {
			json.NewEncoder(w).Encode(map[string]any{
				"embeddings": [][]float64{
					{0.1, 0.2, 0.3},
				},
			})
		} else {
			json.NewEncoder(w).Encode(map[string]any{
				"embeddings": [][]float64{
					{0.4, 0.5, 0.6},
				},
			})
		}
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	embeddings, err := c.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount != 2 {
		t.Errorf("expected 2 sequential calls, got %d", callCount)
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

func TestChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["stream"] != true {
			t.Errorf("expected stream: true, got %v", req["stream"])
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher := w.(http.Flusher)

		// Ollama streams newline-delimited JSON (not SSE).
		chunks := []string{
			`{"message":{"role":"assistant","content":"Hello"},"done":false}`,
			`{"message":{"role":"assistant","content":" world"},"done":false}`,
			`{"message":{"role":"assistant","content":""},"done":true}`,
		}
		for _, chunk := range chunks {
			fmt.Fprintln(w, chunk)
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
		}
	}

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

func TestChatStreamWithToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher := w.(http.Flusher)

		// Stream a response that contains a tool call JSON.
		chunks := []string{
			`{"message":{"role":"assistant","content":"{\"tool\":\"get_weather\",\"arguments\":{\"location\":\"NYC\"}}"},"done":false}`,
			`{"message":{"role":"assistant","content":""},"done":true}`,
		}
		for _, chunk := range chunks {
			fmt.Fprintln(w, chunk)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Weather?"}}}},
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
				if chunk.ToolUse.ID != "ollama-tool-1" {
					t.Errorf("expected ollama-tool-1, got %s", chunk.ToolUse.ID)
				}
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
	if !gotDone {
		t.Error("did not receive done chunk")
	}
}

func TestProviderAndModel(t *testing.T) {
	c, err := New(llm.Options{
		Model: "llama3",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	if c.Provider() != "ollama" {
		t.Errorf("expected provider ollama, got %s", c.Provider())
	}
	if c.Model() != "llama3" {
		t.Errorf("expected model llama3, got %s", c.Model())
	}
}

func TestErrorMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]any{
			"error": "model not found",
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
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("expected ProviderError")
	}
	if pe.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", pe.StatusCode)
	}
}

func TestDefaultBaseURL(t *testing.T) {
	c, err := New(llm.Options{
		Model: "llama3",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	oc := c.(*client)
	if oc.baseURL != "http://localhost:11434" {
		t.Errorf("expected default base URL http://localhost:11434, got %s", oc.baseURL)
	}
}

// ---------- New tests for coverage ----------

func TestUsageAndAddUsage(t *testing.T) {
	c, err := New(llm.Options{Model: "llama3"})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	oc := c.(*client)

	// Initially zero.
	usage := oc.Usage()
	if usage.PromptTokens != 0 || usage.CompletionTokens != 0 || usage.TotalTokens != 0 {
		t.Errorf("expected zero initial usage, got %+v", usage)
	}

	// Add some usage.
	oc.addUsage(llm.TokenUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	})

	usage = oc.Usage()
	if usage.PromptTokens != 10 {
		t.Errorf("expected PromptTokens=10, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 20 {
		t.Errorf("expected CompletionTokens=20, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 30 {
		t.Errorf("expected TotalTokens=30, got %d", usage.TotalTokens)
	}

	// Add more usage - should accumulate.
	oc.addUsage(llm.TokenUsage{
		PromptTokens:     5,
		CompletionTokens: 5,
		TotalTokens:      10,
	})

	usage = oc.Usage()
	if usage.PromptTokens != 15 {
		t.Errorf("expected PromptTokens=15, got %d", usage.PromptTokens)
	}
	if usage.TotalTokens != 40 {
		t.Errorf("expected TotalTokens=40, got %d", usage.TotalTokens)
	}
}

func TestConvertMessage_ImageBase64(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "look:"},
			{Type: llm.BlockImage, Image: &llm.ImageContent{
				Data:      []byte{0x01, 0x02, 0x03},
				MediaType: "image/png",
			}},
		},
	}
	out, err := convertMessage(m)
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	if out[0].Content != "look:" {
		t.Errorf("content = %q", out[0].Content)
	}
	if len(out[0].Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(out[0].Images))
	}
	// Base64-encoded {0x01,0x02,0x03} -> "AQID".
	if out[0].Images[0] != "AQID" {
		t.Errorf("image = %q, want AQID", out[0].Images[0])
	}
}

func TestConvertMessage_ImageURLRejected(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			{Type: llm.BlockImage, Image: &llm.ImageContent{URL: "https://example.com/x.png"}},
		},
	}
	_, err := convertMessage(m)
	if err == nil {
		t.Fatal("expected error for URL-only image")
	}
	if !strings.Contains(err.Error(), "URL image input") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestConvertMessage_ToolResultBecomesToolRoleMessage(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "tu_1", Text: "ok"},
		},
	}
	out, err := convertMessage(m)
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	if out[0].Role != "tool" || out[0].Content != "ok" {
		t.Errorf("got %+v", out[0])
	}
}

func TestConvertMessage_ToolUseSerialisedIntoContent(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
				Name:  "get_weather",
				Input: json.RawMessage(`{"city":"NYC"}`),
			}},
		},
	}
	out, err := convertMessage(m)
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	if !strings.Contains(out[0].Content, `"tool":"get_weather"`) ||
		!strings.Contains(out[0].Content, `"city":"NYC"`) {
		t.Errorf("tool call not serialised into content: %q", out[0].Content)
	}
}

func TestEmbedEmptyEmbeddings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return a valid 200 response but with empty embeddings slice.
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float64{},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for empty embeddings response")
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProviderError, got %T: %v", err, err)
	}
	if pe.Message != "no embedding data returned" {
		t.Errorf("unexpected error message: %s", pe.Message)
	}
}

func TestChatStreamErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": "unauthorized",
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	_, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error for non-2xx status")
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProviderError, got %T: %v", err, err)
	}
	if pe.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected status 401, got %d", pe.StatusCode)
	}
}

func TestChatStreamWithToolsNotAToolCall(t *testing.T) {
	// When tools are provided but the accumulated content is NOT a valid tool call,
	// the stream should emit a text chunk with the full content.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher := w.(http.Flusher)

		chunks := []string{
			`{"message":{"role":"assistant","content":"This is just plain text, not a tool call."},"done":false}`,
			`{"message":{"role":"assistant","content":""},"done":true}`,
		}
		for _, chunk := range chunks {
			fmt.Fprintln(w, chunk)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "What is 2+2?"}}}},
		Tools: []llm.Tool{
			{
				Name:        "calculator",
				Description: "Perform calculations",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gotText string
	var gotDone bool
	var gotToolStart bool
	for chunk := range stream.Chunks {
		switch chunk.Type {
		case llm.ChunkText:
			gotText = chunk.Text
		case llm.ChunkDone:
			gotDone = true
		case llm.ChunkToolUseStart:
			gotToolStart = true
		}
	}

	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	if gotToolStart {
		t.Error("did not expect tool_use_start for plain text response")
	}
	if gotText != "This is just plain text, not a tool call." {
		t.Errorf("expected plain text content, got %q", gotText)
	}
	if !gotDone {
		t.Error("did not receive done chunk")
	}
}

func TestMapErrorEmptyBody(t *testing.T) {
	// When error body is empty (or has no message), mapError should fall through
	// to "HTTP <status>" format.
	err := mapError(http.StatusGatewayTimeout, []byte{})
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProviderError, got %T", err)
	}
	if pe.Message != "HTTP 504" {
		t.Errorf("expected message 'HTTP 504', got %q", pe.Message)
	}
	if pe.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("expected status 504, got %d", pe.StatusCode)
	}
}

func TestChatStreamPopulatesUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"hi"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":7,"eval_count":3}` + "\n"))
	}))
	defer server.Close()

	c, err := llm.NewClient(providerName, llm.Options{
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
		t.Fatal("done chunk has nil Usage")
	}
	if done.Usage.PromptTokens != 7 || done.Usage.CompletionTokens != 3 || done.Usage.TotalTokens != 10 {
		t.Errorf("usage = %+v, want PromptTokens=7 CompletionTokens=3 TotalTokens=10", done.Usage)
	}
}

func TestExtractJSONFromText(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "direct JSON",
			input:    `{"tool":"get_weather","arguments":{"location":"NYC"}}`,
			expected: `{"tool":"get_weather","arguments":{"location":"NYC"}}`,
		},
		{
			name:     "JSON wrapped in text",
			input:    `Sure, let me help. {"tool":"get_weather","arguments":{"location":"NYC"}} Done.`,
			expected: `{"tool":"get_weather","arguments":{"location":"NYC"}}`,
		},
		{
			name:     "no JSON",
			input:    "Just plain text without any braces",
			expected: "",
		},
		{
			name:     "unclosed brace",
			input:    `{"tool":"get_weather"`,
			expected: "",
		},
		{
			name:     "nested braces",
			input:    `{"tool":"test","arguments":{"nested":{"deep":"value"}}}`,
			expected: `{"tool":"test","arguments":{"nested":{"deep":"value"}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSONFromText(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
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
		json.NewEncoder(w).Encode(map[string]any{
			"message":           map[string]any{"role": "assistant", "content": "{}"},
			"done":              true,
			"prompt_eval_count": 1,
			"eval_count":        1,
		})
	}))
	defer srv.Close()

	c, err := llm.NewClient(providerName, llm.Options{
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

	if captured["format"] != "json" {
		t.Errorf("format = %v, want \"json\"", captured["format"])
	}
}

func TestChatStopSequences(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message":           map[string]any{"role": "assistant", "content": "ok"},
			"done":              true,
			"prompt_eval_count": 1,
			"eval_count":        1,
		})
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		BaseURL: srv.URL, Model: "test", Retry: llm.RetryConfig{Disabled: true},
	})
	_, _ = c.Chat(context.Background(), llm.ChatRequest{
		Messages:      []llm.Message{llm.UserText("hi")},
		StopSequences: []string{"END", "STOP"},
	})
	opts, ok := captured["options"].(map[string]any)
	if !ok {
		t.Fatalf("options missing or wrong type: %+v", captured["options"])
	}
	stops, ok := opts["stop"].([]any)
	if !ok || len(stops) != 2 {
		t.Fatalf("options.stop missing or wrong length: %+v", opts["stop"])
	}
	if stops[0] != "END" || stops[1] != "STOP" {
		t.Errorf("options.stop = %v", stops)
	}
}

func TestStripThinkTags(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no tags", "plain answer", "plain answer"},
		{"empty", "", ""},
		{
			name: "single block before answer",
			in:   "<think>I should call get_weather.</think>\n\nThe weather is sunny.",
			want: "The weather is sunny.",
		},
		{
			name: "case-insensitive tags",
			in:   "<Think>reasoning</THINK>final",
			want: "final",
		},
		{
			name: "multiple blocks",
			in:   "<think>step 1</think>middle<think>step 2</think>end",
			want: "middleend",
		},
		{
			name: "unterminated open tag drops remainder",
			in:   "answer prefix<think>never closes",
			want: "answer prefix",
		},
		{
			name: "embedded JSON inside thinking is removed",
			in:   `<think>Maybe call {"tool":"a"}.</think>{"tool":"b","arguments":{}}`,
			want: `{"tool":"b","arguments":{}}`,
		},
		{
			name: "trims trailing whitespace after close",
			in:   "<think>x</think>   \n\n   answer",
			want: "answer",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := stripThinkTags(tc.in)
			if got != tc.want {
				t.Errorf("stripThinkTags(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestParseChatResponseStripsThinkingFromText(t *testing.T) {
	resp := (&client{}).parseChatResponse(&ollamaChatResponse{
		Message: ollamaRespMessage{
			Role:    "assistant",
			Content: "<think>let me think</think>\n\nThe answer is 42.",
		},
		Done:            true,
		PromptEvalCount: 5,
		EvalCount:       7,
	}, false)
	if len(resp.Content) != 1 || resp.Content[0].Type != llm.BlockText {
		t.Fatalf("unexpected content blocks: %+v", resp.Content)
	}
	if resp.Content[0].Text != "The answer is 42." {
		t.Errorf("text = %q, want %q", resp.Content[0].Text, "The answer is 42.")
	}
}

func TestTryParseToolCallIgnoresJSONInsideThinking(t *testing.T) {
	// The model "thinks out loud" with example JSON, then emits a real
	// tool call. Without thinking-tag stripping, the brace matcher would
	// pick up the example call inside <think>...</think>.
	content := `<think>I could try {"tool":"wrong","arguments":{}}.</think>` +
		`{"tool":"get_weather","arguments":{"city":"NYC"}}`

	resp := (&client{}).parseChatResponse(&ollamaChatResponse{
		Message: ollamaRespMessage{Role: "assistant", Content: content},
		Done:    true,
	}, true)
	if resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("StopReason = %v, want StopReasonToolUse", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != llm.BlockToolUse {
		t.Fatalf("unexpected content blocks: %+v", resp.Content)
	}
	if resp.Content[0].ToolUse.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", resp.Content[0].ToolUse.Name, "get_weather")
	}
}

func TestPing(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "llama3"}}})
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		if err := c.Ping(context.Background()); err != nil {
			t.Errorf("Ping returned %v, want nil", err)
		}
	})
	t.Run("propagates upstream error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"down"}`))
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		err := c.Ping(context.Background())
		if err == nil {
			t.Fatal("expected error from Ping, got nil")
		}
		if !errors.Is(err, llm.ErrProviderError) {
			t.Errorf("err = %v, want ErrProviderError", err)
		}
	})
}

func TestResetUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"message":           map[string]any{"role": "assistant", "content": "hi"},
			"done":              true,
			"prompt_eval_count": 3,
			"eval_count":        4,
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
				{"name": "llama3:latest"},
				{"name": "deepseek-r1:14b"},
			},
		})
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	infos, err := c.ListModelsWithMetadata(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(infos) != 2 {
		t.Fatalf("got %d infos, want 2: %+v", len(infos), infos)
	}
	for _, info := range infos {
		if len(info.Capabilities) != 2 {
			t.Errorf("ollama capabilities for %q = %v, want 2 entries", info.ID, info.Capabilities)
		}
	}
}

func TestNewClientRejectsInvalidBaseURL(t *testing.T) {
	_, err := New(llm.Options{Model: "test", BaseURL: "not-a-valid-url"})
	if err == nil {
		t.Fatal("want error for invalid BaseURL")
	}
	if !strings.Contains(err.Error(), "ollama") {
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
			"message":           map[string]any{"role": "assistant", "content": "ok"},
			"done":              true,
			"prompt_eval_count": 1,
			"eval_count":        1,
		})
	}))
	defer srv.Close()

	var got llm.RetryEvent
	c, err := New(llm.Options{
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
