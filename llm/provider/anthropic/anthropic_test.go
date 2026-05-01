//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package anthropic

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
		Model:   "claude-sonnet-4-20250514",
		BaseURL: url,
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
		if r.URL.Path != "/messages" {
			t.Errorf("expected /messages, got %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("expected x-api-key test-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("expected anthropic-version 2023-06-01, got %s", r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("anthropic-beta") != "prompt-caching-2024-07-31" {
			t.Errorf("expected anthropic-beta prompt-caching-2024-07-31, got %s", r.Header.Get("anthropic-beta"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected application/json content type, got %s", r.Header.Get("Content-Type"))
		}

		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "claude-sonnet-4-20250514" {
			t.Errorf("expected model claude-sonnet-4-20250514, got %v", req["model"])
		}
		msgs := req["messages"].([]any)
		if len(msgs) != 1 {
			t.Errorf("expected 1 message, got %d", len(msgs))
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{
					"type": "text",
					"text": "Hello!",
				},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
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

func TestChatWithToolsAndCacheControl(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify tools are in Anthropic format with cache_control.
		tools := req["tools"].([]any)
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		tool := tools[0].(map[string]any)
		if tool["name"] != "get_weather" {
			t.Errorf("expected name get_weather, got %v", tool["name"])
		}
		if tool["description"] != "Get weather for a location" {
			t.Errorf("expected description, got %v", tool["description"])
		}

		// Verify cache_control is present.
		cc, ok := tool["cache_control"].(map[string]any)
		if !ok {
			t.Fatal("expected cache_control on tool")
		}
		if cc["type"] != "ephemeral" {
			t.Errorf("expected cache_control type ephemeral, got %v", cc["type"])
		}

		// Return a tool call response with cache token usage.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{
					"type": "tool_use",
					"id":   "toolu_123",
					"name": "get_weather",
					"input": map[string]any{
						"location": "NYC",
					},
				},
			},
			"stop_reason": "tool_use",
			"usage": map[string]any{
				"input_tokens":                10,
				"output_tokens":               8,
				"cache_creation_input_tokens": 100,
				"cache_read_input_tokens":     50,
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	resp, err := c.Chat(context.Background(), WithToolCaching(llm.ChatRequest{
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
	}))
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
	if resp.Content[0].ToolUse.ID != "toolu_123" {
		t.Errorf("expected toolu_123, got %s", resp.Content[0].ToolUse.ID)
	}
	if resp.Content[0].ToolUse.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", resp.Content[0].ToolUse.Name)
	}

	var input map[string]string
	json.Unmarshal(resp.Content[0].ToolUse.Input, &input)
	if input["location"] != "NYC" {
		t.Errorf("expected NYC, got %s", input["location"])
	}

	// Verify cache token usage.
	if resp.Usage.CacheCreationInputTokens != 100 {
		t.Errorf("expected 100 cache creation tokens, got %d", resp.Usage.CacheCreationInputTokens)
	}
	if resp.Usage.CacheReadInputTokens != 50 {
		t.Errorf("expected 50 cache read tokens, got %d", resp.Usage.CacheReadInputTokens)
	}
}

func TestEmbedNotSupported(t *testing.T) {
	c, err := New(llm.Options{
		APIKey: "test-key",
		Model:  "claude-sonnet-4-20250514",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	_, err = c.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", err)
	}

	_, err = c.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", err)
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
				{"id": "claude-sonnet-4-20250514", "type": "model"},
				{"id": "claude-3-opus-20240229", "type": "model"},
				{"id": "some-internal-thing", "type": "internal"},
			},
		})
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should filter to type=="model" only.
	expected := map[string]bool{
		"claude-sonnet-4-20250514": true,
		"claude-3-opus-20240229":   true,
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

func TestChatAuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Invalid API key",
				"type":    "authentication_error",
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

func TestCumulativeUsage(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{
					"type": "text",
					"text": "response",
				},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":                10,
				"output_tokens":               5,
				"cache_creation_input_tokens": 20,
				"cache_read_input_tokens":     30,
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
	if usage.CacheCreationInputTokens != 20 {
		t.Errorf("expected 20 cache creation tokens, got %d", usage.CacheCreationInputTokens)
	}
	if usage.CacheReadInputTokens != 30 {
		t.Errorf("expected 30 cache read tokens, got %d", usage.CacheReadInputTokens)
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
	if usage.CacheCreationInputTokens != 40 {
		t.Errorf("expected 40 cache creation tokens, got %d", usage.CacheCreationInputTokens)
	}
	if usage.CacheReadInputTokens != 60 {
		t.Errorf("expected 60 cache read tokens, got %d", usage.CacheReadInputTokens)
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

		events := []string{
			`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`,
			`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
			`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
			`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
			`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
		}
		for _, event := range events {
			w.Write([]byte(event + "\n\n"))
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
				if chunk.Usage.PromptTokens != 10 {
					t.Errorf("expected 10 prompt tokens, got %d", chunk.Usage.PromptTokens)
				}
				if chunk.Usage.CompletionTokens != 5 {
					t.Errorf("expected 5 completion tokens, got %d", chunk.Usage.CompletionTokens)
				}
				if chunk.Usage.TotalTokens != 15 {
					t.Errorf("expected 15 total tokens, got %d", chunk.Usage.TotalTokens)
				}
			}
		}
	}

	// Check for errors.
	if streamErr := <-stream.Err; streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	fullText := ""
	for _, p := range textParts {
		fullText += p
	}
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

		events := []string{
			`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}`,
			`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather"}}`,
			`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}`,
			`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"NYC\"}"}}`,
			`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
			`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`,
			`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
		}
		for _, event := range events {
			w.Write([]byte(event + "\n\n"))
			flusher.Flush()
		}
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
				if chunk.ToolUse.ID != "toolu_abc" {
					t.Errorf("expected toolu_abc, got %s", chunk.ToolUse.ID)
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

func TestProviderAndModel(t *testing.T) {
	c, err := New(llm.Options{
		APIKey: "test-key",
		Model:  "claude-sonnet-4-20250514",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	if c.Provider() != "anthropic" {
		t.Errorf("expected provider anthropic, got %s", c.Provider())
	}
	if c.Model() != "claude-sonnet-4-20250514" {
		t.Errorf("expected model claude-sonnet-4-20250514, got %s", c.Model())
	}
}

// ---------- Unit tests for internal conversion functions ----------

func TestConvertMessage_AssistantContentBlocks(t *testing.T) {
	input := json.RawMessage(`{"key":"value"}`)
	m := llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockText, Text: "Hello"},
			{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
				ID:    "tool-1",
				Name:  "my_tool",
				Input: input,
			}},
		},
	}

	result := convertMessage(m)

	if result.Role != "assistant" {
		t.Errorf("expected role assistant, got %s", result.Role)
	}
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" || result.Content[0].Text != "Hello" {
		t.Errorf("first block: %+v", result.Content[0])
	}
	if result.Content[1].Type != "tool_use" {
		t.Errorf("expected second block type tool_use, got %s", result.Content[1].Type)
	}
	if result.Content[1].ID != "tool-1" {
		t.Errorf("expected tool id tool-1, got %s", result.Content[1].ID)
	}
	if result.Content[1].Name != "my_tool" {
		t.Errorf("expected tool name my_tool, got %s", result.Content[1].Name)
	}
	if string(result.Content[1].Input) != string(input) {
		t.Errorf("expected input passthrough, got %s", string(result.Content[1].Input))
	}
}

func TestConvertBlock_NilToolUseEmitsBareBlock(t *testing.T) {
	// A tool_use block with nil ToolUse becomes a bare wire block
	// (no id/name/input). This is tolerated by Anthropic and avoids
	// a silent drop that would shift downstream block indexing.
	out := convertBlock(llm.ContentBlock{Type: llm.BlockToolUse, ToolUse: nil})
	if out.Type != "tool_use" {
		t.Errorf("type = %q, want tool_use", out.Type)
	}
	if out.ID != "" || out.Name != "" || len(out.Input) != 0 {
		t.Errorf("expected zero ID/Name/Input on nil ToolUse, got %+v", out)
	}
}

func TestConvertMessage_ToolRoleBecomesUserWithToolResult(t *testing.T) {
	m := llm.Message{
		Role: llm.RoleTool,
		Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "tid-1", Text: "result1", IsError: false},
			{Type: llm.BlockToolResult, ToolUseID: "tid-2", Text: "oops", IsError: true},
		},
	}

	result := convertMessage(m)

	if result.Role != "user" {
		t.Errorf("expected role user, got %s", result.Role)
	}
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "tool_result" {
		t.Errorf("type = %q, want tool_result", result.Content[0].Type)
	}
	if result.Content[0].ToolUseID != "tid-1" {
		t.Errorf("tool_use_id = %q", result.Content[0].ToolUseID)
	}
	if result.Content[0].Content != "result1" {
		t.Errorf("content = %q", result.Content[0].Content)
	}
	if result.Content[0].IsError {
		t.Errorf("first block IsError should be false")
	}
	if !result.Content[1].IsError {
		t.Errorf("second block IsError should be true")
	}
}

func TestConvertBlock_ImageBase64(t *testing.T) {
	out := convertBlock(llm.ContentBlock{
		Type: llm.BlockImage,
		Image: &llm.ImageContent{
			Data:      []byte{0x89, 0x50},
			MediaType: "image/png",
		},
	})
	if out.Type != "image" {
		t.Errorf("type = %q, want image", out.Type)
	}
	if out.Source == nil || out.Source.Type != "base64" {
		t.Fatalf("expected base64 source, got %+v", out.Source)
	}
	if out.Source.MediaType != "image/png" {
		t.Errorf("media_type = %q", out.Source.MediaType)
	}
	if len(out.Source.Data) != 2 {
		t.Errorf("data not preserved: %v", out.Source.Data)
	}
}

func TestConvertBlock_ImageURL(t *testing.T) {
	out := convertBlock(llm.ContentBlock{
		Type:  llm.BlockImage,
		Image: &llm.ImageContent{URL: "https://example.com/cat.png"},
	})
	if out.Source == nil || out.Source.Type != "url" {
		t.Fatalf("expected url source, got %+v", out.Source)
	}
	if out.Source.URL != "https://example.com/cat.png" {
		t.Errorf("url = %q", out.Source.URL)
	}
}

func TestConvertBlock_CacheControl(t *testing.T) {
	out := convertBlock(llm.ContentBlock{
		Type:         llm.BlockText,
		Text:         "cached",
		CacheControl: &llm.CacheControl{Type: "ephemeral"},
	})
	if out.CacheControl == nil || out.CacheControl.Type != "ephemeral" {
		t.Errorf("cache_control = %+v", out.CacheControl)
	}
}

func TestConvertMessage_RoleMapping(t *testing.T) {
	cases := []struct {
		in   llm.Role
		want string
	}{
		{llm.RoleUser, "user"},
		{llm.RoleAssistant, "assistant"},
		{llm.RoleTool, "user"},
		{llm.RoleSystem, "user"},
	}
	for _, c := range cases {
		got := convertMessage(llm.Message{Role: c.in}).Role
		if got != c.want {
			t.Errorf("role %q -> %q, want %q", c.in, got, c.want)
		}
	}
}

func TestMapError_InvalidRequest(t *testing.T) {
	body := []byte(`{"error":{"message":"bad param","type":"invalid_request_error"}}`)
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
	if pe.Message != "bad param" {
		t.Errorf("expected message 'bad param', got %q", pe.Message)
	}
}

func TestMapError_ProviderError(t *testing.T) {
	body := []byte(`{"error":{"message":"internal server error","type":"server_error"}}`)
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
	if pe.Message != "internal server error" {
		t.Errorf("expected message 'internal server error', got %q", pe.Message)
	}
}

func TestMapError_EmptyBody(t *testing.T) {
	// When body is empty/unparseable, message falls back to "HTTP <status>"
	err := mapError(503, []byte(``))
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
		t.Errorf("expected message 'HTTP 503', got %q", pe.Message)
	}
}

// ---------- End unit tests for internal conversion functions ----------

func TestExplicitZeroTemperatureReachesWire(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": "ok"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  1,
				"output_tokens": 1,
			},
		})
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

func TestAnthropicMaxTokensFallbackWhenNilEverywhere(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": "ok"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  1,
				"output_tokens": 1,
			},
		})
	}))
	defer srv.Close()

	// Note: pass nil MaxTokens via Options{}.WithDefaults() — but that
	// would set it to 4096. To trigger the fallback path, we must
	// bypass WithDefaults. Instead we go through llm.NewClient which
	// applies WithDefaults internally — so Options.MaxTokens will be
	// 4096 (default), and ChatRequest.MaxTokens is nil, so the
	// per-request precedence chain selects the client default of 4096.
	//
	// To actually exercise the "both nil" fallback, we'd need to set
	// both opts.MaxTokens AND req.MaxTokens to nil despite NewClient.
	// That can't happen via the public API once WithDefaults runs.
	//
	// Instead, the test asserts the EFFECTIVE on-wire value is 4096
	// when the consumer doesn't specify MaxTokens — which is the
	// observable behaviour we care about.
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

	mt, ok := captured["max_tokens"]
	if !ok {
		t.Fatalf("max_tokens missing from wire body — Anthropic API requires it: %+v", captured)
	}
	mtF, _ := mt.(float64)
	if int(mtF) != 4096 {
		t.Errorf("max_tokens on wire = %v, want 4096", mtF)
	}
}

func TestChatJSONResponseFormat(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content":     []map[string]any{{"type": "text", "text": "{}"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
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
	_, err = c.Chat(context.Background(), llm.ChatRequest{
		Messages:       []llm.Message{llm.UserText("hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSON},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	// Anthropic has no native JSON mode; the directive is injected into the
	// system prompt blocks.
	systemRaw, ok := captured["system"]
	if !ok {
		t.Fatalf("system field missing from wire body: %+v", captured)
	}
	systemBlocks, ok := systemRaw.([]any)
	if !ok {
		t.Fatalf("system is %T, want []any: %v", systemRaw, systemRaw)
	}
	found := false
	for _, blk := range systemBlocks {
		b, ok := blk.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := b["text"].(string); ok && strings.Contains(text, "valid JSON") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected system block containing 'valid JSON', got: %+v", systemBlocks)
	}
}

func TestChatWithSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// System prompt should be an array of objects.
		system := req["system"].([]any)
		if len(system) != 1 {
			t.Fatalf("expected 1 system block, got %d", len(system))
		}
		sysBlock := system[0].(map[string]any)
		if sysBlock["type"] != "text" {
			t.Errorf("expected type text, got %v", sysBlock["type"])
		}
		// Per-request system prompt should be used.
		if sysBlock["text"] != "You are a pirate." {
			t.Errorf("expected per-request system prompt, got %v", sysBlock["text"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": "Ahoy!"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
			},
		})
	}))
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "claude-sonnet-4-20250514",
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

func TestChatToolChoice(t *testing.T) {
	// anthropicToolChoiceMode maps each ToolChoiceMode to the expected Anthropic wire "type".
	// ToolChoiceNone falls back to "auto" since Anthropic has no native "none" mode.
	cases := []struct {
		name     string
		choice   llm.ToolChoice
		wantType string
		wantName string
	}{
		{"auto", llm.ToolChoice{Mode: llm.ToolChoiceAuto}, "auto", ""},
		{"none_fallback", llm.ToolChoice{Mode: llm.ToolChoiceNone}, "auto", ""},
		{"required", llm.ToolChoice{Mode: llm.ToolChoiceRequired}, "any", ""},
		{"specific", llm.ToolChoice{Mode: llm.ToolChoiceSpecific, Name: "search"}, "tool", "search"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &captured)
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"content":     []map[string]any{{"type": "text", "text": "ok"}},
					"stop_reason": "end_turn",
					"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
				})
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

			tcRaw, ok := captured["tool_choice"]
			if !ok {
				t.Fatalf("tool_choice missing from wire body: %+v", captured)
			}
			tcMap, ok := tcRaw.(map[string]any)
			if !ok {
				t.Fatalf("tool_choice should be a map, got %T: %v", tcRaw, tcRaw)
			}
			if tcMap["type"] != tc.wantType {
				t.Errorf("type = %v, want %v", tcMap["type"], tc.wantType)
			}
			if tc.wantName != "" {
				if tcMap["name"] != tc.wantName {
					t.Errorf("name = %v, want %v", tcMap["name"], tc.wantName)
				}
			} else {
				if _, hasName := tcMap["name"]; hasName {
					t.Errorf("unexpected name field in tool_choice: %v", tcMap["name"])
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
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
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
	stops, ok := captured["stop_sequences"].([]any)
	if !ok || len(stops) != 2 {
		t.Fatalf("stop_sequences missing or wrong length: %+v", captured["stop_sequences"])
	}
	if stops[0] != "END" || stops[1] != "STOP" {
		t.Errorf("stop_sequences = %v", stops)
	}
}

func TestChatExtendedThinkingSetsThinkingField(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		APIKey: "k", BaseURL: srv.URL, Model: "test", Retry: llm.RetryConfig{Disabled: true},
	})
	req := WithExtendedThinking(llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("think hard")},
	}, 8000)
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	thinking, ok := captured["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking field missing from request: %+v", captured)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want \"enabled\"", thinking["type"])
	}
	if budget, _ := thinking["budget_tokens"].(float64); int(budget) != 8000 {
		t.Errorf("thinking.budget_tokens = %v, want 8000", thinking["budget_tokens"])
	}
}

func TestChatNoExtensionOmitsThinkingField(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
	}))
	defer srv.Close()

	c, _ := llm.NewClient(providerName, llm.Options{
		APIKey: "k", BaseURL: srv.URL, Model: "test", Retry: llm.RetryConfig{Disabled: true},
	})
	if _, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("hi")},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, present := captured["thinking"]; present {
		t.Errorf("thinking field was sent without WithExtendedThinking: %+v", captured["thinking"])
	}
}

func TestPing(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "claude-3-5-sonnet", "type": "model"}}})
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
			"content":     []map[string]any{{"type": "text", "text": "hi"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 10, "output_tokens": 5},
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
			"data": []map[string]any{
				{"id": "claude-3-5-sonnet-20241022", "type": "model"},
				{"id": "claude-3-haiku-20240307", "type": "model"},
				{"id": "future-model-x", "type": "model"},
			},
		})
	}))
	defer srv.Close()
	c := newTestClient(t, srv.URL)
	infos, err := c.ListModelsWithMetadata(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(infos) != 3 {
		t.Fatalf("got %d infos, want 3: %+v", len(infos), infos)
	}
	byID := map[string]llm.ModelInfo{}
	for _, info := range infos {
		byID[info.ID] = info
	}
	// Known prefix → known capability set including Vision.
	sonnet := byID["claude-3-5-sonnet-20241022"]
	hasVision := false
	for _, c := range sonnet.Capabilities {
		if c == llm.ModelCapabilityVision {
			hasVision = true
		}
	}
	if !hasVision {
		t.Errorf("sonnet capabilities missing Vision: %v", sonnet.Capabilities)
	}
	// Unknown model falls back to [Chat, Streaming].
	unknown := byID["future-model-x"]
	if len(unknown.Capabilities) != 2 {
		t.Errorf("unknown model fallback caps = %v, want 2 entries", unknown.Capabilities)
	}
}

func TestNewClientRejectsInvalidBaseURL(t *testing.T) {
	_, err := New(llm.Options{APIKey: "k", Model: "test", BaseURL: "not-a-valid-url"})
	if err == nil {
		t.Fatal("want error for invalid BaseURL")
	}
	if !strings.Contains(err.Error(), "anthropic") {
		t.Errorf("error should mention provider name: %v", err)
	}
}

func TestNewClientPlumbsOnRetryHook(t *testing.T) {
	// Server fails twice then succeeds; we verify the OnRetry hook
	// fires with translated llm.RetryEvent values.
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
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

func TestNormalizeStopReason(t *testing.T) {
	cases := []struct {
		in   string
		want llm.StopReason
	}{
		{"end_turn", llm.StopReasonEndTurn},
		{"max_tokens", llm.StopReasonMaxTokens},
		{"stop_sequence", llm.StopReasonStopSequence},
		{"tool_use", llm.StopReasonToolUse},
		{"unrecognised_value", llm.StopReasonEndTurn},
		{"", llm.StopReasonEndTurn},
	}
	for _, tc := range cases {
		if got := normalizeStopReason(tc.in); got != tc.want {
			t.Errorf("normalizeStopReason(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
