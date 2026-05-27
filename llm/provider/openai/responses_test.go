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
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

func TestUseResponsesAPI_AutoDetect(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"gpt-4o", false},
		{"gpt-4o-mini", false},
		{"gpt-3.5-turbo", false},
		{"gpt-5", true},
		{"gpt-5-turbo", true},
		{"o1", true},
		{"o1-preview", true},
		{"o1-mini", true},
		{"o3", true},
		{"o3-mini", true},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := useResponsesAPI(tc.model, nil); got != tc.want {
				t.Errorf("useResponsesAPI(%q, nil) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

func TestUseResponsesAPI_ExtensionOverride(t *testing.T) {
	cases := []struct {
		name  string
		model string
		ext   *bool
		want  bool
	}{
		{"force-on-for-gpt-4o", "gpt-4o", llm.Bool(true), true},
		{"force-off-for-gpt-5", "gpt-5", llm.Bool(false), false},
		{"nil-falls-back-to-auto-gpt-4o", "gpt-4o", nil, false},
		{"nil-falls-back-to-auto-gpt-5", "gpt-5", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			exts := []llm.ProviderExtension{Extension{ResponsesAPI: tc.ext}}
			if got := useResponsesAPI(tc.model, exts); got != tc.want {
				t.Errorf("useResponsesAPI(%q, ext=%v) = %v, want %v", tc.model, tc.ext, got, tc.want)
			}
		})
	}
}

// responsesEchoServer asserts the request landed on /responses and
// returns a minimal successful response. The decoded body is written
// to *captured for assertion in the caller.
func responsesEchoServer(t *testing.T, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("expected /responses, got %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "completed",
			"output": []map[string]any{{
				"type": "message",
				"role": "assistant",
				"content": []map[string]any{{
					"type": "output_text",
					"text": "Hello!",
				}},
			}},
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
				"total_tokens":  15,
			},
		})
	}))
}

func TestResponsesAPI_RoutesGPT5(t *testing.T) {
	var captured map[string]any
	srv := responsesEchoServer(t, &captured)
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if captured["model"] != "gpt-5" {
		t.Errorf("model = %v, want gpt-5", captured["model"])
	}
	if _, ok := captured["input"]; !ok {
		t.Errorf("input missing from request body: %#v", captured)
	}
	if _, ok := captured["max_output_tokens"]; !ok {
		t.Errorf("max_output_tokens missing from request body: %#v", captured)
	}
	if _, ok := captured["messages"]; ok {
		t.Errorf("messages should not appear on a /responses request: %#v", captured)
	}

	if len(resp.Content) != 1 || resp.Content[0].Type != llm.BlockText {
		t.Fatalf("expected one text block, got %#v", resp.Content)
	}
	if resp.Content[0].Text != "Hello!" {
		t.Errorf("text = %q, want %q", resp.Content[0].Text, "Hello!")
	}
	if resp.StopReason != llm.StopReasonEndTurn {
		t.Errorf("StopReason = %v, want %v", resp.StopReason, llm.StopReasonEndTurn)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v, want {10 5 15}", resp.Usage)
	}
}

func TestResponsesAPI_ExtensionForcesResponsesOnGPT4o(t *testing.T) {
	var captured map[string]any
	srv := responsesEchoServer(t, &captured)
	defer srv.Close()

	c, err := New(llm.Options{
		APIKey:     "test-key",
		Model:      "gpt-4o",
		BaseURL:    srv.URL,
		Retry:      llm.RetryConfig{Disabled: true},
		Extensions: []llm.ProviderExtension{Extension{ResponsesAPI: llm.Bool(true)}},
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	if _, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Hi")},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if captured["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", captured["model"])
	}
}

func TestResponsesAPI_TranslatesSystemPromptToInstructions(t *testing.T) {
	var captured map[string]any
	srv := responsesEchoServer(t, &captured)
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if _, err := c.Chat(context.Background(), llm.ChatRequest{
		SystemPrompt: "You are concise.",
		Messages:     []llm.Message{llm.UserText("Hi")},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if captured["instructions"] != "You are concise." {
		t.Errorf("instructions = %v, want %q", captured["instructions"], "You are concise.")
	}
}

func TestResponsesAPI_ToolsWireShape(t *testing.T) {
	var captured map[string]any
	srv := responsesEchoServer(t, &captured)
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if _, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Weather?")},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	tools, ok := captured["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %#v", captured["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want function", tool["type"])
	}
	// Responses API puts name/description/parameters at the top level
	// of each tool — there is no nested "function" wrapper.
	if tool["name"] != "get_weather" {
		t.Errorf("tool name = %v, want get_weather", tool["name"])
	}
	if _, ok := tool["function"]; ok {
		t.Errorf("Responses API tools must not have a nested 'function' wrapper: %#v", tool)
	}
}

func TestResponsesAPI_ParsesToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "completed",
			"output": []map[string]any{{
				"type":      "function_call",
				"call_id":   "call_xyz",
				"name":      "get_weather",
				"arguments": `{"location":"NYC"}`,
			}},
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
				"total_tokens":  15,
			},
		})
	}))
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Weather?")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != llm.BlockToolUse {
		t.Fatalf("expected one tool_use block, got %#v", resp.Content)
	}
	tu := resp.Content[0].ToolUse
	if tu == nil {
		t.Fatal("ToolUse is nil")
	}
	if tu.ID != "call_xyz" || tu.Name != "get_weather" {
		t.Errorf("ToolUse = %+v, want id=call_xyz name=get_weather", tu)
	}
	if string(tu.Input) != `{"location":"NYC"}` {
		t.Errorf("ToolUse.Input = %s, want {\"location\":\"NYC\"}", string(tu.Input))
	}
	if resp.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason = %v, want %v", resp.StopReason, llm.StopReasonToolUse)
	}
}

func TestResponsesAPI_MaxOutputTokensStopReason(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":             "incomplete",
			"incomplete_details": map[string]any{"reason": "max_output_tokens"},
			"output": []map[string]any{{
				"type":    "message",
				"role":    "assistant",
				"content": []map[string]any{{"type": "output_text", "text": "partial"}},
			}},
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
				"total_tokens":  15,
			},
		})
	}))
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.StopReason != llm.StopReasonMaxTokens {
		t.Errorf("StopReason = %v, want %v", resp.StopReason, llm.StopReasonMaxTokens)
	}
}

func TestResponsesAPI_AssistantAndToolHistory(t *testing.T) {
	var captured map[string]any
	srv := responsesEchoServer(t, &captured)
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	if _, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			llm.UserText("Weather?"),
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{
						ID: "call_1", Name: "get_weather",
						Input: json.RawMessage(`{"location":"NYC"}`),
					}},
				},
			},
			llm.ToolResultMessage("call_1", "72F sunny", false),
		},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	input, ok := captured["input"].([]any)
	if !ok {
		t.Fatalf("input missing or wrong shape: %#v", captured["input"])
	}
	if len(input) != 3 {
		t.Fatalf("expected 3 input items (user, function_call, function_call_output), got %d: %#v", len(input), input)
	}
	if input[1].(map[string]any)["type"] != "function_call" {
		t.Errorf("item 1 type = %v, want function_call", input[1].(map[string]any)["type"])
	}
	if input[2].(map[string]any)["type"] != "function_call_output" {
		t.Errorf("item 2 type = %v, want function_call_output", input[2].(map[string]any)["type"])
	}
	if input[2].(map[string]any)["output"] != "72F sunny" {
		t.Errorf("item 2 output = %v, want %q", input[2].(map[string]any)["output"], "72F sunny")
	}
}

func TestResponsesAPI_RejectsStopSequences(t *testing.T) {
	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: "https://example.invalid",
		Retry:   llm.RetryConfig{Disabled: true},
	})
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages:      []llm.Message{llm.UserText("Hi")},
		StopSequences: []string{"END"},
	})
	if err == nil {
		t.Fatal("expected error for StopSequences on Responses API")
	}
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", err)
	}
}

func TestResponsesAPI_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != true {
			t.Errorf("expected stream: true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`{"type":"response.created","response":{}}`,
			`{"type":"response.output_text.delta","delta":"Hello"}`,
			`{"type":"response.output_text.delta","delta":" world"}`,
			`{"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`,
		}
		for _, ev := range events {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}

	var text []string
	var gotDone bool
	for chunk := range stream.Chunks {
		switch chunk.Type {
		case llm.ChunkText:
			text = append(text, chunk.Text)
		case llm.ChunkDone:
			gotDone = true
			if chunk.Usage == nil || chunk.Usage.PromptTokens != 5 || chunk.Usage.CompletionTokens != 2 {
				t.Errorf("done usage = %+v, want {5 2 7}", chunk.Usage)
			}
		}
	}
	if err := <-stream.Err; err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if got := strings.Join(text, ""); got != "Hello world" {
		t.Errorf("text = %q, want %q", got, "Hello world")
	}
	if !gotDone {
		t.Error("missing ChunkDone")
	}
}

func TestResponsesAPI_StreamToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`{"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_abc","name":"get_weather"}}`,
			`{"type":"response.function_call_arguments.delta","delta":"{\"loc"}`,
			`{"type":"response.function_call_arguments.delta","delta":"ation\":\"NYC\"}"}`,
			`{"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":8,"total_tokens":18}}}`,
		}
		for _, ev := range events {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	stream, err := c.ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Weather?")},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}

	var sawStart, sawDelta, sawDone bool
	for chunk := range stream.Chunks {
		switch chunk.Type {
		case llm.ChunkToolUseStart:
			sawStart = true
			if chunk.ToolUse == nil || chunk.ToolUse.Name != "get_weather" || chunk.ToolUse.ID != "call_abc" {
				t.Errorf("ToolUseStart = %+v, want id=call_abc name=get_weather", chunk.ToolUse)
			}
		case llm.ChunkToolUseDelta:
			sawDelta = true
			if chunk.Partial == "" {
				t.Error("ChunkToolUseDelta: Partial must be non-empty")
			}
		case llm.ChunkDone:
			sawDone = true
		}
	}
	if err := <-stream.Err; err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if !sawStart {
		t.Error("did not receive ChunkToolUseStart")
	}
	if !sawDelta {
		t.Error("did not receive ChunkToolUseDelta")
	}
	if !sawDone {
		t.Error("did not receive ChunkDone")
	}
}

func TestResponsesAPI_AccumulatesUsage(t *testing.T) {
	srv := responsesEchoServer(t, &map[string]any{})
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	for i := 0; i < 2; i++ {
		if _, err := c.Chat(context.Background(), llm.ChatRequest{
			Messages: []llm.Message{llm.UserText("Hi")},
		}); err != nil {
			t.Fatalf("Chat %d: %v", i, err)
		}
	}
	u := c.Usage()
	if u.PromptTokens != 20 || u.CompletionTokens != 10 || u.TotalTokens != 30 {
		t.Errorf("cumulative usage = %+v, want {20 10 30}", u)
	}
}

func TestResponsesAPI_AuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()

	c, _ := New(llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-5",
		BaseURL: srv.URL,
		Retry:   llm.RetryConfig{Disabled: true},
	})
	_, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.UserText("Hi")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuthentication) {
		t.Errorf("expected ErrAuthentication, got %v", err)
	}
}
