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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/toolresult"
)

// toolErrorResults is a successful result, a failed one with text and
// a failed one without, alongside the content each should be sent as.
var toolErrorResults = []struct {
	block llm.ContentBlock
	want  string
}{
	{llm.ToolResultBlock("c1", "12 degrees", false), "12 degrees"},
	{llm.ToolResultBlock("c2", `no such city "Atlantis"`, true), `{"error":"no such city \"Atlantis\""}`},
	{llm.ToolResultBlock("c3", "", true), `{"error":"` + toolresult.UnspecifiedError + `"}`},
}

// toolErrorMessages returns a conversation whose tool results are held
// in a message of the given role: RoleTool, or RoleUser (the Anthropic
// convention, which the provider splits into tool results).
func toolErrorMessages(role llm.Role) []llm.Message {
	resultMsg := llm.Message{Role: role}
	calls := llm.Message{Role: llm.RoleAssistant}
	for _, r := range toolErrorResults {
		resultMsg.Content = append(resultMsg.Content, r.block)
		calls.Content = append(calls.Content, llm.ContentBlock{
			Type:    llm.BlockToolUse,
			ToolUse: &llm.ToolUse{ID: r.block.ToolUseID, Name: "weather", Input: json.RawMessage(`{}`)},
		})
	}
	return []llm.Message{llm.UserText("Weather?"), calls, resultMsg}
}

// toolErrorServer captures each request body and answers it in the
// shape of the endpoint and mode requested, so that both Chat and
// ChatStream complete normally.
func toolErrorServer(t *testing.T, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*captured = nil
		if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
			t.Errorf("decode request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		stream := (*captured)["stream"] == true
		responses := r.URL.Path == "/responses"
		switch {
		case stream && responses:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"type":"response.output_text.delta","delta":"ok"}` + "\n\n" +
				`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` + "\n\n"))
		case stream:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n" +
				"data: [DONE]\n\n"))
		case responses:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		}
	}))
}

// sentToolResults extracts the tool results from a captured request
// body, keyed by call ID: role:"tool" messages on Chat Completions, or
// function_call_output items on the Responses API.
func sentToolResults(t *testing.T, body map[string]any, responses bool) map[string]any {
	t.Helper()
	key, roleKey, roleVal, idKey, contentKey := "messages", "role", "tool", "tool_call_id", "content"
	if responses {
		key, roleKey, roleVal, idKey, contentKey = "input", "type", "function_call_output", "call_id", "output"
	}
	items, _ := body[key].([]any)
	got := map[string]any{}
	for _, it := range items {
		m, _ := it.(map[string]any)
		if m[roleKey] == roleVal {
			got[m[idKey].(string)] = m[contentKey]
		}
	}
	return got
}

// TestToolResultIsErrorSent checks that a failed tool result reaches
// both endpoints as {"error": "<text>"}, on Chat and ChatStream, for
// results held in a tool message and in a user message, whilst a
// successful result is sent unchanged.
func TestToolResultIsErrorSent(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, role := range []llm.Role{llm.RoleTool, llm.RoleUser} {
				name := fmt.Sprintf("responses=%v/stream=%v/%s", responses, stream, role)
				t.Run(name, func(t *testing.T) {
					var captured map[string]any
					srv := toolErrorServer(t, &captured)
					defer srv.Close()

					c := toolErrorClient(t, srv.URL, responses)
					runChat(t, c, llm.ChatRequest{Messages: toolErrorMessages(role)}, stream)
					checkToolResults(t, sentToolResults(t, captured, responses))
				})
			}
		}
	}
}

// toolErrorClient returns a client for url that uses the Responses API
// when responses is set, and Chat Completions otherwise.
func toolErrorClient(t *testing.T, url string, responses bool) llm.Client {
	t.Helper()
	opts := llm.Options{
		APIKey:  "test-key",
		Model:   "gpt-4o",
		BaseURL: url,
		Retry:   llm.RetryConfig{Disabled: true},
	}
	if responses {
		opts.Extensions = forceResponses
	}
	c, err := New(opts)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

// runChat sends req with Chat, or with ChatStream when stream is set,
// draining the stream, and fails the test on any error.
func runChat(t *testing.T, c llm.Client, req llm.ChatRequest, stream bool) {
	t.Helper()
	if !stream {
		if _, err := c.Chat(context.Background(), req); err != nil {
			t.Fatalf("Chat: %v", err)
		}
		return
	}
	s, err := c.ChatStream(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for range s.Chunks {
	}
	if err := <-s.Err; err != nil {
		t.Fatalf("stream error: %v", err)
	}
}

// checkToolResults compares the tool results sent, keyed by call ID,
// with toolErrorResults.
func checkToolResults(t *testing.T, got map[string]any) {
	t.Helper()
	if len(got) != len(toolErrorResults) {
		t.Fatalf("sent %d tool results, want %d: %v", len(got), len(toolErrorResults), got)
	}
	for _, r := range toolErrorResults {
		if got[r.block.ToolUseID] != r.want {
			t.Errorf("%s content = %v, want %s", r.block.ToolUseID, got[r.block.ToolUseID], r.want)
		}
	}
}
