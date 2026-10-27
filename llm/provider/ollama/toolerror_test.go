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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/toolresult"
)

// TestToolResultIsErrorSent checks that a failed tool result is sent as
// {"error": "<text>"} on Chat and ChatStream, whilst a successful
// result is sent unchanged.
func TestToolResultIsErrorSent(t *testing.T) {
	want := []string{
		"12 degrees",
		`{"error":"no such city \"Atlantis\""}`,
		`{"error":"` + toolresult.UnspecifiedError + `"}`,
	}
	msgs := []llm.Message{
		llm.UserText("Weather?"),
		{Role: llm.RoleTool, Content: []llm.ContentBlock{
			llm.ToolResultBlock("c1", "12 degrees", false),
			llm.ToolResultBlock("c2", `no such city "Atlantis"`, true),
			llm.ToolResultBlock("c3", "", true),
		}},
	}

	for _, stream := range []bool{false, true} {
		name := "chat"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			var captured struct {
				Messages []ollamaMessage `json:"messages"`
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
					t.Errorf("decode request body: %v", err)
					http.Error(w, "bad request", http.StatusBadRequest)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"))
			}))
			defer srv.Close()

			c := newTestClient(t, srv.URL)
			req := llm.ChatRequest{Messages: msgs}
			if stream {
				s, err := c.ChatStream(context.Background(), req)
				if err != nil {
					t.Fatalf("ChatStream: %v", err)
				}
				for range s.Chunks {
				}
				if err := <-s.Err; err != nil {
					t.Fatalf("stream error: %v", err)
				}
			} else if _, err := c.Chat(context.Background(), req); err != nil {
				t.Fatalf("Chat: %v", err)
			}

			var got []string
			for _, m := range captured.Messages {
				if m.Role == "tool" {
					got = append(got, m.Content)
				}
			}
			if len(got) != len(want) {
				t.Fatalf("sent %d tool messages, want %d: %q", len(got), len(want), got)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("tool message %d = %s, want %s", i, got[i], want[i])
				}
			}
		})
	}
}
