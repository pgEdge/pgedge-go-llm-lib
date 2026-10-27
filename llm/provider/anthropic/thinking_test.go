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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// thinkingResponse is a tool-use turn in the shape Anthropic returns
// with thinking on: a summarised thinking block, one whose text was
// omitted, a redacted block and the tool call.
const thinkingResponse = `{"content":[` +
	`{"type":"thinking","thinking":"Check the weather.","signature":"sig-A"},` +
	`{"type":"thinking","thinking":"","signature":"sig-B"},` +
	`{"type":"redacted_thinking","data":"opaque-data"},` +
	`{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"London"}}` +
	`],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`

// wantThinkingBlocks is the content parsed from thinkingResponse, and
// from the equivalent stream.
var wantThinkingBlocks = []llm.ContentBlock{
	{Type: llm.BlockThinking, Thinking: &llm.Thinking{Text: "Check the weather.", Signature: "sig-A"}},
	{Type: llm.BlockThinking, Thinking: &llm.Thinking{Signature: "sig-B"}},
	{Type: llm.BlockRedactedThinking, Thinking: &llm.Thinking{Data: "opaque-data"}},
	{Type: llm.BlockToolUse, ToolUse: &llm.ToolUse{ID: "tu_1", Name: "get_weather", Input: json.RawMessage(`{"city":"London"}`)}},
}

// wantThinkingWire is how wantThinkingBlocks go back on the wire: each
// block exactly as received, the empty thinking text included.
const wantThinkingWire = `[` +
	`{"type":"thinking","thinking":"Check the weather.","signature":"sig-A"},` +
	`{"type":"thinking","thinking":"","signature":"sig-B"},` +
	`{"type":"redacted_thinking","data":"opaque-data"},` +
	`{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"London"}}]`

// thinkingStream is thinkingResponse as Anthropic streams it.
var thinkingStream = []string{
	`{"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Check the "}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weather."}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-A"}}`,
	`{"type":"content_block_stop","index":0}`,
	`{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sig-B"}}`,
	`{"type":"content_block_stop","index":1}`,
	`{"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking","data":"opaque-data"}}`,
	`{"type":"content_block_stop","index":2}`,
	`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}`,
	`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"London\"}"}}`,
	`{"type":"content_block_stop","index":3}`,
	`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}`,
	`{"type":"message_stop"}`,
}

func TestChatKeepsThinkingBlocks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(thinkingResponse))
	}))
	defer srv.Close()

	resp, err := newTestClient(t, srv.URL).Chat(context.Background(), hiRequest)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if !reflect.DeepEqual(resp.Content, wantThinkingBlocks) {
		t.Errorf("Content = %+v, want %+v", resp.Content, wantThinkingBlocks)
	}
}

func TestChatStreamKeepsThinkingBlocks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range thinkingStream {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
		}
	}))
	defer srv.Close()

	s, err := newTestClient(t, srv.URL).ChatStream(context.Background(), hiRequest)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	resp, err := s.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if !reflect.DeepEqual(resp.Content, wantThinkingBlocks) {
		t.Errorf("Content = %+v, want %+v", resp.Content, wantThinkingBlocks)
	}
}

func TestThinkingBlocksSentBackUnchanged(t *testing.T) {
	msg := convertMessage(llm.Message{Role: llm.RoleAssistant, Content: wantThinkingBlocks})
	got, err := json.Marshal(msg.Content)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(got) != wantThinkingWire {
		t.Errorf("wire = %s\nwant   %s", got, wantThinkingWire)
	}
}

// signatureRejected is the preserved-thinking rejection Anthropic
// returns when the history ahead of a replayed thinking block changed,
// cut short: the rest names a setting and what changed, and its
// wording is documented as liable to change.
const signatureRejected = `{"type":"error","error":{"type":"invalid_request_error","message":` +
	`"messages.1.content.0: Invalid ` + "`signature`" + ` in ` + "`thinking`" + ` block. The block is bound to a different conversation."}}`

// thinkingHistory is a tool-use turn being answered: the assistant
// message carries thinking blocks, then the tool result follows.
func thinkingHistory() llm.ChatRequest {
	return llm.ChatRequest{
		ReasoningEffort: llm.ReasoningEffortHigh,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Weather?"}}},
			{Role: llm.RoleAssistant, Content: wantThinkingBlocks},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockToolResult, ToolUseID: "tu_1", Text: "Rain"}}},
		},
	}
}

// assistantBlockTypes returns the block types of the recorded request's
// assistant message.
func assistantBlockTypes(t *testing.T, body map[string]any) []string {
	t.Helper()
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v, want 3", msgs)
	}
	m, _ := msgs[1].(map[string]any)
	content, _ := m["content"].([]any)
	types := make([]string, 0, len(content))
	for _, b := range content {
		bm, _ := b.(map[string]any)
		s, _ := bm["type"].(string)
		types = append(types, s)
	}
	return types
}

func TestSignatureRejectionStripsThinkingBlocks(t *testing.T) {
	// The documented recovery: send once more without any thinking
	// blocks, keeping the turn's other blocks, and do not learn
	// anything about the thinking type, which the next call still
	// sends along with the blocks.
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t, signatureRejected)
		c := newEffortClient(t, srv.URL, llm.Options{})
		for i := 0; i < 2; i++ {
			if err := sendEffort(t, c, stream, thinkingHistory()); err != nil {
				t.Fatalf("call %d: %v", i, err)
			}
		}
		got := requests()
		if len(got) != 3 {
			t.Fatalf("requests = %d, want 3", len(got))
		}
		want := [][]string{
			{"thinking", "thinking", "redacted_thinking", "tool_use"},
			{"tool_use"},
			{"thinking", "thinking", "redacted_thinking", "tool_use"},
		}
		for i, body := range got {
			if types := assistantBlockTypes(t, body); !reflect.DeepEqual(types, want[i]) {
				t.Errorf("request %d assistant blocks = %v, want %v", i, types, want[i])
			}
			if thinkingType(body) != "adaptive" {
				t.Errorf("request %d thinking = %v, want adaptive", i, body["thinking"])
			}
		}
	})
}

func TestSignatureRejectionRetriedOnce(t *testing.T) {
	srv, requests := effortServer(t, signatureRejected, signatureRejected)
	c := newEffortClient(t, srv.URL, llm.Options{})
	err := sendEffort(t, c, false, thinkingHistory())
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("want ErrInvalidRequest, got %v", err)
	}
	if got := requests(); len(got) != 2 {
		t.Errorf("requests = %d, want 2", len(got))
	}
}

func TestModifiedThinkingRejectionNotLearned(t *testing.T) {
	// A rejection of the replayed blocks is about this conversation,
	// so it must not stop the client sending adaptive thinking.
	modified := `{"type":"error","error":{"type":"invalid_request_error","message":` +
		"\"`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified.\"}}"
	srv, requests := effortServer(t, modified)
	c := newEffortClient(t, srv.URL, llm.Options{})
	if err := sendEffort(t, c, false, thinkingHistory()); !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("want ErrInvalidRequest, got %v", err)
	}
	if err := sendEffort(t, c, false, thinkingHistory()); err != nil {
		t.Fatalf("second call: %v", err)
	}
	got := requests()
	if len(got) != 2 || thinkingType(got[1]) != "adaptive" {
		t.Errorf("want adaptive thinking still sent: %v", got)
	}
}

func TestSignatureRejectionWithoutBlocksNotRetried(t *testing.T) {
	// With no blocks to remove there is nothing to retry, and the
	// message names thinking, so it must not be put down to the type
	// either.
	srv, requests := effortServer(t, signatureRejected)
	c := newEffortClient(t, srv.URL, llm.Options{})
	req := thinkingHistory()
	req.Messages[1].Content = req.Messages[1].Content[3:]
	if err := sendEffort(t, c, false, req); err == nil {
		t.Fatal("want an error")
	}
	if got := requests(); len(got) != 1 {
		t.Errorf("requests = %d, want 1", len(got))
	}
}
