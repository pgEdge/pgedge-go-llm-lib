//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"
)

func TestStreamChunkTypeConstants(t *testing.T) {
	cases := []struct {
		got  StreamChunkType
		want string
	}{
		{ChunkText, "text"},
		{ChunkToolUseStart, "tool_use_start"},
		{ChunkToolUseDelta, "tool_use_delta"},
		{ChunkDone, "done"},
	}
	for _, c := range cases {
		if string(c.got) != c.want {
			t.Errorf("StreamChunkType %q != %q", c.got, c.want)
		}
	}
}

func TestStreamChunkJSON(t *testing.T) {
	chunk := StreamChunk{
		Type:    ChunkToolUseDelta,
		Partial: `{"foo":`,
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"tool_use_delta","partial":"{\"foo\":"}`
	if string(data) != want {
		t.Errorf("got %s\nwant %s", data, want)
	}
}

func TestStreamChunkJSONOmitEmpty(t *testing.T) {
	chunk := StreamChunk{Type: ChunkText, Text: "hi"}
	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"text","text":"hi"}`
	if string(data) != want {
		t.Errorf("got %s\nwant %s", data, want)
	}
}

func TestStreamChunkDoneCarriesUsage(t *testing.T) {
	chunk := StreamChunk{
		Type:  ChunkDone,
		Usage: &TokenUsage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded StreamChunk
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.Type != ChunkDone {
		t.Errorf("type = %q, want %q", decoded.Type, ChunkDone)
	}
	if decoded.Usage == nil || decoded.Usage.TotalTokens != 15 {
		t.Errorf("usage not round-tripped: %+v", decoded.Usage)
	}
}

func TestStreamConsumption(t *testing.T) {
	chunks := make(chan StreamChunk, 3)
	errCh := make(chan error, 1)

	chunks <- StreamChunk{Type: ChunkText, Text: "Hello "}
	chunks <- StreamChunk{Type: ChunkText, Text: "world"}
	chunks <- StreamChunk{
		Type:  ChunkDone,
		Usage: &TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
	close(chunks)
	close(errCh)

	stream := &Stream{Chunks: chunks, Err: errCh}

	var fullText string
	var finalUsage *TokenUsage
	for chunk := range stream.Chunks {
		if chunk.Type == ChunkText {
			fullText += chunk.Text
		}
		if chunk.Type == ChunkDone {
			finalUsage = chunk.Usage
		}
	}

	if fullText != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", fullText)
	}
	if finalUsage == nil || finalUsage.TotalTokens != 15 {
		t.Errorf("unexpected usage: %+v", finalUsage)
	}

	if err, ok := <-stream.Err; ok && err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStreamWithToolUse(t *testing.T) {
	chunks := make(chan StreamChunk, 3)
	errCh := make(chan error, 1)

	toolUse := &ToolUse{ID: "tu_1", Name: "get_weather"}
	chunks <- StreamChunk{Type: ChunkToolUseStart, ToolUse: toolUse}
	chunks <- StreamChunk{Type: ChunkToolUseDelta, Partial: `{"city":"London"}`}
	chunks <- StreamChunk{Type: ChunkDone}
	close(chunks)
	close(errCh)

	stream := &Stream{Chunks: chunks, Err: errCh}

	var gotToolStart bool
	for chunk := range stream.Chunks {
		if chunk.Type == ChunkToolUseStart && chunk.ToolUse != nil {
			gotToolStart = true
			if chunk.ToolUse.Name != "get_weather" {
				t.Errorf("expected get_weather, got %s", chunk.ToolUse.Name)
			}
		}
		if chunk.Type == ChunkToolUseDelta {
			if chunk.Partial == "" {
				t.Error("ChunkToolUseDelta: Partial must be non-empty")
			}
			if chunk.Text != "" {
				t.Errorf("ChunkToolUseDelta: Text must be empty, got %q", chunk.Text)
			}
		}
	}
	if !gotToolStart {
		t.Error("expected tool_use_start chunk")
	}
}

func TestStreamRecvDeliversChunks(t *testing.T) {
	chunks := make(chan StreamChunk, 2)
	errCh := make(chan error, 1)
	chunks <- StreamChunk{Type: ChunkText, Text: "hello"}
	chunks <- StreamChunk{Type: ChunkDone, Usage: &TokenUsage{TotalTokens: 1}}
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}

	c1, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv #1: %v", err)
	}
	if c1.Type != ChunkText || c1.Text != "hello" {
		t.Errorf("Recv #1: %+v", c1)
	}

	c2, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv #2: %v", err)
	}
	if c2.Type != ChunkDone {
		t.Errorf("Recv #2: %+v", c2)
	}

	_, err = s.Recv()
	if !errors.Is(err, io.EOF) {
		t.Errorf("Recv #3: want io.EOF, got %v", err)
	}
}

func TestStreamRecvSurfacesErrors(t *testing.T) {
	chunks := make(chan StreamChunk)
	errCh := make(chan error, 1)
	errCh <- errors.New("upstream exploded")
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}

	_, err := s.Recv()
	if err == nil || err.Error() != "upstream exploded" {
		t.Errorf("Recv: want upstream error, got %v", err)
	}
}

func TestStreamRecvEOFOnEmptyClose(t *testing.T) {
	chunks := make(chan StreamChunk)
	errCh := make(chan error)
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}

	_, err := s.Recv()
	if !errors.Is(err, io.EOF) {
		t.Errorf("Recv: want io.EOF, got %v", err)
	}
}

func TestStreamRecvDrainsBufferedErrorAfterChunksClose(t *testing.T) {
	// Reproduce the path where the outer select picks Chunks (closed)
	// and the inner non-blocking select drains a buffered error from
	// Err. We force this by leaving Chunks closed empty so the outer
	// select cannot pick it for a value, but the closed-channel arm
	// is always ready. errCh has a buffered error and remains open
	// (not closed) so the outer select cannot pick it via close —
	// only the buffered value is available, and Go's fairness rules
	// will eventually have the inner drain catch it.
	//
	// Run with -count to validate determinism: the buffered error
	// must always win over io.EOF regardless of which arm of the
	// outer select fires first.
	for i := 0; i < 50; i++ {
		chunks := make(chan StreamChunk)
		errCh := make(chan error, 1)
		errCh <- errors.New("buffered failure")
		close(chunks)
		// Note: errCh stays open with the buffered error in it.

		s := &Stream{Chunks: chunks, Err: errCh}

		_, err := s.Recv()
		if err == nil || err.Error() != "buffered failure" {
			t.Fatalf("iter %d: want buffered failure, got %v", i, err)
		}
	}
}

func TestStreamCollect(t *testing.T) {
	chunks := make(chan StreamChunk, 6)
	errCh := make(chan error, 1)
	chunks <- StreamChunk{Type: ChunkText, Text: "hello "}
	chunks <- StreamChunk{Type: ChunkText, Text: "world"}
	chunks <- StreamChunk{Type: ChunkToolUseStart, ToolUse: &ToolUse{ID: "tu_1", Name: "search"}}
	chunks <- StreamChunk{Type: ChunkToolUseDelta, Partial: `{"q":`}
	chunks <- StreamChunk{Type: ChunkToolUseDelta, Partial: `"hi"}`}
	chunks <- StreamChunk{Type: ChunkDone, Usage: &TokenUsage{TotalTokens: 5}}
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}
	resp, err := s.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if resp.Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if len(resp.Content) < 2 {
		t.Fatalf("got %d blocks, want >=2", len(resp.Content))
	}
	if resp.Content[0].Type != BlockText || resp.Content[0].Text != "hello world" {
		t.Errorf("text block: %+v", resp.Content[0])
	}
	if resp.Content[1].Type != BlockToolUse || resp.Content[1].ToolUse == nil {
		t.Fatalf("tool block: %+v", resp.Content[1])
	}
	if resp.Content[1].ToolUse.ID != "tu_1" || resp.Content[1].ToolUse.Name != "search" {
		t.Errorf("tool ID/name: %+v", resp.Content[1].ToolUse)
	}
	if string(resp.Content[1].ToolUse.Input) != `{"q":"hi"}` {
		t.Errorf("input = %q", resp.Content[1].ToolUse.Input)
	}
}

func TestStreamCollectEmptyStream(t *testing.T) {
	chunks := make(chan StreamChunk)
	errCh := make(chan error)
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}
	resp, err := s.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(resp.Content) != 0 {
		t.Errorf("expected empty Content, got %+v", resp.Content)
	}
}

func TestStreamCollectPartialThenError(t *testing.T) {
	chunks := make(chan StreamChunk, 2)
	errCh := make(chan error, 1)
	chunks <- StreamChunk{Type: ChunkText, Text: "partial"}
	errCh <- io.ErrUnexpectedEOF
	close(chunks)
	close(errCh)

	s := &Stream{Chunks: chunks, Err: errCh}
	resp, err := s.Collect(context.Background())
	if err == nil {
		t.Fatal("expected error from underlying stream")
	}
	// Even on error, the partial response is returned.
	if len(resp.Content) != 1 || resp.Content[0].Text != "partial" {
		t.Errorf("partial content lost: %+v", resp.Content)
	}
}

func TestStreamCollectCancellation(t *testing.T) {
	chunks := make(chan StreamChunk) // never sends
	errCh := make(chan error)
	// don't close — Recv will block

	s := &Stream{Chunks: chunks, Err: errCh}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	_, err := s.Collect(ctx)
	if err == nil {
		t.Fatal("expected ctx.Err(), got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// TestStreamCollectDrainsAllBlockTypesAfterError exercises the
// drain-after-error loop in Collect over each of the inner switch
// arms (text, tool-use start, tool-use delta, done). Recv's select is
// non-deterministic when both Chunks and Err are ready, so the test
// loops to give every arm a chance to fire at least once across
// iterations. Coverage accumulates regardless of which branch each
// individual iteration takes.
func TestStreamCollectDrainsAllBlockTypesAfterError(t *testing.T) {
	for i := 0; i < 200; i++ {
		chunks := make(chan StreamChunk, 8)
		errCh := make(chan error, 1)

		// Pre-populate a tool-use start, an args delta, an additional
		// text chunk, and a done so the drain loop has something to
		// pull from in every arm of its inner switch.
		chunks <- StreamChunk{Type: ChunkText, Text: "leading "}
		chunks <- StreamChunk{Type: ChunkToolUseStart, ToolUse: &ToolUse{
			ID:    "tu-drain",
			Name:  "do_something",
			Input: []byte(`{"start":1`),
		}}
		chunks <- StreamChunk{Type: ChunkToolUseDelta, Partial: `,"end":2}`}
		chunks <- StreamChunk{Type: ChunkDone, Usage: &TokenUsage{TotalTokens: 9}}
		errCh <- io.ErrUnexpectedEOF
		close(chunks)
		close(errCh)

		s := &Stream{Chunks: chunks, Err: errCh}
		resp, err := s.Collect(context.Background())

		// Whether Collect returns an error (drain branch) or nil
		// (chunks-then-EOF-with-err-already-consumed branch), the
		// assembled response must always contain BOTH the leading
		// text and the tool-use block with concatenated input. A
		// dropped chunk would mean the drain logic regressed.
		_ = err
		var sawText, sawTool bool
		for _, b := range resp.Content {
			switch b.Type {
			case BlockText:
				if b.Text == "leading " {
					sawText = true
				}
			case BlockToolUse:
				if b.ToolUse != nil && string(b.ToolUse.Input) == `{"start":1,"end":2}` {
					sawTool = true
				}
			}
		}
		if !sawText || !sawTool {
			t.Fatalf("iter %d: drained content lost; sawText=%v sawTool=%v content=%+v",
				i, sawText, sawTool, resp.Content)
		}
	}
}
