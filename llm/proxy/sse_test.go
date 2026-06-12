//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package proxy

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// statusErr is a minimal error type implementing HTTPStatus() int, used
// to prove httpStatusOf traverses wrapped errors via errors.As.
type statusErr struct{ code int }

func (statusErr) Error() string     { return "teapot" }
func (e statusErr) HTTPStatus() int { return e.code }

// TestHTTPStatusOfWrapped verifies httpStatusOf reaches an HTTPStatus()
// implementation buried under %w wrapping rather than only matching a
// direct type assertion.
func TestHTTPStatusOfWrapped(t *testing.T) {
	wrapped := fmt.Errorf("context: %w", statusErr{code: 418})
	if got := httpStatusOf(wrapped, http.StatusBadRequest); got != 418 {
		t.Fatalf("httpStatusOf(wrapped) = %d, want 418", got)
	}

	// A plain error with no HTTPStatus method falls back.
	if got := httpStatusOf(errors.New("plain"), http.StatusBadRequest); got != http.StatusBadRequest {
		t.Fatalf("httpStatusOf(plain) = %d, want 400", got)
	}

	// A direct (unwrapped) implementer still resolves.
	if got := httpStatusOf(statusErr{code: 403}, http.StatusBadRequest); got != 403 {
		t.Fatalf("httpStatusOf(direct) = %d, want 403", got)
	}
}

// fakeFlushBuffer captures writes and counts Flush calls.
type fakeFlushBuffer struct {
	bytes.Buffer
	flushes int
}

func (b *fakeFlushBuffer) Flush() { b.flushes++ }

func TestSSEWriterEmitsChunkAndDone(t *testing.T) {
	chunks := make(chan llm.StreamChunk, 3)
	errCh := make(chan error, 1)
	chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "hi"}
	chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 1}}
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	resp, err := writeSSE(buf, stream)
	if err != nil {
		t.Fatalf("writeSSE: %v", err)
	}
	if resp == nil {
		t.Fatal("writeSSE returned nil response")
	}
	if resp.Usage.TotalTokens != 1 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != llm.BlockText || resp.Content[0].Text != "hi" {
		t.Errorf("accumulated Content = %+v, want one BlockText 'hi'", resp.Content)
	}

	out := buf.String()
	if !strings.Contains(out, `data: {"type":"text","text":"hi"}`) {
		t.Errorf("missing text chunk in: %s", out)
	}
	if !strings.Contains(out, `event: done`) {
		t.Errorf("missing done event in: %s", out)
	}
	if buf.flushes < 2 {
		t.Errorf("flushes = %d, want >= 2", buf.flushes)
	}
}

func TestSSEWriterEmitsErrorEvent(t *testing.T) {
	chunks := make(chan llm.StreamChunk)
	errCh := make(chan error, 1)
	errCh <- errors.New("boom")
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	_, err := writeSSE(buf, stream)
	if err == nil {
		t.Fatal("expected error to be returned")
	}

	out := buf.String()
	if !strings.Contains(out, `event: error`) {
		t.Errorf("missing error event in: %s", out)
	}
	if !strings.Contains(out, `boom`) {
		t.Errorf("error message missing in: %s", out)
	}
}

// Verify writeSSE's writer interface is exactly what http.ResponseWriter satisfies.
var _ io.Writer = (*fakeFlushBuffer)(nil)

func TestSSEWriterEmitsSyntheticDoneOnEarlyEOF(t *testing.T) {
	chunks := make(chan llm.StreamChunk, 1)
	errCh := make(chan error, 1)
	chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "hello"}
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	_, err := writeSSE(buf, stream)
	if err != nil {
		t.Fatalf("writeSSE: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, `data: {"type":"text","text":"hello"}`) {
		t.Errorf("missing text chunk: %s", out)
	}
	if !strings.Contains(out, `event: done`) {
		t.Errorf("missing synthetic done: %s", out)
	}
}

// TestSSEWriterAccumulatesToolUseAcrossDeltas verifies that writeSSE
// folds tool-use start + delta chunks into a single ContentBlock with
// concatenated Input, mirroring Stream.Collect's assembly. This is
// the path proxy.OnResponse relies on to surface tool-call payloads
// for streaming requests.
func TestSSEWriterAccumulatesToolUseAcrossDeltas(t *testing.T) {
	chunks := make(chan llm.StreamChunk, 5)
	errCh := make(chan error, 1)
	chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "calling "}
	chunks <- llm.StreamChunk{Type: llm.ChunkToolUseStart, ToolUse: &llm.ToolUse{
		ID:   "tu_42",
		Name: "get_weather",
	}}
	chunks <- llm.StreamChunk{Type: llm.ChunkToolUseDelta, Partial: `{"city":"`}
	chunks <- llm.StreamChunk{Type: llm.ChunkToolUseDelta, Partial: `London"}`}
	chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 9}}
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	resp, err := writeSSE(buf, stream)
	if err != nil {
		t.Fatalf("writeSSE: %v", err)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("Content = %+v, want 2 blocks (text + tool_use)", resp.Content)
	}
	if resp.Content[0].Type != llm.BlockText || resp.Content[0].Text != "calling " {
		t.Errorf("block 0 = %+v, want BlockText 'calling '", resp.Content[0])
	}
	tu := resp.Content[1]
	if tu.Type != llm.BlockToolUse || tu.ToolUse == nil {
		t.Fatalf("block 1 = %+v, want BlockToolUse", tu)
	}
	if tu.ToolUse.ID != "tu_42" || tu.ToolUse.Name != "get_weather" {
		t.Errorf("ToolUse = %+v, want ID=tu_42 Name=get_weather", tu.ToolUse)
	}
	if string(tu.ToolUse.Input) != `{"city":"London"}` {
		t.Errorf("ToolUse.Input = %q, want %q", string(tu.ToolUse.Input), `{"city":"London"}`)
	}
	if resp.Usage.TotalTokens != 9 {
		t.Errorf("Usage = %+v, want TotalTokens=9", resp.Usage)
	}
}

// TestSSEWriterReturnsNonNilResponseOnError verifies that writeSSE
// always returns a non-nil response, even when the stream errors
// before any data arrives, so OnResponse hooks can rely on
// dereferencing it without a nil check.
func TestSSEWriterReturnsNonNilResponseOnError(t *testing.T) {
	chunks := make(chan llm.StreamChunk)
	errCh := make(chan error, 1)
	errCh <- errors.New("stream died before any chunks")
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	resp, err := writeSSE(buf, stream)
	if err == nil {
		t.Fatal("expected error from writeSSE")
	}
	if resp == nil {
		t.Fatal("response must be non-nil even on stream error")
	}
	// No chunks were delivered, so Content is empty and Usage is zero.
	if len(resp.Content) != 0 {
		t.Errorf("Content = %+v, want empty", resp.Content)
	}
	if resp.Usage != (llm.TokenUsage{}) {
		t.Errorf("Usage = %+v, want zero value", resp.Usage)
	}
	if !strings.Contains(buf.String(), "event: error") {
		t.Errorf("missing error event: %s", buf.String())
	}
}

func TestDisplayNameFor(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"anthropic", "Anthropic"},
		{"openai", "OpenAI"},
		{"weirdcustom", "weirdcustom"},
	}
	for _, tc := range cases {
		if got := displayNameFor(tc.input); got != tc.want {
			t.Errorf("displayNameFor(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestSSEWriterDoesNotDoubleEmitDone(t *testing.T) {
	chunks := make(chan llm.StreamChunk, 2)
	errCh := make(chan error, 1)
	chunks <- llm.StreamChunk{Type: llm.ChunkText, Text: "hi"}
	chunks <- llm.StreamChunk{Type: llm.ChunkDone, Usage: &llm.TokenUsage{TotalTokens: 7}}
	close(chunks)
	close(errCh)

	stream := &llm.Stream{Chunks: chunks, Err: errCh}
	buf := &fakeFlushBuffer{}

	_, err := writeSSE(buf, stream)
	if err != nil {
		t.Fatalf("writeSSE: %v", err)
	}

	out := buf.String()
	count := strings.Count(out, "event: done")
	if count != 1 {
		t.Errorf("got %d done events, want exactly 1\n%s", count, out)
	}
}
