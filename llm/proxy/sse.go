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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/redact"
)

// flusher is the subset of http.Flusher we need. Mock-friendly.
type flusher interface {
	Flush()
}

// writeSSE serialises a *llm.Stream onto w as Server-Sent Events and
// concurrently assembles the same chunks into a *llm.ChatResponse so
// proxy hooks can see the full response payload.
//
// Wire format:
//
//	data: <json-encoded StreamChunk>\n\n   (one per non-done chunk)
//	event: done\ndata: <json>\n\n          (final chunk; data is the StreamChunk with Usage)
//	event: error\ndata: {"error":"..."}\n\n   (only on stream error)
//
// The returned *ChatResponse is always non-nil. On stream error it
// carries whatever chunks were already received; on early EOF without
// an explicit ChunkDone its Usage is the zero value. The caller is
// responsible for setting response headers (Content-Type,
// Cache-Control) before calling.
//
// Any secrets given are stripped from an error event before it reaches
// the wire, alongside anything else credential-shaped; see
// Proxy.redactError for why this layer exists even though provider
// errors are already redacted at source.
func writeSSE(w io.Writer, stream *llm.Stream, secrets ...string) (*llm.ChatResponse, error) {
	resp := &llm.ChatResponse{}
	sawDone := false

	flush := func() {
		if f, ok := w.(flusher); ok {
			f.Flush()
		}
	}

	// Accumulator state mirrors Stream.Collect: text and tool-use
	// deltas merge into their owning content block, flushed when a
	// boundary chunk arrives.
	var textBuf strings.Builder
	var toolBuf strings.Builder
	var currentTool *llm.ToolUse

	flushText := func() {
		if textBuf.Len() > 0 {
			resp.Content = append(resp.Content, llm.ContentBlock{Type: llm.BlockText, Text: textBuf.String()})
			textBuf.Reset()
		}
	}
	flushTool := func() {
		if currentTool != nil {
			currentTool.Input = json.RawMessage(toolBuf.String())
			resp.Content = append(resp.Content, llm.ContentBlock{Type: llm.BlockToolUse, ToolUse: currentTool})
			currentTool = nil
			toolBuf.Reset()
		}
	}
	accumulate := func(chunk llm.StreamChunk) {
		switch chunk.Type {
		case llm.ChunkText:
			flushTool()
			textBuf.WriteString(chunk.Text)
		case llm.ChunkToolUseStart:
			flushText()
			flushTool()
			if chunk.ToolUse != nil {
				currentTool = &llm.ToolUse{ID: chunk.ToolUse.ID, Name: chunk.ToolUse.Name, Input: chunk.ToolUse.Input}
				if len(currentTool.Input) > 0 {
					toolBuf.Write(currentTool.Input)
				}
			}
		case llm.ChunkToolUseDelta:
			toolBuf.WriteString(chunk.Partial)
		}
	}

	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			if !sawDone {
				// Synthesise a done event so SSE consumers always see
				// a terminator, even when the upstream stream closes
				// without an explicit ChunkDone.
				payload, _ := json.Marshal(llm.StreamChunk{Type: llm.ChunkDone})
				fmt.Fprintf(w, "event: done\ndata: %s\n\n", payload)
				flush()
			}
			flushText()
			flushTool()
			return resp, nil
		}
		if err != nil {
			payload, _ := json.Marshal(ErrorResponse{Error: redact.Message(err.Error(), secrets...)})
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
			flush()
			flushText()
			flushTool()
			return resp, err
		}

		payload, mErr := json.Marshal(chunk)
		if mErr != nil {
			flushText()
			flushTool()
			return resp, fmt.Errorf("marshal chunk: %w", mErr)
		}

		if chunk.Type == llm.ChunkDone {
			if chunk.Usage != nil {
				resp.Usage = *chunk.Usage
			}
			fmt.Fprintf(w, "event: done\ndata: %s\n\n", payload)
			flush()
			flushText()
			flushTool()
			// sawDone is intentionally not flipped here; this branch
			// returns immediately and only the EOF branch reads it.
			return resp, nil
		}

		accumulate(chunk)
		fmt.Fprintf(w, "data: %s\n\n", payload)
		flush()
	}
}
