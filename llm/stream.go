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
	"strings"
)

// StreamChunkType identifies the kind of streaming event emitted by a
// provider. Use the constants below — providers must not emit
// arbitrary string values.
type StreamChunkType string

const (
	// ChunkText carries an incremental text delta in StreamChunk.Text.
	ChunkText StreamChunkType = "text"

	// ChunkToolUseStart announces a new tool call. ToolUse.ID and
	// ToolUse.Name are populated; ToolUse.Input may be empty (deltas
	// follow) or fully populated for providers that send arguments
	// in one shot (Gemini, Ollama).
	ChunkToolUseStart StreamChunkType = "tool_use_start"

	// ChunkToolUseDelta carries an incremental fragment of the
	// current tool call's JSON arguments in StreamChunk.Partial.
	// Consumers concatenate Partial fragments to reconstruct the
	// full ToolUse.Input.
	ChunkToolUseDelta StreamChunkType = "tool_use_delta"

	// ChunkDone is the final chunk emitted by a stream. Usage is
	// always non-nil; if the upstream provider does not report
	// token counts, Usage is the zero TokenUsage.
	ChunkDone StreamChunkType = "done"
)

// Stream represents a streaming response from an LLM provider.
//
// Consumers should prefer Stream.Recv over reading the channels
// directly — Recv handles channel-close coordination and surfaces
// stream errors as Go errors. The Chunks and Err fields remain
// exported for advanced callers that need select-driven cancellation.
type Stream struct {
	Chunks <-chan StreamChunk
	Err    <-chan error
}

// StreamChunk is one event in a streaming response. The Type field
// determines which other fields are populated:
//
//	ChunkText            -> Text
//	ChunkToolUseStart    -> ToolUse
//	ChunkToolUseDelta    -> Partial
//	ChunkDone            -> Usage
type StreamChunk struct {
	Type    StreamChunkType `json:"type"`
	Text    string          `json:"text,omitempty"`
	ToolUse *ToolUse        `json:"tool_use,omitempty"`
	Partial string          `json:"partial,omitempty"`
	Usage   *TokenUsage     `json:"usage,omitempty"`
}

// Recv returns the next chunk from the stream.
//
// At end of stream Recv returns the zero StreamChunk and io.EOF. If
// the stream recorded an error, that error is returned instead of
// io.EOF — the error may arrive before, during, or after chunks.
// On any non-nil error return, the returned StreamChunk is the zero
// value.
//
// Once Recv returns a non-nil error (including io.EOF), subsequent
// calls may block; callers should stop calling Recv after a terminal
// return.
//
// Recv is not safe for concurrent use from multiple goroutines.
//
// Recv is the recommended way to consume a Stream; the underlying
// Chunks/Err channels remain exported for advanced callers that need
// select-driven cancellation.
//
// See also Collect, which drains the stream and assembles a ChatResponse.
func (s *Stream) Recv() (StreamChunk, error) {
	select {
	case chunk, ok := <-s.Chunks:
		if ok {
			return chunk, nil
		}
		// Chunks channel closed — check for a buffered error.
		select {
		case err, ok := <-s.Err:
			if ok && err != nil {
				return StreamChunk{}, err
			}
		default:
		}
		return StreamChunk{}, io.EOF
	case err, ok := <-s.Err:
		if ok && err != nil {
			return StreamChunk{}, err
		}
		// Err closed without an error; drain remaining chunks normally.
		chunk, ok := <-s.Chunks
		if !ok {
			return StreamChunk{}, io.EOF
		}
		return chunk, nil
	}
}

// Collect drains the stream and returns the assembled ChatResponse.
// Text deltas are concatenated; tool-use deltas are buffered into the
// preceding tool-use block's Input. The final TokenUsage is taken
// from the ChunkDone chunk.
//
// If the stream errors mid-flight, Collect returns the partial
// response up to the error along with the error.
//
// Collect respects ctx cancellation: if ctx is already cancelled
// when Collect is invoked, it returns immediately with ctx.Err().
// If ctx is cancelled mid-drain, it returns the partial response
// and ctx.Err().
func (s *Stream) Collect(ctx context.Context) (*ChatResponse, error) {
	resp := &ChatResponse{}
	var textBuf strings.Builder
	var toolBuf strings.Builder
	var currentTool *ToolUse

	flushText := func() {
		if textBuf.Len() > 0 {
			resp.Content = append(resp.Content, ContentBlock{Type: BlockText, Text: textBuf.String()})
			textBuf.Reset()
		}
	}
	flushTool := func() {
		if currentTool != nil {
			currentTool.Input = json.RawMessage(toolBuf.String())
			resp.Content = append(resp.Content, ContentBlock{Type: BlockToolUse, ToolUse: currentTool})
			currentTool = nil
			toolBuf.Reset()
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			flushText()
			flushTool()
			return resp, err
		}
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			flushText()
			flushTool()
			return resp, nil
		}
		if err != nil {
			// Drain any remaining buffered chunks so callers get the
			// partial response even when the error races ahead of data.
			for {
				select {
				case c, ok := <-s.Chunks:
					if !ok {
						goto doneWithDrain
					}
					switch c.Type {
					case ChunkText:
						flushTool()
						textBuf.WriteString(c.Text)
					case ChunkToolUseStart:
						flushText()
						flushTool()
						if c.ToolUse != nil {
							currentTool = &ToolUse{ID: c.ToolUse.ID, Name: c.ToolUse.Name, Input: c.ToolUse.Input, Signature: c.ToolUse.Signature}
							if len(currentTool.Input) > 0 {
								toolBuf.Write(currentTool.Input)
							}
						}
					case ChunkToolUseDelta:
						toolBuf.WriteString(c.Partial)
					case ChunkDone:
						if c.Usage != nil {
							resp.Usage = *c.Usage
						}
					}
				default:
					goto doneWithDrain
				}
			}
		doneWithDrain:
			flushText()
			flushTool()
			return resp, err
		}

		switch chunk.Type {
		case ChunkText:
			flushTool()
			textBuf.WriteString(chunk.Text)
		case ChunkToolUseStart:
			flushText()
			flushTool()
			if chunk.ToolUse != nil {
				currentTool = &ToolUse{ID: chunk.ToolUse.ID, Name: chunk.ToolUse.Name, Input: chunk.ToolUse.Input, Signature: chunk.ToolUse.Signature}
				if len(currentTool.Input) > 0 {
					toolBuf.Write(currentTool.Input)
				}
			}
		case ChunkToolUseDelta:
			toolBuf.WriteString(chunk.Partial)
		case ChunkDone:
			flushText()
			flushTool()
			if chunk.Usage != nil {
				resp.Usage = *chunk.Usage
			}
			return resp, nil
		}
	}
}
