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

	// ChunkThinkingStart announces a new block of the model's
	// reasoning in StreamChunk.Thinking. A redacted block arrives
	// whole, with Thinking.Data populated; a thinking block's text and
	// signature usually follow in ChunkThinkingDelta chunks.
	ChunkThinkingStart StreamChunkType = "thinking_start"

	// ChunkThinkingDelta carries an incremental fragment of the
	// current thinking block's Text or Signature in
	// StreamChunk.Thinking. Consumers concatenate each field
	// separately to reconstruct the block.
	ChunkThinkingDelta StreamChunkType = "thinking_delta"

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
//	ChunkThinkingStart   -> Thinking
//	ChunkThinkingDelta   -> Thinking
//	ChunkDone            -> Usage
type StreamChunk struct {
	Type     StreamChunkType `json:"type"`
	Text     string          `json:"text,omitempty"`
	ToolUse  *ToolUse        `json:"tool_use,omitempty"`
	Thinking *Thinking       `json:"thinking,omitempty"`
	Partial  string          `json:"partial,omitempty"`
	Usage    *TokenUsage     `json:"usage,omitempty"`
}

// ThinkingAccumulator assembles ChunkThinkingStart and
// ChunkThinkingDelta chunks into thinking content blocks. Stream.Collect
// uses one, and it is exported for consumers that assemble a response
// from chunks themselves, such as the proxy's SSE bridge. The zero
// value is ready to use.
type ThinkingAccumulator struct {
	current   *Thinking
	text      strings.Builder
	signature strings.Builder
}

// Add applies a thinking chunk. A ChunkThinkingStart returns the block
// it ends, if one was in progress, so the caller can append it to the
// response; chunks of any other type are ignored.
func (a *ThinkingAccumulator) Add(c StreamChunk) (ContentBlock, bool) {
	switch c.Type {
	case ChunkThinkingStart:
		done, ok := a.Flush()
		a.current = &Thinking{}
		if c.Thinking != nil {
			a.current.Data = c.Thinking.Data
			a.text.WriteString(c.Thinking.Text)
			a.signature.WriteString(c.Thinking.Signature)
		}
		return done, ok
	case ChunkThinkingDelta:
		if a.current != nil && c.Thinking != nil {
			a.text.WriteString(c.Thinking.Text)
			a.signature.WriteString(c.Thinking.Signature)
		}
	}
	return ContentBlock{}, false
}

// Flush returns the block in progress, if any, and resets the
// accumulator. A block with Data set is a BlockRedactedThinking, and
// any other a BlockThinking.
func (a *ThinkingAccumulator) Flush() (ContentBlock, bool) {
	if a.current == nil {
		return ContentBlock{}, false
	}
	t := a.current
	t.Text = a.text.String()
	t.Signature = a.signature.String()
	a.current = nil
	a.text.Reset()
	a.signature.Reset()
	if t.Data != "" {
		return ContentBlock{Type: BlockRedactedThinking, Thinking: t}, true
	}
	return ContentBlock{Type: BlockThinking, Thinking: t}, true
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
	var thinking ThinkingAccumulator

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
	flushThinking := func() {
		if b, ok := thinking.Flush(); ok {
			resp.Content = append(resp.Content, b)
		}
	}
	flushAll := func() {
		flushText()
		flushTool()
		flushThinking()
	}
	accumulate := func(c StreamChunk) {
		switch c.Type {
		case ChunkText:
			flushTool()
			flushThinking()
			textBuf.WriteString(c.Text)
		case ChunkToolUseStart:
			flushAll()
			if c.ToolUse != nil {
				currentTool = &ToolUse{ID: c.ToolUse.ID, Name: c.ToolUse.Name, Input: c.ToolUse.Input, Signature: c.ToolUse.Signature}
				if len(currentTool.Input) > 0 {
					toolBuf.Write(currentTool.Input)
				}
			}
		case ChunkToolUseDelta:
			toolBuf.WriteString(c.Partial)
		case ChunkThinkingStart:
			flushAll()
			thinking.Add(c)
		case ChunkThinkingDelta:
			thinking.Add(c)
		case ChunkDone:
			if c.Usage != nil {
				resp.Usage = *c.Usage
			}
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			flushAll()
			return resp, err
		}
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			flushAll()
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
					accumulate(c)
				default:
					goto doneWithDrain
				}
			}
		doneWithDrain:
			flushAll()
			return resp, err
		}

		accumulate(chunk)
		if chunk.Type == ChunkDone {
			flushAll()
			return resp, nil
		}
	}
}
