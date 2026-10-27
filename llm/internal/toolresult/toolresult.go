//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package toolresult holds the representation of a failed tool result
// shared by the providers whose wire format has no field to mark one.
//
// Anthropic sends llm.ContentBlock.IsError as is_error. OpenAI's tool
// message (Chat Completions), its function_call_output item (Responses
// API) and Ollama's tool message carry only the result text, so a
// failure would otherwise reach the model looking exactly like a
// success. Those providers send the text as {"error": "<text>"}
// instead, mirroring the "error" key the Gemini provider uses in its
// functionResponse.
package toolresult

import (
	"bytes"
	"encoding/json"
)

// UnspecifiedError is sent when a tool result is flagged as an error
// but carries no text. An empty payload would be indistinguishable from
// a successful call that returned nothing, so something explicit is
// sent instead.
const UnspecifiedError = "tool execution failed; no error message was provided"

// Content returns the text to send for a tool result: text unchanged
// when isError is false, otherwise a JSON object holding text (or
// UnspecifiedError when text is empty) under an "error" key.
func Content(text string, isError bool) string {
	if !isError {
		return text
	}
	if text == "" {
		text = UnspecifiedError
	}

	// HTML escaping is turned off so that an error such as "x < y"
	// reaches the model as written rather than as "x \u003c y".
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	// Encoding a struct with a single string field cannot fail.
	_ = enc.Encode(struct {
		Error string `json:"error"`
	}{text})
	return string(bytes.TrimSuffix(buf.Bytes(), []byte("\n")))
}
