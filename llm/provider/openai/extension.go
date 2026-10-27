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
	"reflect"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Extension carries OpenAI-specific tunables that the unified Client
// API does not surface directly. Attach via llm.Options.Extensions
// (client-level); the OpenAI provider reads it on each Embed,
// EmbedBatch, Chat, and ChatStream call. ExtraBody is also read from a
// ChatRequest's Extensions. Other providers ignore it.
type Extension struct {
	// EmbeddingDimensions, when non-zero, asks OpenAI to return a
	// lower-dimensional vector from a model that supports truncation
	// (e.g. text-embedding-3-small accepts 256..1536; text-embedding-3-
	// large accepts up to 3072). Zero (default) omits the parameter and
	// the model returns its native vector size.
	EmbeddingDimensions int

	// ResponsesAPI overrides automatic Chat-Completions vs. Responses
	// routing. By default (nil), the provider uses /v1/chat/completions
	// and switches the client to /v1/responses the first time OpenAI
	// reports that the model is only available there. Set to
	// llm.Bool(true) to force the Responses API, or llm.Bool(false) to
	// force the Chat Completions API for every Chat / ChatStream call
	// and disable the automatic switch.
	ResponsesAPI *bool

	// ExtraBody holds fields to add to the JSON body of Chat and
	// ChatStream requests, on both /v1/chat/completions and
	// /v1/responses, for fields the library does not model (such as a
	// gateway's own options). Each value is encoded with encoding/json
	// and set as a top-level field after the library builds the body,
	// so a key the library also sets is replaced by the extra value. An
	// error that names one of these fields is returned to the caller
	// without the client adjusting and resending the request.
	//
	// It is read from a request's Extensions and from
	// llm.Options.Extensions; the request's map is merged over the
	// client's, key by key, and both over llm.Options.ExtraBody.
	ExtraBody map[string]any
}

// ProviderName returns "openai" so this extension is locatable in a
// generic []llm.ProviderExtension by callers and by the OpenAI client.
func (Extension) ProviderName() string { return providerName }

// findExtension locates an openai.Extension in a generic
// []llm.ProviderExtension, accepting both value and pointer forms.
// Returns nil when no matching extension is present. Nil and
// typed-nil pointer entries are skipped defensively so a malformed
// Options.Extensions slice never panics here.
func findExtension(exts []llm.ProviderExtension) *Extension {
	for _, e := range exts {
		if e == nil {
			continue
		}
		if rv := reflect.ValueOf(e); rv.Kind() == reflect.Pointer && rv.IsNil() {
			continue
		}
		if e.ProviderName() != providerName {
			continue
		}
		if ext, ok := e.(Extension); ok {
			return &ext
		}
		if ext, ok := e.(*Extension); ok {
			return ext
		}
	}
	return nil
}
