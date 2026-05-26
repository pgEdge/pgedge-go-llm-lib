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
	"reflect"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Extension carries Ollama-specific tunables that the unified Client
// API does not surface directly. Attach via llm.Options.Extensions
// (client-level); the Ollama provider reads it on each Embed and
// EmbedBatch call. Other providers ignore it.
type Extension struct {
	// EmbedContextLength, when non-zero, sets Ollama's "num_ctx" on
	// /api/embed requests. Use this to raise the model's context
	// window beyond its compiled default when chunks routinely exceed
	// it (e.g. nomic-embed-text ships with a 2048-token default and
	// rejects longer inputs with "the input length exceeds the
	// context length"). Zero (the default) omits "num_ctx" from the
	// request body and Ollama uses the model's default.
	EmbedContextLength int

	// EmbedTruncateOnOverflow, when true, makes Embed and EmbedBatch
	// retry a request whose input exceeded the model's context window
	// (or crashed Ollama's model runner with HTTP 500) at progressively
	// smaller fractions of the original input — 75%, 50%, then 25% —
	// each cut at a word boundary. If all three truncated retries also
	// fail, the original error is returned. When false (default), such
	// errors surface immediately.
	EmbedTruncateOnOverflow bool
}

// ProviderName returns "ollama" so this extension is locatable in a
// generic []llm.ProviderExtension by callers and by the Ollama client.
func (Extension) ProviderName() string { return providerName }

// findExtension locates an ollama.Extension in a generic
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
