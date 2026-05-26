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
// (client-level); the OpenAI provider reads it on each Embed and
// EmbedBatch call. Other providers ignore it.
type Extension struct {
	// EmbeddingDimensions, when non-zero, asks OpenAI to return a
	// lower-dimensional vector from a model that supports truncation
	// (e.g. text-embedding-3-small accepts 256..1536; text-embedding-3-
	// large accepts up to 3072). Zero (default) omits the parameter and
	// the model returns its native vector size.
	EmbeddingDimensions int
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
