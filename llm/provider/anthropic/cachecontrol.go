//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package anthropic

import "github.com/pgEdge/pgedge-go-llm-lib/llm"

// Extension carries Anthropic-specific request options. Attach via
// WithToolCaching, WithSystemCaching, WithExtendedThinking, or by
// appending an Extension value to ChatRequest.Extensions directly.
type Extension struct {
	// CacheToolsThrough is the index (inclusive) of the last tool
	// to mark as a cacheable prefix. Setting it to len(tools)-1
	// caches the entire tools block. Negative values disable tool
	// caching.
	CacheToolsThrough int

	// CacheSystem marks the system prompt as a cacheable prefix.
	// When true, the last system block on the wire gets
	// cache_control: ephemeral. Has no effect when the request has
	// no system prompt.
	CacheSystem bool

	// ExtendedThinking enables Anthropic's extended-thinking mode.
	ExtendedThinking bool

	// BudgetTokens caps the thinking-mode token budget. Ignored
	// unless ExtendedThinking is true.
	BudgetTokens int
}

// ProviderName implements llm.ProviderExtension.
func (Extension) ProviderName() string { return "anthropic" }

// WithToolCaching returns a copy of req with an Anthropic extension
// attached that marks the entire Tools block for prompt caching.
// Other providers ignore the extension. If req has no tools, the
// returned request is unchanged.
func WithToolCaching(req llm.ChatRequest) llm.ChatRequest {
	if len(req.Tools) == 0 {
		return req
	}
	return withAnthropicExtension(req, func(ext *Extension) {
		ext.CacheToolsThrough = len(req.Tools) - 1
	})
}

// WithSystemCaching returns a copy of req with an Anthropic extension
// attached that marks the system prompt as a cacheable prefix. Other
// providers ignore the extension. If req has no system prompt the
// extension is still attached but has no effect on the wire.
func WithSystemCaching(req llm.ChatRequest) llm.ChatRequest {
	return withAnthropicExtension(req, func(ext *Extension) {
		ext.CacheSystem = true
	})
}

// WithExtendedThinking returns a copy of req with extended-thinking
// mode enabled at the given token budget.
func WithExtendedThinking(req llm.ChatRequest, budgetTokens int) llm.ChatRequest {
	return withAnthropicExtension(req, func(ext *Extension) {
		ext.ExtendedThinking = true
		ext.BudgetTokens = budgetTokens
	})
}

// withAnthropicExtension returns a copy of req with mutate applied to
// its Anthropic Extension — either an existing entry (merged in
// place) or a freshly-appended one. This makes the With* helpers
// composable in any order: llm.FindExtension returns only the first
// matching extension, so all Anthropic-specific flags must live on a
// single Extension.
func withAnthropicExtension(req llm.ChatRequest, mutate func(*Extension)) llm.ChatRequest {
	out := req
	out.Extensions = append([]llm.ProviderExtension(nil), req.Extensions...)
	for i, e := range out.Extensions {
		if ext, ok := e.(Extension); ok {
			mutate(&ext)
			out.Extensions[i] = ext
			return out
		}
	}
	// CacheToolsThrough defaults to -1 so a freshly-created Extension
	// doesn't accidentally enable tool[0] caching when only system or
	// extended-thinking flags are requested.
	ext := Extension{CacheToolsThrough: -1}
	mutate(&ext)
	out.Extensions = append(out.Extensions, ext)
	return out
}
