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
// WithToolCaching, WithExtendedThinking, or by appending an
// Extension value to ChatRequest.Extensions directly.
type Extension struct {
	// CacheToolsThrough is the index (inclusive) of the last tool
	// to mark as a cacheable prefix. Setting it to len(tools)-1
	// caches the entire tools block. Negative values disable tool
	// caching.
	CacheToolsThrough int

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
	out := req
	out.Extensions = append([]llm.ProviderExtension(nil), req.Extensions...)
	out.Extensions = append(out.Extensions, Extension{CacheToolsThrough: len(req.Tools) - 1})
	return out
}

// WithExtendedThinking returns a copy of req with extended-thinking
// mode enabled at the given token budget.
func WithExtendedThinking(req llm.ChatRequest, budgetTokens int) llm.ChatRequest {
	out := req
	out.Extensions = append([]llm.ProviderExtension(nil), req.Extensions...)
	out.Extensions = append(out.Extensions, Extension{
		ExtendedThinking: true,
		BudgetTokens:     budgetTokens,
	})
	return out
}
