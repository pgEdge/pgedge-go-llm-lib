//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package anthropic

import (
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

func TestExtensionProviderName(t *testing.T) {
	ext := Extension{}
	if ext.ProviderName() != "anthropic" {
		t.Errorf("ProviderName() = %q, want \"anthropic\"", ext.ProviderName())
	}
}

func TestWithToolCachingAttachesExtension(t *testing.T) {
	tools := []llm.Tool{{Name: "a"}, {Name: "b"}}
	req := llm.ChatRequest{Tools: tools}
	out := WithToolCaching(req)

	if len(out.Tools) != 2 {
		t.Errorf("tools changed: %v", out.Tools)
	}
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext == nil {
		t.Fatal("no anthropic extension attached")
	}
	if ext.CacheToolsThrough != 1 { // last index
		t.Errorf("CacheToolsThrough = %d, want 1", ext.CacheToolsThrough)
	}
}

func TestWithToolCachingNoToolsNoOp(t *testing.T) {
	req := llm.ChatRequest{Tools: nil}
	out := WithToolCaching(req)
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext != nil {
		t.Errorf("WithToolCaching attached extension to no-tools request: %+v", ext)
	}
}

func TestWithExtendedThinkingAttachesExtension(t *testing.T) {
	req := llm.ChatRequest{}
	out := WithExtendedThinking(req, 16000)
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext == nil {
		t.Fatal("no anthropic extension attached")
	}
	if !ext.ExtendedThinking || ext.BudgetTokens != 16000 {
		t.Errorf("got %+v", ext)
	}
}

func TestExtensionPreservesOtherExtensions(t *testing.T) {
	// We use a dummy Extension value to verify that prior extensions
	// are preserved by WithToolCaching rather than overwritten.
	req := llm.ChatRequest{
		Tools:      []llm.Tool{{Name: "x"}},
		Extensions: []llm.ProviderExtension{Extension{CacheToolsThrough: -1}},
	}
	out := WithToolCaching(req)
	if len(out.Extensions) != 2 {
		t.Errorf("got %d extensions, want 2", len(out.Extensions))
	}
}
