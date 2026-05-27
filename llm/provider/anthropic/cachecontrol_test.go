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

func TestWithSystemCachingAttachesExtension(t *testing.T) {
	req := llm.ChatRequest{SystemPrompt: "you are a helpful assistant"}
	out := WithSystemCaching(req)

	if out.SystemPrompt != req.SystemPrompt {
		t.Errorf("SystemPrompt changed: %q", out.SystemPrompt)
	}
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext == nil {
		t.Fatal("no anthropic extension attached")
	}
	if !ext.CacheSystem {
		t.Errorf("CacheSystem = false, want true")
	}
	// Ensure WithSystemCaching does not inadvertently enable tool
	// caching for the first tool by leaving CacheToolsThrough at its
	// zero value.
	if ext.CacheToolsThrough >= 0 {
		t.Errorf("CacheToolsThrough = %d, want < 0", ext.CacheToolsThrough)
	}
}

func TestWithSystemCachingNoSystemPromptStillAttaches(t *testing.T) {
	// WithSystemCaching attaches the extension regardless; the wire
	// builder is responsible for skipping the marker when there is no
	// system block to tag.
	req := llm.ChatRequest{}
	out := WithSystemCaching(req)
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext == nil || !ext.CacheSystem {
		t.Fatalf("expected CacheSystem extension, got %+v", ext)
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

func TestWithHelpersComposeOrderIndependently(t *testing.T) {
	// FindExtension returns only the first anthropic Extension; the
	// helpers must therefore merge into a single Extension rather than
	// appending new ones, so any composition order preserves all flags.
	tools := []llm.Tool{{Name: "a"}, {Name: "b"}}
	base := llm.ChatRequest{Tools: tools, SystemPrompt: "sys"}

	for _, tc := range []struct {
		name string
		fn   func(llm.ChatRequest) llm.ChatRequest
	}{
		{"tool->system->thinking", func(r llm.ChatRequest) llm.ChatRequest {
			return WithExtendedThinking(WithSystemCaching(WithToolCaching(r)), 1000)
		}},
		{"system->tool->thinking", func(r llm.ChatRequest) llm.ChatRequest {
			return WithExtendedThinking(WithToolCaching(WithSystemCaching(r)), 1000)
		}},
		{"thinking->system->tool", func(r llm.ChatRequest) llm.ChatRequest {
			return WithToolCaching(WithSystemCaching(WithExtendedThinking(r, 1000)))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := tc.fn(base)
			if n := len(out.Extensions); n != 1 {
				t.Fatalf("got %d anthropic extensions, want 1", n)
			}
			ext := llm.FindExtension[Extension](out, "anthropic")
			if ext == nil {
				t.Fatal("no anthropic extension found")
			}
			if ext.CacheToolsThrough != len(tools)-1 {
				t.Errorf("CacheToolsThrough = %d, want %d", ext.CacheToolsThrough, len(tools)-1)
			}
			if !ext.CacheSystem {
				t.Errorf("CacheSystem = false, want true")
			}
			if !ext.ExtendedThinking || ext.BudgetTokens != 1000 {
				t.Errorf("ExtendedThinking=%v, BudgetTokens=%d", ext.ExtendedThinking, ext.BudgetTokens)
			}
		})
	}
}

// otherProviderExt is a non-anthropic ProviderExtension used to
// verify that the With* helpers preserve unrelated extensions.
type otherProviderExt struct{}

func (otherProviderExt) ProviderName() string { return "other" }

func TestExtensionPreservesOtherExtensions(t *testing.T) {
	// The With* helpers must preserve non-anthropic extensions while
	// merging into a single anthropic Extension.
	req := llm.ChatRequest{
		Tools: []llm.Tool{{Name: "x"}},
		Extensions: []llm.ProviderExtension{
			otherProviderExt{},
			Extension{CacheToolsThrough: -1},
		},
	}
	out := WithToolCaching(req)

	if len(out.Extensions) != 2 {
		t.Fatalf("got %d extensions, want 2 (one other + one merged anthropic)", len(out.Extensions))
	}
	if _, ok := out.Extensions[0].(otherProviderExt); !ok {
		t.Errorf("first extension changed: %T", out.Extensions[0])
	}
	ext := llm.FindExtension[Extension](out, "anthropic")
	if ext == nil || ext.CacheToolsThrough != 0 {
		t.Errorf("anthropic extension not merged with tool index: %+v", ext)
	}
}
