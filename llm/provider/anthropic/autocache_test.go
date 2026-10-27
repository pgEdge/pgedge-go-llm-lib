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
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// cacheControlRejected is a 400 in the shape Anthropic returns from an
// endpoint that does not accept the top-level marker.
const cacheControlRejected = `{"type":"error","error":{"type":"invalid_request_error","message":"cache_control: Extra inputs are not permitted"}}`

// topLevelCacheType returns the top-level cache_control type, or "".
func topLevelCacheType(body map[string]any) string {
	cc, _ := body["cache_control"].(map[string]any)
	s, _ := cc["type"].(string)
	return s
}

func TestPromptCachingWireFormat(t *testing.T) {
	cases := []struct {
		name          string
		client, req   llm.PromptCaching
		wantTopMarker bool
	}{
		{"unset", "", "", false},
		{"request auto", "", llm.PromptCachingAuto, true},
		{"client auto", llm.PromptCachingAuto, "", true},
		{"request off overrides client auto", llm.PromptCachingAuto, llm.PromptCachingOff, false},
		{"request auto overrides client off", llm.PromptCachingOff, llm.PromptCachingAuto, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, requests := effortServer(t)
			c := newEffortClient(t, srv.URL, llm.Options{PromptCaching: tc.client})
			if err := sendEffort(t, c, false, llm.ChatRequest{PromptCaching: tc.req}); err != nil {
				t.Fatalf("call: %v", err)
			}
			body := requests()[0]
			if got := topLevelCacheType(body) == "ephemeral"; got != tc.wantTopMarker {
				t.Errorf("top-level cache_control = %v, want %v: %v", body["cache_control"], tc.wantTopMarker, body)
			}
			if !tc.wantTopMarker {
				if _, has := body["cache_control"]; has {
					t.Errorf("cache_control should be omitted: %v", body)
				}
			}
		})
	}
}

func TestPromptCachingKeepsBlockMarkers(t *testing.T) {
	srv, requests := effortServer(t)
	c := newEffortClient(t, srv.URL, llm.Options{})
	req := llm.ChatRequest{
		SystemPrompt:  "You are terse.",
		Tools:         []llm.Tool{{Name: "lookup", Description: "Look something up", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		PromptCaching: llm.PromptCachingAuto,
	}
	req = WithSystemCaching(WithToolCaching(req))
	if err := sendEffort(t, c, false, req); err != nil {
		t.Fatalf("call: %v", err)
	}
	body := requests()[0]
	if topLevelCacheType(body) != "ephemeral" {
		t.Errorf("want the top-level marker: %v", body)
	}
	tools, _ := body["tools"].([]any)
	system, _ := body["system"].([]any)
	if len(tools) != 1 || len(system) != 1 {
		t.Fatalf("want one tool and one system block: %v", body)
	}
	for _, block := range []any{tools[0], system[0]} {
		if m, _ := block.(map[string]any); m["cache_control"] == nil {
			t.Errorf("block marker missing: %v", block)
		}
	}
}

func TestPromptCachingUnknownRejectedBeforeSending(t *testing.T) {
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t)
		c := newEffortClient(t, srv.URL, llm.Options{})
		err := sendEffort(t, c, stream, llm.ChatRequest{PromptCaching: "always"})
		var pe *llm.ProviderError
		if !errors.As(err, &pe) || !errors.Is(err, llm.ErrInvalidRequest) || pe.Provider != "anthropic" {
			t.Errorf("want an anthropic ErrInvalidRequest, got %v", err)
		}
		if got := requests(); len(got) != 0 {
			t.Errorf("requests = %d, want 0", len(got))
		}
	})
}

func TestPromptCachingRejected(t *testing.T) {
	// An endpoint that refuses the top-level marker has it omitted from
	// then on.
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t, cacheControlRejected)
		c := newEffortClient(t, srv.URL, llm.Options{PromptCaching: llm.PromptCachingAuto})
		for i := 0; i < 2; i++ {
			if err := sendEffort(t, c, stream, llm.ChatRequest{}); err != nil {
				t.Fatalf("call: %v", err)
			}
		}
		got := requests()
		if len(got) != 3 {
			t.Fatalf("requests = %d, want 3", len(got))
		}
		if topLevelCacheType(got[0]) != "ephemeral" || got[1]["cache_control"] != nil || got[2]["cache_control"] != nil {
			t.Errorf("cache_control should be sent, then omitted: %v", got)
		}
	})
}

func TestPromptCachingRejectionWithBlockMarkersNotRetried(t *testing.T) {
	// With a system marker also on the wire, a rejection naming
	// cache_control may be about that marker, so it is not retried.
	srv, requests := effortServer(t, cacheControlRejected)
	c := newEffortClient(t, srv.URL, llm.Options{PromptCaching: llm.PromptCachingAuto})
	err := sendEffort(t, c, false, WithSystemCaching(llm.ChatRequest{SystemPrompt: "You are terse."}))
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("want ErrInvalidRequest, got %v", err)
	}
	if got := requests(); len(got) != 1 {
		t.Errorf("requests = %d, want 1", len(got))
	}
}

func TestPromptCachingLeftOutWhenBreakpointsUsed(t *testing.T) {
	// Four block markers use up Anthropic's breakpoints, so the
	// top-level marker would be rejected and is left out; three leave
	// room for it.
	for _, marked := range []int{3, 4} {
		srv, requests := effortServer(t)
		c := newEffortClient(t, srv.URL, llm.Options{PromptCaching: llm.PromptCachingAuto})
		var blocks []llm.ContentBlock
		for i := 0; i < marked; i++ {
			blocks = append(blocks, llm.ContentBlock{
				Type: llm.BlockText, Text: "part",
				CacheControl: &llm.CacheControl{Type: string(llm.CacheControlEphemeral)},
			})
		}
		req := llm.ChatRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: blocks}}}
		if _, err := c.Chat(context.Background(), req); err != nil {
			t.Fatalf("call: %v", err)
		}
		if got := topLevelCacheType(requests()[0]) == "ephemeral"; got != (marked < 4) {
			t.Errorf("%d block markers: top-level marker sent = %v", marked, got)
		}
	}
}
