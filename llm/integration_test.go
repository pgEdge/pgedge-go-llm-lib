//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/gemini"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
)

func TestAllProvidersRegistered(t *testing.T) {
	providers := []string{"anthropic", "openai", "gemini", "ollama"}
	for _, p := range providers {
		client, err := llm.NewClient(p, llm.Options{
			APIKey: "test-key",
			Model:  "test-model",
		})
		if err != nil {
			t.Errorf("provider %q: unexpected error: %v", p, err)
			continue
		}
		if client.Provider() != p {
			t.Errorf("expected provider %q, got %q", p, client.Provider())
		}
		if client.Model() != "test-model" {
			t.Errorf("expected model test-model, got %q", client.Model())
		}
	}
}

func TestAllProvidersZeroInitialUsage(t *testing.T) {
	providers := []string{"anthropic", "openai", "gemini", "ollama"}
	for _, p := range providers {
		client, err := llm.NewClient(p, llm.Options{
			APIKey: "test-key",
			Model:  "test-model",
		})
		if err != nil {
			t.Errorf("provider %q: %v", p, err)
			continue
		}
		usage := client.Usage()
		if usage.TotalTokens != 0 {
			t.Errorf("provider %q: expected zero initial usage", p)
		}
	}
}

func TestAnthropicEmbedNotSupported(t *testing.T) {
	client, err := llm.NewClient("anthropic", llm.Options{
		APIKey: "test-key",
		Model:  "test-model",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, embedErr := client.Embed(context.TODO(), "hello")
	if !errors.Is(embedErr, llm.ErrNotSupported) {
		t.Errorf("expected ErrNotSupported, got %v", embedErr)
	}
}

func TestDefaultOptionsApplied(t *testing.T) {
	providers := []string{"anthropic", "openai", "gemini", "ollama"}
	for _, p := range providers {
		_, err := llm.NewClient(p, llm.Options{
			APIKey: "test-key",
			Model:  "test-model",
		})
		if err != nil {
			t.Errorf("provider %q should accept default options: %v", p, err)
		}
	}
}

func TestPing(t *testing.T) {
	for _, p := range []string{"anthropic", "openai", "gemini", "ollama"} {
		t.Run(p, func(t *testing.T) {
			c, err := llm.NewClient(p, llm.Options{
				APIKey: "k", Model: "test",
				Retry: llm.RetryConfig{Disabled: true},
			})
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			// Ping is allowed to error (no real upstream) — we only
			// care that the method exists and doesn't panic.
			_ = c.Ping(context.Background())
		})
	}
}

func TestResetUsage(t *testing.T) {
	c, _ := llm.NewClient("openai", llm.Options{
		APIKey: "k", Model: "test",
		Retry: llm.RetryConfig{Disabled: true},
	})
	c.ResetUsage()
	u := c.Usage()
	if u.TotalTokens != 0 {
		t.Errorf("after ResetUsage, TotalTokens = %d", u.TotalTokens)
	}
}
