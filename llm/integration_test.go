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
	"os"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/gemini"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
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

func TestIntegrationVoyage(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in -short mode")
	}
	key := os.Getenv("VOYAGE_API_KEY")
	if key == "" {
		t.Skip("VOYAGE_API_KEY not set; skipping Voyage integration test")
	}
	c, err := llm.NewClient("voyage", llm.Options{APIKey: key, Model: "voyage-3.5-lite"})
	if err != nil {
		t.Fatal(err)
	}
	vec, err := c.Embed(context.Background(), "the quick brown fox")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) == 0 {
		t.Fatal("expected non-empty embedding")
	}

	rerankClient, err := llm.NewClient("voyage", llm.Options{APIKey: key, Model: "rerank-2.5-lite"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := rerankClient.Rerank(context.Background(), llm.RerankRequest{
		Query: "kittens",
		Documents: []string{
			"Cats are small carnivorous mammals.",
			"The Eiffel Tower is in Paris.",
			"A kitten is a juvenile cat.",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Results) == 0 {
		t.Fatal("expected at least one result")
	}
	top := res.Results[0].Index
	if top != 0 && top != 2 {
		t.Errorf("expected the cat-related sentences (index 0 or 2) to rank highest, got index %d", top)
	}
}
