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
	providers := []string{"anthropic", "openai", "gemini", "ollama", "voyage"}
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
	providers := []string{"anthropic", "openai", "gemini", "ollama", "voyage"}
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

// pickModel returns the value of envVar if set, otherwise fallback.
func pickModel(envVar, fallback string) string {
	if v := os.Getenv(envVar); v != "" {
		return v
	}
	return fallback
}

// helloChat exercises a provider's Chat method with a minimal prompt.
// On success the response must have at least one non-empty text block.
func helloChat(t *testing.T, c llm.Client) {
	t.Helper()
	maxTokens := 32
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "Say hi in one word."}},
		}},
		MaxTokens: &maxTokens,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty response content")
	}
	hasText := false
	for _, b := range resp.Content {
		if b.Type == llm.BlockText && b.Text != "" {
			hasText = true
			break
		}
	}
	if !hasText {
		t.Fatalf("expected at least one non-empty text block, got %+v", resp.Content)
	}
	if resp.Usage.TotalTokens == 0 {
		t.Errorf("expected non-zero usage, got %+v", resp.Usage)
	}
}

// helloEmbed exercises a provider's Embed method. Returned vector must be non-empty.
func helloEmbed(t *testing.T, c llm.Client) {
	t.Helper()
	vec, err := c.Embed(context.Background(), "the quick brown fox")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("expected non-empty embedding")
	}
}

func TestIntegrationAnthropic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in -short mode")
	}
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping Anthropic integration test")
	}
	c, err := llm.NewClient("anthropic", llm.Options{
		APIKey: key,
		Model:  pickModel("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloChat(t, c)
}

func TestIntegrationOpenAI(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in -short mode")
	}
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set; skipping OpenAI integration test")
	}
	chatClient, err := llm.NewClient("openai", llm.Options{
		APIKey: key,
		Model:  pickModel("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloChat(t, chatClient)

	embedClient, err := llm.NewClient("openai", llm.Options{
		APIKey: key,
		Model:  pickModel("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloEmbed(t, embedClient)
}

func TestIntegrationGemini(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in -short mode")
	}
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set; skipping Gemini integration test")
	}
	chatClient, err := llm.NewClient("gemini", llm.Options{
		APIKey: key,
		Model:  pickModel("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloChat(t, chatClient)

	embedClient, err := llm.NewClient("gemini", llm.Options{
		APIKey: key,
		Model:  pickModel("GEMINI_EMBED_MODEL", "text-embedding-004"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloEmbed(t, embedClient)
}

func TestIntegrationOllama(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in -short mode")
	}
	base := os.Getenv("OLLAMA_BASE_URL")
	if base == "" {
		t.Skip("OLLAMA_BASE_URL not set; skipping Ollama integration test")
	}
	chatClient, err := llm.NewClient("ollama", llm.Options{
		BaseURL: base,
		Model:   pickModel("OLLAMA_CHAT_MODEL", "llama3.2"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloChat(t, chatClient)

	embedClient, err := llm.NewClient("ollama", llm.Options{
		BaseURL: base,
		Model:   pickModel("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
	})
	if err != nil {
		t.Fatal(err)
	}
	helloEmbed(t, embedClient)
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
