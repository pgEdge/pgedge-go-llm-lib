//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package voyage_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/voyage"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, llm.Client) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := llm.NewClient("voyage", llm.Options{
		APIKey:  "test-key",
		Model:   "voyage-3.5-lite",
		BaseURL: srv.URL, // ValidateBaseURL trims any trailing slash
	})
	if err != nil {
		t.Fatal(err)
	}
	return srv, c
}

func TestVoyageRegistered(t *testing.T) {
	names := llm.RegisteredProviders()
	found := false
	for _, n := range names {
		if n == "voyage" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("voyage not in registered providers: %v", names)
	}
}

func TestChatUnsupported(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
	_, err := c.Chat(context.Background(), llm.ChatRequest{})
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported, got %v", err)
	}
}

func TestChatStreamUnsupported(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
	_, err := c.ChatStream(context.Background(), llm.ChatRequest{})
	if !errors.Is(err, llm.ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported, got %v", err)
	}
}

func TestListModelsContainsExpectedFamilies(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
	names, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"voyage-3.5", "voyage-multimodal-3", "rerank-2.5"}
	for _, w := range want {
		found := false
		for _, n := range names {
			if n == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected %q in catalogue, got %v", w, names)
		}
	}
}

func TestListModelsCapabilityFilterReranking(t *testing.T) {
	_, c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {})
	infos, err := c.ListModelsWithMetadata(context.Background(),
		llm.WithCapabilities(llm.ModelCapabilityReranking))
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 2 {
		t.Fatalf("expected 2 rerank models, got %d (%v)", len(infos), infos)
	}
}

func TestNewWithoutAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	_, err := llm.NewClient("voyage", llm.Options{})
	if err == nil {
		t.Fatal("expected error when no API key supplied")
	}
}
