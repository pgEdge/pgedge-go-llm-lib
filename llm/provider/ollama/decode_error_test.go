//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package ollama

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// malformedServer answers every request with a 200 whose body is
// truncated JSON.
func malformedServer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":`))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestChatMalformedSuccessIsAnError(t *testing.T) {
	c := newTestClient(t, malformedServer(t))
	resp, err := c.Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.BlockText, Text: "hi"}}},
		},
	})
	if !errors.Is(err, llm.ErrProviderError) || !strings.Contains(err.Error(), "could not be decoded") {
		t.Fatalf("resp, err = %+v, %v; want a decode error, not an empty reply", resp, err)
	}
}

func TestListModelsMalformedSuccessIsAnError(t *testing.T) {
	c := newTestClient(t, malformedServer(t))
	if _, err := c.ListModels(context.Background()); !errors.Is(err, llm.ErrProviderError) {
		t.Fatalf("err = %v, want ErrProviderError", err)
	}
}

func TestEmbedMalformedSuccessIsAnError(t *testing.T) {
	c := newTestClient(t, malformedServer(t))
	if _, err := c.Embed(context.Background(), "hi"); !errors.Is(err, llm.ErrProviderError) {
		t.Fatalf("err = %v, want ErrProviderError", err)
	}
}
