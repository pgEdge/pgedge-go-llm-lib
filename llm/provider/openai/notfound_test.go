//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package openai

import (
	"errors"
	"net/http"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// A 404, typically a model that does not exist or that the key cannot
// use, maps to ErrNotFound rather than the ErrProviderError catch-all,
// and keeps its status code.
func TestStatusNotFoundMapsToErrNotFound(t *testing.T) {
	err := (&client{}).mapError(http.StatusNotFound, []byte(`{"error":{"message":"The model example-model does not exist","type":"invalid_request_error"}}`))
	if !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
	if errors.Is(err, llm.ErrProviderError) {
		t.Errorf("a 404 should no longer match ErrProviderError: %v", err)
	}
	var pe *llm.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProviderError, got %T", err)
	}
	if pe.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", pe.StatusCode)
	}
}
