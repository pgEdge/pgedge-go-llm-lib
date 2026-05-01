//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm

import (
	"errors"
	"testing"
)

func TestNewClientUnknownProvider(t *testing.T) {
	_, err := NewClient("nonexistent", Options{})
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
	if !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
}

func TestNewClientEmptyProvider(t *testing.T) {
	_, err := NewClient("", Options{})
	if err == nil {
		t.Fatal("expected error for empty provider")
	}
}

func TestRegisteredProviders(t *testing.T) {
	// Register a sentinel and verify it appears in the list.
	RegisterProvider("test-sentinel", func(o Options) (Client, error) {
		return nil, ErrInvalidRequest
	})
	got := RegisteredProviders()
	found := false
	for _, n := range got {
		if n == "test-sentinel" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("test-sentinel not in RegisteredProviders(): %v", got)
	}
	// Result should be sorted.
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Errorf("RegisteredProviders() not sorted: %v", got)
			break
		}
	}
}
