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

func TestProviderErrorIs(t *testing.T) {
	err := &ProviderError{
		Err:        ErrRateLimit,
		StatusCode: 429,
		Message:    "too many requests",
		Provider:   "openai",
	}
	if !errors.Is(err, ErrRateLimit) {
		t.Error("expected errors.Is(err, ErrRateLimit) to be true")
	}
	if errors.Is(err, ErrAuthentication) {
		t.Error("expected errors.Is(err, ErrAuthentication) to be false")
	}
}

func TestProviderErrorMessage(t *testing.T) {
	err := &ProviderError{
		Err:        ErrAuthentication,
		StatusCode: 401,
		Message:    "invalid api key",
		Provider:   "anthropic",
	}
	msg := err.Error()
	if msg == "" {
		t.Error("expected non-empty error message")
	}
	expected := "anthropic (401): invalid api key"
	if msg != expected {
		t.Errorf("expected %q, got %q", expected, msg)
	}
}

func TestProviderErrorUnwrap(t *testing.T) {
	err := &ProviderError{
		Err:        ErrProviderError,
		StatusCode: 500,
		Message:    "internal server error",
		Provider:   "gemini",
	}
	unwrapped := errors.Unwrap(err)
	if !errors.Is(unwrapped, ErrProviderError) {
		t.Errorf("expected ErrProviderError, got %v", unwrapped)
	}
}

func TestProviderErrorWithStatus(t *testing.T) {
	e := &ProviderError{
		Provider:   "openai",
		StatusCode: 401,
		Message:    "invalid api key",
		Err:        ErrAuthentication,
	}
	want := "openai (401): invalid api key"
	if got := e.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestProviderErrorWithoutStatus(t *testing.T) {
	e := &ProviderError{
		Provider: "openai",
		Message:  "unknown provider",
		Err:      ErrInvalidRequest,
	}
	want := "openai: unknown provider"
	if got := e.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestSentinelErrors(t *testing.T) {
	sentinels := []error{
		ErrNotSupported,
		ErrAuthentication,
		ErrRateLimit,
		ErrInvalidRequest,
		ErrProviderError,
	}
	for i, a := range sentinels {
		for j, b := range sentinels {
			if i != j && errors.Is(a, b) {
				t.Errorf("sentinel errors %d and %d should not match", i, j)
			}
		}
	}
}
