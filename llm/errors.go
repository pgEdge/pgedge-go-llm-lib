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
	"fmt"
)

// Sentinel errors used to categorise provider failures. All
// provider-side errors are wrapped in *ProviderError whose Unwrap
// returns one of these values, so callers should use errors.Is rather
// than direct comparison.
var (
	// ErrNotSupported indicates the provider does not implement the
	// requested operation (e.g. Embed on Anthropic).
	ErrNotSupported = errors.New("operation not supported by provider")
	// ErrAuthentication indicates a 401 or 403 from the upstream API.
	ErrAuthentication = errors.New("authentication failed")
	// ErrRateLimit indicates a 429 from the upstream API; the retry
	// layer has already exhausted its budget when this is returned.
	ErrRateLimit = errors.New("rate limit exceeded")
	// ErrInvalidRequest indicates a 400 from the upstream API or a
	// malformed library-level request.
	ErrInvalidRequest = errors.New("invalid request")
	// ErrProviderError is the catch-all for upstream errors that do
	// not match a more specific sentinel.
	ErrProviderError = errors.New("provider error")
)

// ProviderError wraps a sentinel error with provider-specific details.
type ProviderError struct {
	Err        error
	StatusCode int
	Message    string
	Provider   string
}

// Error implements the error interface.
func (e *ProviderError) Error() string {
	if e.StatusCode == 0 {
		return fmt.Sprintf("%s: %s", e.Provider, e.Message)
	}
	return fmt.Sprintf("%s (%d): %s", e.Provider, e.StatusCode, e.Message)
}

// Unwrap returns the sentinel error so errors.Is/errors.As work as
// expected against the values defined in this package.
func (e *ProviderError) Unwrap() error {
	return e.Err
}
