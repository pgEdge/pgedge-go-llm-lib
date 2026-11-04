//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package typesafe

import (
	"errors"
	"net/http"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// A 404, typically a model that does not exist or that the key cannot
// use, maps to ErrNotFound rather than the ErrProviderError catch-all.
func TestStatusNotFoundMapsToErrNotFound(t *testing.T) {
	if err := statusToErr(http.StatusNotFound); !errors.Is(err, llm.ErrNotFound) {
		t.Errorf("statusToErr(404) = %v, want ErrNotFound", err)
	}
}
