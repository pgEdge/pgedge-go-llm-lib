//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package all registers all built-in LLM providers.
// Import this package to make all providers available via llm.NewClient:
//
//	import (
//	    "github.com/pgEdge/pgedge-go-llm-lib/llm"
//	    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/all"
//	)
package all

import (
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/anthropic"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/gemini"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/ollama"
	_ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/openai"
)
