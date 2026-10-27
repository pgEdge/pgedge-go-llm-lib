//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package gemini

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Gemini models differ in which thinking levels they accept: earlier
// models take a thinking budget rather than a level, and some reject
// particular levels. Rather than keep a list of models, the client
// sends the level as built and, when Gemini rejects it, resends the
// request without it and remembers the rejection. Each client is bound
// to one model, so later calls skip the failed attempt.

// thinkingLevels maps each llm.ReasoningEffort to the Gemini thinking
// level sent for it. Gemini's levels run from "minimal" to "high", so
// None (which Gemini cannot express) is sent as the lowest level and
// XHigh and Max as the highest.
var thinkingLevels = map[llm.ReasoningEffort]string{
	llm.ReasoningEffortNone:    "minimal",
	llm.ReasoningEffortMinimal: "minimal",
	llm.ReasoningEffortLow:     "low",
	llm.ReasoningEffortMedium:  "medium",
	llm.ReasoningEffortHigh:    "high",
	llm.ReasoningEffortXHigh:   "high",
	llm.ReasoningEffortMax:     "high",
}

// thinkingLevel returns the thinking level to send for req, or "" when
// the request and client leave reasoning effort unset or the model has
// rejected that level. The error, for an unknown reasoning effort, wraps
// llm.ErrInvalidRequest.
func (c *client) thinkingLevel(req llm.ChatRequest) (string, error) {
	effort, err := req.EffectiveReasoningEffort(c.opts.ReasoningEffort)
	if err != nil {
		var pe *llm.ProviderError
		if errors.As(err, &pe) {
			pe.Provider = providerName
		}
		return "", err
	}
	level := thinkingLevels[effort]
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.rejectedThinking[level] {
		return "", nil
	}
	return level, nil
}

// sentThinkingLevel returns the thinking level a built request carries,
// or "" when it carries none.
func sentThinkingLevel(gReq geminiRequest) string {
	cfg, _ := gReq.GenerationConfig["thinkingConfig"].(map[string]any)
	level, _ := cfg["thinkingLevel"].(string)
	return level
}

// adaptToRejection reports whether a failed generateContent response is
// Gemini refusing the thinking level the request carried, in which case
// the client records the level as rejected, so that it is omitted from
// this and every later request, and the caller resends once.
func (c *client) adaptToRejection(status int, body []byte, sent geminiRequest) bool {
	level := sentThinkingLevel(sent)
	if status != http.StatusBadRequest || level == "" {
		return false
	}
	var errResp geminiErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil ||
		!strings.Contains(strings.ToLower(errResp.Error.Message), "thinking") {
		return false
	}

	c.mu.Lock()
	if c.rejectedThinking == nil {
		c.rejectedThinking = map[string]bool{}
	}
	first := !c.rejectedThinking[level]
	c.rejectedThinking[level] = true
	c.mu.Unlock()

	// Log only the first time, so concurrent calls that all hit the
	// rejection produce a single record.
	if first && c.opts.Logger != nil {
		c.opts.Logger.Debug("adjusted request after provider rejected a parameter",
			"provider", providerName,
			"model", c.model,
			"param", "thinkingLevel",
			"action", "omitted "+level)
	}
	return true
}
