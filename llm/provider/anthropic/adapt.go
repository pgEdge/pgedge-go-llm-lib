//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package anthropic

import (
	"encoding/json"
	"net/http"
	"strings"
)

// temperatureRejections are the phrases that, alongside the word
// "temperature", mark a 400 as the model refusing the parameter outright
// (e.g. "`temperature` is deprecated for this model.") rather than
// objecting to its value, such as an out-of-range error.
var temperatureRejections = []string{
	"deprecated", "not supported", "unsupported", "not allowed", "not accepted",
}

// rejectsTemperature reports whether an Anthropic error message says the
// model does not accept a temperature at all.
func rejectsTemperature(msg string) bool {
	msg = strings.ToLower(msg)
	if !strings.Contains(msg, "temperature") {
		return false
	}
	for _, p := range temperatureRejections {
		if strings.Contains(msg, p) {
			return true
		}
	}
	return false
}

// omitTemperature reports whether this client has learned that its model
// rejects temperature.
func (c *client) omitTemperature() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dropTemperature
}

// adaptToRejection inspects a failed /messages response and reports
// whether the request should be rebuilt and resent without temperature.
// It only adapts when the request actually carried a temperature and the
// model rejected it. When the request also enabled extended thinking,
// the rejection may be down to that per-request setting rather than the
// model, so the caller omits temperature for this call only; otherwise
// the client omits it from every later request, since a client is bound
// to a single model. As the resent request carries no temperature, a
// caller loops at most once.
func (c *client) adaptToRejection(status int, body []byte, sent *anthropicChatRequest) bool {
	if status != http.StatusBadRequest || sent.Temperature == nil {
		return false
	}
	var errResp anthropicErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil ||
		errResp.Error.Type != "invalid_request_error" ||
		!rejectsTemperature(errResp.Error.Message) {
		return false
	}

	if sent.Thinking != nil {
		c.logAdjustment("omitted for this request")
		return true
	}

	c.mu.Lock()
	learned := !c.dropTemperature
	c.dropTemperature = true
	c.mu.Unlock()

	// Log only the first time, so concurrent calls that all hit the
	// rejection produce a single record.
	if learned {
		c.logAdjustment("omitted")
	}
	return true
}

// logAdjustment records at Debug level that temperature was dropped.
func (c *client) logAdjustment(action string) {
	if c.opts.Logger != nil {
		c.opts.Logger.Debug("adjusted request after provider rejected a parameter",
			"provider", providerName,
			"model", c.model,
			"param", "temperature",
			"action", action)
	}
}
