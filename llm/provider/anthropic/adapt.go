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
	"errors"
	"net/http"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
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

// maxSends bounds the requests one Chat or ChatStream call makes, so a
// chain of adjustments (temperature, thinking type and effort level,
// each rejected in turn) can complete without looping.
const maxSends = 4

// Adjustment names, used as the "param" attribute in log records.
const (
	adjustTemperature = "temperature"
	adjustThinking    = "thinking"
	adjustEffort      = "effort"
)

// checkReasoningEffort returns an error for a request whose reasoning
// effort, or the client default it falls back to, is not a recognised
// level, so that it fails before any network call.
func (c *client) checkReasoningEffort(req llm.ChatRequest) error {
	_, err := req.EffectiveReasoningEffort(c.opts.ReasoningEffort)
	var pe *llm.ProviderError
	if errors.As(err, &pe) {
		pe.Provider = providerName
	}
	return err
}

// effortLevels maps each llm.ReasoningEffort that Anthropic sends as
// output_config.effort to its wire value. Anthropic has no "minimal",
// so it is sent as the lowest level, "low". None is not here: it turns
// thinking off rather than setting an effort.
var effortLevels = map[llm.ReasoningEffort]string{
	llm.ReasoningEffortMinimal: "low",
	llm.ReasoningEffortLow:     "low",
	llm.ReasoningEffortMedium:  "medium",
	llm.ReasoningEffortHigh:    "high",
	llm.ReasoningEffortXHigh:   "xhigh",
	llm.ReasoningEffortMax:     "max",
}

// applyReasoningEffort sets thinking and output_config on aReq from the
// request's reasoning effort (or the client default). A level sends
// thinking {type: "adaptive"} with output_config {effort}; None sends
// thinking {type: "disabled"}. Thinking already set by the Extension is
// left alone, and anything the model has rejected is omitted.
//
// Anthropic documents thinking as incompatible with any temperature
// other than 1, so a request that carries adaptive thinking leaves out
// any other temperature rather than being sent only to be rejected.
func (c *client) applyReasoningEffort(aReq *anthropicChatRequest, req llm.ChatRequest) {
	c.setReasoningEffort(aReq, req)
	if aReq.Thinking != nil && aReq.Thinking.Type == "adaptive" &&
		aReq.Temperature != nil && *aReq.Temperature != 1 {
		aReq.Temperature = nil
	}
}

// setReasoningEffort does the work of applyReasoningEffort apart from
// the temperature.
func (c *client) setReasoningEffort(aReq *anthropicChatRequest, req llm.ChatRequest) {
	effort, err := req.EffectiveReasoningEffort(c.opts.ReasoningEffort)
	if err != nil || effort == llm.ReasoningEffortDefault {
		return
	}

	thinking := "adaptive"
	if effort == llm.ReasoningEffortNone {
		thinking = "disabled"
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if aReq.Thinking == nil && !c.rejectedThinking[thinking] {
		aReq.Thinking = &anthropicThinking{Type: thinking}
	}
	if level, ok := effortLevels[effort]; ok && !c.rejectedEffort[level] {
		aReq.OutputConfig = &anthropicOutputCfg{Effort: level}
	}
}

// adaptToRejection inspects a failed /messages response and returns the
// adjustment that lets the request be rebuilt and resent, or "" when
// there is none. It only adapts when the request actually carried the
// parameter the model rejected:
//
//   - Temperature: when the request also carried thinking that may
//     be what the model objects to, the caller omits temperature for
//     this call only; otherwise the client omits it from every later
//     request, since a client is bound to a single model. See
//     temperatureRejectedForThinking.
//   - Effort level or thinking type set from ReasoningEffort: the client
//     omits that level or type from every later request.
//
// Thinking the Extension set ("enabled") is the caller's explicit
// choice and is never dropped.
func (c *client) adaptToRejection(status int, body []byte, sent *anthropicChatRequest) string {
	if status != http.StatusBadRequest {
		return ""
	}
	var errResp anthropicErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil ||
		errResp.Error.Type != "invalid_request_error" {
		return ""
	}
	msg := strings.ToLower(errResp.Error.Message)

	switch {
	case sent.Temperature != nil && rejectsTemperature(msg):
		if temperatureRejectedForThinking(sent, msg) {
			c.logAdjustment(adjustTemperature, "omitted for this request")
			return adjustTemperature
		}
		if c.learn(&c.dropTemperature) {
			c.logAdjustment(adjustTemperature, "omitted")
		}
		return adjustTemperature

	// Checked before thinking, since an effort rejection can mention
	// thinking ("output_config.effort 'xhigh' is not supported when
	// thinking is disabled"). A rejection of a thinking type can mention
	// effort too, pointing at the fields to use instead, so one naming
	// thinking.type is left to the thinking case.
	case sent.OutputConfig != nil && strings.Contains(msg, "effort") &&
		!strings.Contains(msg, "thinking.type"):
		if c.learnRejected(&c.rejectedEffort, sent.OutputConfig.Effort) {
			c.logAdjustment(adjustEffort, "omitted "+sent.OutputConfig.Effort)
		}
		return adjustEffort

	case sent.Thinking != nil && sent.Thinking.Type != "enabled" &&
		strings.Contains(msg, "thinking") && !strings.Contains(msg, "temperature"):
		if c.learnRejected(&c.rejectedThinking, sent.Thinking.Type) {
			c.logAdjustment(adjustThinking, "omitted "+sent.Thinking.Type)
		}
		return adjustThinking
	}
	return ""
}

// temperatureRejectedForThinking reports whether a temperature
// rejection may be down to the thinking the request carried rather
// than the model. Thinking from the Extension is a per-request choice,
// so any rejection alongside it is treated that way, as before
// ReasoningEffort existed. Thinking from ReasoningEffort is usually a
// client default sent on every request, so the rejection is put down
// to it only when the message mentions thinking; otherwise a model that
// refuses temperature outright would cost an extra request on every
// call.
func temperatureRejectedForThinking(sent *anthropicChatRequest, msg string) bool {
	if sent.Thinking == nil {
		return false
	}
	return sent.Thinking.Type == "enabled" || strings.Contains(msg, "thinking")
}

// learn sets a learned flag and reports whether it was newly set, so
// that concurrent calls that all hit the same rejection log once.
func (c *client) learn(flag *bool) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	first := !*flag
	*flag = true
	return first
}

// learnRejected adds value to a learned set and reports whether it was
// newly added.
func (c *client) learnRejected(set *map[string]bool, value string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if *set == nil {
		*set = map[string]bool{}
	}
	first := !(*set)[value]
	(*set)[value] = true
	return first
}

// logAdjustment records an adjustment at Debug level.
func (c *client) logAdjustment(param, action string) {
	if c.opts.Logger != nil {
		c.opts.Logger.Debug("adjusted request after provider rejected a parameter",
			"provider", providerName,
			"model", c.model,
			"param", param,
			"action", action)
	}
}

// retryAfterRejection reports whether a call should rebuild and resend
// its request after a rejection: when adaptToRejection finds an
// adjustment the call has not already applied, and the call has sends
// left. The adjustment is recorded in applied, so each is applied at
// most once per call.
func (c *client) retryAfterRejection(status int, body []byte, sent *anthropicChatRequest,
	applied map[string]bool, sends int) bool {
	adj := c.adaptToRejection(status, body, sent)
	if adj == "" || applied[adj] {
		return false
	}
	applied[adj] = true
	return sends < maxSends
}
