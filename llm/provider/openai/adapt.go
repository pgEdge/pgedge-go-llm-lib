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
	"encoding/json"
	"errors"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// OpenAI models differ in which parameters and endpoint they accept,
// and the differences change with every model generation. Rather than
// keep a list of models, the client sends the request as built and,
// when OpenAI rejects something the library put in it, adjusts the
// request, retries, and remembers the adjustment. Each client is bound
// to one model, so later calls skip the failed attempt.

// maxSends bounds the requests one Chat or ChatStream call makes, so a
// chain of adjustments (for example rerouted to /v1/responses, then
// temperature and reasoning effort rejected there) can complete without
// looping.
const maxSends = 4

// Adjustment names, used as the "param" attribute in log records.
const (
	adjustMaxTokens   = "max_tokens"
	adjustTemperature = "temperature"
	adjustEndpoint    = "endpoint"
	adjustEffort      = "reasoning_effort"
)

// actionEffortAll is the logged action when the model refuses the
// reasoning effort parameter itself rather than one level.
const actionEffortAll = "omitted at every level"

// learned records the adjustments the client has made after OpenAI
// rejected a request.
type learned struct {
	maxCompletionTokens bool // send max_completion_tokens, not max_tokens
	dropTemperature     bool // omit temperature
	responsesAPI        bool // route Chat / ChatStream to /v1/responses

	// rejectedEffort holds the reasoning effort levels the model has
	// refused, which are then omitted; effortUnsupported records that
	// it refused the parameter itself, so every level is omitted.
	rejectedEffort    map[llm.ReasoningEffort]bool
	effortUnsupported bool
}

// rejection describes a non-2xx response to a Chat or ChatStream
// request, together with what that request carried, so a rejection is
// acted on only when it names something the library sent.
type rejection struct {
	status      int
	body        []byte
	responses   bool // sent to /v1/responses
	maxTokens   bool // carried max_tokens
	temperature bool // carried temperature

	effort llm.ReasoningEffort // reasoning effort carried, if any
}

func (c *client) learnedFlags() learned {
	c.learnedMu.Lock()
	defer c.learnedMu.Unlock()
	return c.learned
}

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

// reasoningEffort returns the reasoning effort to send for req: the
// request's level, else the client default, or the zero value when
// neither is set or the model has rejected that level. Requests are
// checked by checkReasoningEffort first, so an unknown level never
// reaches here.
func (c *client) reasoningEffort(req llm.ChatRequest) llm.ReasoningEffort {
	effort, err := req.EffectiveReasoningEffort(c.opts.ReasoningEffort)
	if err != nil || effort == llm.ReasoningEffortDefault {
		return llm.ReasoningEffortDefault
	}
	c.learnedMu.Lock()
	defer c.learnedMu.Unlock()
	if c.learned.effortUnsupported || c.learned.rejectedEffort[effort] {
		return llm.ReasoningEffortDefault
	}
	return effort
}

// forcedRoute returns the explicit openai.Extension{ResponsesAPI: ...}
// setting, or nil when routing is left to the client.
func (c *client) forcedRoute() *bool {
	if ext := findExtension(c.opts.Extensions); ext != nil {
		return ext.ResponsesAPI
	}
	return nil
}

// useResponsesAPI reports whether a Chat / ChatStream call should be
// routed to /v1/responses instead of /v1/chat/completions. An explicit
// extension setting wins; otherwise the client uses /v1/responses only
// once OpenAI has said the model requires it.
func (c *client) useResponsesAPI() bool {
	if forced := c.forcedRoute(); forced != nil {
		return *forced
	}
	return c.learnedFlags().responsesAPI
}

// classify returns the adjustment a rejection calls for, with the
// action to log, or empty strings when the rejection is not one the
// client can resolve by changing the request.
func (c *client) classify(rej *rejection) (param, action string) {
	var errResp openaiErrorResponse
	_ = json.Unmarshal(rej.body, &errResp) // best-effort; an undecodable body matches nothing
	e := errResp.Error

	switch {
	case rejectsMaxTokens(rej, e.Param, e.Code):
		return adjustMaxTokens, "sent as max_completion_tokens"
	case rejectsTemperature(rej, e.Param, e.Code):
		return adjustTemperature, "omitted"
	case rejectsReasoningEffort(rej, e.Param, e.Message):
		if e.Code == "unsupported_value" {
			return adjustEffort, "omitted " + string(rej.effort)
		}
		if e.Code == "unsupported_parameter" || e.Code == "unknown_parameter" {
			return adjustEffort, actionEffortAll
		}
	case c.requiresResponsesAPI(rej, e.Param, e.Message):
		return adjustEndpoint, "routed to /v1/responses"
	}
	return "", ""
}

// rejectsMaxTokens reports whether the model refused a max_tokens the
// request carried, as models that take max_completion_tokens do.
func rejectsMaxTokens(rej *rejection, param, code string) bool {
	return rej.status == 400 && rej.maxTokens &&
		param == "max_tokens" && code == "unsupported_parameter"
}

// rejectsTemperature reports whether the model refused a temperature
// the request carried, as models that accept only their default do.
func rejectsTemperature(rej *rejection, param, code string) bool {
	if rej.status != 400 || !rej.temperature || param != "temperature" {
		return false
	}
	return code == "unsupported_value" || code == "unsupported_parameter"
}

// rejectsReasoningEffort reports whether a 400 names the reasoning
// effort the request carried. OpenAI names it in param on some errors
// and only in the message (with param null) on others, so either is
// accepted; the caller tells a refused level from a refused parameter
// by the error code.
func rejectsReasoningEffort(rej *rejection, param, msg string) bool {
	if rej.status != 400 || rej.effort == llm.ReasoningEffortDefault {
		return false
	}
	switch param {
	case "reasoning_effort", "reasoning.effort", "reasoning":
		return true
	case "":
		return strings.Contains(msg, "reasoning_effort") || strings.Contains(msg, "reasoning.effort")
	}
	return false
}

// requiresResponsesAPI reports whether Chat Completions refused the
// model because it is only available through /v1/responses, and the
// caller has not pinned the endpoint.
func (c *client) requiresResponsesAPI(rej *rejection, param, msg string) bool {
	if rej.responses || param != "" || (rej.status != 400 && rej.status != 404) {
		return false
	}
	return c.forcedRoute() == nil && strings.Contains(msg, "only supported in v1/responses")
}

// learn records an adjustment on the client and logs it at Debug level.
// Only the first time an adjustment is learned is logged, so concurrent
// calls that all hit the same rejection produce a single record.
func (c *client) learn(param, action string, rej *rejection) {
	if param == adjustEffort {
		c.learnEffort(action, rej)
		return
	}
	var flag *bool
	c.learnedMu.Lock()
	switch param {
	case adjustMaxTokens:
		flag = &c.learned.maxCompletionTokens
	case adjustTemperature:
		flag = &c.learned.dropTemperature
	case adjustEndpoint:
		flag = &c.learned.responsesAPI
	}
	first := !*flag
	*flag = true
	c.learnedMu.Unlock()

	if first {
		c.logAdjustment(param, action)
	}
}

// learnEffort records that the model refused the reasoning effort rej
// carried: that level alone for a refused value, or every level when
// the model refused the parameter itself.
func (c *client) learnEffort(action string, rej *rejection) {
	all := action == actionEffortAll
	c.learnedMu.Lock()
	var first bool
	if all {
		first = !c.learned.effortUnsupported
		c.learned.effortUnsupported = true
	} else {
		if c.learned.rejectedEffort == nil {
			c.learned.rejectedEffort = map[llm.ReasoningEffort]bool{}
		}
		first = !c.learned.rejectedEffort[rej.effort]
		c.learned.rejectedEffort[rej.effort] = true
	}
	c.learnedMu.Unlock()

	if first {
		c.logAdjustment(adjustEffort, action)
	}
}

// logAdjustment records an adjustment at Debug level.
func (c *client) logAdjustment(param, action string) {
	if c.opts.Logger != nil {
		c.opts.Logger.Debug("adjusted request after provider rejected a parameter",
			"provider", providerName, "model", c.model,
			"param", param, "action", action)
	}
}

// sendWithAdjustments calls send, routed as the client currently
// decides, and retries after each rejection that an adjustment can
// resolve. Each adjustment is applied at most once per call, and at
// most maxSends requests are made. send returns a rejection for a
// non-2xx response and an error for anything else that failed.
func sendWithAdjustments[T any](c *client, send func(responses bool) (T, *rejection, error)) (T, error) {
	applied := map[string]bool{}
	for sends := 1; ; sends++ {
		out, rej, err := send(c.useResponsesAPI())
		if rej == nil {
			return out, err
		}
		param, action := c.classify(rej)
		if param == "" || applied[param] {
			var zero T
			return zero, c.mapError(rej.status, rej.body)
		}
		// Learn even from the final send, so the next call is built
		// correctly rather than hitting the same rejection first.
		applied[param] = true
		c.learn(param, action, rej)
		if sends == maxSends {
			var zero T
			return zero, c.mapError(rej.status, rej.body)
		}
	}
}
