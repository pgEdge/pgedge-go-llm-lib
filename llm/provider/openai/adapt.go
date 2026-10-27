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

// Logged actions for reasoning effort rejections. actionEffortAll is
// for a model that refuses the parameter itself rather than one level;
// actionEffortThisCall is for a rejection that may be down to the tools
// the request carried (see classify).
const (
	actionEffortAll      = "omitted at every level"
	actionEffortThisCall = "omitted for this request"
)

// learned records the adjustments the client has made after OpenAI
// rejected a request.
type learned struct {
	maxCompletionTokens bool // send max_completion_tokens, not max_tokens
	responsesAPI        bool // route Chat / ChatStream to /v1/responses

	// Temperature is omitted once the model has rejected it, but only
	// alongside the kind of reasoning effort the rejected request
	// carried: OpenAI's reasoning models accept temperature at effort
	// "none" and refuse it at every other level, so a rejection at
	// "high" says nothing about requests at "none". See
	// temperatureFlag.
	dropTemperature          bool // with no reasoning effort sent
	dropTemperatureNone      bool // with reasoning effort "none"
	dropTemperatureReasoning bool // with any other reasoning effort

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
	tools       bool // carried tool definitions

	effort llm.ReasoningEffort // reasoning effort carried, if any
}

// temperatureFlag returns the learned flag that records a temperature
// rejection alongside the given reasoning effort. The caller must hold
// learnedMu.
func (l *learned) temperatureFlag(effort llm.ReasoningEffort) *bool {
	switch effort {
	case llm.ReasoningEffortDefault:
		return &l.dropTemperature
	case llm.ReasoningEffortNone:
		return &l.dropTemperatureNone
	}
	return &l.dropTemperatureReasoning
}

// omitTemperature reports whether the model has rejected temperature
// alongside the given reasoning effort, so it is left out.
func (c *client) omitTemperature(effort llm.ReasoningEffort) bool {
	c.learnedMu.Lock()
	defer c.learnedMu.Unlock()
	return *c.learned.temperatureFlag(effort)
}

// temperatureAction returns the logged action for a temperature
// rejection alongside the given reasoning effort.
func temperatureAction(effort llm.ReasoningEffort) string {
	switch effort {
	case llm.ReasoningEffortDefault:
		return "omitted"
	case llm.ReasoningEffortNone:
		return "omitted at reasoning effort none"
	}
	return "omitted at reasoning effort above none"
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

// effortOrder lists the reasoning effort levels from lowest to highest.
var effortOrder = []llm.ReasoningEffort{
	llm.ReasoningEffortNone,
	llm.ReasoningEffortMinimal,
	llm.ReasoningEffortLow,
	llm.ReasoningEffortMedium,
	llm.ReasoningEffortHigh,
	llm.ReasoningEffortXHigh,
	llm.ReasoningEffortMax,
}

// towardsMedium returns the level next to effort in the direction of
// medium, or the zero value for medium itself. A rejected level is
// replaced by its nearest neighbour this way rather than omitted, since
// omitting it applies the model's default, which for OpenAI's reasoning
// models is medium or higher and so further from what the caller asked
// for than any neighbour.
func towardsMedium(effort llm.ReasoningEffort) llm.ReasoningEffort {
	const medium = 3 // index of llm.ReasoningEffortMedium in effortOrder
	for i, e := range effortOrder {
		if e != effort {
			continue
		}
		switch {
		case i < medium:
			return effortOrder[i+1]
		case i > medium:
			return effortOrder[i-1]
		}
	}
	return llm.ReasoningEffortDefault
}

// reasoningEffort returns the reasoning effort to send for req: the
// request's level, else the client default. A level the model has
// rejected is replaced by the nearest one towards medium that it has
// not (see towardsMedium), and the zero value is returned when neither
// is set, when omit is true, or when the model refuses the parameter
// or every level from there to medium. Requests are checked by
// checkReasoningEffort first, so an unknown level never reaches here.
func (c *client) reasoningEffort(req llm.ChatRequest, omit bool) llm.ReasoningEffort {
	effort, err := req.EffectiveReasoningEffort(c.opts.ReasoningEffort)
	if omit || err != nil || effort == llm.ReasoningEffortDefault {
		return llm.ReasoningEffortDefault
	}
	c.learnedMu.Lock()
	defer c.learnedMu.Unlock()
	if c.learned.effortUnsupported {
		return llm.ReasoningEffortDefault
	}
	for effort != llm.ReasoningEffortDefault && c.learned.rejectedEffort[effort] {
		effort = towardsMedium(effort)
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
		return adjustTemperature, temperatureAction(rej.effort)
	case rejectsReasoningEffort(rej, e.Param, e.Message):
		// Some models accept tools on Chat Completions only at effort
		// "none", and a rejection naming reasoning_effort only in its
		// message may be about that combination rather than the level,
		// so it is not remembered for requests without tools.
		if e.Param == "" && rej.tools {
			return adjustEffort, actionEffortThisCall
		}
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
		flag = c.learned.temperatureFlag(rej.effort)
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
// resolve. Each adjustment is applied at most once per call (a
// rejected reasoning effort level counts separately for each level, so
// a call can step through neighbouring levels), and at most maxSends
// requests are made. send returns a rejection for a non-2xx response
// and an error for anything else that failed; omitEffort tells it to
// leave reasoning effort out of this call's requests.
func sendWithAdjustments[T any](c *client, send func(responses, omitEffort bool) (T, *rejection, error)) (T, error) {
	applied := map[string]bool{}
	omitEffort := false
	for sends := 1; ; sends++ {
		out, rej, err := send(c.useResponsesAPI(), omitEffort)
		if rej == nil {
			return out, err
		}
		param, action := c.classify(rej)
		key := param + "/" + action
		if param == "" || applied[key] {
			var zero T
			return zero, c.mapError(rej.status, rej.body)
		}
		applied[key] = true
		if action == actionEffortThisCall {
			omitEffort = true
			c.logAdjustment(param, action)
		} else {
			// Learn even from the final send, so the next call is
			// built correctly rather than hitting the same rejection
			// first.
			c.learn(param, action, rej)
		}
		if sends == maxSends {
			var zero T
			return zero, c.mapError(rej.status, rej.body)
		}
	}
}
