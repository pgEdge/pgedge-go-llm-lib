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
	"strings"
)

// OpenAI models differ in which parameters and endpoint they accept,
// and the differences change with every model generation. Rather than
// keep a list of models, the client sends the request as built and,
// when OpenAI rejects something the library put in it, adjusts the
// request, retries, and remembers the adjustment. Each client is bound
// to one model, so later calls skip the failed attempt.

// maxSends bounds the requests one Chat or ChatStream call makes, so a
// chain of adjustments (for example rerouted to /v1/responses, then
// temperature rejected there) can complete without looping.
const maxSends = 3

// Adjustment names, used as the "param" attribute in log records.
const (
	adjustMaxTokens   = "max_tokens"
	adjustTemperature = "temperature"
	adjustEndpoint    = "endpoint"
)

// learned records the adjustments the client has made after OpenAI
// rejected a request.
type learned struct {
	maxCompletionTokens bool // send max_completion_tokens, not max_tokens
	dropTemperature     bool // omit temperature
	responsesAPI        bool // route Chat / ChatStream to /v1/responses
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
}

func (c *client) learnedFlags() learned {
	c.learnedMu.Lock()
	defer c.learnedMu.Unlock()
	return c.learned
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
	case rej.status == 400 && rej.maxTokens &&
		e.Param == "max_tokens" && e.Code == "unsupported_parameter":
		return adjustMaxTokens, "sent as max_completion_tokens"
	case rej.status == 400 && rej.temperature && e.Param == "temperature" &&
		(e.Code == "unsupported_value" || e.Code == "unsupported_parameter"):
		return adjustTemperature, "omitted"
	case (rej.status == 400 || rej.status == 404) && !rej.responses &&
		c.forcedRoute() == nil && strings.Contains(e.Message, "v1/responses"):
		return adjustEndpoint, "routed to /v1/responses"
	}
	return "", ""
}

// learn records an adjustment on the client and logs it at Debug level.
func (c *client) learn(param, action string) {
	c.learnedMu.Lock()
	switch param {
	case adjustMaxTokens:
		c.learned.maxCompletionTokens = true
	case adjustTemperature:
		c.learned.dropTemperature = true
	case adjustEndpoint:
		c.learned.responsesAPI = true
	}
	c.learnedMu.Unlock()

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
		if param == "" || applied[param] || sends == maxSends {
			var zero T
			return zero, c.mapError(rej.status, rej.body)
		}
		applied[param] = true
		c.learn(param, action)
	}
}
