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
	"fmt"
	"strings"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// extraBody returns the extra body fields for req: llm.Options.ExtraBody,
// then the ExtraBody of an openai.Extension on llm.Options.Extensions,
// then that of one on the request, each merged over the one before key
// by key. It returns nil when none of them sets a field.
func (c *client) extraBody(req llm.ChatRequest) map[string]any {
	var out map[string]any
	add := func(fields map[string]any) {
		for k, v := range fields {
			if out == nil {
				out = make(map[string]any, len(fields))
			}
			out[k] = v
		}
	}
	add(c.opts.ExtraBody)
	if ext := findExtension(c.opts.Extensions); ext != nil {
		add(ext.ExtraBody)
	}
	if ext := findExtension(req.Extensions); ext != nil {
		add(ext.ExtraBody)
	}
	return out
}

// checkExtraBody reports an extra field whose value cannot be encoded
// as JSON, before anything is sent.
func checkExtraBody(extra map[string]any) error {
	for k, v := range extra {
		if _, err := json.Marshal(v); err != nil {
			return &llm.ProviderError{
				Err:      llm.ErrInvalidRequest,
				Message:  fmt.Sprintf("extra body field %q: %v", k, err),
				Provider: providerName,
			}
		}
	}
	return nil
}

// withExtraBody returns body with the extra fields set at the top level
// of its JSON encoding, replacing any field of the same name. With no
// extra fields it returns body unchanged.
func withExtraBody(body any, extra map[string]any) (any, error) {
	if len(extra) == 0 {
		return body, nil
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	fields := map[string]json.RawMessage{}
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, fmt.Errorf("merge extra body fields: %w", err)
	}
	for k, v := range extra {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("extra body field %q: %w", k, err)
		}
		fields[k] = raw
	}
	return fields, nil
}

// namesExtraField reports whether a rejection names one of the extra
// fields the request carried, either in the error's param (the field
// itself or a path inside it) or, when param is empty, in the message,
// which is where OpenAI names the field on some errors.
func namesExtraField(body []byte, extra map[string]any) bool {
	if len(extra) == 0 {
		return false
	}
	var errResp openaiErrorResponse
	_ = json.Unmarshal(body, &errResp) // best-effort; an undecodable body names nothing
	param, msg := errResp.Error.Param, errResp.Error.Message
	for k := range extra {
		if param == "" {
			if strings.Contains(msg, k) {
				return true
			}
			continue
		}
		if param == k || strings.HasPrefix(param, k+".") || strings.HasPrefix(param, k+"[") {
			return true
		}
	}
	return false
}
