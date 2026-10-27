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
	"errors"
	"reflect"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// gatewayCachingRejected names an extra field in param.
const gatewayCachingRejected = `{"error":{"message":"Unknown parameter: 'providerOptions.gateway'.","type":"invalid_request_error","param":"providerOptions.gateway","code":"unknown_parameter"}}`

// endpoints runs fn once against /v1/chat/completions and once against
// /v1/responses, each in Chat and ChatStream mode.
func endpoints(t *testing.T, fn func(t *testing.T, responses, stream bool)) {
	for _, responses := range []bool{false, true} {
		name := "chat_completions"
		if responses {
			name = "responses"
		}
		t.Run(name, func(t *testing.T) {
			modes(t, func(t *testing.T, stream bool) { fn(t, responses, stream) })
		})
	}
}

// newExtraBodyClient returns a client with a client-default temperature
// and the given client-level extra fields, routed as responses says.
func newExtraBodyClient(t *testing.T, url string, responses bool, opts, ext map[string]any) llm.Client {
	t.Helper()
	c, err := New(llm.Options{
		APIKey:      "test-key",
		Model:       "gpt-test-1",
		BaseURL:     url,
		Temperature: llm.Float(0.7),
		ExtraBody:   opts,
		Retry:       llm.RetryConfig{Disabled: true},
		Extensions: []llm.ProviderExtension{
			Extension{ResponsesAPI: llm.Bool(responses), ExtraBody: ext},
		},
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

func TestExtraBodyOnTheWire(t *testing.T) {
	endpoints(t, func(t *testing.T, responses, stream bool) {
		srv, requests := adaptServer(t)
		c := newExtraBodyClient(t, srv.URL, responses,
			map[string]any{"from_options": "options", "shared": "options", "level": "options"},
			map[string]any{"shared": "extension", "level": "extension"})

		req := llm.ChatRequest{Extensions: []llm.ProviderExtension{&Extension{ExtraBody: map[string]any{
			"level":           "request",
			"providerOptions": map[string]any{"gateway": map[string]any{"caching": "auto"}},
		}}}}
		if _, err := call(t, c, stream, req); err != nil {
			t.Fatalf("call: %v", err)
		}

		body := requests()[0].body
		want := map[string]any{
			"from_options":    "options",
			"shared":          "extension",
			"level":           "request",
			"providerOptions": map[string]any{"gateway": map[string]any{"caching": "auto"}},
		}
		for k, v := range want {
			if !reflect.DeepEqual(body[k], v) {
				t.Errorf("%s = %v, want %v", k, body[k], v)
			}
		}
		// The fields the library builds are still there.
		if body["model"] != "gpt-test-1" || body["temperature"] != 0.7 || (body["stream"] == true) != stream {
			t.Errorf("library fields missing or changed: %v", body)
		}
	})
}

func TestExtraBodyReplacesLibraryFields(t *testing.T) {
	endpoints(t, func(t *testing.T, responses, stream bool) {
		srv, requests := adaptServer(t)
		c := newExtraBodyClient(t, srv.URL, responses, nil, nil)

		extra := map[string]any{"temperature": 0.2, "reasoning_effort": "high", "reasoning": map[string]any{"effort": "high", "summary": "auto"}}
		req := llm.ChatRequest{
			ReasoningEffort: llm.ReasoningEffortLow,
			Extensions:      []llm.ProviderExtension{Extension{ExtraBody: extra}},
		}
		if _, err := call(t, c, stream, req); err != nil {
			t.Fatalf("call: %v", err)
		}

		body := requests()[0].body
		for k, v := range extra {
			if !reflect.DeepEqual(body[k], v) {
				t.Errorf("%s = %v, want %v", k, body[k], v)
			}
		}
	})
}

func TestExtraBodyUnencodableRejectedBeforeSending(t *testing.T) {
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t)
		c := newExtraBodyClient(t, srv.URL, false, map[string]any{"bad": make(chan int)}, nil)

		_, err := call(t, c, stream, llm.ChatRequest{})
		var pe *llm.ProviderError
		if !errors.As(err, &pe) || !errors.Is(err, llm.ErrInvalidRequest) || pe.Provider != "openai" {
			t.Errorf("want an openai ErrInvalidRequest, got %v", err)
		}
		if got := requests(); len(got) != 0 {
			t.Errorf("requests = %d, want 0", len(got))
		}
	})
}

func TestExtraBodyRejectionNotRetried(t *testing.T) {
	cases := []struct {
		name  string
		extra map[string]any
		reply string
	}{
		// The field, or a path inside it, in param.
		{"param", map[string]any{"providerOptions": map[string]any{"gateway": map[string]any{"caching": "auto"}}}, gatewayCachingRejected},
		// A library field replaced by an extra one, named only in
		// the message: without the bypass this would drop
		// reasoning_effort and resend.
		{"message", map[string]any{"reasoning_effort": "max"}, effortParamRejected},
		{"replaced library field", map[string]any{"temperature": 0.2}, temperatureValueRejected},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			modes(t, func(t *testing.T, stream bool) {
				srv, requests := adaptServer(t, scriptedReply{400, tc.reply})
				c := newExtraBodyClient(t, srv.URL, false, nil, nil)
				req := llm.ChatRequest{
					ReasoningEffort: llm.ReasoningEffortLow,
					Extensions:      []llm.ProviderExtension{Extension{ExtraBody: tc.extra}},
				}
				if _, err := call(t, c, stream, req); !errors.Is(err, llm.ErrInvalidRequest) {
					t.Errorf("want ErrInvalidRequest, got %v", err)
				}
				if got := requests(); len(got) != 1 {
					t.Errorf("requests = %d, want 1", len(got))
				}
			})
		})
	}
}

func TestExtraBodyKeptWhenLibraryFieldAdjusted(t *testing.T) {
	endpoints(t, func(t *testing.T, responses, stream bool) {
		reply := temperatureValueRejected
		srv, requests := adaptServer(t, scriptedReply{400, reply})
		c := newExtraBodyClient(t, srv.URL, responses, map[string]any{"prompt_cache_key": "tenant-a"}, nil)

		if _, err := call(t, c, stream, llm.ChatRequest{}); err != nil {
			t.Fatalf("call: %v", err)
		}
		got := requests()
		if len(got) != 2 {
			t.Fatalf("requests = %d, want 2", len(got))
		}
		if got[1].has("temperature") || got[1].body["prompt_cache_key"] != "tenant-a" {
			t.Errorf("retry should omit temperature and keep the extra field: %v", got[1].body)
		}
	})
}

func TestNamesExtraField(t *testing.T) {
	extra := map[string]any{"metadata": nil}
	cases := []struct {
		body string
		want bool
	}{
		{`{"error":{"param":"metadata"}}`, true},
		{`{"error":{"param":"metadata.tenant"}}`, true},
		{`{"error":{"param":"metadata[0]"}}`, true},
		{`{"error":{"param":null,"message":"Unknown parameter: 'metadata'."}}`, true},
		{`{"error":{"param":"metadata_extra"}}`, false},
		{`{"error":{"param":"temperature","message":"metadata"}}`, false},
		{`not json`, false},
	}
	for _, tc := range cases {
		if got := namesExtraField([]byte(tc.body), extra); got != tc.want {
			t.Errorf("namesExtraField(%s) = %v, want %v", tc.body, got, tc.want)
		}
	}
	if namesExtraField([]byte(`{"error":{"param":"metadata"}}`), nil) {
		t.Error("no extra fields should name nothing")
	}
}
