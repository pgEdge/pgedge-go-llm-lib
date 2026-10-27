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
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Rejection messages in the shapes Anthropic returns.
const (
	adaptiveRejected = `{"type":"error","error":{"type":"invalid_request_error","message":"adaptive thinking is not supported on this model"}}`
	disabledRejected = `{"type":"error","error":{"type":"invalid_request_error","message":"\"thinking.type.disabled\" is not supported for this model."}}`
	enabledRejected  = `{"type":"error","error":{"type":"invalid_request_error","message":"\"thinking.type.enabled\" is not supported for this model. Use \"thinking.type.adaptive\" and \"output_config.effort\" instead."}}`
	effortRejected   = `{"type":"error","error":{"type":"invalid_request_error","message":"output_config.effort 'max' is not supported on this model."}}`
)

// effortServer answers each request with the next scripted rejection
// and, once the script is used up, with success. It records each
// request body.
func effortServer(t *testing.T, rejections ...string) (*httptest.Server, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var got []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		mu.Lock()
		got = append(got, body)
		n := len(got)
		mu.Unlock()

		if n <= len(rejections) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(rejections[n-1]))
			return
		}
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}` + "\n\n"))
			_, _ = w.Write([]byte(`data: {"type":"message_stop"}` + "\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), got...)
	}
}

func newEffortClient(t *testing.T, url string, opts llm.Options) llm.Client {
	t.Helper()
	opts.APIKey = "test-key"
	opts.Model = "claude-test-model"
	opts.BaseURL = url
	opts.Retry = llm.RetryConfig{Disabled: true}
	c, err := New(opts)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

// sendEffort makes a Chat or ChatStream call with req, or with
// hiRequest's messages when req has none, and returns its error.
func sendEffort(t *testing.T, c llm.Client, stream bool, req llm.ChatRequest) error {
	t.Helper()
	if req.Messages == nil {
		req.Messages = hiRequest.Messages
	}
	if !stream {
		_, err := c.Chat(context.Background(), req)
		return err
	}
	s, err := c.ChatStream(context.Background(), req)
	if err != nil {
		return err
	}
	for range s.Chunks {
	}
	return <-s.Err
}

func effortModes(t *testing.T, fn func(t *testing.T, stream bool)) {
	t.Run("Chat", func(t *testing.T) { fn(t, false) })
	t.Run("ChatStream", func(t *testing.T) { fn(t, true) })
}

// thinkingType and effortLevel read the fields from a recorded body,
// returning "" when absent.
func thinkingType(body map[string]any) string {
	th, _ := body["thinking"].(map[string]any)
	s, _ := th["type"].(string)
	return s
}

func effortLevel(body map[string]any) string {
	oc, _ := body["output_config"].(map[string]any)
	s, _ := oc["effort"].(string)
	return s
}

func TestReasoningEffortWireFormat(t *testing.T) {
	cases := []struct {
		effort       llm.ReasoningEffort
		thinking     string
		outputEffort string
	}{
		{llm.ReasoningEffortDefault, "", ""},
		{llm.ReasoningEffortNone, "disabled", ""},
		{llm.ReasoningEffortMinimal, "adaptive", "low"},
		{llm.ReasoningEffortLow, "adaptive", "low"},
		{llm.ReasoningEffortMedium, "adaptive", "medium"},
		{llm.ReasoningEffortHigh, "adaptive", "high"},
		{llm.ReasoningEffortXHigh, "adaptive", "xhigh"},
		{llm.ReasoningEffortMax, "adaptive", "max"},
	}
	for _, tc := range cases {
		t.Run(string(tc.effort), func(t *testing.T) {
			srv, requests := effortServer(t)
			c := newEffortClient(t, srv.URL, llm.Options{})
			if err := sendEffort(t, c, false, llm.ChatRequest{ReasoningEffort: tc.effort}); err != nil {
				t.Fatalf("call: %v", err)
			}
			body := requests()[0]
			if thinkingType(body) != tc.thinking || effortLevel(body) != tc.outputEffort {
				t.Errorf("thinking=%q effort=%q, want %q, %q", thinkingType(body), effortLevel(body), tc.thinking, tc.outputEffort)
			}
			if th, ok := body["thinking"].(map[string]any); ok {
				if _, has := th["budget_tokens"]; has {
					t.Errorf("adaptive or disabled thinking should carry no budget: %v", th)
				}
			}
			if tc.outputEffort == "" {
				if _, has := body["output_config"]; has {
					t.Errorf("output_config should be omitted: %v", body)
				}
			}
		})
	}
}

func TestReasoningEffortClientDefault(t *testing.T) {
	srv, requests := effortServer(t)
	c := newEffortClient(t, srv.URL, llm.Options{ReasoningEffort: llm.ReasoningEffortMedium})
	for _, req := range []llm.ChatRequest{{}, {ReasoningEffort: llm.ReasoningEffortHigh}} {
		if err := sendEffort(t, c, false, req); err != nil {
			t.Fatalf("call: %v", err)
		}
	}
	got := requests()
	if effortLevel(got[0]) != "medium" || effortLevel(got[1]) != "high" {
		t.Errorf("effort = %q, %q; want medium, high", effortLevel(got[0]), effortLevel(got[1]))
	}
}

func TestReasoningEffortWithExtendedThinking(t *testing.T) {
	// Thinking from the Extension wins; the effort level is still sent.
	srv, requests := effortServer(t)
	c := newEffortClient(t, srv.URL, llm.Options{})
	req := WithExtendedThinking(llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}, 8000)
	if err := sendEffort(t, c, false, req); err != nil {
		t.Fatalf("call: %v", err)
	}
	body := requests()[0]
	th, _ := body["thinking"].(map[string]any)
	if th["type"] != "enabled" || th["budget_tokens"] != float64(8000) || effortLevel(body) != "high" {
		t.Errorf("want enabled thinking with budget 8000 and effort high: %v", body)
	}
}

func TestReasoningEffortUnknownRejectedBeforeSending(t *testing.T) {
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t)
		c := newEffortClient(t, srv.URL, llm.Options{})
		err := sendEffort(t, c, stream, llm.ChatRequest{ReasoningEffort: "extreme"})
		var pe *llm.ProviderError
		if !errors.As(err, &pe) || !errors.Is(err, llm.ErrInvalidRequest) || pe.Provider != "anthropic" {
			t.Errorf("want an anthropic ErrInvalidRequest, got %v", err)
		}
		if got := requests(); len(got) != 0 {
			t.Errorf("requests = %d, want 0", len(got))
		}
	})
}

func TestAdaptiveThinkingRejected(t *testing.T) {
	// Adaptive thinking is dropped and remembered; the effort stays.
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t, adaptiveRejected)
		c := newEffortClient(t, srv.URL, llm.Options{})
		req := llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}
		for i := 0; i < 2; i++ {
			if err := sendEffort(t, c, stream, req); err != nil {
				t.Fatalf("call: %v", err)
			}
		}
		got := requests()
		if len(got) != 3 {
			t.Fatalf("requests = %d, want 3", len(got))
		}
		if thinkingType(got[0]) != "adaptive" || thinkingType(got[1]) != "" || thinkingType(got[2]) != "" {
			t.Errorf("thinking should be sent, then omitted: %v", got)
		}
		if effortLevel(got[1]) != "high" || effortLevel(got[2]) != "high" {
			t.Errorf("effort should still be sent: %v", got)
		}
	})
}

func TestEffortLevelRejected(t *testing.T) {
	// A refused level is omitted from then on; other levels are sent.
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t, effortRejected)
		c := newEffortClient(t, srv.URL, llm.Options{})
		for _, e := range []llm.ReasoningEffort{llm.ReasoningEffortMax, llm.ReasoningEffortMax, llm.ReasoningEffortHigh} {
			if err := sendEffort(t, c, stream, llm.ChatRequest{ReasoningEffort: e}); err != nil {
				t.Fatalf("call: %v", err)
			}
		}
		got := requests()
		if len(got) != 4 {
			t.Fatalf("requests = %d, want 4", len(got))
		}
		if effortLevel(got[0]) != "max" || effortLevel(got[1]) != "" || effortLevel(got[2]) != "" || effortLevel(got[3]) != "high" {
			t.Errorf("effort = %q, %q, %q, %q", effortLevel(got[0]), effortLevel(got[1]), effortLevel(got[2]), effortLevel(got[3]))
		}
		if thinkingType(got[1]) != "adaptive" {
			t.Errorf("thinking should be untouched: %v", got[1])
		}
	})
}

func TestDisabledThinkingRejected(t *testing.T) {
	srv, requests := effortServer(t, disabledRejected)
	c := newEffortClient(t, srv.URL, llm.Options{})
	if err := sendEffort(t, c, false, llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortNone}); err != nil {
		t.Fatalf("call: %v", err)
	}
	got := requests()
	if len(got) != 2 || thinkingType(got[0]) != "disabled" {
		t.Fatalf("want disabled thinking, then a retry: %v", got)
	}
	if _, has := got[1]["thinking"]; has {
		t.Errorf("thinking should be omitted on the retry: %v", got[1])
	}
}

func TestEffortRejectionsChained(t *testing.T) {
	// Temperature, thinking, effort, the top-level cache_control marker
	// and the replayed thinking blocks refused in turn take six sends,
	// the most one call makes. The temperature is 1, the only value
	// Anthropic allows alongside thinking, so it is sent with adaptive
	// thinking.
	effortModes(t, func(t *testing.T, stream bool) {
		srv, requests := effortServer(t, temperatureDeprecatedBody, adaptiveRejected, effortRejected,
			cacheControlRejected, signatureRejected)
		c := newEffortClient(t, srv.URL, llm.Options{Temperature: llm.Float(1), PromptCaching: llm.PromptCachingAuto})
		req := thinkingHistory()
		req.ReasoningEffort = llm.ReasoningEffortMax
		if err := sendEffort(t, c, stream, req); err != nil {
			t.Fatalf("call: %v", err)
		}
		got := requests()
		if len(got) != maxSends {
			t.Fatalf("requests = %d, want %d", len(got), maxSends)
		}
		last := got[maxSends-1]
		for _, key := range []string{"temperature", "thinking", "output_config", "cache_control"} {
			if _, has := last[key]; has {
				t.Errorf("final request should omit %s: %v", key, last)
			}
		}
		if types := assistantBlockTypes(t, last); !reflect.DeepEqual(types, []string{"tool_use"}) {
			t.Errorf("final request assistant blocks = %v, want only tool_use", types)
		}
	})
}

func TestExtendedThinkingRejectionNotRetried(t *testing.T) {
	// Thinking the caller asked for through the Extension is never
	// dropped, and a rejection of it that points at output_config.effort
	// does not drop the effort level either.
	srv, requests := effortServer(t, enabledRejected)
	c := newEffortClient(t, srv.URL, llm.Options{})
	req := WithExtendedThinking(llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}, 8000)
	err := sendEffort(t, c, false, req)
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("want ErrInvalidRequest, got %v", err)
	}
	if got := requests(); len(got) != 1 {
		t.Errorf("requests = %d, want 1", len(got))
	}
}

func TestTemperatureRejectionWithAdaptiveThinking(t *testing.T) {
	// With thinking from ReasoningEffort, a temperature rejection that
	// does not mention thinking is the model's, so it is remembered;
	// one that does is dropped for that call only. The temperature is
	// 1, since any other value is left out alongside adaptive thinking.
	for _, tc := range []struct {
		name       string
		body       string
		remembered bool
	}{
		{"model rejects temperature", temperatureDeprecatedBody, true},
		{"thinking conflict", `{"type":"error","error":{"type":"invalid_request_error","message":"temperature is not supported when thinking is enabled"}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, requests := effortServer(t, tc.body)
			c := newEffortClient(t, srv.URL, llm.Options{
				Temperature:     llm.Float(1),
				ReasoningEffort: llm.ReasoningEffortHigh,
			})
			for i := 0; i < 2; i++ {
				if err := sendEffort(t, c, false, llm.ChatRequest{}); err != nil {
					t.Fatalf("call: %v", err)
				}
			}
			got := requests()
			if len(got) != 3 {
				t.Fatalf("requests = %d, want 3", len(got))
			}
			if _, sent := got[2]["temperature"]; sent == tc.remembered {
				t.Errorf("second call temperature sent = %v, want %v", sent, !tc.remembered)
			}
		})
	}
}

func TestTemperatureOmittedWithAdaptiveThinking(t *testing.T) {
	// Anthropic allows no temperature other than 1 alongside thinking,
	// so one is left out of requests that carry adaptive thinking, and
	// sent as before otherwise.
	for _, tc := range []struct {
		name string
		temp float64
		req  llm.ChatRequest
		sent bool
	}{
		{"adaptive", 0.7, llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}, false},
		{"adaptive at 1", 1, llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}, true},
		{"disabled", 0.7, llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortNone}, true},
		{"no effort", 0.7, llm.ChatRequest{}, true},
		{"extended thinking", 0.7, WithExtendedThinking(llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh}, 8000), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			effortModes(t, func(t *testing.T, stream bool) {
				srv, requests := effortServer(t)
				c := newEffortClient(t, srv.URL, llm.Options{Temperature: llm.Float(tc.temp)})
				if err := sendEffort(t, c, stream, tc.req); err != nil {
					t.Fatalf("call: %v", err)
				}
				got := requests()
				if len(got) != 1 {
					t.Fatalf("requests = %d, want 1", len(got))
				}
				if _, sent := got[0]["temperature"]; sent != tc.sent {
					t.Errorf("temperature sent = %v, want %v: %v", sent, tc.sent, got[0])
				}
			})
		})
	}
}
