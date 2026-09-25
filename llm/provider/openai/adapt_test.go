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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Rejection bodies in the shapes OpenAI returns.
const (
	maxTokensRejected = `{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}`

	temperatureValueRejected = `{"error":{"message":"Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.","type":"invalid_request_error","param":"temperature","code":"unsupported_value"}}`

	temperatureParamRejected = `{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model.","type":"invalid_request_error","param":"temperature","code":"unsupported_parameter"}}`

	responsesOnly = `{"error":{"message":"This model is only supported in v1/responses and not in v1/chat/completions.","type":"invalid_request_error","param":null,"code":null}}`

	messagesInvalid = `{"error":{"message":"Invalid value for 'messages'.","type":"invalid_request_error","param":"messages","code":"invalid_value"}}`
)

type scriptedReply struct {
	status int
	body   string
}

type recordedRequest struct {
	path string
	body map[string]any
}

// has reports whether the recorded request body carried key.
func (r recordedRequest) has(key string) bool {
	_, ok := r.body[key]
	return ok
}

// adaptServer answers each request with the next scripted reply and,
// once the script is used up, with a success in the shape of the
// endpoint and streaming mode requested. requests returns what the
// server has received so far.
func adaptServer(t *testing.T, replies ...scriptedReply) (srv *httptest.Server, requests func() []recordedRequest) {
	t.Helper()
	var mu sync.Mutex
	var got []recordedRequest
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		mu.Lock()
		got = append(got, recordedRequest{path: r.URL.Path, body: body})
		n := len(got)
		mu.Unlock()

		if n <= len(replies) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(replies[n-1].status)
			_, _ = w.Write([]byte(replies[n-1].body))
			return
		}

		stream := body["stream"] == true
		switch {
		case r.URL.Path == "/chat/completions" && stream:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"))
		case r.URL.Path == "/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		case stream:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"type":"response.output_text.delta","delta":"ok"}` + "\n\n" + `data: {"type":"response.completed"}` + "\n\n"))
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
		}
	}))
	t.Cleanup(srv.Close)
	return srv, func() []recordedRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]recordedRequest(nil), got...)
	}
}

// newAdaptClient returns a client with a client-default temperature,
// so every request carries both max_tokens and temperature.
func newAdaptClient(t *testing.T, url string, logger *slog.Logger, exts ...llm.ProviderExtension) llm.Client {
	t.Helper()
	c, err := New(llm.Options{
		APIKey:      "test-key",
		Model:       "gpt-test-1",
		BaseURL:     url,
		Temperature: llm.Float(0.7),
		Retry:       llm.RetryConfig{Disabled: true},
		Logger:      logger,
		Extensions:  exts,
	})
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

// call makes one Chat or ChatStream call and returns the text received.
func call(t *testing.T, c llm.Client, stream bool, req llm.ChatRequest) (string, error) {
	t.Helper()
	if req.Messages == nil {
		req.Messages = []llm.Message{llm.UserText("Hi")}
	}
	if !stream {
		resp, err := c.Chat(context.Background(), req)
		if err != nil {
			return "", err
		}
		return resp.Content[0].Text, nil
	}
	s, err := c.ChatStream(context.Background(), req)
	if err != nil {
		return "", err
	}
	var text strings.Builder
	for chunk := range s.Chunks {
		text.WriteString(chunk.Text)
	}
	return text.String(), <-s.Err
}

// modes runs fn once for Chat and once for ChatStream.
func modes(t *testing.T, fn func(t *testing.T, stream bool)) {
	t.Run("Chat", func(t *testing.T) { fn(t, false) })
	t.Run("ChatStream", func(t *testing.T) { fn(t, true) })
}

// mustSucceed makes a call that is expected to succeed with "ok".
func mustSucceed(t *testing.T, c llm.Client, stream bool) {
	t.Helper()
	text, err := call(t, c, stream, llm.ChatRequest{})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if text != "ok" {
		t.Fatalf("text = %q, want ok", text)
	}
}

// wantTokenField fails the test unless the request carried the token
// limit (4096) in field alone, and not in the other token field.
func wantTokenField(t *testing.T, r recordedRequest, field string) {
	t.Helper()
	other := "max_completion_tokens"
	if field == other {
		other = "max_tokens"
	}
	if r.body[field] != float64(4096) || r.has(other) {
		t.Errorf("request should carry %s=4096 only: %v", field, r.body)
	}
}

func TestAdapt_MaxTokensRejected(t *testing.T) {
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t, scriptedReply{400, maxTokensRejected})
		c := newAdaptClient(t, srv.URL, nil)

		mustSucceed(t, c, stream)
		got := requests()
		if len(got) != 2 {
			t.Fatalf("requests = %d, want 2", len(got))
		}
		wantTokenField(t, got[0], "max_tokens")
		wantTokenField(t, got[1], "max_completion_tokens")

		// The adjustment is remembered: the next call makes one request.
		mustSucceed(t, c, stream)
		got = requests()
		if len(got) != 3 {
			t.Fatalf("requests = %d, want 3", len(got))
		}
		wantTokenField(t, got[2], "max_completion_tokens")
	})
}

// wantPaths fails the test unless every request went to path.
func wantPaths(t *testing.T, got []recordedRequest, path string) {
	t.Helper()
	for _, r := range got {
		if r.path != path {
			t.Errorf("path = %s, want %s", r.path, path)
		}
	}
}

func TestAdapt_TemperatureRejected(t *testing.T) {
	endpoints := []struct {
		name string
		path string
		exts []llm.ProviderExtension
	}{
		{"chat-completions", "/chat/completions", nil},
		{"responses", "/responses", forceResponses},
	}
	bodies := map[string]string{
		"unsupported_value":     temperatureValueRejected,
		"unsupported_parameter": temperatureParamRejected,
	}
	for _, ep := range endpoints {
		for code, body := range bodies {
			t.Run(ep.name+"/"+code, func(t *testing.T) {
				modes(t, func(t *testing.T, stream bool) {
					srv, requests := adaptServer(t, scriptedReply{400, body})
					c := newAdaptClient(t, srv.URL, nil, ep.exts...)

					mustSucceed(t, c, stream)
					got := requests()
					if len(got) != 2 {
						t.Fatalf("requests = %d, want 2", len(got))
					}
					if got[0].body["temperature"] != 0.7 {
						t.Errorf("first request temperature = %v, want 0.7", got[0].body["temperature"])
					}
					if got[1].has("temperature") {
						t.Errorf("retry should omit temperature: %v", got[1].body)
					}

					// Omitted from now on, even when set per request.
					if _, err := call(t, c, stream, llm.ChatRequest{Temperature: llm.Float(0.2)}); err != nil {
						t.Fatalf("second call: %v", err)
					}
					got = requests()
					if len(got) != 3 || got[2].has("temperature") {
						t.Errorf("second call should send one request without temperature: %v", got)
					}
					wantPaths(t, got, ep.path)
				})
			})
		}
	}
}

func TestAdapt_ResponsesOnlyRerouted(t *testing.T) {
	for _, status := range []int{404, 400} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			modes(t, func(t *testing.T, stream bool) {
				srv, requests := adaptServer(t, scriptedReply{status, responsesOnly})
				c := newAdaptClient(t, srv.URL, nil)

				mustSucceed(t, c, stream)
				got := requests()
				if len(got) != 2 || got[0].path != "/chat/completions" || got[1].path != "/responses" {
					t.Fatalf("want /chat/completions then /responses, got %v", got)
				}
				if !got[1].has("input") || !got[1].has("max_output_tokens") {
					t.Errorf("rerouted request should use the Responses shape: %v", got[1].body)
				}

				mustSucceed(t, c, stream)
				got = requests()
				if len(got) != 3 || got[2].path != "/responses" {
					t.Errorf("second call should go straight to /responses: %v", got)
				}
			})
		})
	}
}

func TestAdapt_ForcedChatCompletionsNotRerouted(t *testing.T) {
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t, scriptedReply{404, responsesOnly})
		c := newAdaptClient(t, srv.URL, nil, Extension{ResponsesAPI: llm.Bool(false)})

		_, err := call(t, c, stream, llm.ChatRequest{})
		if !errors.Is(err, llm.ErrProviderError) || !strings.Contains(err.Error(), "v1/responses") {
			t.Errorf("want the provider's 404 error, got %v", err)
		}
		if got := requests(); len(got) != 1 {
			t.Errorf("requests = %d, want 1", len(got))
		}
	})
}

func TestAdapt_ReroutedRequestChecksResponsesLimits(t *testing.T) {
	// Stop sequences cannot be sent to /v1/responses, so a reroute
	// must report ErrNotSupported rather than silently drop them.
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t, scriptedReply{404, responsesOnly})
		c := newAdaptClient(t, srv.URL, nil)

		_, err := call(t, c, stream, llm.ChatRequest{StopSequences: []string{"END"}})
		if !errors.Is(err, llm.ErrNotSupported) {
			t.Errorf("want ErrNotSupported, got %v", err)
		}
		if got := requests(); len(got) != 1 {
			t.Errorf("requests = %d, want 1", len(got))
		}
	})
}

func TestAdapt_ChainedAdjustments(t *testing.T) {
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t,
			scriptedReply{404, responsesOnly},
			scriptedReply{400, temperatureValueRejected},
		)
		c := newAdaptClient(t, srv.URL, nil)

		mustSucceed(t, c, stream)
		got := requests()
		if len(got) != 3 {
			t.Fatalf("requests = %d, want 3", len(got))
		}
		if got[0].path != "/chat/completions" || got[1].path != "/responses" || got[2].path != "/responses" {
			t.Errorf("paths = %s, %s, %s", got[0].path, got[1].path, got[2].path)
		}
		if got[1].body["temperature"] != 0.7 || got[2].has("temperature") {
			t.Errorf("temperature should be sent, then omitted: %v, %v", got[1].body, got[2].body)
		}
	})
}

func TestAdapt_StopsAfterThreeSends(t *testing.T) {
	// Three different adjustments would need a fourth request; the
	// call gives up and returns the third rejection instead, but still
	// learns from it, so the next call succeeds first time.
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t,
			scriptedReply{400, maxTokensRejected},
			scriptedReply{404, responsesOnly},
			scriptedReply{400, temperatureValueRejected},
		)
		c := newAdaptClient(t, srv.URL, nil)

		_, err := call(t, c, stream, llm.ChatRequest{})
		if !errors.Is(err, llm.ErrInvalidRequest) || !strings.Contains(err.Error(), "temperature") {
			t.Errorf("want the temperature rejection, got %v", err)
		}
		if got := requests(); len(got) != 3 {
			t.Errorf("requests = %d, want 3", len(got))
		}

		mustSucceed(t, c, stream)
		got := requests()
		if len(got) != 4 {
			t.Fatalf("requests = %d, want 4", len(got))
		}
		if got[3].path != "/responses" || got[3].has("temperature") {
			t.Errorf("follow-up call should go to /responses without temperature: %s %v", got[3].path, got[3].body)
		}
	})
}

func TestAdapt_UnrelatedRejectionNotRetried(t *testing.T) {
	modes(t, func(t *testing.T, stream bool) {
		srv, requests := adaptServer(t, scriptedReply{400, messagesInvalid})
		c := newAdaptClient(t, srv.URL, nil)

		_, err := call(t, c, stream, llm.ChatRequest{})
		if !errors.Is(err, llm.ErrInvalidRequest) {
			t.Errorf("want ErrInvalidRequest, got %v", err)
		}
		if got := requests(); len(got) != 1 {
			t.Errorf("requests = %d, want 1", len(got))
		}
	})
}

func TestAdapt_RejectionOfUnsentParameterNotRetried(t *testing.T) {
	cases := []struct {
		name string
		body string
		opts llm.Options
	}{
		// No temperature configured, so none is sent.
		{"temperature", temperatureValueRejected, llm.Options{}},
		// /v1/responses sends max_output_tokens, never max_tokens.
		{"max_tokens", maxTokensRejected, llm.Options{Extensions: forceResponses}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			modes(t, func(t *testing.T, stream bool) {
				srv, requests := adaptServer(t, scriptedReply{400, tc.body})
				opts := tc.opts
				opts.APIKey, opts.Model, opts.BaseURL = "test-key", "gpt-test-1", srv.URL
				opts.Retry = llm.RetryConfig{Disabled: true}
				c, err := New(opts)
				if err != nil {
					t.Fatalf("create client: %v", err)
				}

				if _, err := call(t, c, stream, llm.ChatRequest{}); !errors.Is(err, llm.ErrInvalidRequest) {
					t.Errorf("want ErrInvalidRequest, got %v", err)
				}
				if got := requests(); len(got) != 1 {
					t.Errorf("requests = %d, want 1", len(got))
				}
			})
		})
	}
}

func TestAdapt_LogsEachAdjustmentAtDebug(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	srv, _ := adaptServer(t,
		scriptedReply{404, responsesOnly},
		scriptedReply{400, temperatureValueRejected},
	)
	c := newAdaptClient(t, srv.URL, logger)
	mustSucceed(t, c, false)
	mustSucceed(t, c, false) // nothing new learned, so nothing logged

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 log records, got %d:\n%s", len(lines), buf.String())
	}
	want := [][]string{
		{"param=endpoint", `action="routed to /v1/responses"`},
		{"param=temperature", "action=omitted"},
	}
	for i, line := range lines {
		common := []string{
			"level=DEBUG",
			`msg="adjusted request after provider rejected a parameter"`,
			"provider=openai",
			"model=gpt-test-1",
		}
		for _, s := range append(common, want[i]...) {
			if !strings.Contains(line, s) {
				t.Errorf("record %d missing %s: %s", i, s, line)
			}
		}
	}
}

func TestAdapt_LogsMaxTokensAdjustment(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	srv, _ := adaptServer(t, scriptedReply{400, maxTokensRejected})
	c := newAdaptClient(t, srv.URL, logger)
	mustSucceed(t, c, true)

	out := buf.String()
	if strings.Count(out, "\n") != 1 ||
		!strings.Contains(out, "param=max_tokens") ||
		!strings.Contains(out, `action="sent as max_completion_tokens"`) {
		t.Errorf("unexpected log output: %s", out)
	}
}
