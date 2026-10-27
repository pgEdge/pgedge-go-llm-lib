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

// thinkingRejected is a 400 in the shape Gemini returns.
const thinkingRejected = `{"error":{"code":400,"message":"Thinking level is not supported for this model.","status":"INVALID_ARGUMENT"}}`

const otherRejected = `{"error":{"code":400,"message":"Invalid value at 'contents'.","status":"INVALID_ARGUMENT"}}`

// thinkingServer answers each request with the next scripted 400 and,
// once the script is used up, with success. It records each request
// body.
func thinkingServer(t *testing.T, rejections ...string) (*httptest.Server, func() []map[string]any) {
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
		reply := `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: " + reply + "\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), got...)
	}
}

func newThinkingClient(t *testing.T, url string, opts llm.Options) llm.Client {
	t.Helper()
	opts.APIKey = "test-key"
	opts.Model = "gemini-test-model"
	opts.BaseURL = url
	opts.Retry = llm.RetryConfig{Disabled: true}
	c, err := New(opts)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	return c
}

func sendThinking(t *testing.T, c llm.Client, stream bool, req llm.ChatRequest) error {
	t.Helper()
	req.Messages = []llm.Message{llm.UserText("Hi")}
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

func thinkingModes(t *testing.T, fn func(t *testing.T, stream bool)) {
	t.Run("Chat", func(t *testing.T) { fn(t, false) })
	t.Run("ChatStream", func(t *testing.T) { fn(t, true) })
}

// sentLevel reads generationConfig.thinkingConfig.thinkingLevel from a
// recorded body, returning "" when absent.
func sentLevel(body map[string]any) string {
	gc, _ := body["generationConfig"].(map[string]any)
	tc, _ := gc["thinkingConfig"].(map[string]any)
	s, _ := tc["thinkingLevel"].(string)
	return s
}

func TestThinkingLevelWireFormat(t *testing.T) {
	cases := map[llm.ReasoningEffort]string{
		llm.ReasoningEffortDefault: "",
		llm.ReasoningEffortNone:    "minimal",
		llm.ReasoningEffortMinimal: "minimal",
		llm.ReasoningEffortLow:     "low",
		llm.ReasoningEffortMedium:  "medium",
		llm.ReasoningEffortHigh:    "high",
		llm.ReasoningEffortXHigh:   "high",
		llm.ReasoningEffortMax:     "high",
	}
	for effort, want := range cases {
		t.Run(string(effort), func(t *testing.T) {
			srv, requests := thinkingServer(t)
			c := newThinkingClient(t, srv.URL, llm.Options{})
			if err := sendThinking(t, c, false, llm.ChatRequest{ReasoningEffort: effort}); err != nil {
				t.Fatalf("call: %v", err)
			}
			body := requests()[0]
			if got := sentLevel(body); got != want {
				t.Errorf("thinkingLevel = %q, want %q", got, want)
			}
			if gc, _ := body["generationConfig"].(map[string]any); want == "" && gc["thinkingConfig"] != nil {
				t.Errorf("thinkingConfig should be omitted: %v", body)
			}
		})
	}
}

func TestThinkingLevelClientDefault(t *testing.T) {
	srv, requests := thinkingServer(t)
	c := newThinkingClient(t, srv.URL, llm.Options{ReasoningEffort: llm.ReasoningEffortLow})
	for _, req := range []llm.ChatRequest{{}, {ReasoningEffort: llm.ReasoningEffortHigh}} {
		if err := sendThinking(t, c, false, req); err != nil {
			t.Fatalf("call: %v", err)
		}
	}
	got := requests()
	if sentLevel(got[0]) != "low" || sentLevel(got[1]) != "high" {
		t.Errorf("thinkingLevel = %q, %q; want low, high", sentLevel(got[0]), sentLevel(got[1]))
	}
}

func TestThinkingLevelRejected(t *testing.T) {
	// A refused level is omitted from then on; other levels are sent.
	thinkingModes(t, func(t *testing.T, stream bool) {
		var buf bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
		srv, requests := thinkingServer(t, thinkingRejected)
		c := newThinkingClient(t, srv.URL, llm.Options{Logger: logger, Temperature: llm.Float(0.5)})
		for _, e := range []llm.ReasoningEffort{llm.ReasoningEffortMinimal, llm.ReasoningEffortMinimal, llm.ReasoningEffortLow} {
			if err := sendThinking(t, c, stream, llm.ChatRequest{ReasoningEffort: e}); err != nil {
				t.Fatalf("call: %v", err)
			}
		}
		got := requests()
		if len(got) != 4 {
			t.Fatalf("requests = %d, want 4", len(got))
		}
		if sentLevel(got[0]) != "minimal" || sentLevel(got[1]) != "" || sentLevel(got[2]) != "" || sentLevel(got[3]) != "low" {
			t.Errorf("thinkingLevel = %q, %q, %q, %q", sentLevel(got[0]), sentLevel(got[1]), sentLevel(got[2]), sentLevel(got[3]))
		}
		if gc, _ := got[1]["generationConfig"].(map[string]any); gc["temperature"] != 0.5 {
			t.Errorf("the rest of generationConfig should be kept: %v", got[1])
		}
		out := strings.TrimSpace(buf.String())
		if strings.Count(out, "\n") != 0 || !strings.Contains(out, "param=thinkingLevel") ||
			!strings.Contains(out, `action="omitted minimal"`) {
			t.Errorf("want one adjustment record, got:\n%s", out)
		}
	})
}

func TestThinkingLevelRejectionRetriedOnce(t *testing.T) {
	srv, requests := thinkingServer(t, thinkingRejected, thinkingRejected)
	c := newThinkingClient(t, srv.URL, llm.Options{})
	err := sendThinking(t, c, false, llm.ChatRequest{ReasoningEffort: llm.ReasoningEffortHigh})
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("want ErrInvalidRequest, got %v", err)
	}
	if got := requests(); len(got) != 2 {
		t.Errorf("requests = %d, want 2", len(got))
	}
}

func TestUnrelatedRejectionNotRetried(t *testing.T) {
	for _, tc := range []struct {
		name   string
		effort llm.ReasoningEffort
		body   string
	}{
		{"other error", llm.ReasoningEffortHigh, otherRejected},
		{"thinking not sent", llm.ReasoningEffortDefault, thinkingRejected},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, requests := thinkingServer(t, tc.body)
			c := newThinkingClient(t, srv.URL, llm.Options{})
			err := sendThinking(t, c, false, llm.ChatRequest{ReasoningEffort: tc.effort})
			if !errors.Is(err, llm.ErrInvalidRequest) {
				t.Errorf("want ErrInvalidRequest, got %v", err)
			}
			if got := requests(); len(got) != 1 {
				t.Errorf("requests = %d, want 1", len(got))
			}
		})
	}
}

func TestThinkingLevelUnknownRejectedBeforeSending(t *testing.T) {
	thinkingModes(t, func(t *testing.T, stream bool) {
		srv, requests := thinkingServer(t)
		c := newThinkingClient(t, srv.URL, llm.Options{})
		err := sendThinking(t, c, stream, llm.ChatRequest{ReasoningEffort: "extreme"})
		var pe *llm.ProviderError
		if !errors.As(err, &pe) || !errors.Is(err, llm.ErrInvalidRequest) || pe.Provider != "gemini" {
			t.Errorf("want a gemini ErrInvalidRequest, got %v", err)
		}
		if got := requests(); len(got) != 0 {
			t.Errorf("requests = %d, want 0", len(got))
		}
	})
}
