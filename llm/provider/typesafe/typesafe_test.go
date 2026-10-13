//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package typesafe

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

const testKey = "ts-test-key-not-real"

func newTestClient(t *testing.T, handler http.HandlerFunc, model string) llm.Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := New(llm.Options{APIKey: testKey, BaseURL: srv.URL, Model: model, Retry: llm.RetryConfig{Disabled: true}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func sampleRequest() llm.DecisionRequest {
	return llm.DecisionRequest{
		Input: "Replication lag on replica-1 is 45 seconds and rising.",
		Questions: []llm.DecisionQuestion{
			{Name: "urgent", Kind: llm.DecisionYesNo, Instructions: "Is this urgent?"},
			{Name: "category", Kind: llm.DecisionChoice, Instructions: "Which category?",
				Options: []llm.DecisionOption{{Value: "replication", Description: "Replication issues"}, {Value: "storage"}}},
			{Name: "severity", Kind: llm.DecisionScore, Instructions: "How severe?",
				Options: []llm.DecisionOption{{Value: "low"}, {Value: "medium", Description: "Needs attention today"}, {Value: "high"}}},
		},
	}
}

// Keys are deliberately in a different order from the request.
const sampleResponse = `{
  "model": "jev-test",
  "answers": {
    "severity": {"type": "score", "score": 1.4, "confidence": 0.7,
                 "legend": {"0": "low", "1": "medium", "2": "high"},
                 "probabilities": {"2": 0.2, "0": 0.1, "1": 0.7}},
    "category": {"type": "choice", "choice": "replication", "confidence": 0.8,
                 "probabilities": {"storage": 0.2, "replication": 0.8}},
    "urgent": {"type": "noul", "noul": 0.9}
  },
  "usage": {"input_tokens": 100, "output_tokens": 3}
}`

func TestDecideRequestShape(t *testing.T) {
	var got map[string]any
	var auth, path string
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		_, _ = w.Write([]byte(sampleResponse))
	}, "")
	if _, err := c.Decide(context.Background(), sampleRequest()); err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if auth != "Bearer "+testKey || path != "/systemone" {
		t.Fatalf("auth=%q path=%q", auth, path)
	}
	want := `{"model":"jev-latest","state":"Replication lag on replica-1 is 45 seconds and rising.","questions":{` +
		`"category":{"type":"choice","instructions":"Which category?","criteria":{"replication":"Replication issues","storage":null}},` +
		`"severity":{"type":"score","instructions":"How severe?","criteria":["low",{"label":"medium","description":"Needs attention today"},"high"]},` +
		`"urgent":{"type":"noul","instructions":"Is this urgent?"}}}`
	gotJSON, _ := json.Marshal(got)
	var wantMap map[string]any
	_ = json.Unmarshal([]byte(want), &wantMap)
	wantJSON, _ := json.Marshal(wantMap)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("request body\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}

func TestDecideReordersAnswers(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(sampleResponse))
	}, "")
	resp, err := c.Decide(context.Background(), sampleRequest())
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if resp.Model != "jev-test" {
		t.Errorf("Model = %q", resp.Model)
	}
	if len(resp.Answers) != 3 || resp.Answers[0].Name != "urgent" || resp.Answers[1].Name != "category" || resp.Answers[2].Name != "severity" {
		t.Fatalf("answers out of order: %+v", resp.Answers)
	}
	if a := resp.Answers[0]; a.Kind != llm.DecisionYesNo || a.Probability != 0.9 {
		t.Errorf("urgent = %+v", a)
	}
	if a := resp.Answers[1]; a.Choice != "replication" || a.Confidence != 0.8 ||
		len(a.Probabilities) != 2 || a.Probabilities[0] != (llm.OptionProbability{Value: "replication", Probability: 0.8}) ||
		a.Probabilities[1] != (llm.OptionProbability{Value: "storage", Probability: 0.2}) {
		t.Errorf("category = %+v", a)
	}
	if a := resp.Answers[2]; a.Score != 1.4 || a.Confidence != 0.7 ||
		len(a.Probabilities) != 3 || a.Probabilities[0] != (llm.OptionProbability{Value: "low", Probability: 0.1}) ||
		a.Probabilities[2] != (llm.OptionProbability{Value: "high", Probability: 0.2}) {
		t.Errorf("severity = %+v", a)
	}
	want := llm.TokenUsage{PromptTokens: 100, CompletionTokens: 3, TotalTokens: 103}
	if resp.Usage != want || c.Usage() != want {
		t.Errorf("usage = %+v, cumulative %+v", resp.Usage, c.Usage())
	}
	if _, err := c.Decide(context.Background(), sampleRequest()); err != nil {
		t.Fatalf("second Decide: %v", err)
	}
	doubled := llm.TokenUsage{PromptTokens: 200, CompletionTokens: 6, TotalTokens: 206}
	if c.Usage() != doubled {
		t.Errorf("cumulative usage after two calls = %+v, want %+v", c.Usage(), doubled)
	}
}

func TestDecideModelSelection(t *testing.T) {
	var model string
	handler := func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Model string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		model = body.Model
		_, _ = w.Write([]byte(sampleResponse))
	}
	c := newTestClient(t, handler, "jev-preview")
	if _, err := c.Decide(context.Background(), sampleRequest()); err != nil || model != "jev-preview" {
		t.Fatalf("Options.Model: model=%q err=%v", model, err)
	}
	req := sampleRequest()
	req.Model = "jev-override"
	if _, err := c.Decide(context.Background(), req); err != nil || model != "jev-override" {
		t.Fatalf("request Model: model=%q err=%v", model, err)
	}
}

func TestDecideBadResponses(t *testing.T) {
	tests := map[string]struct {
		body    string
		leak    string // must not appear in the error message
		wantTot int
	}{
		"missing answer": {body: `{"model":"m","answers":{"urgent":{"type":"noul","noul":0.5},"category":{"type":"choice","choice":"storage","probabilities":{}}},"usage":{}}`},
		"wrong type": {body: `{"model":"m","answers":{"urgent":{"type":"bogus-type-xyz","choice":"x"},"category":{"type":"choice","choice":"storage"},"severity":{"type":"score","score":1}},"usage":{"input_tokens":7,"output_tokens":2}}`,
			leak: "bogus-type-xyz", wantTot: 9},
		"choice not in options": {body: `{"model":"m","answers":{"urgent":{"type":"noul","noul":0.5},"category":{"type":"choice","choice":"bogus-choice-xyz"},"severity":{"type":"score","score":1}},"usage":{"input_tokens":7,"output_tokens":2}}`,
			leak: "bogus-choice-xyz", wantTot: 9},
		"malformed": {body: `{"model":"leak-marker-xyz":`, leak: "leak-marker-xyz"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(tt.body)) }, "")
			_, err := c.Decide(context.Background(), sampleRequest())
			if !errors.Is(err, llm.ErrProviderError) {
				t.Fatalf("err = %v, want ErrProviderError", err)
			}
			if tt.leak != "" && strings.Contains(err.Error(), tt.leak) {
				t.Errorf("message relays response text: %q", err.Error())
			}
			if name == "malformed" {
				var pe *llm.ProviderError
				if !errors.As(err, &pe) || pe.StatusCode != 200 {
					t.Errorf("StatusCode = %+v, want 200", pe)
				}
			}
			if got := c.Usage().TotalTokens; got != tt.wantTot {
				t.Errorf("Usage().TotalTokens = %d, want %d", got, tt.wantTot)
			}
		})
	}
}

func TestDecideRejectsBeforeNetwork(t *testing.T) {
	called := false
	c := newTestClient(t, func(http.ResponseWriter, *http.Request) { called = true }, "")

	withImage := sampleRequest()
	withImage.Images = []llm.ImageContent{{URL: "https://example.com/chart.png"}}
	if _, err := c.Decide(context.Background(), withImage); !errors.Is(err, llm.ErrNotSupported) {
		t.Errorf("images: err = %v, want ErrNotSupported", err)
	}

	invalid := sampleRequest()
	invalid.Questions[0].Name = ""
	_, err := c.Decide(context.Background(), invalid)
	var pe *llm.ProviderError
	if !errors.Is(err, llm.ErrInvalidRequest) || !errors.As(err, &pe) || pe.Provider != "typesafe" {
		t.Errorf("invalid: err = %v, want ErrInvalidRequest from typesafe", err)
	}

	tooManyLevels := sampleRequest()
	tooManyLevels.Questions[2].Options = nil
	for i := 0; i < 11; i++ {
		tooManyLevels.Questions[2].Options = append(tooManyLevels.Questions[2].Options,
			llm.DecisionOption{Value: string(rune('a' + i))})
	}
	if _, err := c.Decide(context.Background(), tooManyLevels); !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("11 levels: err = %v, want ErrInvalidRequest", err)
	}
	if called {
		t.Fatal("server was called for a request that should fail locally")
	}
}

func TestDecideErrorMapping(t *testing.T) {
	tests := []struct {
		status  int
		body    string
		want    error
		message string
	}{
		{401, `{"detail":"Invalid API key"}`, llm.ErrAuthentication, "Invalid API key"},
		{400, `{"detail":{"error_type":"api_usage_error","message":"Unknown model: nope"}}`, llm.ErrInvalidRequest, "Unknown model: nope"},
		{422, `{"detail":[{"loc":["body","questions","urgent","type"],"msg":"Input should be 'noul'","type":"literal_error"}]}`, llm.ErrInvalidRequest, "body.questions.urgent.type: Input should be 'noul'"},
		{429, `{"detail":"Too many requests"}`, llm.ErrRateLimit, "Too many requests"},
		{500, `not json`, llm.ErrProviderError, "typesafe: HTTP 500"},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			c := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}, "")
			_, err := c.Decide(context.Background(), sampleRequest())
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.message) {
				t.Errorf("err = %q, want it to contain %q", err.Error(), tt.message)
			}
			var pe *llm.ProviderError
			if !errors.As(err, &pe) || pe.StatusCode != tt.status {
				t.Errorf("StatusCode = %+v", pe)
			}
		})
	}
}

func TestListModelsAndPing(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/models" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"jev-latest","description":"Latest","release_date":"2026-01-01"},{"name":"jev-preview"}]}`))
	}, "")
	names, err := c.ListModels(context.Background())
	if err != nil || len(names) != 2 || names[0] != "jev-latest" {
		t.Fatalf("ListModels = %v, %v", names, err)
	}
	infos, err := c.ListModelsWithMetadata(context.Background(), llm.WithCapabilities(llm.ModelCapabilityDecisions))
	if err != nil || len(infos) != 2 {
		t.Fatalf("ListModelsWithMetadata(decisions) = %v, %v", infos, err)
	}
	if infos, _ := c.ListModelsWithMetadata(context.Background(), llm.WithCapabilities(llm.ModelCapabilityChat)); len(infos) != 0 {
		t.Errorf("chat filter returned %v", infos)
	}
	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("Ping: %v", err)
	}
}

func TestUnsupportedMethods(t *testing.T) {
	c := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, "")
	ctx := context.Background()
	checks := map[string]error{}
	_, checks["Chat"] = c.Chat(ctx, llm.ChatRequest{})
	_, checks["ChatStream"] = c.ChatStream(ctx, llm.ChatRequest{})
	_, checks["Embed"] = c.Embed(ctx, "x")
	_, checks["EmbedBatch"] = c.EmbedBatch(ctx, []string{"x"})
	_, checks["EmbedMultimodal"] = c.EmbedMultimodal(ctx, llm.MultimodalEmbedRequest{})
	_, checks["Rerank"] = c.Rerank(ctx, llm.RerankRequest{})
	for name, err := range checks {
		if !errors.Is(err, llm.ErrNotSupported) {
			t.Errorf("%s: err = %v, want ErrNotSupported", name, err)
		}
	}
}

func TestNewKeyHandling(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	if _, err := New(llm.Options{}); !errors.Is(err, llm.ErrInvalidRequest) {
		t.Errorf("no key: err = %v, want ErrInvalidRequest", err)
	}
	t.Setenv("TYPESAFE_API_KEY", "env-key-not-real")
	c, err := New(llm.Options{})
	if err != nil {
		t.Fatalf("env key: %v", err)
	}
	if c.Provider() != "typesafe" {
		t.Errorf("Provider() = %q", c.Provider())
	}
	if _, err := New(llm.Options{APIKey: "k", BaseURL: "ftp://example.com"}); err == nil {
		t.Error("expected an error for a non-HTTP base URL")
	}
}
