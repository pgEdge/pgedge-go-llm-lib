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

func newDecideClient(t *testing.T, handler http.HandlerFunc) llm.Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c, err := New(llm.Options{APIKey: "sk-test-not-real", BaseURL: srv.URL, Model: "gpt-4o-mini", Retry: llm.RetryConfig{Disabled: true}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func decideRequest() llm.DecisionRequest {
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

// Answers deliberately out of order; probabilities out of option order.
const decideResponseBody = `{
  "model": "gpt-6-luna-test",
  "answers": [
    {"type": "score", "name": "severity", "score": 1.4, "confidence": 0.7,
     "probabilities": [{"value": 2, "label": "high", "probability": 0.2}, {"value": 0, "label": "low", "probability": 0.1}, {"value": 1, "label": "medium", "probability": 0.7}]},
    {"type": "predicate", "name": "urgent", "probability": 0.9},
    {"type": "choice", "name": "category", "choice": "replication", "confidence": 0.8,
     "probabilities": [{"value": "storage", "probability": 0.2}, {"value": "replication", "probability": 0.8}]}
  ],
  "usage": {"input_tokens": 100, "output_tokens": 3, "total_tokens": 103}
}`

func TestDecideScoreMatchesByIndex(t *testing.T) {
	body := `{"model":"m","answers":[{"type":"score","name":"severity","score":1,"confidence":1,
"probabilities":[{"value":0,"label":"LOW","probability":0.1},{"value":1,"label":"Medium!","probability":0.8},{"value":2,"label":"x","probability":0.1},{"value":7,"label":"low","probability":0.9},{"value":-1,"label":"high","probability":0.9}]}],"usage":{}}`
	c := newDecideClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	})
	resp, err := c.Decide(context.Background(), llm.DecisionRequest{
		Input: "x",
		Questions: []llm.DecisionQuestion{{Name: "severity", Kind: llm.DecisionScore, Instructions: "How severe?",
			Options: []llm.DecisionOption{{Value: "low"}, {Value: "medium"}, {Value: "high"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := resp.Answers[0].Probabilities
	want := []llm.OptionProbability{{Value: "low", Probability: 0.1}, {Value: "medium", Probability: 0.8}, {Value: "high", Probability: 0.1}}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("probabilities = %+v, want %+v", got, want)
		}
	}
}

func TestDecideRequestShape(t *testing.T) {
	var got, path string
	c := newDecideClient(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		_, _ = w.Write([]byte(decideResponseBody))
	})
	if _, err := c.Decide(context.Background(), decideRequest()); err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if path != "/decisions" {
		t.Errorf("path = %q", path)
	}
	want := `{"model":"gpt-6-luna","input":"Replication lag on replica-1 is 45 seconds and rising.","questions":[` +
		`{"type":"predicate","name":"urgent","instructions":"Is this urgent?"},` +
		`{"type":"choice","name":"category","instructions":"Which category?","choices":[{"value":"replication","description":"Replication issues"},{"value":"storage"}]},` +
		`{"type":"score","name":"severity","instructions":"How severe?","levels":[{"label":"low"},{"label":"medium","description":"Needs attention today"},{"label":"high"}]}]}`
	if got != want {
		t.Fatalf("request body\n got: %s\nwant: %s", got, want)
	}
}

func TestDecideImageInput(t *testing.T) {
	var body struct {
		Input []struct {
			Role    string `json:"role"`
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				ImageURL string `json:"image_url"`
			} `json:"content"`
		} `json:"input"`
	}
	c := newDecideClient(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = w.Write([]byte(decideResponseBody))
	})
	req := decideRequest()
	req.Images = []llm.ImageContent{
		{URL: "https://example.com/chart.png"},
		{Data: []byte("fake-png"), MediaType: "image/png"},
	}
	if _, err := c.Decide(context.Background(), req); err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if len(body.Input) != 1 || body.Input[0].Role != "user" || len(body.Input[0].Content) != 3 {
		t.Fatalf("input = %+v", body.Input)
	}
	parts := body.Input[0].Content
	if parts[0].Type != "input_text" || parts[0].Text != req.Input {
		t.Errorf("text part = %+v", parts[0])
	}
	if parts[1].Type != "input_image" || parts[1].ImageURL != "https://example.com/chart.png" {
		t.Errorf("url image part = %+v", parts[1])
	}
	if parts[2].Type != "input_image" || parts[2].ImageURL != "data:image/png;base64,ZmFrZS1wbmc=" {
		t.Errorf("data image part = %+v", parts[2])
	}
}

func TestDecideResponseMapping(t *testing.T) {
	c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(decideResponseBody))
	})
	resp, err := c.Decide(context.Background(), decideRequest())
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if resp.Model != "gpt-6-luna-test" || len(resp.Answers) != 3 {
		t.Fatalf("resp = %+v", resp)
	}
	if a := resp.Answers[0]; a.Name != "urgent" || a.Kind != llm.DecisionYesNo || a.Probability != 0.9 {
		t.Errorf("urgent = %+v", a)
	}
	if a := resp.Answers[1]; a.Name != "category" || a.Choice != "replication" || a.Confidence != 0.8 ||
		a.Probabilities[0] != (llm.OptionProbability{Value: "replication", Probability: 0.8}) ||
		a.Probabilities[1] != (llm.OptionProbability{Value: "storage", Probability: 0.2}) {
		t.Errorf("category = %+v", a)
	}
	if a := resp.Answers[2]; a.Name != "severity" || a.Score != 1.4 ||
		a.Probabilities[0] != (llm.OptionProbability{Value: "low", Probability: 0.1}) ||
		a.Probabilities[2] != (llm.OptionProbability{Value: "high", Probability: 0.2}) {
		t.Errorf("severity = %+v", a)
	}
	want := llm.TokenUsage{PromptTokens: 100, CompletionTokens: 3, TotalTokens: 103}
	if resp.Usage != want || c.Usage() != want {
		t.Errorf("usage = %+v / %+v", resp.Usage, c.Usage())
	}
}

func TestDecideRefusal(t *testing.T) {
	c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","answers":[
			{"type":"predicate","name":"urgent","probability":0.4},
			{"type":"refusal","name":"category"},
			{"type":"score","name":"severity","score":0,"confidence":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]}],"usage":{}}`))
	})
	resp, err := c.Decide(context.Background(), decideRequest())
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if a := resp.Answers[1]; !a.Refused || a.Name != "category" || a.Kind != llm.DecisionChoice || a.Choice != "" {
		t.Errorf("refused answer = %+v", a)
	}
	if resp.Answers[0].Refused || resp.Answers[0].Probability != 0.4 {
		t.Errorf("other answer = %+v", resp.Answers[0])
	}
}

func TestDecideBadResponses(t *testing.T) {
	tests := []struct{ name, body, want string }{
		{"missing answer", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4}],"usage":{}}`, "has no answer for question"},
		{"wrong type", `{"model":"m","answers":[{"type":"choice","name":"urgent","choice":"x"},{"type":"refusal","name":"category"},{"type":"refusal","name":"severity"}],"usage":{}}`, "unexpected type"},
		{"non-string choice", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":true},{"type":"refusal","name":"severity"}],"usage":{}}`, "non-string choice"},
		{"choice not in options", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"other"},{"type":"refusal","name":"severity"}],"usage":{}}`, "not one of its options"},
		{"duplicate answer", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"predicate","name":"urgent","probability":0.5},{"type":"refusal","name":"category"},{"type":"refusal","name":"severity"}],"usage":{}}`, "more than one answer for question"},
		{"score out of range", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"refusal","name":"category"},{"type":"score","name":"severity","score":2.5,"confidence":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]}],"usage":{}}`, "missing or out-of-range score"},
		{"negative score", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"refusal","name":"category"},{"type":"score","name":"severity","score":-0.1,"confidence":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]}],"usage":{}}`, "missing or out-of-range score"},
		{"missing probability", `{"model":"m","answers":[{"type":"predicate","name":"urgent"},{"type":"refusal","name":"category"},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range probability"},
		{"probability out of range", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":7.5},{"type":"refusal","name":"category"},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range probability"},
		{"missing score", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"refusal","name":"category"},{"type":"score","name":"severity","confidence":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]}],"usage":{}}`, "missing or out-of-range score"},
		{"choice missing confidence", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","probabilities":[{"value":"replication","probability":0.4},{"value":"storage","probability":0.6}]},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range confidence"},
		{"choice confidence out of range", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","confidence":7.5,"probabilities":[{"value":"replication","probability":0.4},{"value":"storage","probability":0.6}]},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range confidence"},
		{"choice missing probabilities", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","confidence":0.6},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range option probability"},
		{"choice probability out of range", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","confidence":0.6,"probabilities":[{"value":"replication","probability":-3},{"value":"storage","probability":4}]},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range option probability"},
		{"choice entry missing probability", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","confidence":0.6,"probabilities":[{"value":"replication"},{"value":"storage","probability":0.6}]},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range option probability"},
		{"choice non-string option value", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"storage","confidence":0.6,"probabilities":[{"value":1,"probability":0.4},{"value":"storage","probability":0.6}]},{"type":"refusal","name":"severity"}],"usage":{}}`, "missing or out-of-range option probability"},
		{"score missing confidence", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"refusal","name":"category"},{"type":"score","name":"severity","score":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]}],"usage":{}}`, "missing or out-of-range confidence"},
		{"score missing level probability", `{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"refusal","name":"category"},{"type":"score","name":"severity","score":1,"confidence":0.5,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.8}]}],"usage":{}}`, "missing or out-of-range option probability"},
		{"malformed", `{"model":`, "could not be decoded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(tt.body)) })
			_, err := c.Decide(context.Background(), decideRequest())
			if !errors.Is(err, llm.ErrProviderError) {
				t.Fatalf("err = %v, want ErrProviderError", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestDecideErrors(t *testing.T) {
	called := false
	c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid 'questions[0].choices': array too short.","type":"invalid_request_error","param":"questions[0].choices","code":"array_below_min_length"}}`))
	})
	_, err := c.Decide(context.Background(), decideRequest())
	if !errors.Is(err, llm.ErrInvalidRequest) {
		t.Fatalf("400: err = %v", err)
	}

	called = false
	bad := decideRequest()
	bad.Questions = nil
	if _, err := c.Decide(context.Background(), bad); !errors.Is(err, llm.ErrInvalidRequest) || called {
		t.Fatalf("invalid request: err = %v, called = %v", err, called)
	}
}

func TestDecideModelOverride(t *testing.T) {
	var model string
	c := newDecideClient(t, func(w http.ResponseWriter, r *http.Request) {
		var b struct{ Model string }
		_ = json.NewDecoder(r.Body).Decode(&b)
		model = b.Model
		_, _ = w.Write([]byte(decideResponseBody))
	})
	req := decideRequest()
	req.Model = "gpt-6-luna-2026-10-01"
	if _, err := c.Decide(context.Background(), req); err != nil || model != req.Model {
		t.Fatalf("model = %q, err = %v", model, err)
	}
}

func TestDecisionModelCapability(t *testing.T) {
	caps := lookupOpenAICapabilities("gpt-6-luna")
	found := false
	for _, c := range caps {
		if c == llm.ModelCapabilityDecisions {
			found = true
		}
	}
	if !found {
		t.Fatalf("gpt-6-luna capabilities = %v, want decisions", caps)
	}
}

func TestMapError422(t *testing.T) {
	c := &client{apiKey: "sk-test-not-real"}
	if err := c.mapError(422, []byte(`{"error":{"message":"bad"}}`)); !errors.Is(err, llm.ErrInvalidRequest) {
		t.Fatalf("422: err = %v", err)
	}
}

func TestDecideChoiceNotInOptions(t *testing.T) {
	c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","answers":[{"type":"predicate","name":"urgent","probability":0.4},{"type":"choice","name":"category","choice":"secret-leak","confidence":1,"probabilities":[{"value":0,"probability":0.2},{"value":1,"probability":0.5},{"value":2,"probability":0.3}]},{"type":"refusal","name":"severity"}],"usage":{"input_tokens":7,"output_tokens":1,"total_tokens":8}}`))
	})
	_, err := c.Decide(context.Background(), decideRequest())
	if !errors.Is(err, llm.ErrProviderError) {
		t.Fatalf("err = %v, want ErrProviderError", err)
	}
	if strings.Contains(err.Error(), "secret-leak") {
		t.Errorf("error embeds response text: %v", err)
	}
	if got := c.Usage().TotalTokens; got != 8 {
		t.Errorf("usage not counted on bad answer: %d", got)
	}
}

func TestDecideMalformedBodyNotEchoed(t *testing.T) {
	c := newDecideClient(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model": "secret-body-text", "answers": 5}`))
	})
	_, err := c.Decide(context.Background(), decideRequest())
	if !errors.Is(err, llm.ErrProviderError) {
		t.Fatalf("err = %v", err)
	}
	if strings.Contains(err.Error(), "secret-body-text") {
		t.Errorf("error embeds body: %v", err)
	}
}
