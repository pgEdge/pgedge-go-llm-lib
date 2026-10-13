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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
	"github.com/pgEdge/pgedge-go-llm-lib/llm/internal/httpclient"
)

// DefaultDecisionModel is the model Decide uses when the request names
// none. Options.Model is deliberately not used: on an OpenAI client it
// names a chat or embedding model.
const DefaultDecisionModel = "gpt-6-luna"

type decisionRequestWire struct {
	Model     string                 `json:"model"`
	Input     any                    `json:"input"`
	Questions []decisionQuestionWire `json:"questions"`
}

type decisionMessageWire struct {
	Role    string                `json:"role"`
	Content []decisionContentWire `json:"content"`
}

type decisionContentWire struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type decisionQuestionWire struct {
	Type         string               `json:"type"`
	Name         string               `json:"name"`
	Instructions string               `json:"instructions"`
	Choices      []decisionChoiceWire `json:"choices,omitempty"`
	Levels       []decisionLevelWire  `json:"levels,omitempty"`
}

type decisionChoiceWire struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

type decisionLevelWire struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

type decisionResponseWire struct {
	Model   string               `json:"model"`
	Answers []decisionAnswerWire `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type decisionAnswerWire struct {
	Type          string                    `json:"type"`
	Name          string                    `json:"name"`
	Probability   float64                   `json:"probability"`
	Choice        json.RawMessage           `json:"choice"`
	Score         float64                   `json:"score"`
	Confidence    float64                   `json:"confidence"`
	Probabilities []decisionProbabilityWire `json:"probabilities"`
}

type decisionProbabilityWire struct {
	Value       json.RawMessage `json:"value"`
	Label       string          `json:"label"`
	Probability float64         `json:"probability"`
}

var decisionWireTypes = map[llm.DecisionKind]string{
	llm.DecisionYesNo:  "predicate",
	llm.DecisionChoice: "choice",
	llm.DecisionScore:  "score",
}

// Decide calls OpenAI's Decisions API (POST /decisions).
func (c *client) Decide(ctx context.Context, req llm.DecisionRequest) (*llm.DecisionResponse, error) {
	if err := req.Validate(); err != nil {
		var pe *llm.ProviderError
		if errors.As(err, &pe) {
			pe.Provider = providerName
		}
		return nil, err
	}
	var raw decisionResponseWire
	status, body, err := httpclient.DoJSON(ctx, c.httpClient, http.MethodPost,
		c.baseURL+"/decisions", c.headers(), buildDecisionRequest(req), &raw)
	if err != nil && status == 0 {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, c.mapError(status, body)
	}
	if err != nil {
		return nil, &llm.ProviderError{
			Err:        llm.ErrProviderError,
			StatusCode: status,
			Message:    "response could not be decoded",
			Provider:   providerName,
		}
	}
	// Count billed tokens before converting, so they are recorded even
	// when an answer turns out to be unusable.
	usage := llm.TokenUsage{
		PromptTokens:     raw.Usage.InputTokens,
		CompletionTokens: raw.Usage.OutputTokens,
		TotalTokens:      raw.Usage.TotalTokens,
	}
	c.addUsage(usage)
	byName := make(map[string]decisionAnswerWire, len(raw.Answers))
	asked := make(map[string]struct{}, len(req.Questions))
	for _, q := range req.Questions {
		asked[q.Name] = struct{}{}
	}
	for _, a := range raw.Answers {
		if _, ok := asked[a.Name]; !ok {
			continue // ignore answers to questions nobody asked
		}
		if _, dup := byName[a.Name]; dup {
			// a.Name is one of the caller's own question names here.
			return nil, decisionResponseError(fmt.Sprintf("response has more than one answer for question %q", a.Name))
		}
		byName[a.Name] = a
	}
	answers := make([]llm.DecisionAnswer, len(req.Questions))
	for i, q := range req.Questions {
		a, err := convertDecisionAnswer(q, byName)
		if err != nil {
			return nil, err
		}
		answers[i] = a
	}
	return &llm.DecisionResponse{Answers: answers, Model: raw.Model, Usage: usage}, nil
}

func buildDecisionRequest(req llm.DecisionRequest) decisionRequestWire {
	model := req.Model
	if model == "" {
		model = DefaultDecisionModel
	}
	wire := decisionRequestWire{Model: model, Input: req.Input}
	if len(req.Images) > 0 {
		var parts []decisionContentWire
		if req.Input != "" {
			parts = append(parts, decisionContentWire{Type: "input_text", Text: req.Input})
		}
		for _, img := range req.Images {
			url := img.URL
			if url == "" {
				url = "data:" + img.MediaType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
			}
			parts = append(parts, decisionContentWire{Type: "input_image", ImageURL: url})
		}
		wire.Input = []decisionMessageWire{{Role: "user", Content: parts}}
	}
	wire.Questions = make([]decisionQuestionWire, len(req.Questions))
	for i, q := range req.Questions {
		wq := decisionQuestionWire{Type: decisionWireTypes[q.Kind], Name: q.Name, Instructions: q.Instructions}
		for _, o := range q.Options {
			if q.Kind == llm.DecisionChoice {
				wq.Choices = append(wq.Choices, decisionChoiceWire{Value: o.Value, Description: o.Description})
			} else {
				wq.Levels = append(wq.Levels, decisionLevelWire{Label: o.Value, Description: o.Description})
			}
		}
		wire.Questions[i] = wq
	}
	return wire
}

// convertDecisionAnswer maps OpenAI's answer to q onto the neutral
// type, putting probabilities in the request's option order.
func convertDecisionAnswer(q llm.DecisionQuestion, byName map[string]decisionAnswerWire) (llm.DecisionAnswer, error) {
	wa, ok := byName[q.Name]
	if !ok {
		return llm.DecisionAnswer{}, decisionResponseError(fmt.Sprintf("response has no answer for question %q", q.Name))
	}
	a := llm.DecisionAnswer{Name: q.Name, Kind: q.Kind}
	if wa.Type == "refusal" {
		a.Refused = true
		return a, nil
	}
	if wa.Type != decisionWireTypes[q.Kind] {
		return llm.DecisionAnswer{}, decisionResponseError(fmt.Sprintf("answer to question %q has an unexpected type", q.Name))
	}
	switch q.Kind {
	case llm.DecisionYesNo:
		a.Probability = wa.Probability
	case llm.DecisionChoice:
		if err := json.Unmarshal(wa.Choice, &a.Choice); err != nil {
			return llm.DecisionAnswer{}, decisionResponseError(fmt.Sprintf("answer to question %q has a non-string choice", q.Name))
		}
		if !decisionHasOption(q.Options, a.Choice) {
			return llm.DecisionAnswer{}, decisionResponseError(fmt.Sprintf("answer to question %q is not one of its options", q.Name))
		}
		a.Confidence = wa.Confidence
		probs := make(map[string]float64, len(wa.Probabilities))
		for _, p := range wa.Probabilities {
			var v string
			if json.Unmarshal(p.Value, &v) == nil {
				probs[v] = p.Probability
			}
		}
		a.Probabilities = orderedProbabilities(q.Options, probs)
	case llm.DecisionScore:
		a.Score, a.Confidence = wa.Score, wa.Confidence
		probs := make(map[string]float64, len(wa.Probabilities))
		for _, p := range wa.Probabilities {
			probs[p.Label] = p.Probability
		}
		a.Probabilities = orderedProbabilities(q.Options, probs)
	}
	return a, nil
}

func decisionHasOption(options []llm.DecisionOption, value string) bool {
	for _, o := range options {
		if o.Value == value {
			return true
		}
	}
	return false
}

func orderedProbabilities(options []llm.DecisionOption, probs map[string]float64) []llm.OptionProbability {
	out := make([]llm.OptionProbability, len(options))
	for i, o := range options {
		out[i] = llm.OptionProbability{Value: o.Value, Probability: probs[o.Value]}
	}
	return out
}

func decisionResponseError(msg string) error {
	return &llm.ProviderError{Err: llm.ErrProviderError, Message: msg, Provider: providerName}
}
