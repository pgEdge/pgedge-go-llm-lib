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
	"errors"
	"fmt"
	"net/http"
	"strconv"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// maxScoreLevels is TypeSafe's limit on the levels of a score question.
const maxScoreLevels = 10

type decideRequest struct {
	Model     string                  `json:"model"`
	State     string                  `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	// Criteria is a map of option to description (or null) for a
	// choice question, and a list of levels for a score question.
	Criteria any `json:"criteria,omitempty"`
}

type scoreLevel struct {
	Label       string `json:"label"`
	Description string `json:"description"`
}

type decideResponse struct {
	Model   string                `json:"model"`
	Answers map[string]wireAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type wireAnswer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul"`
	Choice        string             `json:"choice"`
	Score         float64            `json:"score"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

var wireTypes = map[llm.DecisionKind]string{
	llm.DecisionYesNo:  "noul",
	llm.DecisionChoice: "choice",
	llm.DecisionScore:  "score",
}

func (c *client) Decide(ctx context.Context, req llm.DecisionRequest) (*llm.DecisionResponse, error) {
	if err := req.Validate(); err != nil {
		var pe *llm.ProviderError
		if errors.As(err, &pe) {
			pe.Provider = providerName
		}
		return nil, err
	}
	if len(req.Images) > 0 {
		return nil, c.notSupported("image input to decision models")
	}
	wire, err := c.buildDecideRequest(req)
	if err != nil {
		return nil, err
	}
	var raw decideResponse
	if err := c.doJSON(ctx, http.MethodPost, "/systemone", wire, &raw); err != nil {
		return nil, err
	}
	answers := make([]llm.DecisionAnswer, len(req.Questions))
	for i, q := range req.Questions {
		a, err := convertAnswer(q, raw.Answers)
		if err != nil {
			return nil, err
		}
		answers[i] = a
	}
	usage := llm.TokenUsage{
		PromptTokens:     raw.Usage.InputTokens,
		CompletionTokens: raw.Usage.OutputTokens,
		TotalTokens:      raw.Usage.InputTokens + raw.Usage.OutputTokens,
	}
	c.addUsage(usage)
	return &llm.DecisionResponse{Answers: answers, Model: raw.Model, Usage: usage}, nil
}

func (c *client) buildDecideRequest(req llm.DecisionRequest) (decideRequest, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		model = DefaultModel
	}
	questions := make(map[string]wireQuestion, len(req.Questions))
	for _, q := range req.Questions {
		wq := wireQuestion{Type: wireTypes[q.Kind], Instructions: q.Instructions}
		switch q.Kind {
		case llm.DecisionChoice:
			criteria := make(map[string]*string, len(q.Options))
			for _, o := range q.Options {
				if o.Description == "" {
					criteria[o.Value] = nil
				} else {
					d := o.Description
					criteria[o.Value] = &d
				}
			}
			wq.Criteria = criteria
		case llm.DecisionScore:
			if len(q.Options) > maxScoreLevels {
				return decideRequest{}, &llm.ProviderError{
					Err:      llm.ErrInvalidRequest,
					Message:  fmt.Sprintf("score question %q has %d levels; TypeSafe allows at most %d", q.Name, len(q.Options), maxScoreLevels),
					Provider: providerName,
				}
			}
			levels := make([]any, len(q.Options))
			for i, o := range q.Options {
				if o.Description == "" {
					levels[i] = o.Value
				} else {
					levels[i] = scoreLevel{Label: o.Value, Description: o.Description}
				}
			}
			wq.Criteria = levels
		}
		questions[q.Name] = wq
	}
	return decideRequest{Model: model, State: req.Input, Questions: questions}, nil
}

// convertAnswer maps TypeSafe's answer to q onto the neutral type,
// putting probabilities in the request's option order. Score
// probabilities are keyed by level index ("0", "1", ...).
func convertAnswer(q llm.DecisionQuestion, answers map[string]wireAnswer) (llm.DecisionAnswer, error) {
	wa, ok := answers[q.Name]
	if !ok {
		return llm.DecisionAnswer{}, badResponse(fmt.Sprintf("response has no answer for question %q", q.Name))
	}
	if wa.Type != wireTypes[q.Kind] {
		return llm.DecisionAnswer{}, badResponse(fmt.Sprintf("answer to question %q has type %q, want %q", q.Name, wa.Type, wireTypes[q.Kind]))
	}
	a := llm.DecisionAnswer{Name: q.Name, Kind: q.Kind}
	switch q.Kind {
	case llm.DecisionYesNo:
		a.Probability = wa.Noul
	case llm.DecisionChoice:
		a.Choice, a.Confidence = wa.Choice, wa.Confidence
		a.Probabilities = make([]llm.OptionProbability, len(q.Options))
		for i, o := range q.Options {
			a.Probabilities[i] = llm.OptionProbability{Value: o.Value, Probability: wa.Probabilities[o.Value]}
		}
	case llm.DecisionScore:
		a.Score, a.Confidence = wa.Score, wa.Confidence
		a.Probabilities = make([]llm.OptionProbability, len(q.Options))
		for i, o := range q.Options {
			a.Probabilities[i] = llm.OptionProbability{Value: o.Value, Probability: wa.Probabilities[strconv.Itoa(i)]}
		}
	}
	return a, nil
}

func badResponse(msg string) error {
	return &llm.ProviderError{Err: llm.ErrProviderError, Message: msg, Provider: providerName}
}
