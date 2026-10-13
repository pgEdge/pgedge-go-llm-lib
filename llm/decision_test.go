//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm

import (
	"errors"
	"fmt"
	"testing"
)

func validDecisionRequest() DecisionRequest {
	return DecisionRequest{
		Input: "Replication lag on replica-1 is 45 seconds and rising.",
		Questions: []DecisionQuestion{
			{Name: "urgent", Kind: DecisionYesNo, Instructions: "Is this urgent?"},
			{Name: "category", Kind: DecisionChoice, Instructions: "Which category?",
				Options: []DecisionOption{{Value: "replication", Description: "Replication issues"}, {Value: "storage"}}},
			{Name: "severity", Kind: DecisionScore, Instructions: "How severe?",
				Options: []DecisionOption{{Value: "low"}, {Value: "medium"}, {Value: "high"}}},
		},
	}
}

func TestDecisionRequestValidateAcceptsValid(t *testing.T) {
	if err := validDecisionRequest().Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	imagesOnly := validDecisionRequest()
	imagesOnly.Input = ""
	imagesOnly.Images = []ImageContent{{URL: "https://example.com/chart.png"}}
	if err := imagesOnly.Validate(); err != nil {
		t.Fatalf("Validate with images only: %v", err)
	}
}

func TestDecisionRequestValidateRejects(t *testing.T) {
	many := make([]DecisionOption, MaxDecisionChoices+1)
	for i := range many {
		many[i] = DecisionOption{Value: fmt.Sprintf("v%d", i)}
	}
	tests := []struct {
		name   string
		mutate func(r *DecisionRequest)
	}{
		{"no input or images", func(r *DecisionRequest) { r.Input = "" }},
		{"no questions", func(r *DecisionRequest) { r.Questions = nil }},
		{"empty name", func(r *DecisionRequest) { r.Questions[0].Name = "" }},
		{"duplicate name", func(r *DecisionRequest) { r.Questions[1].Name = "urgent" }},
		{"empty instructions", func(r *DecisionRequest) { r.Questions[0].Instructions = "" }},
		{"unknown kind", func(r *DecisionRequest) { r.Questions[0].Kind = "maybe" }},
		{"yes_no with options", func(r *DecisionRequest) {
			r.Questions[0].Options = []DecisionOption{{Value: "yes"}}
		}},
		{"choice with one option", func(r *DecisionRequest) {
			r.Questions[1].Options = r.Questions[1].Options[:1]
		}},
		{"choice with too many options", func(r *DecisionRequest) { r.Questions[1].Options = many }},
		{"choice duplicate value", func(r *DecisionRequest) {
			r.Questions[1].Options[1].Value = "replication"
		}},
		{"choice empty value", func(r *DecisionRequest) { r.Questions[1].Options[1].Value = "" }},
		{"score with one level", func(r *DecisionRequest) {
			r.Questions[2].Options = r.Questions[2].Options[:1]
		}},
		{"score empty label", func(r *DecisionRequest) { r.Questions[2].Options[0].Value = "" }},
		{"score duplicate label", func(r *DecisionRequest) { r.Questions[2].Options[2].Value = "low" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := validDecisionRequest()
			tt.mutate(&req)
			err := req.Validate()
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("Validate() = %v, want ErrInvalidRequest", err)
			}
			var pe *ProviderError
			if !errors.As(err, &pe) {
				t.Fatalf("Validate() error is %T, want *ProviderError", err)
			}
		})
	}
}

func TestDecisionResponseAnswer(t *testing.T) {
	resp := &DecisionResponse{Answers: []DecisionAnswer{
		{Name: "urgent", Kind: DecisionYesNo, Probability: 0.9},
		{Name: "category", Kind: DecisionChoice, Choice: "storage"},
	}}
	a, ok := resp.Answer("category")
	if !ok || a.Choice != "storage" {
		t.Fatalf("Answer(category) = %+v, %v", a, ok)
	}
	if _, ok := resp.Answer("missing"); ok {
		t.Fatal("Answer(missing) reported ok")
	}
	var nilResp *DecisionResponse
	if _, ok := nilResp.Answer("urgent"); ok {
		t.Fatal("Answer on nil response reported ok")
	}
}
