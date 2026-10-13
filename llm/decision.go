//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package llm

import "fmt"

// DecisionKind identifies the shape of a decision question.
type DecisionKind string

// DecisionKind values. The names are deliberately provider-neutral:
// TypeSafe calls a yes/no question "noul" and OpenAI calls it
// "predicate".
const (
	// DecisionYesNo asks a question whose answer is the probability
	// that the statement is true.
	DecisionYesNo DecisionKind = "yes_no"
	// DecisionChoice asks the model to pick one of a set of options.
	DecisionChoice DecisionKind = "choice"
	// DecisionScore asks the model to rate on an ordered scale of
	// levels, lowest first.
	DecisionScore DecisionKind = "score"
)

// MaxDecisionChoices is the largest number of options a choice
// question may carry; both supported APIs share this limit.
const MaxDecisionChoices = 255

// DecisionOption is one option of a choice question, or one level of
// a score question.
type DecisionOption struct {
	// Value is the option returned when it is chosen (choice), or the
	// level's label (score). Required and unique within the question.
	Value string
	// Description optionally tells the model what the option means.
	Description string
}

// DecisionQuestion is one question in a DecisionRequest.
type DecisionQuestion struct {
	// Name identifies the question and its answer. Required and unique
	// within the request.
	Name string
	// Kind selects the question type.
	Kind DecisionKind
	// Instructions is the question put to the model. Required.
	Instructions string
	// Options holds the options of a choice question (2 to
	// MaxDecisionChoices) or the levels of a score question (at least
	// 2, lowest first). Must be empty for a yes/no question.
	Options []DecisionOption
}

// DecisionRequest asks a decision model one or more questions about
// the same input.
type DecisionRequest struct {
	// Model overrides the client's default decision model for this
	// request.
	Model string
	// Input is the text the questions are about. Required unless
	// Images is set.
	Input string
	// Images optionally accompanies Input. Not every provider accepts
	// images; those that do not return ErrNotSupported.
	Images []ImageContent
	// Questions to answer; at least one.
	Questions []DecisionQuestion
	// Extensions carries provider-specific options.
	Extensions []ProviderExtension
}

// OptionProbability is the probability the model assigned to one
// option of a choice or score question.
type OptionProbability struct {
	// Value is the option's Value (choice) or the level's label (score).
	Value       string
	Probability float64
}

// DecisionAnswer is the answer to one DecisionQuestion.
type DecisionAnswer struct {
	// Name is the question's Name.
	Name string
	// Kind is the question's Kind.
	Kind DecisionKind
	// Refused is true when the provider declined to answer this
	// question; every other field except Name and Kind is then zero.
	Refused bool
	// Probability is, for a yes/no question, the probability that the
	// answer is yes.
	Probability float64
	// Choice is, for a choice question, the Value of the selected
	// option.
	Choice string
	// Score is, for a score question, the probability-weighted level
	// index, where 0 is the first (lowest) level.
	Score float64
	// Confidence is the provider's confidence in Choice or Score.
	Confidence float64
	// Probabilities lists, for a choice or score question, the
	// probability of each option, in the order the request gave them.
	Probabilities []OptionProbability
}

// DecisionResponse holds the answers to a DecisionRequest.
type DecisionResponse struct {
	// Answers holds one answer per question, in request order.
	Answers []DecisionAnswer
	// Model is the model the provider reports having used.
	Model string
	// Usage is the token usage for this request.
	Usage TokenUsage
}

// Answer returns the answer to the named question.
func (r *DecisionResponse) Answer(name string) (DecisionAnswer, bool) {
	if r == nil {
		return DecisionAnswer{}, false
	}
	for _, a := range r.Answers {
		if a.Name == name {
			return a, true
		}
	}
	return DecisionAnswer{}, false
}

// Validate checks the request for errors that every provider would
// reject, so that they fail before any network call. The error is a
// *ProviderError wrapping ErrInvalidRequest, with Provider unset;
// providers fill it in.
func (r DecisionRequest) Validate() error {
	if r.Input == "" && len(r.Images) == 0 {
		return invalidDecision("decision request needs Input or Images")
	}
	if len(r.Questions) == 0 {
		return invalidDecision("decision request needs at least one question")
	}
	seen := make(map[string]struct{}, len(r.Questions))
	for i, q := range r.Questions {
		if q.Name == "" {
			return invalidDecision(fmt.Sprintf("question %d has no Name", i))
		}
		if _, dup := seen[q.Name]; dup {
			return invalidDecision(fmt.Sprintf("duplicate question name %q", q.Name))
		}
		seen[q.Name] = struct{}{}
		if q.Instructions == "" {
			return invalidDecision(fmt.Sprintf("question %q has no Instructions", q.Name))
		}
		if err := validateDecisionOptions(q); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionOptions(q DecisionQuestion) error {
	switch q.Kind {
	case DecisionYesNo:
		if len(q.Options) != 0 {
			return invalidDecision(fmt.Sprintf("yes/no question %q takes no Options", q.Name))
		}
		return nil
	case DecisionChoice:
		if len(q.Options) < 2 || len(q.Options) > MaxDecisionChoices {
			return invalidDecision(fmt.Sprintf("choice question %q needs 2 to %d options, got %d",
				q.Name, MaxDecisionChoices, len(q.Options)))
		}
	case DecisionScore:
		if len(q.Options) < 2 {
			return invalidDecision(fmt.Sprintf("score question %q needs at least 2 levels, got %d",
				q.Name, len(q.Options)))
		}
	default:
		return invalidDecision(fmt.Sprintf("question %q has unknown Kind %q", q.Name, q.Kind))
	}
	values := make(map[string]struct{}, len(q.Options))
	for _, o := range q.Options {
		if o.Value == "" {
			return invalidDecision(fmt.Sprintf("question %q has an option with no Value", q.Name))
		}
		if _, dup := values[o.Value]; dup {
			return invalidDecision(fmt.Sprintf("question %q has duplicate option %q", q.Name, o.Value))
		}
		values[o.Value] = struct{}{}
	}
	return nil
}

func invalidDecision(msg string) error {
	return &ProviderError{Err: ErrInvalidRequest, Message: msg}
}
