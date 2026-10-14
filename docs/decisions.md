# Decisions

`Client.Decide` asks a decision model one or more questions about a piece of
input and returns probabilities rather than free text. Use it for
classification and triage, where the probability behind an answer matters as
much as the answer itself: routing an alert, flagging a support ticket, or
scoring a document against a rubric. For open-ended generation, use `Chat`
instead.

## Supported providers

| Provider | Default model | Input |
|---|---|---|
| `typesafe` | `jev-latest` (also `jev-preview`) | Text only |
| `openai` | `gpt-6-luna` (public beta) | Text and images |

Every other provider returns `ErrNotSupported` from `Decide`.

TypeSafe reads its key from `Options.APIKey`, `Options.APIKeyFile` or the
`TYPESAFE_API_KEY` environment variable.

## Asking questions

```go
import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/pgEdge/pgedge-go-llm-lib/llm"
    _ "github.com/pgEdge/pgedge-go-llm-lib/llm/provider/typesafe"
)

func main() {
    client, err := llm.NewClient("typesafe", llm.Options{
        APIKey: os.Getenv("TYPESAFE_API_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }

    resp, err := client.Decide(context.Background(), llm.DecisionRequest{
        Input: "Alert: replication lag on the standby has grown from 2 seconds to 45 seconds over ten minutes and is still rising.",
        Questions: []llm.DecisionQuestion{
            {
                Name:         "urgent",
                Kind:         llm.DecisionYesNo,
                Instructions: "Does this need attention within the hour?",
            },
            {
                Name:         "category",
                Kind:         llm.DecisionChoice,
                Instructions: "Which area does this alert concern?",
                Options: []llm.DecisionOption{
                    {Value: "replication"},
                    {Value: "storage"},
                    {Value: "connections"},
                },
            },
            {
                Name:         "severity",
                Kind:         llm.DecisionScore,
                Instructions: "How severe is this?",
                Options: []llm.DecisionOption{
                    {Value: "low"},
                    {Value: "medium", Description: "Degraded but working"},
                    {Value: "high"},
                },
            },
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    if a, ok := resp.Answer("urgent"); ok && !a.Refused {
        fmt.Printf("urgent: %.2f\n", a.Probability)
    }
    if a, ok := resp.Answer("category"); ok && !a.Refused {
        fmt.Printf("category: %s (confidence %.2f)\n", a.Choice, a.Confidence)
    }
    if a, ok := resp.Answer("severity"); ok && !a.Refused {
        fmt.Printf("severity score: %.2f\n", a.Score)
    }
}
```

`resp.Answers` holds one answer per question, in request order, and
`resp.Answer(name)` finds one by name. `resp.Usage` reports the tokens used,
which are also added to the client's cumulative `Usage()`.

## Question kinds

| Kind | Options | Fields of `DecisionAnswer` that are set |
|---|---|---|
| `DecisionYesNo` | None | `Probability`: the probability that the answer is yes. |
| `DecisionChoice` | 2 to 255 | `Choice`: the `Value` of the selected option; `Confidence`; `Probabilities`. |
| `DecisionScore` | 2 or more levels, lowest first | `Score`: the probability-weighted level index; `Confidence`; `Probabilities`. |

`Name` and `Kind` are always set, and are copied from the question.

## Option ordering

`Probabilities` follows the order in which the request listed the options,
and each entry carries the option's `Value`. The levels of a score question
are listed lowest first, and `Score` is a zero-based weighted index across
them. With three levels, a `Score` of 0 means the lowest level is certain, a
`Score` of 2 means the highest is certain, and a value of 1.4 lies between
the middle and highest levels.

## Refusals

OpenAI may decline to answer an individual question. The answer then has
`Refused` set to true, and every field other than `Name`, `Kind` and `Refused` is zero.
Check `Refused` before using an answer. TypeSafe does not refuse questions.

## Validation and limits

`DecisionRequest.Validate` runs before any network call, and the following
rules apply to every provider:

- the request needs `Input` or `Images`, and at least one question.
- each image sets exactly one of `URL` or `Data`, and an image with `Data`
  needs a `MediaType`.
- each question needs a unique `Name`, a `Kind` and `Instructions`.
- a yes/no question takes no options.
- a choice question takes 2 to 255 options (`llm.MaxDecisionChoices`).
- a score question takes at least 2 levels.
- option values are required and unique within a question.

TypeSafe additionally accepts at most 10 levels in a score question, and
rejects more with `ErrInvalidRequest`.

## Selecting a model

For TypeSafe, the model is `DecisionRequest.Model` if set, then
`Options.Model`, then `jev-latest`.

For OpenAI, the model is `DecisionRequest.Model` if set, and otherwise
`gpt-6-luna`. `Options.Model` is deliberately ignored, because on an OpenAI
client it names a chat or embedding model that cannot make decisions.

Use `client.ListModelsWithMetadata(ctx, llm.WithCapabilities(llm.ModelCapabilityDecisions))`
to discover decision-capable models.

## Images

OpenAI accepts images in `DecisionRequest.Images`, sent either by URL or as
inline data. TypeSafe accepts text only, and returns `ErrNotSupported` when a
request includes images.

## Errors

| Error | Cause |
|---|---|
| `ErrInvalidRequest` | The request failed validation (before any network call), exceeded a provider limit, or was rejected by the provider with HTTP 422. |
| `ErrNotSupported` | The provider does not implement `Decide`, or the request includes images and the provider accepts text only. |
| `ErrProviderError` | The provider returned a malformed response, such as a missing answer, an answer of the wrong kind, a choice that was not one of the request's options, a missing or out-of-range yes/no probability, a missing score or one outside the range of level indices, or a missing or out-of-range confidence or option probability on a choice or score answer. |
| `ErrAuthentication`, `ErrRateLimit` | As for every other method. |

Token usage is counted even when the answers turn out to be malformed, since
the provider has already charged for them.

`Decide` is not exposed through the HTTP proxy in `llm/proxy`.
