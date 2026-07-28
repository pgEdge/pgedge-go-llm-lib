//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package redact

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// Every credential in this file is synthetic. Nothing here is, or has
// ever been, a working key.
const (
	fakeOpenAIKey    = "sk-proj-T3stK3yNotReal0000000000000000000000000000000000AbCd"
	fakeAnthropicKey = "sk-ant-api03-T3stK3yNotReal000000000000000000000000000000WxYz"
	fakeGeminiKey    = "AIzaT3stK3yNotReal00000000000000000000A"
	fakeVoyageKey    = "pa-T3stK3yNotReal000000000000000000000000Qq77"
)

func TestMessageRedactsExactSecret(t *testing.T) {
	msg := "authentication failed for " + fakeOpenAIKey + " please retry"
	got := Message(msg, fakeOpenAIKey)

	if strings.Contains(got, fakeOpenAIKey) {
		t.Fatalf("secret survived redaction: %q", got)
	}
	if !strings.Contains(got, Placeholder) {
		t.Fatalf("expected a placeholder, got %q", got)
	}
	if !strings.Contains(got, "please retry") {
		t.Fatalf("surrounding text was destroyed: %q", got)
	}
}

func TestMessageRedactsMaskedEcho(t *testing.T) {
	// The shape of a provider echoing a key back with the middle
	// masked: a leading fragment and a trailing fragment survive, and
	// both must go. The exact masking style varies between providers
	// and over time, which is why each variant is covered rather than
	// one canonical form being assumed.
	variants := []string{
		"sk-proj-T3stK3y****************************************AbCd",
		"sk-proj-T3stK3y...AbCd",
		"sk-proj-T3stK3y…AbCd",
		"sk-proj-T3stK3y**AbCd",
	}

	for _, echo := range variants {
		t.Run(echo, func(t *testing.T) {
			msg := "Incorrect API key provided: " + echo +
				". You can find your API key at https://example.invalid/keys."
			got := Message(msg, fakeOpenAIKey)

			for _, frag := range []string{"T3stK3y", "AbCd"} {
				if strings.Contains(got, frag) {
					t.Errorf("fragment %q survived redaction: %q", frag, got)
				}
			}
			if !strings.Contains(got, "Incorrect API key provided:") {
				t.Errorf("leading context lost: %q", got)
			}
			if !strings.Contains(got, "https://example.invalid/keys") {
				t.Errorf("trailing context lost: %q", got)
			}
			if n := strings.Count(got, Placeholder); n != 1 {
				t.Errorf("want the echo collapsed to 1 placeholder, got %d: %q", n, got)
			}
		})
	}
}

func TestMessageRedactsKeyShapesWithoutKnowingTheSecret(t *testing.T) {
	cases := []struct {
		name string
		msg  string
		gone []string
	}{
		{
			name: "openai",
			msg:  "upstream rejected " + fakeOpenAIKey,
			gone: []string{"sk-proj-", "AbCd"},
		},
		{
			name: "anthropic",
			msg:  "invalid x-api-key " + fakeAnthropicKey,
			gone: []string{"sk-ant-", "WxYz"},
		},
		{
			name: "gemini",
			msg:  "API key not valid: " + fakeGeminiKey,
			gone: []string{"AIzaT3st"},
		},
		{
			name: "voyage",
			msg:  "bad credential " + fakeVoyageKey,
			gone: []string{"pa-T3st", "Qq77"},
		},
		{
			name: "bearer header",
			msg:  "rejected Authorization: Bearer abcdef0123456789",
			gone: []string{"abcdef0123456789"},
		},
		{
			name: "labelled value",
			msg:  `provider said api_key=abcdef0123456789 was revoked`,
			gone: []string{"abcdef0123456789"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// No secrets passed: shape patterns alone must carry it.
			got := Message(tc.msg)
			for _, frag := range tc.gone {
				if strings.Contains(got, frag) {
					t.Errorf("fragment %q survived shape redaction: %q", frag, got)
				}
			}
		})
	}
}

func TestMessageKeepsLabelsForDiagnosis(t *testing.T) {
	got := Message("rejected Authorization: Bearer abcdef0123456789")
	if !strings.Contains(strings.ToLower(got), "bearer") {
		t.Errorf("want the scheme retained for diagnosis, got %q", got)
	}

	got = Message("api_key=abcdef0123456789 was revoked")
	if !strings.Contains(got, "api_key=") {
		t.Errorf("want the label retained for diagnosis, got %q", got)
	}
	if !strings.Contains(got, "was revoked") {
		t.Errorf("want the trailing context retained, got %q", got)
	}
}

func TestMessagePreservesInnocentText(t *testing.T) {
	// A message with nothing credential-shaped in it must come
	// through untouched, including model names, request IDs, and
	// hyphenated prose, which are precisely what an operator needs
	// when diagnosing a failure.
	cases := []string{
		"model gpt-4o-mini does not exist or you do not have access to it",
		"rate limit exceeded, request req_abc123def456 rejected",
		"the request was not well-formed: messages must not be empty",
		"context length 200000 exceeded by 1423 tokens",
		"HTTP 503",
		"",
	}

	for _, msg := range cases {
		t.Run(msg, func(t *testing.T) {
			if got := Message(msg, fakeOpenAIKey); got != msg {
				t.Errorf("innocent text was altered:\n want %q\n  got %q", msg, got)
			}
		})
	}
}

func TestMessageIgnoresShortSecretsForFragmentMatching(t *testing.T) {
	// A short junk key must not turn every occurrence of an ordinary
	// word into a placeholder by fragment matching. It is still
	// removed where it appears verbatim, which is the correct
	// trade-off: the word is redacted only when it genuinely is the
	// configured credential.
	got := Message("the latest request failed during testing", "test")
	if strings.Contains(got, "the la") == false {
		t.Fatalf("message mangled: %q", got)
	}
	if !strings.Contains(got, "request failed during") {
		t.Errorf("unrelated words were redacted: %q", got)
	}
}

func TestMessageIsIdempotent(t *testing.T) {
	msg := "Incorrect API key provided: sk-proj-T3stK3y****AbCd. Try again."

	once := Message(msg, fakeOpenAIKey)
	twice := Message(once, fakeOpenAIKey)

	if once != twice {
		t.Errorf("not idempotent:\n once %q\ntwice %q", once, twice)
	}
}

func TestMessageHandlesMultipleSecrets(t *testing.T) {
	msg := "first " + fakeOpenAIKey + " then " + fakeGeminiKey
	got := Message(msg, fakeOpenAIKey, fakeGeminiKey)

	if strings.Contains(got, fakeOpenAIKey) || strings.Contains(got, fakeGeminiKey) {
		t.Fatalf("a secret survived: %q", got)
	}
	if !strings.Contains(got, "first ") || !strings.Contains(got, " then ") {
		t.Errorf("structure lost: %q", got)
	}
}

func TestMessageNoSecretsIsSafe(t *testing.T) {
	msg := "plain provider failure"
	if got := Message(msg); got != msg {
		t.Errorf("want %q unchanged, got %q", msg, got)
	}
	if got := Message(msg, ""); got != msg {
		t.Errorf("empty secret should be skipped, got %q", got)
	}
}

// TestMessageInvariantOverGeneratedEchoes exercises the core invariant
// across a few thousand generated cases rather than the handful a
// table can hold: whatever recognisable part of a secret a provider
// chooses to echo, and wherever in the message it lands, no
// substantial run of that secret comes out the far side.
//
// The seed is fixed, so a counterexample is a permanent, reproducible
// failure rather than an intermittent one. This is deliberately
// stdlib-only: the library carries no third-party dependencies, and a
// property-testing framework is not worth being the first.
func TestMessageInvariantOverGeneratedEchoes(t *testing.T) {
	const keyAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

	rng := rand.New(rand.NewSource(20260728))

	prefixes := []string{"", "sk-", "sk-proj-", "AIza", "pa-"}
	masks := []string{"****", "...", "…", "************", "**"}
	templates := []string{
		"Incorrect API key provided: %s. Check your configuration.",
		"authentication failed (%s)",
		"%s is not authorised for this resource",
		"the credential %s has been revoked, generate a new one",
	}

	for i := 0; i < 3000; i++ {
		// Build a synthetic secret of a realistic length.
		n := 32 + rng.Intn(32)
		var sb strings.Builder
		sb.WriteString(prefixes[rng.Intn(len(prefixes))])
		for j := 0; j < n; j++ {
			sb.WriteByte(keyAlphabet[rng.Intn(len(keyAlphabet))])
		}
		secret := sb.String()

		// Choose what the provider echoes back.
		var echo string
		switch rng.Intn(5) {
		case 0: // the whole key
			echo = secret
		case 1: // masked middle, first 7 and last 4 surviving
			echo = secret[:7] + masks[rng.Intn(len(masks))] + secret[len(secret)-4:]
		case 2: // a leading fragment only
			echo = secret[:minEdgeFragmentLen+rng.Intn(8)]
		case 3: // a trailing fragment only
			echo = secret[len(secret)-(minEdgeFragmentLen+rng.Intn(8)):]
		case 4: // an interior run
			runLen := minInteriorFragmentLen + rng.Intn(4)
			start := 1 + rng.Intn(len(secret)-runLen-1)
			echo = secret[start : start+runLen]
		}

		msg := fmt.Sprintf(templates[rng.Intn(len(templates))], echo)
		got := Message(msg, secret)

		if strings.Contains(got, echo) {
			t.Fatalf("iteration %d: echoed fragment survived\n secret %q\n   echo %q\n    got %q",
				i, secret, echo, got)
		}
		if run := longestSharedRun(got, secret); run >= minInteriorFragmentLen {
			t.Fatalf("iteration %d: %d-character run of the secret survived\n secret %q\n    got %q",
				i, run, secret, got)
		}
	}
}

// longestSharedRun returns the length of the longest contiguous
// substring that msg and secret have in common.
func longestSharedRun(msg, secret string) int {
	best := 0
	for i := range secret {
		for j := i + best + 1; j <= len(secret); j++ {
			if strings.Contains(msg, secret[i:j]) {
				if j-i > best {
					best = j - i
				}
			} else {
				break
			}
		}
	}
	return best
}
