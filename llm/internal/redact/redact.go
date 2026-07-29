//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package redact strips credentials out of strings that are destined
// for an error message, a log line, or an HTTP response body.
//
// It exists because provider APIs routinely quote part of the
// credential you just sent them back at you when authentication fails.
// OpenAI, for one, characteristically includes a partially masked form
// of the submitted key in the "message" field of a 401 body. Relaying
// that field verbatim into an error, as this library previously did,
// hands a fragment of the operator's real API key to whoever can see
// the error — which, for a service built on top of this library, is
// typically an anonymous HTTP client.
//
// Redaction is unconditional and there is no way to switch it off. The
// unredacted text is never retained anywhere, so no caller can log it
// by accident.
package redact

import (
	"regexp"
	"strings"
)

// Placeholder replaces every redacted run. It is deliberately
// conspicuous so that a redaction is obvious in a log or a bug report.
const Placeholder = "[REDACTED]"

const (
	// minExactLen is the shortest secret we will look for verbatim.
	// Below this the risk of mangling ordinary prose outweighs the
	// benefit, and a credential that short is not a credential.
	minExactLen = 4

	// minFragmentSecretLen is the shortest secret eligible for
	// fragment matching. Real provider keys run to 40 characters and
	// beyond; the floor stops a junk or placeholder key such as
	// "test" from redacting every occurrence of that word.
	minFragmentSecretLen = 12

	// minInteriorFragmentLen is the shortest message fragment we will
	// redact on the strength of it appearing somewhere inside a
	// secret.
	minInteriorFragmentLen = 6

	// minEdgeFragmentLen is the shortest message fragment we will
	// redact on the strength of it being a prefix or suffix of a
	// secret. It is lower than minInteriorFragmentLen because a
	// masked echo characteristically preserves only the first and
	// last few characters, and matching a key's own edge is far
	// stronger evidence than matching somewhere in its middle.
	minEdgeFragmentLen = 4
)

// fragmentChars matches a maximal run of the characters that appear in
// provider API keys. Runs of these are the candidate fragments that
// pass 2 tests against the configured secrets.
var fragmentChars = regexp.MustCompile(`[A-Za-z0-9_-]+`)

// keyShapePatterns match tokens that look like credentials regardless
// of whether we hold the corresponding secret. This catches a key the
// caller never handed us: one read from a provider-side configuration,
// or echoed out of a nested call we did not make.
//
// Each pattern tolerates the masking characters (*, ., and the
// ellipsis) that providers substitute for the middle of a key, so a
// partially starred echo is consumed as a single unit rather than
// leaving unmatched fragments on either side.
var keyShapePatterns = []*regexp.Regexp{
	// OpenAI, including the sk-proj- and sk-ant- (Anthropic) forms. The
	// leading \b stops this matching inside an ordinary compound word
	// or identifier such as "task-1234567890" or "disk-quota-exceeded",
	// where "sk-" would otherwise appear mid-word.
	regexp.MustCompile(`\bsk-[A-Za-z0-9_*.\x{2026}-]{8,}`),
	// Google AI Studio / Gemini.
	regexp.MustCompile(`\bAIza[A-Za-z0-9_*.\x{2026}-]{8,}`),
	// Voyage AI.
	regexp.MustCompile(`\bpa-[A-Za-z0-9_*.\x{2026}-]{20,}`),
	// An HTTP Authorization value quoted into a message.
	regexp.MustCompile(`(?i)\b(bearer|basic)\s+[A-Za-z0-9+/=_*.\x{2026}-]{8,}`),
	// A labelled credential, e.g. `api_key: hunter2` or
	// `Authorization=Bearer xyz`.
	regexp.MustCompile(`(?i)\b(api[_ -]?key|api[_ -]?token|access[_ -]?token|auth[_ -]?token|authorization|secret|token|password|passwd)\b\s*[:=]\s*"?[^\s"',;]{8,}"?`),
}

// bearerPrefix and labelledValue let the two "keep the label, redact
// the value" patterns above put their label back after substitution.
var (
	bearerPrefix  = regexp.MustCompile(`(?i)^(bearer|basic)\s+`)
	labelledValue = regexp.MustCompile(`(?i)^([^\s:=]+(?:[_ -][^\s:=]+)*)(\s*[:=]\s*)`)
)

// collapse joins placeholders that are separated only by the
// characters a provider uses to mask the middle of a key, so that a
// single echoed credential yields a single placeholder rather than one
// per surviving fragment.
var collapse = regexp.MustCompile(`(?:` + regexp.QuoteMeta(Placeholder) + `)(?:[*.\x{2026}]+(?:` + regexp.QuoteMeta(Placeholder) + `))+`)

// Message returns msg with any of the given secrets, any recognisable
// fragment of them, and anything otherwise shaped like a credential
// replaced by Placeholder.
//
// Secrets are the credentials the caller knows it sent, typically the
// configured API key. Passing none is valid and still applies the
// shape-based patterns, which is the right call for a provider that
// authenticates with no key at all, such as a local Ollama.
//
// The result is safe to put in front of an untrusted reader. Message is
// idempotent: redacting an already-redacted string changes nothing.
func Message(msg string, secrets ...string) string {
	if msg == "" {
		return msg
	}

	out := msg

	// Pass 1: exact substring replacement. Cheap, no false
	// positives, and it covers a provider that echoes the key whole.
	for _, s := range secrets {
		if len(s) < minExactLen {
			continue
		}
		out = strings.ReplaceAll(out, s, Placeholder)
	}

	// Pass 2: fragment overlap against the secrets we hold. This is
	// what catches a truncated or masked echo, and it needs no
	// knowledge of how any particular provider formats one.
	out = redactFragments(out, secrets)

	// Pass 3: shape-based patterns, for credentials we were never
	// given.
	for _, re := range keyShapePatterns {
		out = re.ReplaceAllStringFunc(out, redactKeyShape)
	}

	return collapse.ReplaceAllString(out, Placeholder)
}

// redactFragments replaces every run of key characters in msg that
// overlaps one of the secrets far enough to be incriminating.
func redactFragments(msg string, secrets []string) string {
	eligible := make([]string, 0, len(secrets))
	for _, s := range secrets {
		if len(s) >= minFragmentSecretLen {
			eligible = append(eligible, s)
		}
	}
	if len(eligible) == 0 {
		return msg
	}

	return fragmentChars.ReplaceAllStringFunc(msg, func(run string) string {
		// Leave our own placeholder alone; it is made of key
		// characters and would otherwise be re-examined.
		if strings.Contains(Placeholder, run) && len(run) >= len("REDACTED") {
			return run
		}
		for _, s := range eligible {
			if fragmentIncriminates(run, s) {
				return Placeholder
			}
		}
		return run
	})
}

// fragmentIncriminates reports whether run overlaps secret by enough
// to conclude that run is part of an echoed credential.
func fragmentIncriminates(run, secret string) bool {
	switch {
	case len(run) >= minInteriorFragmentLen:
		return strings.Contains(secret, run)
	case len(run) >= minEdgeFragmentLen:
		return strings.HasPrefix(secret, run) || strings.HasSuffix(secret, run)
	default:
		return false
	}
}

// redactKeyShape replaces a shape-matched token, preserving any label
// or scheme that identifies what was redacted. Knowing that an
// Authorization header was involved is useful; the value never is.
func redactKeyShape(match string) string {
	if p := bearerPrefix.FindString(match); p != "" {
		return p + Placeholder
	}
	if m := labelledValue.FindStringSubmatch(match); m != nil {
		return m[1] + m[2] + Placeholder
	}
	return Placeholder
}
