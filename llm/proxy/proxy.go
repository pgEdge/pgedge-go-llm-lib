//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package proxy

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/pgEdge/pgedge-go-llm-lib/llm"
)

// Config configures a Proxy.
//
// Providers maps provider names (matching values registered with
// llm.RegisterProvider — typically "anthropic", "openai", "gemini",
// "ollama") to the Options used to construct a fresh client per
// request. A provider with no entry here is unreachable through the
// proxy regardless of whether it is registered globally.
type Config struct {
	DefaultProvider string
	Providers       map[string]llm.Options

	// Hooks are invoked for telemetry / auth side-effects. They run
	// synchronously in the request goroutine; keep them fast.
	OnRequest  func(r *http.Request, info RequestInfo)
	OnResponse func(r *http.Request, info ResponseInfo)
	OnError    func(r *http.Request, info ErrorInfo)

	// Authorize is invoked before request parsing on every endpoint.
	// Return a non-nil error to reject the request (default status 401;
	// implement HTTPStatus() int on the error to override). Authorize
	// MAY mutate the request — e.g. attaching context values for
	// downstream hooks.
	//
	// Authorize fires for every endpoint including GET /v1/providers,
	// /v1/models, and /v1/health.
	Authorize func(*http.Request) error

	// RequestIDHeader is the header name to read for an incoming
	// request ID and write on outgoing responses. Defaults to
	// "X-Request-ID" when empty. Set to "-" to disable request-ID
	// propagation entirely.
	RequestIDHeader string
}

// RequestInfo describes an incoming chat request, supplied to OnRequest.
//
// Request is the fully-resolved llm.ChatRequest the proxy is about to
// hand to the provider, with messages, tools, system prompt,
// tool-choice, response format, and stop sequences populated. It is
// nil only when the request failed to parse before this hook fires
// (in which case OnError fires instead).
//
// Hooks that want to log prompt content, tool definitions, or
// per-request overrides read this field. The pointer is to a value
// owned by the proxy for the lifetime of the request — do not mutate
// it from inside the hook.
type RequestInfo struct {
	Provider  string
	Model     string
	Stream    bool
	RequestID string // Empty if RequestIDHeader was set to "-"
	Request   *llm.ChatRequest
}

// ErrorInfo describes a request that produced an error response,
// supplied to OnError.
type ErrorInfo struct {
	Provider   string // empty if the error occurred before provider resolution
	Model      string // empty if the error occurred before model resolution
	Stream     bool   // true if the error occurred on a streaming endpoint
	StatusCode int
	Err        error
	RequestID  string // Empty if RequestIDHeader was set to "-"
}

// ResponseInfo describes a completed chat response, supplied to OnResponse.
//
// Response is the assembled llm.ChatResponse the proxy returned to
// the client: content blocks, stop reason, and token usage. For
// streaming requests it is built up from the SSE chunks (text deltas
// concatenated, tool-use deltas folded into their start block) using
// the same logic as Stream.Collect, so consumers see one ChatResponse
// shape regardless of whether the underlying call was streaming or
// not. Response is non-nil whenever the upstream call completed,
// even when the stream errored mid-flight (in which case it carries
// the partial response). Read-only — owned by the proxy.
type ResponseInfo struct {
	Provider   string
	Model      string
	Stream     bool
	Usage      llm.TokenUsage
	StatusCode int
	RequestID  string // Empty if RequestIDHeader was set to "-"
	Response   *llm.ChatResponse
}

// AuthError indicates an Authorize hook rejected the request. Embed
// the underlying error and an optional HTTP status (defaults to 401).
type AuthError struct {
	Err    error
	Status int
}

func (e *AuthError) Error() string { return e.Err.Error() }
func (e *AuthError) Unwrap() error { return e.Err }

// HTTPStatus returns the HTTP status code the proxy will respond
// with when this error is returned from an Authorize hook. Defaults
// to 401 when Status is unset.
func (e *AuthError) HTTPStatus() int {
	if e.Status == 0 {
		return http.StatusUnauthorized
	}
	return e.Status
}

type contextKey string

const requestIDKey contextKey = "proxy.request-id"

// RequestIDFromContext returns the request ID attached by the proxy
// for this request, or empty string if request-ID propagation is
// disabled.
func RequestIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey).(string)
	return v
}

// requestIDHeaderName returns the configured header name (defaults
// to X-Request-ID).
func (p *Proxy) requestIDHeaderName() string {
	h := p.cfg.RequestIDHeader
	if h == "" {
		return "X-Request-ID"
	}
	return h
}

// ensureRequestID attaches a request ID to the context (generating
// one if missing) and returns the augmented request and the ID.
// Returns the original request and "" when request-ID propagation
// is disabled (RequestIDHeader == "-").
func (p *Proxy) ensureRequestID(r *http.Request) (*http.Request, string) {
	header := p.cfg.RequestIDHeader
	if header == "-" {
		return r, ""
	}
	if header == "" {
		header = "X-Request-ID"
	}
	id := r.Header.Get(header)
	if id == "" {
		id = randomRequestID()
	}
	ctx := context.WithValue(r.Context(), requestIDKey, id)
	return r.WithContext(ctx), id
}

func randomRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fallback to a less random ID; never panic.
		return fmt.Sprintf("req-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// authorize runs the Authorize hook (if configured). Returns true
// if the request should proceed, false if it was rejected (writeError
// was called with an appropriate status).
func (p *Proxy) authorize(w http.ResponseWriter, r *http.Request) bool {
	if p.cfg.Authorize == nil {
		return true
	}
	if err := p.cfg.Authorize(r); err != nil {
		status := http.StatusUnauthorized
		if hs, ok := err.(interface{ HTTPStatus() int }); ok {
			status = hs.HTTPStatus()
		}
		p.writeError(w, r, ErrorInfo{StatusCode: status, Err: err})
		return false
	}
	return true
}

// Proxy serves HTTP requests against the LLM provider abstraction.
// Create one with New and mount p.Handler() on an http.ServeMux.
type Proxy struct {
	cfg Config
}

func encodeJSON(w io.Writer, v any) error {
	return json.NewEncoder(w).Encode(v)
}

// New creates a Proxy from the given Config. Call Handler on the
// result to get the http.Handler that serves:
//
//	GET  /v1/health                — ping all configured providers
//	GET  /v1/providers             — list configured providers
//	GET  /v1/models                — list model names for a provider
//	POST /v1/chat                  — non-streaming chat completion
//	POST /v1/chat/stream           — streaming chat (SSE)
//	POST /v1/embed                 — text embeddings (single or batch)
//	POST /v1/embed/multimodal      — multimodal embeddings
func New(cfg Config) *Proxy {
	providers := make(map[string]llm.Options, len(cfg.Providers))
	for k, v := range cfg.Providers {
		providers[k] = v
	}
	cfg.Providers = providers
	return &Proxy{cfg: cfg}
}

// Handler returns the http.Handler for this proxy.
//
// Mount it under whatever path prefix you like — typically "/api"
// using http.StripPrefix.
func (p *Proxy) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", p.handleHealth)
	mux.HandleFunc("GET /v1/providers", p.handleProviders)
	mux.HandleFunc("GET /v1/models", p.handleModels)
	mux.HandleFunc("POST /v1/chat", p.handleChat)
	mux.HandleFunc("POST /v1/chat/stream", p.handleChatStream)
	mux.HandleFunc("POST /v1/embed", p.handleEmbed)
	mux.HandleFunc("POST /v1/embed/multimodal", p.handleEmbedMultimodal)
	return mux
}

// writeError writes a JSON error response and invokes OnError if set.
func (p *Proxy) writeError(w http.ResponseWriter, r *http.Request, info ErrorInfo) {
	if p.cfg.OnError != nil {
		p.cfg.OnError(r, info)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(info.StatusCode)
	_ = encodeJSON(w, ErrorResponse{Error: info.Err.Error()})
}
