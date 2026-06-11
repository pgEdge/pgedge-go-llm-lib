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
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestContentBlockText(t *testing.T) {
	b := ContentBlock{Type: BlockText, Text: "hello"}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"text","text":"hello"}`
	if string(data) != want {
		t.Errorf("got %s\nwant %s", data, want)
	}
}

func TestContentBlockToolUse(t *testing.T) {
	b := ContentBlock{
		Type: BlockToolUse,
		ToolUse: &ToolUse{
			ID:    "tu_1",
			Name:  "get_weather",
			Input: json.RawMessage(`{"city":"Reykjavik"}`),
		},
	}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(data, []byte(`"type":"tool_use"`)) {
		t.Errorf("missing type tag: %s", data)
	}
	if !bytes.Contains(data, []byte(`"name":"get_weather"`)) {
		t.Errorf("missing name: %s", data)
	}
}

func TestContentBlockToolResult(t *testing.T) {
	b := ContentBlock{
		Type:      BlockToolResult,
		ToolUseID: "tu_1",
		Text:      "It's 4 degrees and rainy.",
	}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(data, []byte(`"type":"tool_result"`)) || !bytes.Contains(data, []byte(`"tool_use_id":"tu_1"`)) {
		t.Errorf("missing tool_result fields: %s", data)
	}
}

func TestMessageContentRoundtrip(t *testing.T) {
	m := Message{
		Role: RoleUser,
		Content: []ContentBlock{
			{Type: BlockText, Text: "What's the weather?"},
		},
	}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out Message
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Role != RoleUser || len(out.Content) != 1 || out.Content[0].Text != "What's the weather?" {
		t.Errorf("round-trip mismatch: %+v", out)
	}
}

func TestToolUseSerialization(t *testing.T) {
	tu := ToolUse{
		ID:    "tu_123",
		Name:  "get_weather",
		Input: json.RawMessage(`{"city":"London"}`),
	}
	data, err := json.Marshal(tu)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}
	var decoded ToolUse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if decoded.ID != "tu_123" || decoded.Name != "get_weather" {
		t.Errorf("unexpected: %+v", decoded)
	}
}

func TestTokenUsageAdd(t *testing.T) {
	a := TokenUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30}
	b := TokenUsage{PromptTokens: 5, CompletionTokens: 15, TotalTokens: 20}
	a.Add(b)
	if a.PromptTokens != 15 || a.CompletionTokens != 35 || a.TotalTokens != 50 {
		t.Errorf("unexpected: %+v", a)
	}
}

func TestTokenUsageAddWithCache(t *testing.T) {
	a := TokenUsage{
		PromptTokens:             10,
		CompletionTokens:         20,
		TotalTokens:              30,
		CacheCreationInputTokens: 5,
		CacheReadInputTokens:     3,
	}
	b := TokenUsage{
		PromptTokens:             5,
		CompletionTokens:         15,
		TotalTokens:              20,
		CacheCreationInputTokens: 2,
		CacheReadInputTokens:     1,
	}
	a.Add(b)
	if a.CacheCreationInputTokens != 7 || a.CacheReadInputTokens != 4 {
		t.Errorf("unexpected cache tokens: %+v", a)
	}
}

func TestOptionsDefaults(t *testing.T) {
	opts := Options{
		APIKey: "test-key",
		Model:  "test-model",
	}
	resolved := opts.WithDefaults()
	if resolved.Temperature == nil || *resolved.Temperature != 0.7 {
		t.Errorf("expected default temperature 0.7, got %v", resolved.Temperature)
	}
	if resolved.MaxTokens == nil || *resolved.MaxTokens != 4096 {
		t.Errorf("expected default max tokens 4096, got %v", resolved.MaxTokens)
	}
}

func TestOptionsNoOverrideExplicit(t *testing.T) {
	temp := 0.3
	maxTok := 1024
	opts := Options{
		APIKey:      "test-key",
		Model:       "test-model",
		Temperature: &temp,
		MaxTokens:   &maxTok,
	}
	resolved := opts.WithDefaults()
	if resolved.Temperature == nil || *resolved.Temperature != 0.3 {
		t.Errorf("expected 0.3, got %v", resolved.Temperature)
	}
	if resolved.MaxTokens == nil || *resolved.MaxTokens != 1024 {
		t.Errorf("expected 1024, got %v", resolved.MaxTokens)
	}
}

func TestOptionsTemperatureZeroPreserved(t *testing.T) {
	zero := 0.0
	opts := Options{Temperature: &zero}.WithDefaults()
	if opts.Temperature == nil {
		t.Fatal("WithDefaults nilled an explicitly-set zero temperature")
	}
	if *opts.Temperature != 0.0 {
		t.Errorf("Temperature = %v, want 0", *opts.Temperature)
	}
}

func TestOptionsTemperatureUnsetGetsDefault(t *testing.T) {
	opts := Options{}.WithDefaults()
	if opts.Temperature == nil {
		t.Fatal("default Temperature should be non-nil")
	}
	if *opts.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", *opts.Temperature)
	}
}

func TestOptionsMaxTokensZeroPreserved(t *testing.T) {
	zero := 0
	opts := Options{MaxTokens: &zero}.WithDefaults()
	if opts.MaxTokens == nil {
		t.Fatal("WithDefaults nilled an explicitly-set zero MaxTokens")
	}
	if *opts.MaxTokens != 0 {
		t.Errorf("MaxTokens = %v, want 0", *opts.MaxTokens)
	}
}

func TestOptionsMaxTokensUnsetGetsDefault(t *testing.T) {
	opts := Options{}.WithDefaults()
	if opts.MaxTokens == nil || *opts.MaxTokens != 4096 {
		t.Errorf("MaxTokens = %v, want 4096", opts.MaxTokens)
	}
}

func TestRetryConfigDefaults(t *testing.T) {
	o := Options{}.WithDefaults()
	if o.Retry.MaxRetries != 5 {
		t.Errorf("MaxRetries = %d, want 5", o.Retry.MaxRetries)
	}
	if o.Retry.InitialBackoff != 2*time.Second {
		t.Errorf("InitialBackoff = %v, want 2s", o.Retry.InitialBackoff)
	}
	if o.Retry.MaxBackoff != 60*time.Second {
		t.Errorf("MaxBackoff = %v, want 60s", o.Retry.MaxBackoff)
	}
	if o.Retry.Disabled {
		t.Errorf("Disabled = true, want false")
	}
}

func TestRetryConfigDisabledRespected(t *testing.T) {
	o := Options{Retry: RetryConfig{Disabled: true}}.WithDefaults()
	if !o.Retry.Disabled {
		t.Errorf("Disabled flag was lost by WithDefaults")
	}
	if o.Retry.MaxRetries != 0 {
		t.Errorf("MaxRetries = %d on disabled config, want 0", o.Retry.MaxRetries)
	}
}

func TestRetryConfigUserValuesPreserved(t *testing.T) {
	custom := RetryConfig{
		MaxRetries:     3,
		InitialBackoff: 500 * time.Millisecond,
		MaxBackoff:     5 * time.Second,
	}
	o := Options{Retry: custom}.WithDefaults()
	if o.Retry.MaxRetries != 3 || o.Retry.InitialBackoff != 500*time.Millisecond || o.Retry.MaxBackoff != 5*time.Second {
		t.Errorf("user values overridden: %+v", o.Retry)
	}
}

func TestRoleConstants(t *testing.T) {
	cases := []struct {
		got  Role
		want string
	}{
		{RoleUser, "user"},
		{RoleAssistant, "assistant"},
		{RoleSystem, "system"},
		{RoleTool, "tool"},
	}
	for _, c := range cases {
		if string(c.got) != c.want {
			t.Errorf("Role %q != %q", c.got, c.want)
		}
	}
}

func TestStopReasonConstants(t *testing.T) {
	cases := []struct {
		got  StopReason
		want string
	}{
		{StopReasonEndTurn, "end_turn"},
		{StopReasonMaxTokens, "max_tokens"},
		{StopReasonStopSequence, "stop_sequence"},
		{StopReasonToolUse, "tool_use"},
		{StopReasonContentFilter, "content_filter"},
		{StopReasonError, "error"},
	}
	for _, c := range cases {
		if string(c.got) != c.want {
			t.Errorf("StopReason %q != %q", c.got, c.want)
		}
	}
}

func TestContentBlockTypeConstants(t *testing.T) {
	cases := []struct {
		got  ContentBlockType
		want string
	}{
		{BlockText, "text"},
		{BlockImage, "image"},
		{BlockDocument, "document"},
		{BlockToolUse, "tool_use"},
		{BlockToolResult, "tool_result"},
	}
	for _, c := range cases {
		if string(c.got) != c.want {
			t.Errorf("ContentBlockType %q != %q", c.got, c.want)
		}
	}
}

func TestCacheControlTypeConstants(t *testing.T) {
	if string(CacheControlEphemeral) != "ephemeral" {
		t.Errorf("CacheControlEphemeral = %q, want \"ephemeral\"", CacheControlEphemeral)
	}
}

func TestUserText(t *testing.T) {
	m := UserText("hi")
	if m.Role != RoleUser || len(m.Content) != 1 || m.Content[0].Text != "hi" {
		t.Errorf("got %+v", m)
	}
	if m.Content[0].Type != BlockText {
		t.Errorf("type = %q, want %q", m.Content[0].Type, BlockText)
	}
}

func TestAssistantText(t *testing.T) {
	m := AssistantText("ok")
	if m.Role != RoleAssistant || m.Content[0].Text != "ok" {
		t.Errorf("got %+v", m)
	}
}

func TestSystemText(t *testing.T) {
	m := SystemText("be helpful")
	if m.Role != RoleSystem || m.Content[0].Text != "be helpful" {
		t.Errorf("got %+v", m)
	}
}

func TestUserBlocks(t *testing.T) {
	m := UserBlocks(TextBlock("hello"), ImageURLBlock("https://example.com/x.png"))
	if m.Role != RoleUser || len(m.Content) != 2 {
		t.Fatalf("got %+v", m)
	}
	if m.Content[0].Type != BlockText || m.Content[0].Text != "hello" {
		t.Errorf("block 0: %+v", m.Content[0])
	}
	if m.Content[1].Type != BlockImage || m.Content[1].Image.URL != "https://example.com/x.png" {
		t.Errorf("block 1: %+v", m.Content[1])
	}
}

func TestAssistantBlocks(t *testing.T) {
	m := AssistantBlocks(TextBlock("here you go"))
	if m.Role != RoleAssistant {
		t.Errorf("role = %q", m.Role)
	}
}

func TestToolResultMessage(t *testing.T) {
	m := ToolResultMessage("tu_1", "Done.", false)
	if m.Role != RoleTool || len(m.Content) != 1 || m.Content[0].Type != BlockToolResult {
		t.Errorf("got %+v", m)
	}
	if m.Content[0].ToolUseID != "tu_1" || m.Content[0].Text != "Done." || m.Content[0].IsError {
		t.Errorf("got %+v", m.Content[0])
	}
}

func TestToolResultMessageWithError(t *testing.T) {
	m := ToolResultMessage("tu_1", "boom", true)
	if !m.Content[0].IsError {
		t.Errorf("IsError = false, want true")
	}
}

func TestImageBlockBase64(t *testing.T) {
	data := []byte("fake-png-bytes")
	b := ImageBlock(data, "image/png")
	if b.Type != BlockImage || b.Image == nil {
		t.Fatalf("got %+v", b)
	}
	if string(b.Image.Data) != "fake-png-bytes" || b.Image.MediaType != "image/png" {
		t.Errorf("got %+v", b.Image)
	}
	if b.Image.URL != "" {
		t.Errorf("URL = %q, want empty", b.Image.URL)
	}
}

func TestDocumentBlock(t *testing.T) {
	data := []byte("%PDF-1.4 fake")
	b := DocumentBlock(data, "application/pdf", "itinerary.pdf")
	if b.Type != BlockDocument || b.Document == nil {
		t.Fatalf("got %+v", b)
	}
	if string(b.Document.Data) != "%PDF-1.4 fake" {
		t.Errorf("data = %q", b.Document.Data)
	}
	if b.Document.MediaType != "application/pdf" {
		t.Errorf("media_type = %q", b.Document.MediaType)
	}
	if b.Document.Filename != "itinerary.pdf" {
		t.Errorf("filename = %q", b.Document.Filename)
	}
	if b.Document.URL != "" {
		t.Errorf("URL = %q, want empty", b.Document.URL)
	}
}

func TestDocumentURLBlock(t *testing.T) {
	b := DocumentURLBlock("https://example.com/a.pdf", "application/pdf", "a.pdf")
	if b.Type != BlockDocument || b.Document == nil {
		t.Fatalf("got %+v", b)
	}
	if b.Document.URL != "https://example.com/a.pdf" {
		t.Errorf("url = %q", b.Document.URL)
	}
	if b.Document.MediaType != "application/pdf" || b.Document.Filename != "a.pdf" {
		t.Errorf("got %+v", b.Document)
	}
	if len(b.Document.Data) != 0 {
		t.Errorf("Data should be empty for URL form, got %v", b.Document.Data)
	}
}

func TestContentBlockDocumentRoundtrip(t *testing.T) {
	b := ContentBlock{
		Type: BlockDocument,
		Document: &DocumentContent{
			Data:      []byte{0x25, 0x50, 0x44, 0x46},
			MediaType: "application/pdf",
			Filename:  "x.pdf",
		},
	}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(data, []byte(`"type":"document"`)) {
		t.Errorf("missing type tag: %s", data)
	}
	var out ContentBlock
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Document == nil || out.Document.MediaType != "application/pdf" || out.Document.Filename != "x.pdf" {
		t.Errorf("round-trip mismatch: %+v", out.Document)
	}
}

func TestToolResultBlock(t *testing.T) {
	b := ToolResultBlock("tu_x", "result", false)
	if b.Type != BlockToolResult || b.ToolUseID != "tu_x" || b.Text != "result" {
		t.Errorf("got %+v", b)
	}
}

type fakeExt struct{ providerName string }

func (f fakeExt) ProviderName() string { return f.providerName }

func TestProviderExtensionInterface(t *testing.T) {
	ext := fakeExt{providerName: "openai"}
	var pe ProviderExtension = ext
	if pe.ProviderName() != "openai" {
		t.Errorf("ProviderName() = %q", pe.ProviderName())
	}
}

func TestFindExtensionFinds(t *testing.T) {
	req := ChatRequest{
		Extensions: []ProviderExtension{
			fakeExt{providerName: "anthropic"},
			fakeExt{providerName: "openai"},
		},
	}
	got := FindExtension[fakeExt](req, "openai")
	if got == nil {
		t.Fatal("FindExtension returned nil for openai")
	}
	if got.providerName != "openai" {
		t.Errorf("got = %+v", got)
	}
}

func TestFindExtensionMissing(t *testing.T) {
	req := ChatRequest{Extensions: []ProviderExtension{fakeExt{providerName: "openai"}}}
	got := FindExtension[fakeExt](req, "anthropic")
	if got != nil {
		t.Errorf("expected nil, got %+v", got)
	}
}

func TestFloatHelper(t *testing.T) {
	p := Float(0.42)
	if p == nil {
		t.Fatal("Float returned nil")
	}
	if *p != 0.42 {
		t.Errorf("*Float(0.42) = %v, want 0.42", *p)
	}
	// A second call must return a distinct pointer so callers can
	// share the helper without aliasing.
	q := Float(0.42)
	if p == q {
		t.Errorf("Float reused pointer across calls")
	}
}

func TestIntHelper(t *testing.T) {
	p := Int(7)
	if p == nil {
		t.Fatal("Int returned nil")
	}
	if *p != 7 {
		t.Errorf("*Int(7) = %v, want 7", *p)
	}
	q := Int(7)
	if p == q {
		t.Errorf("Int reused pointer across calls")
	}
}

func TestTokenUsageCacheSavingsPercent(t *testing.T) {
	cases := []struct {
		name string
		u    TokenUsage
		want float64
	}{
		{"zero", TokenUsage{}, 0},
		{"no cache", TokenUsage{PromptTokens: 100}, 0},
		{"half cached", TokenUsage{PromptTokens: 50, CacheReadInputTokens: 50}, 50},
		{"with creation", TokenUsage{PromptTokens: 0, CacheReadInputTokens: 90, CacheCreationInputTokens: 10}, 90},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.u.CacheSavingsPercent(); got != c.want {
				t.Fatalf("CacheSavingsPercent() = %v, want %v", got, c.want)
			}
		})
	}
}

func TestToolChoiceJSONTags(t *testing.T) {
	tc := ToolChoice{Mode: ToolChoiceSpecific, Name: "get_weather"}
	b, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	want := `{"mode":"specific","name":"get_weather"}`
	if got != want {
		t.Fatalf("marshal mismatch:\n got %s\nwant %s", got, want)
	}

	var rt ToolChoice
	if err := json.Unmarshal([]byte(`{"mode":"auto"}`), &rt); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if rt.Mode != ToolChoiceAuto {
		t.Fatalf("mode = %q, want %q", rt.Mode, ToolChoiceAuto)
	}
	if rt.Name != "" {
		t.Fatalf("name = %q, want empty", rt.Name)
	}
}

func TestModelInfoDimensionsJSON(t *testing.T) {
	b, _ := json.Marshal(ModelInfo{ID: "m", Dimensions: 1536})
	if !strings.Contains(string(b), `"dimensions":1536`) {
		t.Fatalf("missing dimensions tag: %s", b)
	}
	b2, _ := json.Marshal(ModelInfo{ID: "m"})
	if strings.Contains(string(b2), "dimensions") {
		t.Fatalf("zero dimensions should be omitted: %s", b2)
	}
}
