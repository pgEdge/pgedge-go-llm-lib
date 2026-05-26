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
	"reflect"
	"testing"
)

func TestNewClientUnknownProvider(t *testing.T) {
	_, err := NewClient("nonexistent", Options{})
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
	if !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
}

func TestNewClientEmptyProvider(t *testing.T) {
	_, err := NewClient("", Options{})
	if err == nil {
		t.Fatal("expected error for empty provider")
	}
}

func TestRegisteredProviders(t *testing.T) {
	// Register a sentinel and verify it appears in the list.
	RegisterProvider("test-sentinel", func(o Options) (Client, error) {
		return nil, ErrInvalidRequest
	})
	got := RegisteredProviders()
	found := false
	for _, n := range got {
		if n == "test-sentinel" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("test-sentinel not in RegisteredProviders(): %v", got)
	}
	// Result should be sorted.
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Errorf("RegisteredProviders() not sorted: %v", got)
			break
		}
	}
}

func TestWithCapabilitiesAccumulates(t *testing.T) {
	cfg := ListModelsConfig{}
	WithCapabilities(ModelCapabilityChat)(&cfg)
	WithCapabilities(ModelCapabilityReranking, ModelCapabilityEmbeddings)(&cfg)
	want := []ModelCapability{
		ModelCapabilityChat,
		ModelCapabilityReranking,
		ModelCapabilityEmbeddings,
	}
	if !reflect.DeepEqual(cfg.Capabilities, want) {
		t.Fatalf("got %v, want %v", cfg.Capabilities, want)
	}
}

func TestFilterModelInfosNoOptionsReturnsAll(t *testing.T) {
	infos := []ModelInfo{
		{ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat}},
		{ID: "b", Capabilities: []ModelCapability{ModelCapabilityEmbeddings}},
	}
	got := FilterModelInfos(infos, ListModelsConfig{})
	if !reflect.DeepEqual(got, infos) {
		t.Fatalf("expected all infos returned unchanged, got %v", got)
	}
}

func TestFilterModelInfosCapabilityAND(t *testing.T) {
	infos := []ModelInfo{
		{ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat, ModelCapabilityTools}},
		{ID: "b", Capabilities: []ModelCapability{ModelCapabilityChat}},
		{ID: "c", Capabilities: []ModelCapability{ModelCapabilityReranking}},
	}
	cfg := ListModelsConfig{Capabilities: []ModelCapability{
		ModelCapabilityChat, ModelCapabilityTools,
	}}
	got := FilterModelInfos(infos, cfg)
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("expected only 'a', got %v", got)
	}
}

func TestFilterModelInfosUnknownCapabilityIsEmpty(t *testing.T) {
	infos := []ModelInfo{
		{ID: "a", Capabilities: []ModelCapability{ModelCapabilityChat}},
	}
	cfg := ListModelsConfig{Capabilities: []ModelCapability{"never-heard-of-it"}}
	got := FilterModelInfos(infos, cfg)
	if len(got) != 0 {
		t.Fatalf("expected empty result, got %v", got)
	}
}
