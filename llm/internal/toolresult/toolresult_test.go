//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

package toolresult

import (
	"encoding/json"
	"testing"
)

func TestContent(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		isError bool
		want    string
	}{
		{"success unchanged", "42", false, "42"},
		{"empty success unchanged", "", false, ""},
		{"success JSON unchanged", `{"temp":12}`, false, `{"temp":12}`},
		{"error wrapped", "file not found", true, `{"error":"file not found"}`},
		{"empty error placeholder", "", true, `{"error":"` + UnspecifiedError + `"}`},
		{"error quoted and not HTML-escaped", `bad "x" < y & z`, true, `{"error":"bad \"x\" < y & z"}`},
		{"JSON error text kept as a string", `{"code":5}`, true, `{"error":"{\"code\":5}"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Content(tt.text, tt.isError)
			if got != tt.want {
				t.Fatalf("Content(%q, %v) = %q, want %q", tt.text, tt.isError, got, tt.want)
			}
			if tt.isError {
				var v map[string]string
				if err := json.Unmarshal([]byte(got), &v); err != nil {
					t.Fatalf("not valid JSON: %v", err)
				}
			}
		})
	}
}
