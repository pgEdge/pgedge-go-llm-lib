/*-------------------------------------------------------------------------
 *
 * pgEdge Go LLM Library
 *
 * Copyright (c) 2025 - 2026, pgEdge, Inc.
 * This software is released under The PostgreSQL License
 *
 *-------------------------------------------------------------------------
 */

package vec

import (
	"math"
	"testing"
)

func TestFloat64ToFloat32(t *testing.T) {
	got := Float64ToFloat32([]float64{0, 1.5, -2.25})
	want := []float32{0, 1.5, -2.25}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if Float64ToFloat32(nil) != nil {
		t.Fatalf("nil input should return nil")
	}
}

func TestNormalize(t *testing.T) {
	got := Normalize([]float32{3, 4}) // L2 norm 5
	if math.Abs(float64(got[0])-0.6) > 1e-6 || math.Abs(float64(got[1])-0.8) > 1e-6 {
		t.Fatalf("normalized = %v, want [0.6 0.8]", got)
	}
	z := Normalize([]float32{0, 0})
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("zero-norm normalize = %v, want [0 0]", z)
	}
	if Normalize(nil) != nil {
		t.Fatalf("nil input should return nil")
	}
}

func TestResize(t *testing.T) {
	if got := Resize([]float32{1, 2, 3}, 2); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("truncate = %v, want [1 2]", got)
	}
	got := Resize([]float32{1, 2}, 4)
	if len(got) != 4 || got[2] != 0 || got[3] != 0 {
		t.Fatalf("pad = %v, want [1 2 0 0]", got)
	}
	if got := Resize([]float32{1, 2}, 2); len(got) != 2 {
		t.Fatalf("same length = %v", got)
	}
	nilOut := Resize(nil, 3)
	if len(nilOut) != 3 || nilOut[0] != 0 || nilOut[1] != 0 || nilOut[2] != 0 {
		t.Fatalf("nil input = %v, want [0 0 0]", nilOut)
	}
}

func TestFloat32ToFloat16(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3C00},
		{-2, 0xC000},
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
	}
	for _, c := range cases {
		if got := Float32ToFloat16([]float32{c.in})[0]; got != c.want {
			t.Fatalf("Float32ToFloat16(%v) = 0x%04X, want 0x%04X", c.in, got, c.want)
		}
	}
	nan := Float32ToFloat16([]float32{float32(math.NaN())})[0]
	if nan&0x7C00 != 0x7C00 || nan&0x03FF == 0 {
		t.Fatalf("NaN encoding = 0x%04X, expected NaN pattern", nan)
	}
	// 1e-8 is too small for float16 (exp < -10): should encode as signed zero.
	sub := Float32ToFloat16([]float32{1e-8})[0]
	if sub&0x7C00 != 0 {
		t.Fatalf("tiny value = 0x%04X, expected subnormal/zero", sub)
	}
	// nil input must return nil.
	if Float32ToFloat16(nil) != nil {
		t.Fatalf("nil input should return nil")
	}
	// 70000.0 overflows float16 range (exp >= 31): encodes as +Inf.
	if got := Float32ToFloat16([]float32{70000.0})[0]; got != 0x7C00 {
		t.Fatalf("overflow 70000 = 0x%04X, want 0x7C00 (+Inf)", got)
	}
	// -70000.0 overflows to -Inf.
	if got := Float32ToFloat16([]float32{-70000.0})[0]; got != 0xFC00 {
		t.Fatalf("overflow -70000 = 0x%04X, want 0xFC00 (-Inf)", got)
	}
	// 1e-5 is a float32 whose float16 representation is a subnormal
	// (exp is -2, within the [-10,0] range); verify the sign bit is zero
	// and the exponent field is zero (subnormal encoding).
	subnormal := Float32ToFloat16([]float32{1e-5})[0]
	if subnormal&0x7C00 != 0 {
		t.Fatalf("subnormal 1e-5 = 0x%04X, expected zero exponent", subnormal)
	}
	if subnormal == 0 {
		t.Fatalf("subnormal 1e-5 = 0, expected non-zero mantissa bits")
	}
	// -1e-5 is the negative counterpart; sign bit must be set.
	negSubnormal := Float32ToFloat16([]float32{-1e-5})[0]
	if negSubnormal&0x8000 == 0 {
		t.Fatalf("negative subnormal -1e-5 = 0x%04X, expected sign bit set", negSubnormal)
	}
	if negSubnormal&0x7C00 != 0 {
		t.Fatalf("negative subnormal -1e-5 = 0x%04X, expected zero exponent", negSubnormal)
	}
	// 1.0015 is a normal float32 (bits 0x3F803127) where the round-bit (bit 12
	// of the 23-bit mantissa) is set and the sticky bits (0x127) are non-zero,
	// so RNE rounds up.  The verified exact result is 0x3C02.
	if got := Float32ToFloat16([]float32{1.0015})[0]; got != 0x3C02 {
		t.Fatalf("round-up 1.0015 = 0x%04X, want 0x3C02", got)
	}
}

func TestFloat32ToFloat16RoundToEven(t *testing.T) {
	// 0x3F801000 is exactly halfway between f16 0x3C00 and 0x3C01;
	// round-to-nearest-even must choose the even value 0x3C00.
	tie := math.Float32frombits(0x3F801000)
	if got := Float32ToFloat16([]float32{tie})[0]; got != 0x3C00 {
		t.Fatalf("RNE tie = 0x%04X, want 0x3C00", got)
	}
	// A value just above the tie must round up to 0x3C01.
	above := math.Float32frombits(0x3F801001)
	if got := Float32ToFloat16([]float32{above})[0]; got != 0x3C01 {
		t.Fatalf("above tie = 0x%04X, want 0x3C01", got)
	}
}
