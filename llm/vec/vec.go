/*-------------------------------------------------------------------------
 *
 * pgEdge Go LLM Library
 *
 * Copyright (c) 2025 - 2026, pgEdge, Inc.
 * This software is released under The PostgreSQL License
 *
 *-------------------------------------------------------------------------
 */

// Package vec provides pure helpers for post-processing embedding vectors:
// type conversion, L2 normalisation, dimension resizing, and IEEE-754
// half-precision (float16) encoding for pgvector halfvec storage. All
// functions are dependency-free and side-effect free.
package vec

import "math"

// Float64ToFloat32 converts a float64 slice to float32. Returns nil for nil
// input.
func Float64ToFloat32(v []float64) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// Normalize returns the L2-normalised copy of v. A zero-norm vector is
// returned unchanged (all zeros), never NaN. Returns nil for nil input.
func Normalize(v []float32) []float32 {
	if v == nil {
		return nil
	}
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if sumSq == 0 {
		return out
	}
	norm := math.Sqrt(sumSq)
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

// Resize returns a copy of v with exactly n elements: truncated if v is
// longer, zero-padded if shorter.
func Resize(v []float32, n int) []float32 {
	out := make([]float32, n)
	copy(out, v)
	return out
}

// Float32ToFloat16 encodes each float32 as the 16-bit IEEE-754 binary16 bit
// pattern used by pgvector halfvec. Handles zero, normals, subnormals,
// infinities, NaN, and round-to-nearest-even for the mantissa.
func Float32ToFloat16(v []float32) []uint16 {
	if v == nil {
		return nil
	}
	out := make([]uint16, len(v))
	for i, f := range v {
		out[i] = float32ToFloat16Bits(f)
	}
	return out
}

func float32ToFloat16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	// Safe: (b>>16)&0x8000 is either 0 or 32768, both fit in uint16.
	sign := uint16((b >> 16) & 0x8000) //nolint:gosec
	// Safe: (b>>23)&0xFF is 0..255, which fits in int32 without overflow.
	exp := int32((b>>23)&0xFF) - 127 + 15 //nolint:gosec // rebias 127 -> 15
	mant := b & 0x7FFFFF

	switch {
	case (b>>23)&0xFF == 0xFF: // Inf or NaN
		if mant == 0 {
			return sign | 0x7C00 // Inf
		}
		// Safe: mant>>13 is at most 10 bits, fits in uint16.
		return sign | 0x7C00 | uint16(mant>>13) | 0x0001 //nolint:gosec // NaN (keep non-zero mantissa)
	case exp >= 0x1F: // overflow -> Inf
		return sign | 0x7C00
	case exp <= 0: // subnormal or zero
		if exp < -10 {
			return sign // too small -> signed zero
		}
		mant |= 0x800000 // restore implicit leading 1
		// Safe: exp is in [-10,0] here, so 14-exp is [14,24], always positive.
		shift := uint32(14 - exp) //nolint:gosec
		// Safe: mant is 24 bits; shift is [14,24], so result fits in uint16.
		half := uint16(mant >> shift) //nolint:gosec
		if mant&(1<<(shift-1)) != 0 {
			half++
		}
		return sign | half
	default: // normal
		// Safe: exp is [1,30] here; exp<<10 is at most 30720, fits in uint16.
		half := sign | uint16(exp<<10) | uint16(mant>>13) //nolint:gosec
		if mant&0x1000 != 0 {
			half++
		}
		return half
	}
}
