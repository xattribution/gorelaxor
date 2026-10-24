package whisper

import "math"

// f16tab widens every half-precision value to float32 (256 KB).
var f16tab = func() []float32 {
	t := make([]float32, 1<<16)
	for i := range t {
		t[i] = f16toF32(uint16(i))
	}
	return t
}()

func f16toF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		man &= 0x3ff
		return math.Float32frombits(sign | e<<23 | man<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
}

// f32toF16 rounds a float32 to the nearest half, ties to even.
func f32toF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xff
	man := b & 0x7fffff
	if exp == 0xff {
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	e := exp - 127 + 15
	if e >= 0x1f {
		return sign | 0x7c00
	}
	if e <= 0 {
		if e < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - e)
		h := man >> shift
		rem := man & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(e)<<10 | man>>13
	rem := man & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return sign | uint16(h)
}

// widen converts a row of halves to float32.
func widen(dst []float32, src []uint16) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = f16tab[v]
	}
}
