package whisper

import "math"

// expf is a fast float32 e^x, accurate to about 2e-7 relative, for the
// softmax and GELU, where math.Exp would cost more than the products.
func expf(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 88 {
		return float32(math.Inf(1))
	}
	// x = n·ln2 + r, |r| ≤ ln2/2
	t := x*1.4426950408889634 + 0.5
	ni := int32(t)
	if float32(ni) > t {
		ni--
	}
	n := float32(ni)
	r := x - n*0.693145751953125 - n*1.428606765330187e-06
	// e^r by a degree-6 polynomial
	p := float32(1.0/720)*r + 1.0/120
	p = p*r + 1.0/24
	p = p*r + 1.0/6
	p = p*r + 0.5
	p = p*r + 1
	p = p*r + 1
	if ni < -126 {
		return p * math.Float32frombits(uint32(ni+127+64)<<23) * 5.421010862427522e-20 // 2^-64
	}
	return p * math.Float32frombits(uint32(ni+127)<<23)
}

// gelu applies the GELU activation in place (tanh form, as whisper.cpp).
func gelu(x []float32) {
	const c = 0.7978845608028654 // √(2/π)
	for i, v := range x {
		u := c * v * (1 + 0.044715*v*v)
		// tanh(u) = 1 - 2/(e^{2u}+1)
		var t float32
		switch {
		case u > 9:
			t = 1
		case u < -9:
			t = -1
		default:
			t = 1 - 2/(expf(2*u)+1)
		}
		x[i] = 0.5 * v * (1 + t)
	}
}

// layerNorm writes (x - mean)/sd · w + b for each row of length n.
func layerNorm(dst, src []float32, rows, n int, w, b []float32) {
	const eps = 1e-5
	for r := 0; r < rows; r++ {
		x := src[r*n : (r+1)*n]
		y := dst[r*n : (r+1)*n]
		var sum float64
		for _, v := range x {
			sum += float64(v)
		}
		mean := float32(sum / float64(n))
		var vs float64
		for _, v := range x {
			d := float64(v - mean)
			vs += d * d
		}
		inv := float32(1 / math.Sqrt(vs/float64(n)+eps))
		for i, v := range x {
			y[i] = (v-mean)*inv*w[i] + b[i]
		}
	}
}

// softmax normalizes x in place.
func softmax(x []float32) {
	mx := float32(math.Inf(-1))
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	s := expSub(x, mx)
	inv := 1 / s
	for i := range x {
		x[i] *= inv
	}
}

func expSubGo(x []float32, mx float32) float32 {
	var s float32
	for i, v := range x {
		e := expf(v - mx)
		x[i] = e
		s += e
	}
	return s
}

func addTo(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

func scale(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}
