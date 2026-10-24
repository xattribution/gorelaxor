package kern

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func quantQ8(v []float32) []byte {
	out := make([]byte, 0, len(v)/QK*Q8Bytes)
	for b := 0; b < len(v); b += QK {
		var amax float32
		for _, x := range v[b : b+QK] {
			amax = max(amax, float32(math.Abs(float64(x))))
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		out = binary.LittleEndian.AppendUint16(out, F32ToF16(d))
		for _, x := range v[b : b+QK] {
			out = append(out, byte(int8(math.Round(float64(x*id)))))
		}
	}
	return out
}

func TestMatMulTypes(t *testing.T) {
	defer setInt(intKind())
	setInt(intNone) // q8int_test.go covers the int8 path
	r := rand.New(rand.NewSource(1))
	for _, sz := range [][3]int{{1, 70, 64}, {3, 33, 96}, {17, 40, 128}, {64, 50, 256}} {
		m, n, k := sz[0], sz[1], sz[2]
		x := make([]float32, m*k)
		wv := make([]float32, n*k)
		bias := make([]float32, n)
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		for i := range wv {
			wv[i] = r.Float32()*2 - 1
		}
		for i := range bias {
			bias[i] = r.Float32()
		}
		h := make([]uint16, len(wv))
		for i, v := range wv {
			h[i] = F32ToF16(v)
		}
		wf, _ := NewF32(n, k, wv)
		wh, _ := NewF16(n, k, h)
		wq, _ := NewQ8(n, k, quantQ8(wv))
		for _, w := range []*Weight{wf, wh, wq} {
			ref := w.Float32()
			got := make([]float32, m*n)
			Linear(got, x, m, w, bias, 2)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var s float64
					for c := 0; c < k; c++ {
						s += float64(x[i*k+c]) * float64(ref[j*k+c])
					}
					s += float64(bias[j])
					if d := math.Abs(s - float64(got[i*n+j])); d > 1e-3 {
						t.Fatalf("type %d %v: [%d,%d] got %g want %g", w.Type, sz, i, j, got[i*n+j], s)
					}
				}
			}
		}
	}
}

func TestDotQ8AsmMatchesGo(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	const k, ld = 1024, 1024 / QK * Q8Bytes
	x := make([]float32, k)
	w := make([]float32, 4*k)
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	for i := range w {
		w[i] = r.Float32()*2 - 1
	}
	q := quantQ8(w)
	var a, b [4]float32
	dotQ8x4(x, q, ld, k, &a)
	dotQ8x4Go(x, q, ld, k, &b)
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > 1e-3 {
			t.Fatalf("row %d: %g vs %g", i, a[i], b[i])
		}
	}
}
