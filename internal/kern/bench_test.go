package kern

import (
	"fmt"
	"math/rand"
	"testing"
)

// Benchmarks for the shapes the 0.6B FastConformer models run: 1024
// wide, 4096 in the feed-forward, a few frames (streaming) or a few
// hundred (offline). Report GFLOP/s with -bench . -benchtime 20x.

func benchWeight(b *testing.B, n, k int, typ Type) *Weight {
	r := rand.New(rand.NewSource(1))
	v := make([]float32, n*k)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	switch typ {
	case Q8_0:
		w, _ := NewQ8(n, k, quantQ8(v))
		return w
	case F16:
		h := make([]uint16, len(v))
		for i, x := range v {
			h[i] = F32ToF16(x)
		}
		w, _ := NewF16(n, k, h)
		return w
	}
	w, _ := NewF32(n, k, v)
	return w
}

func BenchmarkLinear(b *testing.B) {
	for _, typ := range []Type{Q8_0, F32} {
		for _, s := range [][3]int{{1, 1024, 1024}, {8, 1024, 1024}, {16, 4096, 1024}, {300, 1024, 1024}, {300, 4096, 1024}, {300, 1024, 4096}} {
			m, n, k := s[0], s[1], s[2]
			name := fmt.Sprintf("%v/m%d_n%d_k%d", map[Type]string{Q8_0: "q8", F32: "f32"}[typ], m, n, k)
			b.Run(name, func(b *testing.B) {
				w := benchWeight(b, n, k, typ)
				x := make([]float32, m*k)
				for i := range x {
					x[i] = float32(i%7) - 3
				}
				out := make([]float32, m*n)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Linear(out, x, m, w, nil, 2)
				}
				b.ReportMetric(float64(2*m*n*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}

func BenchmarkKernel6x16(b *testing.B) {
	k := 256
	a := make([]float32, 6*k)
	p := make([]float32, k*nr)
	c := make([]float32, 6*nr)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kernel6x16(k, a, k, p, c, nr)
	}
	b.ReportMetric(float64(2*6*16*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

func BenchmarkPackQ8Panels(b *testing.B) {
	w := benchWeight(b, 4096, 1024, Q8_0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packQ8Panels(w.q, 4096, 1024, 1)
	}
	b.ReportMetric(float64(4096*1024)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mweights/s")
}
