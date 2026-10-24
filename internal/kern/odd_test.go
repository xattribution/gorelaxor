package kern

import (
	"math"
	"math/rand"
	"testing"
)

func refMat(x []float32, m, k int, w []float32, n int, bias []float32) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for c := 0; c < k; c++ {
				s += float64(x[i*k+c]) * float64(w[j*k+c])
			}
			if bias != nil {
				s += float64(bias[j])
			}
			out[i*n+j] = s
		}
	}
	return out
}

func TestOddSizesBothPaths(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	defer setAVX(true)
	defer setInt(intKind())
	for _, avx := range []bool{true, false} {
		setAVX(avx)
		setInt(intNone) // q8int_test.go covers the int8 path
		for _, m := range []int{1, 2, 5, 6, 7, 16, 17, 23, 49} {
			for _, n := range []int{1, 3, 4, 5, 15, 16, 17, 33} {
				for _, k := range []int{1, 7, 15, 16, 17, 31, 32, 33, 96, 257, 300, 544} {
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
					ws := []*Weight{}
					wf, _ := NewF32(n, k, wv)
					wh, _ := NewF16(n, k, h)
					ws = append(ws, wf, wh)
					if k%QK == 0 {
						wq, _ := NewQ8(n, k, quantQ8(wv))
						ws = append(ws, wq)
					}
					for _, w := range ws {
						ref := refMat(x, m, k, w.Float32(), n, bias)
						for _, th := range []int{1, 3} {
							got := make([]float32, m*n)
							Linear(got, x, m, w, bias, th)
							for i := range got {
								tol := 1e-4 * (1 + float64(k))
								if d := math.Abs(ref[i] - float64(got[i])); d > tol {
									t.Fatalf("avx=%v type=%d m=%d n=%d k=%d th=%d [%d]: got %g want %g", avx, w.Type, m, n, k, th, i, got[i], ref[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestDequantAsmExact(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for _, nb := range []int{1, 2, 3, 7, 33} {
		v := make([]float32, nb*QK)
		for i := range v {
			v[i] = (r.Float32()*2 - 1) * 1000
		}
		q := quantQ8(v)
		// odd scales including subnormal/negative halves
		q[0], q[1] = 0x01, 0x80
		a := make([]float32, nb*QK)
		b := make([]float32, nb*QK)
		dequantQ8(a, q)
		dequantQ8Go(b, q)
		for i := range a {
			if a[i] != b[i] && !(math.IsNaN(float64(a[i])) && math.IsNaN(float64(b[i]))) {
				t.Fatalf("nb=%d [%d]: asm %g go %g", nb, i, a[i], b[i])
			}
		}
		var o1, o2 [4]float32
		x := make([]float32, nb*QK)
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		qq := append(append(append(append([]byte{}, q...), q...), q...), q...)
		dotQ8x4(x, qq, len(q), nb*QK, &o1)
		dotQ8x4Go(x, qq, len(q), nb*QK, &o2)
		for i := range o1 {
			if math.Abs(float64(o1[i]-o2[i])) > 1e-2*float64(nb) {
				t.Fatalf("dot nb=%d r=%d: %g vs %g", nb, i, o1[i], o2[i])
			}
		}
	}
}

// Q8 row ending exactly at the end of the slice: asm must not read past it.
func TestQ8TightBuffer(t *testing.T) {
	nb := 3
	v := make([]float32, nb*QK)
	q := quantQ8(v)
	q4 := make([]byte, 4*len(q))
	for r := 0; r < 4; r++ {
		copy(q4[r*len(q):], q)
	}
	x := make([]float32, nb*QK)
	var o [4]float32
	dotQ8x4(x, q4[:len(q4):len(q4)], len(q), nb*QK, &o)
	dst := make([]float32, nb*QK)
	dequantQ8(dst, q[:len(q):len(q)])
}
