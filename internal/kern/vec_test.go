package kern

import (
	"math"
	"math/rand"
	"testing"
)

func TestVectorOpsMatchGo(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	for _, n := range []int{0, 1, 7, 8, 9, 100, 1027} {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * 8)
			y[i] = float32(r.NormFloat64())
		}
		if n > 2 {
			x[0], x[1] = -120, 120 // past where e^x is finite
		}
		a, b := append([]float32(nil), x...), append([]float32(nil), x...)
		siluGo(a)
		SiLU(b)
		near(t, "silu", a, b)
		a, b = append([]float32(nil), x...), append([]float32(nil), x...)
		reluGo(a)
		ReLU(b)
		near(t, "relu", a, b)
		a, b = append([]float32(nil), x...), append(append([]float32(nil), x...), 5)
		axpyGo(a, y, 0.5)
		AddScaled(b, y, 0.5)
		near(t, "add", a, b[:n])
		if b[n] != 5 {
			t.Fatal("AddScaled wrote past the source's length")
		}
	}
}

func near(t *testing.T, what string, want, got []float32) {
	t.Helper()
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-6*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s %d of %d: %g, want %g", what, i, len(want), got[i], want[i])
		}
	}
}
