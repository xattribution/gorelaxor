package kern

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantU8AsmMatchesGo(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const nb = 9
	x := make([]float32, nb*QK)
	for i := range x {
		x[i] = float32(r.NormFloat64()) * float32(1+i/QK)
	}
	clear(x[2*QK : 3*QK]) // a block of zeros
	x[5*QK] = -40         // the extreme is negative
	want, got := make([]byte, len(x)), make([]byte, len(x))
	ws, gs := make([]float32, nb), make([]float32, nb)
	quantU8Go(want, ws, x, nb)
	quantU8(got, gs, x, nb)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("value %d: %d, want %d", i, got[i], want[i])
		}
	}
	for b := range ws {
		if ws[b] != gs[b] {
			t.Fatalf("scale %d: %g, want %g", b, gs[b], ws[b])
		}
	}
	if got[5*QK] != 1 || gs[2] != 0 || got[2*QK] != 128 {
		t.Fatalf("extremes: %d %g %d", got[5*QK], gs[2], got[2*QK])
	}
}

// Every int8 kernel this processor has gives what the portable one does.
func TestInt8TileKernels(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	const nb, n = 5, 16
	k := nb * QK
	wv := make([]float32, n*k)
	for i := range wv {
		wv[i] = r.Float32()*2 - 1
	}
	pan := packQ8Panels(quantQ8(wv), n, k, 1)
	a := make([]byte, tileRows*k+7)
	for i := range a {
		a[i] = byte(r.Intn(256))
	}
	as := make([]float32, tileRows*nb)
	for i := range as {
		as[i] = r.Float32()
	}
	const ldc = 19
	base := make([]float32, tileRows*ldc)
	for i := range base {
		base[i] = r.Float32()
	}
	want := append([]float32(nil), base...)
	q8tileGo(nb, a, k+1, as, pan.b, want, ldc)
	defer setInt(intKind())
	for _, mode := range []int{intEVEX, intVEX} {
		if !setInt(mode) {
			t.Logf("this processor has no %s", map[int]string{intEVEX: "AVX512-VNNI", intVEX: "AVX-VNNI"}[mode])
			continue
		}
		got := append([]float32(nil), base...)
		q8tile(nb, a, k+1, as, pan.b, got, ldc)
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-4*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("mode %d, element %d: %g, want %g", mode, i, got[i], want[i])
			}
		}
	}
}

// Products through the int8 path stay within what rounding the
// activations to 8 bits allows, at sizes that aren't whole tiles.
func TestInt8MatMul(t *testing.T) {
	defer setInt(intKind())
	modes := []int{intGo}
	if intKind() != intNone {
		modes = append(modes, intKind())
	}
	r := rand.New(rand.NewSource(5))
	for _, sz := range [][3]int{{2, 1, 32}, {5, 15, 64}, {6, 16, 96}, {7, 17, 32}, {13, 40, 128}, {50, 33, 256}, {100, 70, 64}} {
		m, n, k := sz[0], sz[1], sz[2]
		x := make([]float32, m*(k+3))
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		wv := make([]float32, n*k)
		for i := range wv {
			wv[i] = r.Float32()*2 - 1
		}
		bias := make([]float32, n)
		for i := range bias {
			bias[i] = r.Float32()
		}
		w, _ := NewQ8(n, k, quantQ8(wv))
		ref := w.Float32()
		for _, mode := range modes {
			setInt(mode)
			for _, th := range []int{1, 3} {
				got := make([]float32, m*(n+2))
				MatMul(got, n+2, x, k+3, m, w, bias, th)
				var se, sr float64
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						s := float64(bias[j])
						for c := 0; c < k; c++ {
							s += float64(x[i*(k+3)+c]) * float64(ref[j*k+c])
						}
						d := float64(got[i*(n+2)+j]) - s
						se += d * d
						sr += s * s
					}
				}
				if rel := math.Sqrt(se / sr); rel > 0.02 {
					t.Errorf("%dx%dx%d mode %d threads %d: relative error %.4f", m, n, k, mode, th, rel)
				}
			}
		}
	}
}

// ToQ8 rounds the way the converter (and ggml) do.
func TestToQ8MatchesConverter(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	const n, k = 9, 96
	v := make([]float32, n*k)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	h := make([]uint16, len(v))
	for i, x := range v {
		h[i] = F32ToF16(x)
		v[i] = F16ToF32(h[i])
	}
	w, _ := NewF16(n, k, h)
	got := w.ToQ8()
	want := quantQ8(v)
	if got.Type != Q8_0 || string(got.q) != string(want) {
		t.Fatal("ToQ8 differs from the converter's rounding")
	}
	if odd, _ := NewF16(1, 40, make([]uint16, 40)); odd.ToQ8() != odd {
		t.Fatal("a row that isn't whole blocks should stay as it is")
	}
}

// A packed weight that drops its Q8_0 rows reads its rows back from the
// panels exactly, multiplies a single row through the int8 kernel, and
// costs about its panels' memory alone.
func TestDropRows(t *testing.T) {
	defer setInt(intKind())
	modes := []int{intGo}
	if intKind() != intNone {
		modes = append(modes, intKind())
	}
	r := rand.New(rand.NewSource(8))
	const n, k = 37, 96
	wv := make([]float32, n*k)
	for i := range wv {
		wv[i] = r.Float32()*2 - 1
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for _, mode := range modes {
		setInt(mode)
		w, _ := NewQ8(n, k, quantQ8(wv))
		ref := w.Float32()
		w.Pack(2)
		before := w.Bytes()
		w.DropRows()
		if w.q != nil || w.Bytes() >= before || w.Bytes() < n*k {
			t.Fatalf("mode %d: %d bytes after dropping the rows, %d before", mode, w.Bytes(), before)
		}
		row := make([]float32, k)
		for j := 0; j < n; j++ {
			w.Row(row, j)
			for c := range row {
				if row[c] != ref[j*k+c] {
					t.Fatalf("mode %d: row %d value %d is %g, want %g", mode, j, c, row[c], ref[j*k+c])
				}
			}
		}
		got := make([]float32, n)
		MatMul(got, n, x, k, 1, w, nil, 1)
		var se, sr float64
		for j := 0; j < n; j++ {
			s := 0.0
			for c := 0; c < k; c++ {
				s += float64(x[c]) * float64(ref[j*k+c])
			}
			se += (float64(got[j]) - s) * (float64(got[j]) - s)
			sr += s * s
		}
		if rel := math.Sqrt(se / sr); rel > 0.02 {
			t.Fatalf("mode %d: a single row's relative error is %.4f", mode, rel)
		}
	}
}
