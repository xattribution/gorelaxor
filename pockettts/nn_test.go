package pockettts

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// These tests need no model files: the streaming pieces against their
// textbook definitions, and the text preparation against what Kyutai's
// prepare_text_prompt returned for the same strings.

func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s: value %d is %g, want %g", what, i, got[i], want[i])
		}
	}
}

// A causal conv run in pieces must equal the whole input convolved after
// zero padding of (k−1)·dil+1−stride.
func TestConvStreaming(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct{ cin, cout, k, stride, dil int }{{3, 5, 3, 1, 1}, {4, 2, 8, 4, 1}, {2, 3, 3, 1, 2}, {6, 4, 1, 1, 1}} {
		w, _ := kern.NewF32(c.cout, c.k*c.cin, randVec(r, c.cout*c.k*c.cin))
		cv := conv{w: w, b: randVec(r, c.cout), cin: c.cin, cout: c.cout, k: c.k, stride: c.stride, dil: c.dil}
		T := 12 * c.stride
		x := randVec(r, T*c.cin)
		P := cv.past()
		pad := append(make([]float32, P*c.cin), x...)
		want := make([]float32, T/c.stride*c.cout)
		for to := 0; to < T/c.stride; to++ {
			for co := 0; co < c.cout; co++ {
				s := cv.b[co]
				for j := 0; j < c.k; j++ {
					for ci := 0; ci < c.cin; ci++ {
						// weight [co][j][ci], as the converter stores it
						s += w.Float32()[co*c.k*c.cin+j*c.cin+ci] * pad[(to*c.stride+j*c.dil)*c.cin+ci]
					}
				}
				want[to*c.cout+co] = s
			}
		}
		prev := make([]float32, P*c.cin)
		var got []float32
		ws := &scratch{}
		for _, n := range []int{2, 1, 5, 4} { // pieces, in strides
			a := len(got) / c.cout * c.stride
			y := make([]float32, n*c.cout)
			got = append(got, cv.run(y, prev, x[a*c.cin:(a+n*c.stride)*c.cin], n*c.stride, ws, 0, 1)...)
		}
		near(t, "conv", got, want, 1e-5)
	}
}

// A transposed conv run in pieces must equal the whole output, with the
// overlap carried between calls.
func TestConvTrStreaming(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	cin, cout, k, S := 3, 2, 8, 4
	w, _ := kern.NewF32(k*cout, cin, randVec(r, k*cout*cin))
	ct := &convtr{w: w, b: randVec(r, cout), cin: cin, cout: cout, k: k, stride: S}
	T := 9
	x := randVec(r, T*cin)
	full := make([]float32, ((T-1)*S+k)*cout)
	for i := range full {
		full[i] = ct.b[i%cout]
	}
	wf := w.Float32()
	for ti := 0; ti < T; ti++ {
		for j := 0; j < k; j++ {
			for co := 0; co < cout; co++ {
				for ci := 0; ci < cin; ci++ {
					full[(ti*S+j)*cout+co] += x[ti*cin+ci] * wf[(j*cout+co)*cin+ci]
				}
			}
		}
	}
	part := make([]float32, (k-S)*cout)
	var got []float32
	ws := &scratch{}
	for a, n := range []int{4, 1, 4} {
		_ = a
		start := len(got) / cout / S
		y := make([]float32, n*S*cout)
		got = append(got, ct.run(y, part, x[start*cin:(start+n)*cin], n, ws, 0, 1)...)
	}
	near(t, "convtr", got, full[:T*S*cout], 1e-5)
}

// Attention with a context window, fed in pieces, must equal softmax
// attention computed directly over each query's window.
func TestAttendWindow(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	d, heads, ctx := 8, 2, 5
	tf := &transformer{d: d, heads: heads, context: ctx, layers: make([]layer, 1)}
	hd := d / heads
	T := 13
	qkv := randVec(r, T*3*d)
	want := make([]float32, T*d)
	for p := 0; p < T; p++ {
		for h := 0; h < heads; h++ {
			lo := max(0, p-ctx+1)
			var sc []float64
			var sum float64
			for j := lo; j <= p; j++ {
				var s float64
				for i := 0; i < hd; i++ {
					s += float64(qkv[p*3*d+h*hd+i] * qkv[j*3*d+d+h*hd+i])
				}
				e := math.Exp(s / math.Sqrt(float64(hd)))
				sc = append(sc, e)
				sum += e
			}
			for j := lo; j <= p; j++ {
				for i := 0; i < hd; i++ {
					want[p*d+h*hd+i] += float32(sc[j-lo] / sum * float64(qkv[j*3*d+2*d+h*hd+i]))
				}
			}
		}
	}
	c := &cache{}
	var got []float32
	pos := 0
	for _, n := range []int{1, 6, 1, 5} {
		part := qkv[pos*3*d : (pos+n)*3*d]
		for i := 0; i < n; i++ {
			row := part[i*3*d : (i+1)*3*d]
			c.k = append(c.k, row[d:2*d]...)
			c.v = append(c.v, row[2*d:]...)
		}
		out := make([]float32, n*d)
		tf.attend(out, part, n, c, pos, 1)
		got = append(got, out...)
		pos += n
	}
	near(t, "attention", got, want, 1e-5)
}

func TestPrepareText(t *testing.T) {
	c := &config{appendPunct: true, capitalize: true}
	for _, tc := range [][3]any{ // from pocket_tts prepare_text_prompt
		{"hello world", "Hello world.", 3},
		{"  hi there  ", "Hi there.", 3},
		{"Is it done?", "Is it done?", 3},
		{"He said \"stop\"", "He said \"stop\".", 3},
		{"a list: one, two,", "A list: one, two.", 3},
		{"ending with dash -", "Ending with dash.", 3},
		{"wait…", "Wait…", 3},
		{"(a parenthetical)", "(a parenthetical).", 3},
		{"x;y;z", "X;y;z.", 3},
		{"line one\nline two", "Line one line two.", 3},
		{"école", "École.", 3},
		{"42 is the answer", "42 is the answer.", 3},
		{"one two three four five", "One two three four five.", 1},
	} {
		got, guess, err := c.prepareText(tc[0].(string))
		if err != nil || got != tc[1].(string) || guess != tc[2].(int) {
			t.Errorf("%q: %q, %d, %v; want %q, %d", tc[0], got, guess, err, tc[1], tc[2])
		}
	}
	if _, _, err := c.prepareText(" \n\t"); err != errEmpty {
		t.Errorf("blank text: %v", err)
	}
}
