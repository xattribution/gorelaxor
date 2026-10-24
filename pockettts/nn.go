package pockettts

import (
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// Activations are time-major: T rows of C values.

// scratch hands out reusable buffers by slot, so a frame's work
// allocates nothing once the buffers have grown.
type scratch struct{ b [][]float32 }

func (s *scratch) get(slot, n int) []float32 {
	for len(s.b) <= slot {
		s.b = append(s.b, nil)
	}
	if cap(s.b[slot]) < n {
		s.b[slot] = make([]float32, n, n+n/4)
	}
	return s.b[slot][:n]
}

// layerNorm is LayerNorm with eps: (x - mean)/sqrt(var + eps)·w + b,
// w and b nil for none.
func layerNorm(dst, src []float32, rows, n int, w, b []float32, eps float64) {
	for r := 0; r < rows; r++ {
		x, y := src[r*n:(r+1)*n], dst[r*n:(r+1)*n]
		var s float64
		for _, v := range x {
			s += float64(v)
		}
		mean := s / float64(n)
		var q float64
		for _, v := range x {
			d := float64(v) - mean
			q += d * d
		}
		inv := 1 / math.Sqrt(q/float64(n)+eps)
		for i, v := range x {
			o := float32((float64(v) - mean) * inv)
			if w != nil {
				o = o*w[i] + b[i]
			}
			y[i] = o
		}
	}
}

// elu is ELU (α = 1): x, or eˣ − 1 below 0.
func elu(x []float32) {
	var e [512]float32
	for o := 0; o < len(x); o += len(e) {
		c := x[o:min(len(x), o+len(e))]
		t := e[:len(c)]
		for i, v := range c {
			t[i] = min(v, 0)
		}
		kern.ExpSub(t, 0)
		for i, v := range c {
			c[i] = max(v, 0) + (t[i] - 1) // eˣ − 1 is exactly 0 at and above 0
		}
	}
}

// rope holds a transformer's rotary frequencies: exp(-ln(period)·2i/D)
// for each pair of a head's D values, in float32 as PyTorch makes them.
func ropeFreqs(headDim int, period float64) []float32 {
	c := float32(-math.Log(period) * 2 / float64(headDim))
	f := make([]float32, headDim/2)
	for i := range f {
		f[i] = float32(math.Exp(float64(float32(i) * c)))
	}
	return f
}

// applyRope rotates each pair (x[2i], x[2i+1]) of every head of one row
// by angle freqs[i]·pos.
func applyRope(x []float32, heads int, freqs []float32, pos int) {
	hd := len(freqs) * 2
	p := float32(pos)
	for i, f := range freqs {
		a := float64(f * p)
		c, s := float32(math.Cos(a)), float32(math.Sin(a))
		for h := 0; h < heads; h++ {
			j := h*hd + 2*i
			r, im := x[j], x[j+1]
			x[j] = r*c - im*s
			x[j+1] = r*s + im*c
		}
	}
}

type layer struct {
	norm1w, norm1b, norm2w, norm2b []float32
	qkv, out, ff1, ff2             *kern.Weight
	ls1, ls2                       []float32 // layer scales, nil for none
}

// transformer is Kyutai's StreamingTransformer: pre-norm layers of
// rotary self-attention and a GELU feed-forward, causal, each query
// seeing at most context positions back (0: all of them).
type transformer struct {
	layers       []layer
	d, heads, ff int
	context      int
	freqs        []float32
}

// cache is one layer's keys (rotated) and values, a row per position;
// start is the absolute position of the first row.
type cache struct {
	k, v  []float32
	start int
}

// tstate is a transformer's streaming state: its layers' caches and the
// next position.
type tstate struct {
	c   []cache
	pos int
}

func (t *transformer) newState() *tstate { return &tstate{c: make([]cache, len(t.layers))} }

// clone copies s with room for extra more values per cache.
func (s *tstate) clone(extra int) *tstate {
	o := &tstate{c: make([]cache, len(s.c)), pos: s.pos}
	for i, c := range s.c {
		o.c[i] = cache{
			k:     append(make([]float32, 0, len(c.k)+extra), c.k...),
			v:     append(make([]float32, 0, len(c.v)+extra), c.v...),
			start: c.start,
		}
	}
	return o
}

// forward runs T rows of x through the transformer in place, after the
// positions st holds, and moves st on by T.
func (t *transformer) forward(x []float32, T int, st *tstate, ws *scratch, threads int) {
	d := t.d
	h := ws.get(0, T*d)
	qkv := ws.get(1, T*3*d)
	att := ws.get(2, T*d)
	ff := ws.get(3, T*t.ff)
	for li := range t.layers {
		l := &t.layers[li]
		c := &st.c[li]
		layerNorm(h, x, T, d, l.norm1w, l.norm1b, 1e-5)
		kern.Linear(qkv, h, T, l.qkv, nil, threads)
		for r := 0; r < T; r++ {
			row := qkv[r*3*d : (r+1)*3*d]
			applyRope(row[:d], t.heads, t.freqs, st.pos+r)
			applyRope(row[d:2*d], t.heads, t.freqs, st.pos+r)
		}
		if t.context > 0 {
			// Keep what the first query of this call can still see.
			if keep := t.context - 1; len(c.k)/d > 4*t.context {
				drop := len(c.k)/d - keep
				c.k = append(c.k[:0], c.k[drop*d:]...)
				c.v = append(c.v[:0], c.v[drop*d:]...)
				c.start += drop
			}
		}
		for r := 0; r < T; r++ {
			row := qkv[r*3*d : (r+1)*3*d]
			c.k = append(c.k, row[d:2*d]...)
			c.v = append(c.v, row[2*d:]...)
		}
		t.attend(att, qkv, T, c, st.pos, threads)
		kern.Linear(h, att, T, l.out, nil, threads)
		addScaled(x, h, l.ls1)
		layerNorm(h, x, T, d, l.norm2w, l.norm2b, 1e-5)
		kern.Linear(ff, h, T, l.ff1, nil, threads)
		kern.GELU(ff)
		kern.Linear(h, ff, T, l.ff2, nil, threads)
		addScaled(x, h, l.ls2)
	}
	st.pos += T
}

// addScaled sets x += s·y, s repeating for each row (nil: x += y).
func addScaled(x, y, s []float32) {
	if s == nil {
		kern.Add(x, y)
		return
	}
	n := len(s)
	for i := range x {
		x[i] += s[i%n] * y[i]
	}
}

// attend writes softmax(q·kᵀ/√D)·v for T queries at positions pos0...,
// each over the cached positions it may see. Per head, the scores of all
// T queries against the keys any of them sees are one product, and so
// are the weighted values when there are several queries.
func (t *transformer) attend(out, qkv []float32, T int, c *cache, pos0, threads int) {
	d, hd := t.d, t.d/t.heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	lo0 := c.start // the first key row 0 sees
	if t.context > 0 {
		lo0 = max(lo0, pos0-t.context+1)
	}
	nk := pos0 + T - lo0
	k0 := (lo0 - c.start) * d
	if T*nk*d < 1<<15 {
		threads = 1
	}
	kern.Tasks(t.heads, threads, func(h int) {
		sc := make([]float32, T*nk)
		kern.MatMulF32(sc, nk, qkv[h*hd:], 3*d, c.k[k0+h*hd:], d, T, nk, hd, 1)
		for r := 0; r < T; r++ {
			p := pos0 + r
			lo := lo0
			if t.context > 0 {
				lo = max(lo, p-t.context+1)
			}
			row := sc[r*nk : (r+1)*nk]
			s := row[lo-lo0 : p+1-lo0]
			mx := float32(math.Inf(-1))
			for j := range s {
				s[j] *= scale
				mx = max(mx, s[j])
			}
			kern.Scale(s, 1/kern.ExpSub(s, mx))
			clear(row[:lo-lo0])
			clear(row[p+1-lo0:])
		}
		if T < 4 {
			for r := 0; r < T; r++ {
				o := out[r*d+h*hd : r*d+(h+1)*hd]
				clear(o)
				for j, w := range sc[r*nk : (r+1)*nk] {
					if w != 0 {
						vo := k0 + j*d + h*hd
						kern.AddScaled(o, c.v[vo:vo+hd], w)
					}
				}
			}
			return
		}
		vt := make([]float32, hd*nk) // this head's values, transposed
		for j := 0; j < nk; j++ {
			v := c.v[k0+j*d+h*hd : k0+j*d+(h+1)*hd]
			for i, x := range v {
				vt[i*nk+j] = x
			}
		}
		kern.MatMulF32(out[h*hd:], d, sc, nk, vt, nk, T, hd, nk, 1)
	})
}

// conv is a causal Conv1d (Kyutai's StreamingConv1d): its weight is
// [cout][k·cin], row t of the input window taps t·stride + j·dil.
type conv struct {
	w                         *kern.Weight
	b                         []float32
	cin, cout, k, stride, dil int
}

// past is how many input rows a streaming conv carries to the next call.
func (c *conv) past() int { return (c.k-1)*c.dil + 1 - c.stride }

// run convolves T rows of x (T a multiple of stride) after the rows prev
// holds, updates prev, and returns T/stride output rows in y.
func (c *conv) run(y []float32, prev []float32, x []float32, T int, ws *scratch, slot, threads int) []float32 {
	P := c.past()
	cin := c.cin
	To := T / c.stride
	y = y[:To*c.cout]
	if c.k == 1 && c.stride == 1 {
		kern.Linear(y, x[:T*cin], T, c.w, c.b, threads)
		return y
	}
	buf := ws.get(slot, (P+T)*cin)
	copy(buf, prev)
	copy(buf[P*cin:], x[:T*cin])
	cols := c.k * cin
	col := ws.get(slot+1, To*cols)
	for t := 0; t < To; t++ {
		for j := 0; j < c.k; j++ {
			s := (t*c.stride + j*c.dil) * cin
			copy(col[t*cols+j*cin:t*cols+(j+1)*cin], buf[s:s+cin])
		}
	}
	kern.Linear(y, col, To, c.w, c.b, threads)
	copy(prev, buf[T*cin:])
	return y
}

// convtr is a ConvTranspose1d (StreamingConvTranspose1d): weight
// [k·cout][cin], so one product gives an input row's k output taps.
type convtr struct {
	w                    *kern.Weight
	b                    []float32
	cin, cout, k, stride int
}

// run writes T·stride output rows to y, adding the overlap partial
// carried from the last call and keeping this call's for the next.
func (c *convtr) run(y []float32, partial []float32, x []float32, T int, ws *scratch, slot, threads int) []float32 {
	co, S := c.cout, c.stride
	z := ws.get(slot, T*c.k*co)
	kern.Linear(z, x[:T*c.cin], T, c.w, nil, threads)
	L := (T-1)*S + c.k
	full := ws.get(slot+1, L*co)
	for i := 0; i < L; i++ {
		copy(full[i*co:(i+1)*co], c.b)
	}
	for t := 0; t < T; t++ {
		for j := 0; j < c.k; j++ {
			kern.Add(full[(t*S+j)*co:(t*S+j+1)*co], z[(t*c.k+j)*co:(t*c.k+j+1)*co])
		}
	}
	PT := c.k - S
	kern.Add(full[:PT*co], partial)
	tail := full[(L-PT)*co:]
	for i := range partial {
		partial[i] = tail[i] - c.b[i%co]
	}
	y = y[:T*S*co]
	copy(y, full[:T*S*co])
	return y
}
