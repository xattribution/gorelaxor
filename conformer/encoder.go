package conformer

import (
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// Tensors here are row-major [time][channels] unless noted.

func silu(x []float32) { kern.SiLU(x) }

func sigmoid(v float32) float32 { return 1 / (1 + kern.Expf(-v)) }

func relu(x []float32) { kern.ReLU(x) }

func addScaled(dst, src []float32, s float32) { kern.AddScaled(dst, src, s) }

func addBias(x []float32, rows int, b []float32) {
	if b == nil {
		return
	}
	n := len(b)
	for r := 0; r < rows; r++ {
		kern.Add(x[r*n:(r+1)*n], b)
	}
}

// subOut is the output length of one stride-2, kernel-3 subsampling
// step: causal models pad two before and one after (NeMo CausalConv2D),
// the others one on each side.
func subOut(n int, causal bool) int {
	if causal {
		return n/2 + 1
	}
	return (n-1)/2 + 1
}

// conv0 is the first subsampling convolution: one input plane [t][f]
// to c channels, written channel-last [t][f][c].
func conv0(in []float32, T, F int, w, b []float32, c int, causal bool, threads int) ([]float32, int, int) {
	pl := 1
	if causal {
		pl = 2
	}
	T1, F1 := subOut(T, causal), subOut(F, causal)
	out := make([]float32, T1*F1*c)
	kern.Parallel(T1, threads, func(lo, hi int) {
		var taps [9]float32
		for t := lo; t < hi; t++ {
			for f := 0; f < F1; f++ {
				for kt := 0; kt < 3; kt++ {
					ti := 2*t + kt - pl
					for kf := 0; kf < 3; kf++ {
						fi := 2*f + kf - pl
						v := float32(0)
						if ti >= 0 && ti < T && fi >= 0 && fi < F {
							v = in[ti*F+fi]
						}
						taps[kt*3+kf] = v
					}
				}
				o := out[(t*F1+f)*c : (t*F1+f+1)*c]
				copy(o, b)
				for j, v := range taps {
					if v != 0 {
						addScaled(o, w[j*c:(j+1)*c], v)
					}
				}
			}
		}
	})
	return out, T1, F1
}

// dwConv2 is a depthwise 3x3 stride-2 convolution on [t][f][c].
func dwConv2(in []float32, T, F, c int, w, b []float32, causal bool, threads int) ([]float32, int, int) {
	pl := 1
	if causal {
		pl = 2
	}
	T1, F1 := subOut(T, causal), subOut(F, causal)
	out := make([]float32, T1*F1*c)
	kern.Parallel(T1, threads, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			for f := 0; f < F1; f++ {
				o := out[(t*F1+f)*c : (t*F1+f+1)*c]
				copy(o, b)
				for kt := 0; kt < 3; kt++ {
					ti := 2*t + kt - pl
					if ti < 0 || ti >= T {
						continue
					}
					for kf := 0; kf < 3; kf++ {
						fi := 2*f + kf - pl
						if fi < 0 || fi >= F {
							continue
						}
						x := in[(ti*F+fi)*c : (ti*F+fi+1)*c]
						wk := w[(kt*3+kf)*c : (kt*3+kf+1)*c]
						for i := range o {
							o[i] += wk[i] * x[i]
						}
					}
				}
			}
		}
	})
	return out, T1, F1
}

// maskRows zeroes rows from valid on, as NeMo's MaskedConvSequential
// does before every layer of the pre-encoder.
func maskRows(x []float32, rows, valid, width int) {
	if valid < rows {
		clear(x[max(valid, 0)*width : rows*width])
	}
}

// subsample runs the 8x depthwise-striding pre-encoder on log-mel
// features [mels][frames], of which the first valid are real, and
// returns [T][d_model] with its valid length.
func (m *Model) subsample(mel []float32, frames, valid, threads int) ([]float32, int, int) {
	nm := m.Mel.cfg.NumMels
	in := make([]float32, frames*nm)
	for f := 0; f < nm; f++ {
		for t := 0; t < min(frames, valid); t++ {
			in[t*nm+f] = mel[f*frames+t]
		}
	}
	causal := m.HP.CausalSub
	c := m.HP.SubChannels
	p := &m.pre
	L := valid
	x, T, F := conv0(in, frames, nm, p.w0, p.b0, c, causal, threads)
	L = subOut(L, causal)
	maskRows(x, T, L, F*c)
	relu(x)
	x, T, F = dwConv2(x, T, F, c, p.w2, p.b2, causal, threads)
	L = subOut(L, causal)
	maskRows(x, T, L, F*c)
	y := make([]float32, T*F*c)
	kern.Linear(y, x, T*F, p.w3, p.b3, threads)
	maskRows(y, T, L, F*c)
	relu(y)
	x, T, F = dwConv2(y, T, F, c, p.w5, p.b5, causal, threads)
	L = subOut(L, causal)
	maskRows(x, T, L, F*c)
	y = make([]float32, T*F*c)
	kern.Linear(y, x, T*F, p.w6, p.b6, threads)
	maskRows(y, T, L, F*c)
	relu(y)
	// [t][f][c] to [t][c*F+f]
	flat := make([]float32, T*c*F)
	for t := 0; t < T; t++ {
		for f := 0; f < F; f++ {
			src := y[(t*F+f)*c : (t*F+f+1)*c]
			dst := flat[t*c*F:]
			for ch, v := range src {
				dst[ch*F+f] = v
			}
		}
	}
	out := make([]float32, T*m.HP.DModel)
	kern.Linear(out, flat, T, p.out, p.outb, threads)
	return out, T, max(0, min(L, T))
}

// relPosEmb returns the sinusoidal embedding for relative positions
// n-1 down to -(n-1) as [2n-1][d] (NeMo RelPositionalEncoding).
func relPosEmb(n, d int) []float32 {
	return relPosRange(n-1, 2*n-1, d)
}

// relPosRange returns rows for positions first, first-1, ... (count rows).
func relPosRange(first, count, d int) []float32 {
	out := make([]float32, count*d)
	for i := 0; i < count; i++ {
		pos := float64(first - i)
		row := out[i*d:]
		for j := 0; j < d; j += 2 {
			div := float32(math.Exp(float64(j) * -(math.Log(10000) / float64(d))))
			a := float64(float32(pos) * div)
			row[j] = float32(math.Sin(a))
			if j+1 < d {
				row[j+1] = float32(math.Cos(a))
			}
		}
	}
	return out
}

// keyRange gives, for query q of T, the keys [lo, hi) it may attend to.
type keyRange func(q int) (lo, hi int)

func (m *Model) offlineKeys(T int) (keyRange, int) {
	h := &m.HP
	switch {
	case h.Causal() && h.AttLeft >= 0 && h.AttRight >= 0:
		chunk := h.AttRight + 1
		left := h.AttLeft / chunk
		return func(q int) (int, int) {
			c := q / chunk
			return max(0, (c-left)*chunk), min(T, (c+1)*chunk)
		}, chunk
	case h.AttLeft >= 0 && h.AttRight >= 0 && T-1 > min(h.AttLeft, h.AttRight):
		return func(q int) (int, int) {
			return max(0, q-h.AttLeft), min(T, q+h.AttRight+1)
		}, 16
	}
	return func(int) (int, int) { return 0, T }, 32
}

// layerState holds the per-call scratch buffers of the encoder.
type layerState struct {
	T, D     int
	ln       []float32
	ff       []float32
	q, k, v  []float32
	qu, qv   []float32
	vt       []float32
	att      []float32
	conv     []float32
	glu      []float32
	posProj  []float32
	threads  int
	keys     keyRange
	blockLen int
	lnConv   bool // the convolution module normalizes with layer norm
	valid    int  // frames before this are real, the rest padding
}

func newLayerState(T, D, dff, threads int) *layerState {
	return &layerState{
		T: T, D: D, threads: threads,
		ln: make([]float32, T*D), ff: make([]float32, T*dff),
		q: make([]float32, T*D), k: make([]float32, T*D), v: make([]float32, T*D),
		qu: make([]float32, T*D), qv: make([]float32, T*D), vt: make([]float32, T*D),
		att: make([]float32, T*D), conv: make([]float32, T*2*D), glu: make([]float32, T*D),
		valid: T,
	}
}

func (s *layerState) feedForward(x []float32, w *ffWeights) {
	T, D := s.T, s.D
	kern.LayerNorm(s.ln, x, T, D, w.normW, w.normB)
	kern.Linear(s.ff, s.ln, T, w.w1, w.b1, s.threads)
	silu(s.ff[:T*w.w1.Rows])
	kern.Linear(s.att, s.ff, T, w.w2, w.b2, s.threads)
	addScaled(x[:T*D], s.att[:T*D], 0.5)
}

// attention is relative-position multi-head self-attention. Queries
// are the normalized rows s.ln (s.T of them); keys and values come from
// kv (Tk normalized rows), whose last s.T rows are the queries' own, so
// query i sits at key position Tk-s.T+i. pp holds the projected
// position embeddings for relative positions Tk-1 down to -(s.T-1). The
// result is added to x.
func (s *layerState) attention(x []float32, b *block, heads int, pp, kv []float32, Tk int) {
	T, D := s.T, s.D
	hd := D / heads
	qoff := Tk - T
	s.grow(Tk)
	kern.Linear(s.q, s.ln, T, b.q, b.qb, s.threads)
	kern.Linear(s.k, kv, Tk, b.k, b.kb, s.threads)
	kern.Linear(s.v, kv, Tk, b.v, b.vb, s.threads)
	for t := 0; t < T; t++ {
		for h := 0; h < heads; h++ {
			o := t*D + h*hd
			for j := 0; j < hd; j++ {
				q := s.q[o+j]
				s.qu[o+j] = q + b.posU[h*hd+j]
				s.qv[o+j] = q + b.posV[h*hd+j]
			}
		}
	}
	for t := 0; t < Tk; t++ {
		for c := 0; c < D; c++ {
			s.vt[c*Tk+t] = s.v[t*D+c]
		}
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	bl := s.blockLen
	nb := (T + bl - 1) / bl
	kern.Tasks(heads*nb, s.threads, func(task int) {
		h, blk := task/nb, task%nb
		q0, q1 := blk*bl, min(T, (blk+1)*bl)
		nq := q1 - q0
		k0, k1 := Tk, 0
		for q := q0; q < q1; q++ {
			lo, hi := s.keys(qoff + q)
			k0, k1 = min(k0, lo), max(k1, hi)
		}
		nk := k1 - k0
		if nk <= 0 {
			clear(s.glu[q0*D : q1*D])
			return
		}
		np := nq + nk - 1
		jlo := Tk - 1 - (qoff + q1 - 1) + k0
		sc := make([]float32, nq*nk)
		bd := make([]float32, nq*np)
		kern.MatMulF32(sc, nk, s.qu[q0*D+h*hd:], D, s.k[k0*D+h*hd:], D, nq, nk, hd, 1)
		kern.MatMulF32(bd, np, s.qv[q0*D+h*hd:], D, pp[jlo*D+h*hd:], D, nq, np, hd, 1)
		for i := 0; i < nq; i++ {
			row := sc[i*nk : (i+1)*nk]
			brow := bd[i*np+nq-1-i:]
			lo, hi := s.keys(qoff + q0 + i)
			mx := float32(math.Inf(-1))
			for kk := range row {
				if k := k0 + kk; k < lo || k >= hi {
					row[kk] = float32(math.Inf(-1))
					continue
				}
				v := (row[kk] + brow[kk]) * scale
				row[kk] = v
				mx = max(mx, v)
			}
			if math.IsInf(float64(mx), -1) {
				clear(row)
				continue
			}
			sum := kern.ExpSub(row, mx)
			kern.Scale(row, 1/sum)
		}
		kern.MatMulF32(s.glu[q0*D+h*hd:], D, sc, nk, s.vt[h*hd*Tk+k0:], Tk, nq, hd, nk, 1)
	})
	kern.Linear(s.att, s.glu, T, b.out, b.outb, s.threads)
	kern.Add(x[:T*D], s.att[:T*D])
}

// grow makes the key and value buffers hold Tk rows.
func (s *layerState) grow(Tk int) {
	if n := Tk * s.D; len(s.k) < n {
		s.k, s.v, s.vt = make([]float32, n), make([]float32, n), make([]float32, n)
	}
}

// convModule is the Conformer convolution module on the normalized
// input s.ln; prev holds the padL frames before this call's first frame
// (nil means zeros), frames from s.valid on are padding, and the result
// is added to x. The gated frames stay in s.glu for the caller's cache.
func (s *layerState) convModule(x []float32, b *block, kernel, padL int, prev []float32) {
	T, D := s.T, s.D
	kern.Linear(s.conv, s.ln, T, b.pw1, b.pw1b, s.threads)
	for t := 0; t < T; t++ {
		a := s.conv[t*2*D : (t+1)*2*D]
		g := s.glu[t*D : (t+1)*D]
		if t >= s.valid {
			clear(g)
			continue
		}
		for c := 0; c < D; c++ {
			g[c] = a[c] * sigmoid(a[D+c])
		}
	}
	dw := s.att[:T*D]
	kern.Parallel(T, s.threads, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			o := dw[t*D : (t+1)*D]
			if b.dwb != nil {
				copy(o, b.dwb)
			} else {
				clear(o)
			}
			for j := 0; j < kernel; j++ {
				ti := t + j - padL
				var src []float32
				switch {
				case ti >= 0 && ti < T:
					src = s.glu[ti*D : (ti+1)*D]
				case ti < 0 && prev != nil && padL+ti >= 0:
					src = prev[(padL+ti)*D : (padL+ti+1)*D]
				default:
					continue
				}
				w := b.dw[j*D : (j+1)*D]
				for c := range o {
					o[c] += w[c] * src[c]
				}
			}
		}
	})
	if s.lnConv {
		kern.LayerNorm(dw, dw, T, D, b.cnW, b.cnB)
	} else {
		for t := 0; t < T; t++ {
			r := dw[t*D : (t+1)*D]
			for c := range r {
				r[c] = r[c]*b.cnW[c] + b.cnB[c]
			}
		}
	}
	silu(dw)
	kern.Linear(s.conv, dw, T, b.pw2, b.pw2b, s.threads)
	kern.Add(x[:T*D], s.conv[:T*D])
}

func (m *Model) convPadLeft() int {
	if m.HP.ConvLeft >= 0 && m.HP.ConvRight >= 0 {
		return m.HP.ConvLeft
	}
	return (m.HP.ConvKernel - 1) / 2
}

// runBlock runs one Conformer block on x in place. pp is the projected
// position table for the attention; kvBefore holds earlier normalized
// attention inputs (streaming) and convPrev the gated frames before
// this call's first frame. kvBefore returns updated to also hold this
// call's attention inputs, uncut.
func (m *Model) runBlock(s *layerState, x []float32, b *block, pp, kvBefore, convPrev []float32, trace func(string, []float32)) []float32 {
	T, D := s.T, s.D
	s.feedForward(x, &b.ff1)
	trace("ff1", x)
	kern.LayerNorm(s.ln, x, T, D, b.normAttnW, b.normAtt)
	kv := s.ln[:T*D]
	if kvBefore != nil {
		kv = append(kvBefore, s.ln[:T*D]...)
	}
	s.attention(x, b, m.HP.Heads, pp, kv, len(kv)/D)
	trace("attn", x)
	kern.LayerNorm(s.ln, x, T, D, b.normConvW, b.normCon)
	s.convModule(x, b, m.HP.ConvKernel, m.convPadLeft(), convPrev)
	trace("conv", x)
	s.feedForward(x, &b.ff2)
	trace("ff2", x)
	kern.LayerNorm(x, x, T, D, b.normOutW, b.normOutB)
	return kv
}

// Encode runs the whole encoder over log-mel features [mels][frames],
// the first valid of them real (Mel.Valid), and returns [T][d_model]
// with T about valid/8, ready for Decode. Streaming models use the
// attention limits NeMo runs a whole recording with (its first trained
// choice); multilingual ones apply the language prompt (WithLanguage).
func (m *Model) Encode(mel []float32, frames, valid, threads int) ([]float32, int) {
	x, T := m.encode(mel, frames, valid, threads, nil)
	return m.applyPrompt(x, T, threads), T
}

func (m *Model) encode(mel []float32, frames, valid, threads int, trace func(string, []float32)) ([]float32, int) {
	if trace == nil {
		trace = func(string, []float32) {}
	}
	x, T, L := m.subsample(mel, frames, valid, threads)
	trace("pre_encode", x)
	return m.encodeEmbeddings(x, T, L, threads, trace)
}

// xscale multiplies pre-encoder output by sqrt(d_model) where the model
// asks for it (NeMo applies it in the position encoding).
func (m *Model) xscale(x []float32) {
	if m.HP.XScaling {
		kern.Scale(x, float32(math.Sqrt(float64(m.HP.DModel))))
	}
}

// encodeEmbeddings runs the Conformer blocks over pre-encoder output x
// [T][d_model] (unscaled), of which the first L rows are real, in place.
func (m *Model) encodeEmbeddings(x []float32, T, L, threads int, trace func(string, []float32)) ([]float32, int) {
	if trace == nil {
		trace = func(string, []float32) {}
	}
	m.xscale(x)
	D := m.HP.DModel
	s := newLayerState(T, D, m.HP.DFF, threads)
	keys, bl := m.offlineKeys(T)
	s.keys = func(q int) (int, int) {
		if q >= L {
			return 0, 0
		}
		lo, hi := keys(q)
		return lo, min(hi, L)
	}
	s.blockLen = bl
	s.valid = L
	s.lnConv = m.HP.ConvNorm == "layer_norm"
	pos := relPosEmb(T, D)
	pp := make([]float32, len(pos))
	for i := range m.blocks {
		b := &m.blocks[i]
		m.injectSpeaker(i, x, T, threads)
		kern.Linear(pp, pos, len(pos)/D, b.pos, nil, threads)
		name := "block." + itoa(i) + "."
		m.runBlock(s, x, b, pp, nil, nil, func(n string, v []float32) { trace(name+n, v) })
		trace(name+"out", x)
	}
	return x[:L*D], L
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

// spkKernel is the multitalker model's speaker kernel at one layer: two
// small feed-forward nets whose output is added to the layer's input
// (NeMo SpeakerKernelMixin, kernel type "ff").
type spkKernel struct {
	w0, w3, bw0, bw3 *kern.Weight
	b0, b3, bb0, bb3 []float32
}

// injectSpeaker applies the speaker kernel before block i, as NeMo's
// pre-layer hook does with no speaker targets set: the speaker branch
// sees x (a mask of ones), the background branch zeros.
func (m *Model) injectSpeaker(i int, x []float32, T, threads int) {
	k := m.spk[i]
	if k == nil || i > 0 { // NeMo's later-layer hooks do nothing without targets
		return
	}
	D := m.HP.DModel
	h := make([]float32, T*k.w0.Rows)
	kern.Linear(h, x, T, k.w0, k.b0, threads)
	relu(h)
	y := make([]float32, T*D)
	kern.Linear(y, h, T, k.w3, k.b3, threads)
	kern.Add(x[:T*D], y)
	if k.bw0 == nil {
		return
	}
	// x·0 through the background net is the same vector for every frame
	hb := make([]float32, k.bw0.Rows)
	copy(hb, k.bb0)
	relu(hb)
	yb := make([]float32, D)
	kern.Linear(yb, hb, 1, k.bw3, k.bb3, 1)
	for t := 0; t < T; t++ {
		kern.Add(x[t*D:(t+1)*D], yb)
	}
}
