package pockettts

import (
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// flowLM is Kyutai's FlowLMModel: a causal transformer over the voice
// prompt, the text's token embeddings and then one audio latent per
// 80 ms frame, whose output conditions a small flow network that turns
// noise into the next latent.
type flowLM struct {
	tf         transformer
	embed      *kern.Weight // [tokens+1][d], the text lookup table
	input      *kern.Weight // [d][ldim], a latent into the transformer
	bos        []float32    // the input in place of the first frame's latent
	bosVoice   []float32    // the position before a voice prompt
	outW, outB []float32    // out_norm
	eosW       *kern.Weight // [1][d]
	eosB       float32
	mean, std  []float32    // latents' normalization, undone before Mimi
	speaker    *kern.Weight // [d][ldim], cloned voice latents into the transformer
	head       flowHead
}

// flowHead is SimpleMLPAdaLN: the latent's noise through residual
// blocks modulated (adaLN) by the transformer's output and the two
// LSD time conditions.
type flowHead struct {
	cond, input     *kern.Weight
	condB, inputB   []float32
	time            [2]timeEmbed
	res             []resMLP
	finalAda, out   *kern.Weight
	finalAdaB, outB []float32
	dim, ldim       int
}

type timeEmbed struct {
	freqs    []float32
	fc1, fc2 *kern.Weight
	b1, b2   []float32
	alpha    []float32
}

type resMLP struct {
	normW, normB []float32
	fc1, fc2     *kern.Weight
	b1, b2       []float32
	ada          *kern.Weight
	adaB         []float32
}

func linear1(dst, x []float32, w *kern.Weight, b []float32) {
	kern.Linear(dst, x, 1, w, b, 1)
}

// embed is TimestepEmbedder: [cos, sin] of t·freqs, an MLP, then the
// head's RMSNorm, which divides by the standard deviation
// (x·α/√(var + 1e-5), var with n−1).
func (e *timeEmbed) embed(t float32) []float32 {
	half := len(e.freqs)
	in := make([]float32, 2*half)
	for i, f := range e.freqs {
		a := float64(t * f)
		in[i] = float32(math.Cos(a))
		in[half+i] = float32(math.Sin(a))
	}
	h := make([]float32, e.fc1.Rows)
	linear1(h, in, e.fc1, e.b1)
	kern.SiLU(h)
	y := make([]float32, e.fc2.Rows)
	linear1(y, h, e.fc2, e.b2)
	var s float64
	for _, v := range y {
		s += float64(v)
	}
	mean := s / float64(len(y))
	var q float64
	for _, v := range y {
		q += (float64(v) - mean) * (float64(v) - mean)
	}
	inv := float32(1 / math.Sqrt(q/float64(len(y)-1)+1e-5))
	for i := range y {
		y[i] *= e.alpha[i] * inv
	}
	return y
}

// timeConds is each LSD step's time conditioning, the mean of the start
// and end times' embeddings.
func (f *flowHead) timeConds(steps int) [][]float32 {
	out := make([][]float32, steps)
	for i := range out {
		s := f.time[0].embed(float32(float64(i) / float64(steps)))
		t := f.time[1].embed(float32(float64(i+1) / float64(steps)))
		for j := range s {
			s[j] = (s[j] + t[j]) / 2
		}
		out[i] = s
	}
	return out
}

// sample is lsd_decode: from noise x0, steps of the flow network
// conditioned on c (the transformer's output), returning the latent.
func (f *flowHead) sample(c, x0 []float32, tc [][]float32, ws *scratch) []float32 {
	y := ws.get(10, f.dim)
	linear1(y, c, f.cond, f.condB)
	cur := append([]float32(nil), x0...)
	yi := ws.get(11, f.dim)
	for _, t := range tc {
		for j := range yi {
			yi[j] = y[j] + t[j]
		}
		v := f.net(yi, cur, ws)
		n := float32(len(tc))
		for j := range cur {
			cur[j] += v[j] / n
		}
	}
	return cur
}

func (f *flowHead) net(y, x []float32, ws *scratch) []float32 {
	D := f.dim
	h := ws.get(12, D)
	linear1(h, x, f.input, f.inputB)
	sy := ws.get(13, D)
	copy(sy, y)
	kern.SiLU(sy)
	mod := ws.get(14, 3*D)
	n := ws.get(15, D)
	m := ws.get(16, D)
	for i := range f.res {
		r := &f.res[i]
		linear1(mod, sy, r.ada, r.adaB)
		shift, scale, gate := mod[:D], mod[D:2*D], mod[2*D:]
		layerNorm(n, h, 1, D, r.normW, r.normB, 1e-6)
		for j := range n {
			n[j] = n[j]*(1+scale[j]) + shift[j]
		}
		linear1(m, n, r.fc1, r.b1)
		kern.SiLU(m)
		linear1(n, m, r.fc2, r.b2)
		for j := range h {
			h[j] += gate[j] * n[j]
		}
	}
	linear1(mod[:2*D], sy, f.finalAda, f.finalAdaB)
	shift, scale := mod[:D], mod[D:2*D]
	layerNorm(n, h, 1, D, nil, nil, 1e-6)
	for j := range n {
		n[j] = n[j]*(1+scale[j]) + shift[j]
	}
	out := make([]float32, f.ldim)
	linear1(out, n, f.out, f.outB)
	return out
}

// run feeds T rows of transformer input x and returns the last row's
// output after out_norm.
func (lm *flowLM) run(x []float32, T int, st *tstate, ws *scratch, threads int) []float32 {
	lm.tf.forward(x, T, st, ws, threads)
	d := lm.tf.d
	out := make([]float32, d)
	layerNorm(out, x[(T-1)*d:T*d], 1, d, lm.outW, lm.outB, 1e-5)
	return out
}

// text feeds the token ids' embeddings and returns the output at the
// last token.
func (lm *flowLM) text(ids []int, st *tstate, ws *scratch, threads int) []float32 {
	d := lm.tf.d
	x := make([]float32, len(ids)*d)
	for i, id := range ids {
		lm.embed.Row(x[i*d:(i+1)*d], id)
	}
	return lm.run(x, len(ids), st, ws, threads)
}

// step feeds one latent (nil: the first frame's) and returns the
// transformer's output and the end-of-speech logit.
func (lm *flowLM) step(latent []float32, st *tstate, ws *scratch, threads int) ([]float32, float32) {
	if latent == nil {
		latent = lm.bos
	}
	x := make([]float32, lm.tf.d)
	kern.Linear(x, latent, 1, lm.input, nil, threads)
	out := lm.run(x, 1, st, ws, threads)
	var e [1]float32
	linear1(e[:], out, lm.eosW, nil)
	return out, e[0] + lm.eosB
}
