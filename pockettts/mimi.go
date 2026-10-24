package pockettts

import (
	"github.com/xattribution/gorelaxor/internal/kern"
)

// Mimi is Kyutai's audio codec at 12.5 latents a second. The decoder
// turns each 32-value latent into 1920 samples (80 ms at 24 kHz): a 1x1
// projection to 512 channels, a depthwise transposed convolution up to
// 200 Hz, a 2-layer transformer seeing 1.25 s back, and SEANet's
// convolutions up to 24 kHz. The encoder, used to clone a voice, runs
// the other way. Everything is causal and runs streaming.

// resBlock is SEANetResnetBlock: x + conv1x1(ELU(conv3(ELU(x)))).
type resBlock struct{ a, b conv }

type seaStep struct {
	c   *conv
	t   *convtr
	r   *resBlock
	elu bool // an ELU before the convolution
}

type mimiDecoder struct {
	quant *kern.Weight // [512][32]
	up    []float32    // [512][2·stride] depthwise
	upS   int
	tf    transformer
	steps []seaStep
}

type mimiEncoder struct {
	steps []seaStep
	tf    transformer
	down  conv // to 12.5 Hz, padded with the first input row ("replicate")
}

// seaState holds each step's carried rows, in the order of steps.
type seaState struct {
	prev [][]float32
	ws   scratch
}

func newSeaState(steps []seaStep) *seaState {
	s := &seaState{}
	add := func(n int) { s.prev = append(s.prev, make([]float32, n)) }
	for _, st := range steps {
		switch {
		case st.c != nil:
			add(st.c.past() * st.c.cin)
		case st.t != nil:
			add((st.t.k - st.t.stride) * st.t.cout)
		default:
			add(st.r.a.past() * st.r.a.cin)
			add(st.r.b.past() * st.r.b.cin)
		}
	}
	return s
}

// runSea runs T rows of x through the steps and returns the output rows.
func runSea(steps []seaStep, s *seaState, x []float32, T, C int, threads int) ([]float32, int) {
	pi := 0
	slot := 0
	for _, st := range steps {
		if st.elu {
			elu(x[:T*C])
		}
		switch {
		case st.c != nil:
			y := s.ws.get(slot, T/st.c.stride*st.c.cout)
			x = st.c.run(y, s.prev[pi], x, T, &s.ws, slot+1, threads)
			T, C = T/st.c.stride, st.c.cout
			pi++
		case st.t != nil:
			y := s.ws.get(slot, T*st.t.stride*st.t.cout)
			x = st.t.run(y, s.prev[pi], x, T, &s.ws, slot+1, threads)
			T, C = T*st.t.stride, st.t.cout
			pi++
		default:
			r := st.r
			v := s.ws.get(slot, T*C)
			copy(v, x[:T*C])
			elu(v)
			h := s.ws.get(slot+1, T*r.a.cout)
			h = r.a.run(h, s.prev[pi], v, T, &s.ws, slot+2, threads)
			elu(h)
			o := s.ws.get(slot+4, T*C)
			o = r.b.run(o, s.prev[pi+1], h, T, &s.ws, slot+5, threads)
			y := s.ws.get(slot+7, T*C)
			for i := range y {
				y[i] = x[i] + o[i]
			}
			x = y
			pi += 2
		}
		slot += 8
	}
	return x, T
}

// decState is the decoder's streaming state, one per chunk of text. Its
// scratch slots 0 to 3 are the transformer's.
type decState struct {
	tf      *tstate
	upPart  []float32
	sea     *seaState
	ws      scratch
	started bool
}

func (m *mimiDecoder) newState() *decState {
	return &decState{
		tf:     m.tf.newState(),
		upPart: make([]float32, m.upS*m.tf.d),
		sea:    newSeaState(m.steps),
	}
}

// decode turns n latents (already scaled back by the flow LM's mean
// and std) into n·1920 samples.
func (m *mimiDecoder) decode(st *decState, lat []float32, n, threads int) []float32 {
	C := m.tf.d
	ldim := m.quant.Cols
	z := st.ws.get(20, n*C)
	kern.Linear(z, lat[:n*ldim], n, m.quant, nil, threads)
	// depthwise ConvTranspose1d, kernel 2·stride, no bias
	S, K := m.upS, 2*m.upS
	L := (n-1)*S + K
	full := st.ws.get(21, L*C)
	clear(full)
	for t := 0; t < n; t++ {
		zr := z[t*C : (t+1)*C]
		for c := 0; c < C; c++ {
			w := m.up[c*K : (c+1)*K]
			v := zr[c]
			for j := 0; j < K; j++ {
				full[(t*S+j)*C+c] += v * w[j]
			}
		}
	}
	kern.Add(full[:S*C], st.upPart)
	copy(st.upPart, full[(L-S)*C:])
	T := n * S
	x := st.ws.get(22, T*C)
	copy(x, full[:T*C])
	m.tf.forward(x, T, st.tf, &st.ws, threads)
	out, _ := runSea(m.steps, st.sea, x, T, C, threads)
	return out
}

// encState is the encoder's streaming state for one clip.
type encState struct {
	sea   *seaState
	tf    *tstate
	down  []float32
	first bool
	ws    scratch
}

func (e *mimiEncoder) newState() *encState {
	return &encState{sea: newSeaState(e.steps), tf: e.tf.newState(), down: make([]float32, e.down.past()*e.down.cin), first: true}
}

// encode turns samples (a multiple of 1920) into latents at 12.5 Hz.
func (e *mimiEncoder) encode(st *encState, pcm []float32, threads int) []float32 {
	x := st.ws.get(20, len(pcm))
	copy(x, pcm)
	y, T := runSea(e.steps, st.sea, x, len(pcm), 1, threads)
	C := e.tf.d
	e.tf.forward(y, T, st.tf, &st.ws, threads)
	if st.first {
		for i := 0; i < e.down.past(); i++ {
			copy(st.down[i*C:(i+1)*C], y[:C])
		}
		st.first = false
	}
	out := make([]float32, T/e.down.stride*e.down.cout)
	return e.down.run(out, st.down, y, T, &st.ws, 10, threads)
}
