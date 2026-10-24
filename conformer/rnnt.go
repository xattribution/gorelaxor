package conformer

import (
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// Token is one emitted subword.
type Token struct {
	ID       int
	Frame    int     // encoder frame (80 ms each) where it was emitted
	Duration int     // frames it covers (TDT); 1 for RNN-T
	P        float32 // softmax probability of the token
}

// DecoderState is the prediction network's memory between calls, so a
// stream can be decoded chunk by chunk.
type DecoderState struct {
	h, c [][]float32
	last int // last emitted token, -1 at the start
}

// NewDecoderState returns the start state.
func (m *Model) NewDecoderState() *DecoderState {
	s := &DecoderState{last: -1}
	for range m.lstm {
		s.h = append(s.h, make([]float32, m.HP.PredHidden))
		s.c = append(s.c, make([]float32, m.HP.PredHidden))
	}
	return s
}

func (s *DecoderState) copyFrom(o *DecoderState) {
	for i := range s.h {
		copy(s.h[i], o.h[i])
		copy(s.c[i], o.c[i])
	}
	s.last = o.last
}

func tanh32(x float32) float32 { return float32(math.Tanh(float64(x))) }

// predict runs one prediction-network step from st (fed st.last) into
// next and returns the top layer's output.
func (m *Model) predict(st, next *DecoderState, gates []float32, tmp []float32) []float32 {
	H := m.HP.PredHidden
	x := make([]float32, H)
	if st.last >= 0 && st.last != m.HP.Blank {
		copy(x, m.embed[st.last*H:(st.last+1)*H])
	}
	for l, ly := range m.lstm {
		kern.Linear(gates, x, 1, ly.wx, ly.b, 1)
		kern.Linear(tmp, st.h[l], 1, ly.wh, nil, 1)
		kern.Add(gates, tmp)
		h, c := next.h[l], next.c[l]
		for j := 0; j < H; j++ {
			i := sigmoid(gates[j])
			f := sigmoid(gates[H+j])
			g := tanh32(gates[2*H+j])
			o := sigmoid(gates[3*H+j])
			c[j] = f*st.c[l][j] + i*g
			h[j] = o * tanh32(c[j])
		}
		x = h
	}
	return x
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func softmaxAt(v []float32, i int) float32 {
	mx := v[argmax(v)]
	var s float64
	for _, x := range v {
		s += math.Exp(float64(x - mx))
	}
	return float32(math.Exp(float64(v[i]-mx)) / s)
}

// joint computes the joint network's logits for one encoder projection
// and one prediction output.
func (m *Model) joint(encProj, pred, hidden, predProj, logits []float32) {
	kern.Linear(predProj, pred, 1, m.jPred, m.jPredB, 1)
	for i := range hidden {
		v := encProj[i] + predProj[i]
		switch m.HP.JointAct {
		case "sigmoid":
			v = sigmoid(v)
		case "tanh":
			v = tanh32(v)
		default:
			v = max(v, 0)
		}
		hidden[i] = v
	}
	kern.Linear(logits, hidden, 1, m.jOut, m.jOutB, 1)
}

// Decode runs greedy RNN-T or TDT decoding over encoder output
// [T][d_model], continuing from st (nil starts fresh). Frames in the
// result are offset by frameOffset.
func (m *Model) Decode(enc []float32, T int, st *DecoderState, frameOffset, threads int) []Token {
	if T == 0 || m.jOut == nil {
		return nil
	}
	if st == nil {
		st = m.NewDecoderState()
	}
	cur := m.NewDecoderState()
	cur.copyFrom(st)
	defer func() { st.copyFrom(cur) }()
	hp := &m.HP
	J := hp.JointHidden
	encProj := make([]float32, T*J)
	kern.Linear(encProj, enc, T, m.jEnc, m.jEncB, threads)
	next := m.NewDecoderState()
	H := hp.PredHidden
	gates, tmp := make([]float32, 4*H), make([]float32, 4*H)
	hidden, predProj := make([]float32, J), make([]float32, J)
	logits := make([]float32, m.jOut.Rows)
	nTok := hp.PredVocab
	tdt := hp.HeadKind == "tdt"

	var out []Token
	var pred []float32
	dirty := true
	step, newSymbols := 0, 0
	maxIters := 16*T + 1024
	for iter := 0; step < T && iter < maxIters; iter++ {
		if dirty {
			pred = m.predict(cur, next, gates, tmp)
			dirty = false
		}
		m.joint(encProj[step*J:(step+1)*J], pred, hidden, predProj, logits)
		tok := argmax(logits[:nTok])
		blank := tok == hp.Blank
		duration := 1
		if tdt {
			duration = hp.Durations[argmax(logits[nTok:nTok+len(hp.Durations)])]
		}
		if !blank {
			out = append(out, Token{ID: tok, Frame: frameOffset + step, Duration: duration, P: softmaxAt(logits[:nTok], tok)})
			next.last = tok
			cur, next = next, cur
			dirty = true
		}
		if !tdt {
			if blank {
				step, newSymbols = step+1, 0
			} else if newSymbols++; hp.MaxSymbols > 0 && newSymbols >= hp.MaxSymbols {
				step, newSymbols = step+1, 0
			}
			continue
		}
		step += duration
		newSymbols++
		if duration != 0 {
			newSymbols = 0
		} else if hp.MaxSymbols > 0 && newSymbols >= hp.MaxSymbols {
			step, newSymbols = step+1, 0
		}
	}
	return out
}
