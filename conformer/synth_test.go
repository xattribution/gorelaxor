package conformer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/xattribution/gorelaxor/internal/kern"
)

type synth struct{ r *rand.Rand }

func (s synth) vec(n int, scale float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(s.r.NormFloat64()) * scale
	}
	return v
}
func (s synth) ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}
func (s synth) mat(rows, cols int) *kern.Weight {
	w, err := kern.NewF32(rows, cols, s.vec(rows*cols, float32(1/math.Sqrt(float64(cols)))))
	if err != nil {
		panic(err)
	}
	return w
}

const synMels = 16

func synthMel(norm string) *Mel {
	m, err := NewMel(MelConfig{SampleRate: 16000, NFFT: 512, WinLength: 400, Hop: 160, NumMels: synMels, FMax: 8000, PreEmphasis: 0.97, Normalize: norm})
	if err != nil {
		panic(err)
	}
	return m
}

// synthModel builds a tiny random FastConformer transducer.
func synthModel(seed int64, streaming bool, head string) *Model {
	s := synth{rand.New(rand.NewSource(seed))}
	D, H, DFF, C, K := 16, 2, 32, 4, 9
	hp := HParams{DModel: D, Heads: H, Layers: 2, DFF: DFF, ConvKernel: K, SubChannels: C, Subsampling: 8,
		AttStyle: "regular", AttLeft: -1, AttRight: -1, ConvLeft: -1, ConvRight: -1, ConvNorm: "batch_norm",
		PreEncodeCache: -1, DropExtra: -1, SamplingFramesFirst: -1,
		HeadKind: head, PredHidden: 8, PredLayers: 1, PredVocab: 11, JointHidden: 8, JointAct: "relu", MaxSymbols: 10, Blank: 10}
	norm := "per_feature"
	if streaming {
		hp.AttStyle, hp.AttLeft, hp.AttRight = "chunked_limited", 70, 13
		hp.ContextChoices = [][2]int{{70, 13}, {70, 6}, {70, 1}, {70, 0}}
		hp.ConvLeft, hp.ConvRight, hp.ConvNorm, hp.CausalSub = K-1, 0, "layer_norm", true
		norm = "none"
	}
	if head == "tdt" {
		hp.Durations = []int{0, 1, 2, 3, 4}
		hp.ExtraOutputs = 5
	}
	m := &Model{HP: hp, Mel: synthMel(norm), maxPosEmb: 5000}
	for i := 0; i < 10; i++ {
		m.Vocab = append(m.Vocab, "▁w"+itoa(i))
	}
	m.Vocab = append(m.Vocab, "<blank>")
	F := synMels
	for i := 0; i < 3; i++ {
		F = subOut(F, hp.CausalSub)
	}
	m.pre = preEncode{w0: s.vec(9*C, 0.3), b0: s.vec(C, 0.1), w2: s.vec(9*C, 0.3), b2: s.vec(C, 0.1),
		w3: s.mat(C, C), b3: s.vec(C, 0.1), w5: s.vec(9*C, 0.3), b5: s.vec(C, 0.1), w6: s.mat(C, C), b6: s.vec(C, 0.1),
		out: s.mat(D, C*F), outb: s.vec(D, 0.1)}
	ff := func() ffWeights {
		return ffWeights{normW: s.ones(D), normB: s.vec(D, 0.1), w1: s.mat(DFF, D), b1: s.vec(DFF, 0.1), w2: s.mat(D, DFF), b2: s.vec(D, 0.1)}
	}
	for i := 0; i < hp.Layers; i++ {
		m.blocks = append(m.blocks, block{ff1: ff(), ff2: ff(), normAttnW: s.ones(D), normAtt: s.vec(D, 0.1),
			q: s.mat(D, D), k: s.mat(D, D), v: s.mat(D, D), out: s.mat(D, D), pos: s.mat(D, D),
			qb: s.vec(D, .1), kb: s.vec(D, .1), vb: s.vec(D, .1), outb: s.vec(D, .1), posU: s.vec(D, .1), posV: s.vec(D, .1),
			normConvW: s.ones(D), normCon: s.vec(D, .1), pw1: s.mat(2*D, D), pw1b: s.vec(2*D, .1), pw2: s.mat(D, D), pw2b: s.vec(D, .1),
			dw: s.vec(K*D, .3), dwb: s.vec(D, .1), cnW: s.ones(D), cnB: s.vec(D, .1), normOutW: s.ones(D), normOutB: s.vec(D, .1)})
	}
	Hd, J := hp.PredHidden, hp.JointHidden
	m.embed = s.vec(hp.PredVocab*Hd, 1)
	m.lstm = []lstmLayer{{wx: s.mat(4*Hd, Hd), wh: s.mat(4*Hd, Hd), b: s.vec(4*Hd, .1)}}
	m.jEnc, m.jEncB = s.mat(J, D), s.vec(J, .1)
	m.jPred, m.jPredB = s.mat(J, Hd), s.vec(J, .1)
	m.jOut, m.jOutB = s.mat(hp.PredVocab+hp.ExtraOutputs, J), s.vec(hp.PredVocab+hp.ExtraOutputs, .1)
	// bias toward blank so decoding behaves like a real model
	m.jOutB[hp.Blank] += 2
	return m
}

func synthDiarizer(seed int64) *Diarizer {
	s := synth{rand.New(rand.NewSource(seed))}
	enc := synthModel(seed, false, "rnnt")
	enc.Mel = synthMel("none")
	D, pd := enc.HP.DModel, 12
	lin := func(r, c int) vadLinear { return vadLinear{w: s.mat(r, c), b: s.vec(r, .1)} }
	d := &Diarizer{Mel: enc.Mel, Speakers: 4, enc: enc, proj: lin(pd, D), fc1: lin(pd, pd), head: lin(4, pd), tfHeads: 2}
	for i := 0; i < 2; i++ {
		d.tf = append(d.tf, tfBlock{q: lin(pd, pd), k: lin(pd, pd), v: lin(pd, pd), out: lin(pd, pd),
			ln1w: s.ones(pd), ln1b: s.vec(pd, .1), ffIn: lin(24, pd), ffOut: lin(pd, 24), ln2w: s.ones(pd), ln2b: s.vec(pd, .1)})
	}
	d.Defaults = DiarParams{ChunkLen: 188, LeftContext: 1, RightContext: 1, FIFOLen: 0, UpdatePeriod: 188, CacheLen: 188,
		SilFramesPerSpk: 3, PredScoreThreshold: 0.25, ScoresBoostLatest: 0.05, SilThreshold: 0.2, StrongBoostRate: 0.75, WeakBoostRate: 1.5, MinPosScoresRate: 0.5}
	return d
}

func synthVAD(seed int64) *VAD {
	s := synth{rand.New(rand.NewSource(seed))}
	v := &VAD{Mel: synthMel("per_feature")}
	lin := func(r, c int) vadLinear { return vadLinear{w: s.mat(r, c), b: s.vec(r, .1)} }
	v.blocks = []vadBlock{
		{in: synMels, out: 8, kernel: 11, stride: 2, dilation: 1, reps: []vadRep{{dw: s.vec(11*synMels, .3), pw: lin(8, synMels)}}},
		{in: 8, out: 8, kernel: 13, stride: 1, dilation: 1, reps: []vadRep{{dw: s.vec(13*8, .3), pw: lin(8, 8)}, {dw: s.vec(13*8, .3), pw: lin(8, 8)}}, res: &vadLinear{w: s.mat(8, 8), b: s.vec(8, .1)}},
	}
	v.dec = []vadLinear{lin(2, 8)}
	v.FrameSeconds = 0.02
	return v
}

func noise(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.1)
	}
	return out
}

func sameToks(a, b []Token) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Frame != b[i].Frame {
			return false
		}
	}
	return true
}

func TestSynthStreamEdges(t *testing.T) {
	m := synthModel(1, true, "rnnt")
	for _, R := range []int{13, 6, 1, 0} {
		for _, n := range []int{0, 1, 159, 160, 161, 255, 256, 257, 319, 320, 1279, 1280, 1281, 1440, 1441, 16000, 17920, 17921, 33333} {
			pcm := noise(n, int64(n))
			s, err := m.NewStream(R, 2)
			if err != nil {
				t.Fatal(err)
			}
			var all []Token
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("R=%d n=%d: panic %v", R, n, p)
					}
				}()
				all = append(s.Write(pcm), s.Close()...)
				// write again after close
				s.Write(pcm)
				s.Close()
			}()
			// in pieces of odd sizes
			for _, step := range []int{1, 7, 160, 1601} {
				if n > 20000 && step < 160 {
					continue
				}
				s2, _ := m.NewStream(R, 2)
				var got []Token
				func() {
					defer func() {
						if p := recover(); p != nil {
							t.Errorf("R=%d n=%d step=%d: panic %v", R, n, step, p)
						}
					}()
					for i := 0; i < n; i += step {
						got = append(got, s2.Write(pcm[i:min(n, i+step)])...)
					}
					got = append(got, s2.Close()...)
				}()
				if !sameToks(all, got) {
					t.Errorf("R=%d n=%d step=%d: %d tokens vs %d in one write", R, n, step, len(got), len(all))
				}
			}
		}
	}
}

func TestSynthStreamMemory(t *testing.T) {
	m := synthModel(1, true, "rnnt")
	s, _ := m.NewStream(13, 2)
	for i := 0; i < 600; i++ { // 10 minutes
		s.Write(noise(16000, int64(i)))
		if i%200 == 199 {
			kv := 0
			for _, k := range s.st.kv {
				kv += len(k)
			}
			t.Logf("%d s: emph %d cap %d, frames %d, kv %d, base %d", i+1, len(s.c.emph), cap(s.c.emph), len(s.c.frames), kv, s.c.base)
		}
	}
}

func TestSynthTranscribeEdges(t *testing.T) {
	for _, head := range []string{"tdt", "rnnt"} {
		m := synthModel(2, false, head)
		v := synthVAD(3)
		for _, n := range []int{0, 1, 159, 160, 319, 320, 321, 479, 480, 481, 1280, 16000} {
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("%s n=%d: panic %v", head, n, p)
					}
				}()
				toks, err := m.Transcribe(noise(n, 1), v, DefaultSegments, 2)
				if err != nil {
					t.Logf("%s n=%d: %v", head, n, err)
				}
				_ = toks
			}()
		}
	}
}

func TestSynthVADEdges(t *testing.T) {
	v := synthVAD(3)
	for _, n := range []int{0, 1, 159, 160, 161, 319, 320, 321, 641, 16000} {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("n=%d: panic %v", n, p)
				}
			}()
			p, err := v.Probs(noise(n, 1), 2)
			t.Logf("n=%d: %d probs, err %v", n, len(p), err)
		}()
	}
	// ProbsLong pieces with tiny remainders
	defer func(b, c float64) { vadBlockSeconds, vadContextSeconds = b, c }(vadBlockSeconds, vadContextSeconds)
	vadBlockSeconds, vadContextSeconds = 2.5, 3
	block := int(2.5 * 16000)
	for _, n := range []int{block + 2*48000 + 1, 2*block + 1, 2*block + 160, 2*block + 319, 3*block - 1, 3 * block} {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("long n=%d: panic %v", n, p)
				}
			}()
			pcm := noise(n, 7)
			long, err := v.ProbsLong(pcm, 2)
			whole, _ := v.Probs(pcm, 2)
			md := 0.0
			for i := range min(len(long), len(whole)) {
				md = max(md, math.Abs(float64(long[i]-whole[i])))
			}
			t.Logf("long n=%d: %d vs whole %d, max diff %.2g err %v", n, len(long), len(whole), md, err)
		}()
	}
}

func TestSynthDiarEdges(t *testing.T) {
	d := synthDiarizer(4)
	for _, p := range []DiarParams{DiarHighLatency, DiarLowLatency, DiarVeryHighLatency} {
		for _, n := range []int{0, 1, 159, 160, 161, 1280, 1281, 16000, 16000*10 + 77} {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("chunk %d n=%d: panic %v", p.ChunkLen, n, r)
					}
				}()
				out, err := d.Diarize(noise(n, 9), p, 2)
				if err != nil {
					t.Fatal(err)
				}
				// pieces
				s, _ := d.NewStream(p, 2)
				var got [][]float32
				pcm := noise(n, 9)
				for i := 0; i < n; i += 1601 {
					got = append(got, s.Write(pcm[i:min(n, i+1601)])...)
				}
				got = append(got, s.Close()...)
				if len(got) != len(out) {
					t.Errorf("chunk %d n=%d: %d frames in pieces vs %d whole", p.ChunkLen, n, len(got), len(out))
				}
				if n == 16000*10+77 {
					t.Logf("chunk %d n=%d: %d frames (%.2f s)", p.ChunkLen, n, len(out), float64(n)/16000/0.08)
				}
			}()
		}
	}
}

func TestSynthDiarMemory(t *testing.T) {
	d := synthDiarizer(4)
	s, _ := d.NewStream(DiarHighLatency, 2)
	for i := 0; i < 300; i++ {
		s.Write(noise(16000, int64(i)))
		if i%100 == 99 {
			t.Logf("%d s: mel %d, emph %d, cache %d, cachePreds %d, fifo %d", i+1, len(s.mel), len(s.emph), len(s.cache), len(s.cachePreds), len(s.fifo))
		}
	}
}
