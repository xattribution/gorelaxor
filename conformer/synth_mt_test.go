package conformer

import (
	"math"
	"math/rand"
	"testing"
)

func synthMT(t *testing.T) *MultiTalker {
	asr := synthModel(1, true, "rnnt")
	s := synth{rand.New(rand.NewSource(8))}
	D := asr.HP.DModel
	asr.spk = map[int]*spkKernel{0: {w0: s.mat(8, D), b0: s.vec(8, .1), w3: s.mat(D, 8), b3: s.vec(D, .1)}}
	d := synthDiarizer(4)
	// speaker head biased up so speakers turn active
	for i := range d.head.b {
		d.head.b[i] += 3
	}
	mt, err := NewMultiTalker(asr, d, 13, 2)
	if err != nil {
		t.Fatal(err)
	}
	return mt
}

func TestSynthMultiTalkerEdges(t *testing.T) {
	for _, n := range []int{0, 1, 159, 160, 1280, 1281, 16000, 16000*30 + 77} {
		for _, step := range []int{1601, 4000} {
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("n=%d step=%d: panic %v", n, step, p)
					}
				}()
				mt := synthMT(t)
				pcm := noise(n, 3)
				var toks []SpeakerToken
				for i := 0; i < n; i += step {
					toks = append(toks, mt.Write(pcm[i:min(n, i+step)])...)
				}
				toks = append(toks, mt.Close()...)
				if n > 100000 {
					t.Logf("n=%d: %d tokens, %d speaker streams, recent %d, fifo %d cache %d frames %d", n, len(toks), len(mt.spk), len(mt.recent), len(mt.ds.fifo), len(mt.ds.cache), mt.frames)
				}
			}()
		}
	}
}

func TestSynthDiarNaN(t *testing.T) {
	d := synthDiarizer(4)
	s, _ := d.NewStream(DiarHighLatency, 2)
	s.Write(noise(16000*30, 1))
	bad := noise(1600, 2)
	bad[10] = float32(math.Inf(1))
	s.Write(bad)
	nanRows := 0
	total := 0
	for i := 0; i < 120; i++ {
		out := s.Write(noise(16000, int64(100+i)))
		if i >= 60 {
			for _, r := range out {
				total++
				if math.IsNaN(float64(r[0])) {
					nanRows++
				}
			}
		}
	}
	cacheNaN := 0
	for _, r := range s.cache {
		if math.IsNaN(float64(r[0])) {
			cacheNaN++
		}
	}
	t.Logf("one Inf sample: %d of %d frames NaN 60-120 s later; cache rows NaN %d of %d; meanSil NaN %v", nanRows, total, cacheNaN, len(s.cache), math.IsNaN(s.meanSil[0]))
}

func TestSynthVADLongNone(t *testing.T) {
	v := synthVAD(3)
	v.Mel = synthMel("none")
	defer func(b, c float64) { vadBlockSeconds, vadContextSeconds = b, c }(vadBlockSeconds, vadContextSeconds)
	vadBlockSeconds, vadContextSeconds = 1, 0.5
	for n := 32000 + 1; n < 32000+16000*3; n += 997 {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("long n=%d: panic %v", n, p)
				}
			}()
			pcm := noise(n, 7)
			long, _ := v.ProbsLong(pcm, 2)
			whole, _ := v.Probs(pcm, 2)
			md := 0.0
			for i := range min(len(long), len(whole)) {
				md = max(md, math.Abs(float64(long[i]-whole[i])))
			}
			if len(long) != len(whole) || md > 1e-5 {
				t.Errorf("long n=%d: %d vs whole %d, max diff %.2g", n, len(long), len(whole), md)
			}
		}()
	}
}
