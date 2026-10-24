package pockettts

import (
	"errors"
	"fmt"
	"math"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// Voice is a speaker: the flow LM's state after hearing a short clip of
// them (each layer's keys and values). A Voice can be shared by
// goroutines; speaking copies it.
type Voice struct {
	st    *tstate
	model string // the source SHA-256 of the model it was made for
}

// Positions is how many transformer positions the voice prompt holds
// (12.5 a second of clip, plus one).
func (v *Voice) Positions() int { return v.st.pos }

// LoadVoice reads a preset voice file made by tools/convert_pocket_tts.py
// for this model.
func (m *Model) LoadVoice(path string) (*Voice, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	if f.String("general.architecture", "") != "pockettts-voice" {
		return nil, fmt.Errorf("pockettts: %s isn't a Pocket TTS voice", path)
	}
	if f.String("pockettts.voice.model.sha256", "") != m.sha {
		return nil, fmt.Errorf("pockettts: %s was made for a different model file", path)
	}
	nl, n := f.Int("pockettts.voice.layers", 0), f.Int("pockettts.voice.positions", 0)
	d := m.lm.tf.d
	if nl != len(m.lm.tf.layers) || n < 1 || n > 4096 {
		return nil, fmt.Errorf("pockettts: %s holds %d layers of %d positions", path, nl, n)
	}
	st := m.lm.tf.newState()
	for i := range st.c {
		for j, p := range []*[]float32{&st.c[i].k, &st.c[i].v} {
			name := fmt.Sprintf("voice.%d.%c", i, "kv"[j])
			t, ok := f.Tensors[name]
			if !ok || len(t.Dims) != 2 || t.Dims[0] != d || t.Dims[1] != n {
				return nil, fmt.Errorf("pockettts: %s has no %d×%d %s", path, n, d, name)
			}
			v, err := f.Float32(name)
			if err != nil {
				return nil, err
			}
			for _, x := range v {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					return nil, fmt.Errorf("pockettts: %s has values that aren't numbers", path)
				}
			}
			*p = v
		}
	}
	st.pos = n
	return &Voice{st: st, model: m.sha}, nil
}

// MaxCloneSeconds is the longest clip CloneVoice hears; Kyutai suggests
// 5 to 30 seconds of one speaker with little noise.
const MaxCloneSeconds = 30

// CloneVoice makes a voice from a clip of one speaker (mono samples at
// rate Hz, any rate): resampled to 24 kHz, its trailing silence cut to
// 80 ms (Kyutai's end_on_pause), then Mimi's encoder and the flow LM.
// Clips longer than MaxCloneSeconds are cut.
func (m *Model) CloneVoice(pcm []float32, rate, threads int) (*Voice, error) {
	if m.enc == nil {
		return nil, errors.New("pockettts: this model file has no voice cloning")
	}
	if rate < 1000 || rate > 768000 {
		return nil, fmt.Errorf("pockettts: a sample rate of %d Hz", rate)
	}
	if lim := MaxCloneSeconds * rate; len(pcm) > lim {
		pcm = pcm[:lim]
	}
	clean := make([]float32, len(pcm))
	for i, v := range pcm {
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			clean[i] = v
		}
	}
	x := resamplePoly(clean, m.cfg.sampleRate, rate)
	x = endOnPause(x, m.cfg.sampleRate)
	if len(x) == 0 {
		return nil, errors.New("pockettts: the clip is empty")
	}
	cond, _ := m.cloneCond(x, threads)
	d := m.lm.tf.d
	n := len(cond) / d
	in := make([]float32, (n+1)*d)
	copy(in, m.lm.bosVoice)
	copy(in[d:], cond)
	st := m.lm.tf.newState()
	m.lm.run(in, n+1, st, &scratch{}, threads)
	return &Voice{st: st, model: m.sha}, nil
}

// cloneCond is Mimi's encoding of x (24 kHz) projected into the flow
// LM: the conditioning a voice prompt feeds it, [frames][d], and the
// latents it came from.
func (m *Model) cloneCond(x []float32, threads int) (cond, lat []float32) {
	frame := m.cfg.frameSamples
	x = append(x[:len(x):len(x)], make([]float32, (frame-len(x)%frame)%frame)...)
	st := m.enc.newState()
	ldim := m.lm.speaker.Cols
	const piece = 12 // frames encoded at a time
	for i := 0; i < len(x); i += piece * frame {
		lat = append(lat, m.enc.encode(st, x[i:min(len(x), i+piece*frame)], threads)...)
	}
	n := len(lat) / ldim
	cond = make([]float32, n*m.lm.tf.d)
	kern.Linear(cond, lat, n, m.lm.speaker, nil, threads)
	return cond, lat
}

// endOnPause is Kyutai's end_on_pause: cut the trailing silence (20 ms
// frames more than 35 dB below the loudest), fade the last 20 ms out and
// add 80 ms of silence.
func endOnPause(x []float32, rate int) []float32 {
	frame := max(1, rate/50)
	n := len(x) / frame
	if n == 0 {
		return x
	}
	db := make([]float64, n)
	top := math.Inf(-1)
	for f := range db {
		var s float32
		for _, v := range x[f*frame : (f+1)*frame] {
			s += v * v
		}
		rms := math.Sqrt(float64(s / float32(frame)))
		db[f] = 20 * math.Log10(rms+1e-12)
		top = max(top, db[f])
	}
	last := 0
	for f, v := range db {
		if v > top-35 {
			last = f
		}
	}
	end := (last + 1) * frame
	out := make([]float32, end, end+rate*2/25)
	copy(out, x[:end])
	fade := min(rate/50, end)
	for i := 0; i < fade; i++ {
		w := float32(1)
		if fade > 1 {
			w = float32(1 - float64(i)/float64(fade-1))
		}
		out[end-fade+i] *= w
	}
	return append(out, make([]float32, rate*2/25)...)
}

// resamplePoly is scipy.signal.resample_poly(x, to/g, from/g), the
// resampler Kyutai's convert_audio uses: a Kaiser-windowed (β = 5) FIR
// low-pass of 20·max(up, down) + 1 taps, computed in float32 as scipy
// does for float32 audio.
func resamplePoly(x []float32, to, from int) []float32 {
	g := gcd(to, from)
	up, down := to/g, from/g
	if up == 1 && down == 1 {
		return append([]float32(nil), x...)
	}
	nIn := len(x)
	nOut := nIn * up / down
	if nIn*up%down != 0 {
		nOut++
	}
	maxRate := max(up, down)
	half := 10 * maxRate
	h := firwinKaiser(2*half+1, 1/float64(maxRate), 5)
	pre := down - half%down
	post := 0
	for outputLen(len(h)+pre+post, nIn, up, down) < nOut+(half+pre)/down {
		post++
	}
	hp := make([]float32, pre+len(h)+post)
	for i, v := range h {
		hp[pre+i] = float32(v) * float32(up)
	}
	skip := (half + pre) / down
	out := make([]float32, nOut)
	for o := range out {
		base := (o + skip) * down // position in the upsampled input
		var acc float32
		for j := base % up; j < len(hp); j += up {
			k := (base - j) / up
			if k < 0 {
				break
			}
			if k < nIn {
				acc += hp[j] * x[k]
			}
		}
		out[o] = acc
	}
	return out
}

func outputLen(lenH, nIn, up, down int) int {
	n := (nIn + (lenH+(up-lenH%up)%up)/up - 1) * up
	need := n / down
	if n%down > 0 {
		need++
	}
	return need
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// firwinKaiser is scipy.signal.firwin(n, cutoff, window=("kaiser", beta)):
// a windowed sinc low-pass scaled to unit gain at 0 Hz.
func firwinKaiser(n int, cutoff, beta float64) []float64 {
	h := make([]float64, n)
	alpha := float64(n-1) / 2
	var sum float64
	for i := range h {
		m := float64(i) - alpha
		x := cutoff * m
		s := 1.0
		if x != 0 {
			s = math.Sin(math.Pi*x) / (math.Pi * x)
		}
		r := (float64(i) - alpha) / alpha
		w := besselI0(beta*math.Sqrt(1-r*r)) / besselI0(beta)
		h[i] = cutoff * s * w
		sum += h[i]
	}
	for i := range h {
		h[i] /= sum
	}
	return h
}

// besselI0 is the modified Bessel function of the first kind, order 0.
func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 200; k++ {
		term *= (x / 2) / float64(k)
		sum += term * term
		if term*term < 1e-18*sum {
			break
		}
	}
	return sum
}
