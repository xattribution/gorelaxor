package conformer

import (
	"errors"
	"fmt"
	"math"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// VAD is NVIDIA's MarbleNet frame voice-activity model
// (Frame_VAD_Multilingual_MarbleNet_v2.0): 1-D separable convolution
// blocks over log-mel features, giving the chance of speech every
// 20 ms. It is a translation of NeMo's ConvASREncoder (JasperBlock) and
// MultiLayerPerceptron, with batch norms folded into the convolutions
// by tools/convert_nemo.py.
type VAD struct {
	Mel    *Mel
	padTo  int
	blocks []vadBlock
	dec    []vadLinear
	// FrameSeconds is the time between outputs.
	FrameSeconds float64
}

type vadLinear struct {
	w *kern.Weight
	b []float32
}

type vadRep struct {
	dw []float32 // depthwise kernel [k][in], nil for a plain 1x1 block
	pw vadLinear
}

type vadBlock struct {
	in, out                  int
	kernel, stride, dilation int
	reps                     []vadRep
	res                      *vadLinear
}

// LoadVAD opens a converted MarbleNet file.
func LoadVAD(path string) (*VAD, error) {
	var v *VAD
	err := checked(func() error {
		var err error
		if v, err = loadVAD(path); err != nil {
			return err
		}
		_, err = v.Probs(make([]float32, v.Mel.cfg.SampleRate), 1)
		return err
	})
	if err != nil {
		return nil, err
	}
	return v, nil
}

func loadVAD(path string) (*VAD, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	if a := f.String("general.architecture", ""); a != "marblenet" {
		return nil, fmt.Errorf("conformer: %q is not a MarbleNet file", a)
	}
	v := &VAD{padTo: f.Int("stt.frontend.pad_to", 0)}
	if v.Mel, err = NewMel(melConfig(f)); err != nil {
		return nil, err
	}
	l := &loader{f: f}
	in := v.Mel.cfg.NumMels
	stride := 1
	n := f.Int("stt.marblenet.n_blocks", 0)
	if n == 0 {
		return nil, errors.New("conformer: the VAD file has no blocks")
	}
	for i := 0; i < n; i++ {
		p := fmt.Sprintf("stt.marblenet.block.%d.", i)
		b := vadBlock{
			in: in, out: f.Int(p+"filters", 0),
			kernel: f.Int(p+"kernel", 1), stride: f.Int(p+"stride", 1), dilation: f.Int(p+"dilation", 1),
		}
		if b.stride > 1 && b.dilation > 1 {
			return nil, errors.New("conformer: a VAD block can't both stride and dilate")
		}
		sep := f.Bool(p+"separable", false)
		if !sep && b.kernel != 1 {
			return nil, errors.New("conformer: only separable VAD blocks may have kernels wider than 1")
		}
		ch := in
		for r := 0; r < f.Int(p+"repeat", 1); r++ {
			t := fmt.Sprintf("vad.blocks.%d.%d.", i, r)
			rep := vadRep{pw: vadLinear{w: l.mat(t+"pw.weight", false), b: l.vec(t+"pw.bias", false)}}
			if sep {
				rep.dw = transposeTaps(l.vec(t+"dw.weight", false), ch, b.kernel)
			}
			b.reps = append(b.reps, rep)
			ch = b.out
		}
		if f.Bool(p+"residual", false) {
			b.res = &vadLinear{w: l.mat(fmt.Sprintf("vad.blocks.%d.res.weight", i), false), b: l.vec(fmt.Sprintf("vad.blocks.%d.res.bias", i), false)}
		}
		stride *= b.stride
		v.blocks = append(v.blocks, b)
		in = b.out
	}
	for i := 0; i < f.Int("stt.marblenet.decoder_layers", 1); i++ {
		v.dec = append(v.dec, vadLinear{w: l.mat(fmt.Sprintf("vad.dec.%d.weight", i), false), b: l.vec(fmt.Sprintf("vad.dec.%d.bias", i), false)})
	}
	if l.err != nil {
		return nil, l.err
	}
	v.FrameSeconds = float64(v.Mel.cfg.Hop*stride) / float64(v.Mel.cfg.SampleRate)
	return v, nil
}

// depthwise1D convolves each channel of x [T][c] with its kernel
// [k][c], NeMo's "same" padding of dilation·(k-1)/2 on each side.
func depthwise1D(x []float32, T, c int, w []float32, k, stride, dil int, threads int) ([]float32, int) {
	pad := dil * (k - 1) / 2
	To := (T+2*pad-dil*(k-1)-1)/stride + 1
	out := make([]float32, To*c)
	kern.Parallel(To, threads, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			o := out[t*c : (t+1)*c]
			for j := 0; j < k; j++ {
				ti := t*stride + j*dil - pad
				if ti < 0 || ti >= T {
					continue
				}
				src := x[ti*c : (ti+1)*c]
				wj := w[j*c : (j+1)*c]
				for i := range o {
					o[i] += wj[i] * src[i]
				}
			}
		}
	})
	return out, To
}

// Probs returns the chance of speech for each 20 ms frame of 16 kHz
// mono audio.
func (v *VAD) Probs(pcm []float32, threads int) ([]float32, error) {
	pcm = finite(pcm)
	mel, frames, err := v.Mel.Compute(pcm, threads)
	if err != nil {
		return nil, err
	}
	valid := v.Mel.Valid(len(pcm))
	if v.padTo > 1 && frames%v.padTo != 0 {
		nf := frames + v.padTo - frames%v.padTo
		padded := make([]float32, v.Mel.cfg.NumMels*nf)
		for j := 0; j < v.Mel.cfg.NumMels; j++ {
			copy(padded[j*nf:], mel[j*frames:(j+1)*frames])
		}
		mel, frames = padded, nf
	}
	logits, T, L := v.forward(mel, frames, valid, threads)
	out := make([]float32, min(T, L))
	nc := len(logits) / T
	for t := range out {
		row := logits[t*nc : (t+1)*nc]
		out[t] = softmaxAt(row, nc-1)
	}
	return out, nil
}

// vadBlock and vadContext size ProbsLong's pieces, in seconds.
var vadBlockSeconds, vadContextSeconds = 600.0, 4.0

// ProbsLong is Probs for recordings of any length: it works through
// ten-minute pieces, each with four seconds of audio either side (more
// than the network's 1.5 s reach), so the result equals Probs while
// memory stays bounded.
func (v *VAD) ProbsLong(pcm []float32, threads int) ([]float32, error) {
	hop := int(v.FrameSeconds*float64(v.Mel.cfg.SampleRate) + 0.5)
	block := int(vadBlockSeconds*float64(v.Mel.cfg.SampleRate)) / hop * hop
	ctx := int(vadContextSeconds*float64(v.Mel.cfg.SampleRate)) / hop * hop
	if len(pcm) <= block+2*ctx {
		return v.Probs(pcm, threads)
	}
	var out []float32
	for o0 := 0; o0 < len(pcm); o0 += block {
		o1 := min(len(pcm), o0+block)
		s0, s1 := max(0, o0-ctx), min(len(pcm), o1+ctx)
		p, err := v.Probs(pcm[s0:s1], threads)
		if err != nil {
			return nil, err
		}
		first := (o0 - s0) / hop
		last := first + (o1-o0+hop-1)/hop
		if o1 == len(pcm) {
			last = len(p)
		}
		out = append(out, p[first:min(last, len(p))]...)
	}
	return out, nil
}

// forward returns the decoder's logits [T][classes] for log-mel input
// [mels][frames], and the valid output length.
func (v *VAD) forward(mel []float32, frames, valid, threads int) ([]float32, int, int) {
	nm := v.Mel.cfg.NumMels
	x := make([]float32, frames*nm)
	for f := 0; f < nm; f++ {
		for t := 0; t < frames; t++ {
			x[t*nm+f] = mel[f*frames+t]
		}
	}
	T, L := frames, valid
	for bi := range v.blocks {
		b := &v.blocks[bi]
		in, inT, inL := x, T, L
		c := b.in
		for r, rep := range b.reps {
			if rep.dw != nil {
				maskRows(x, T, L, c)
				x, T = depthwise1D(x, T, c, rep.dw, b.kernel, b.stride, b.dilation, threads)
				pad := b.dilation * (b.kernel - 1) / 2
				L = (L+2*pad-b.dilation*(b.kernel-1)-1)/b.stride + 1
			}
			maskRows(x, T, L, c)
			y := make([]float32, T*b.out)
			kern.Linear(y, x, T, rep.pw.w, rep.pw.b, threads)
			x, c = y, b.out
			if r < len(b.reps)-1 {
				relu(x)
			}
		}
		if b.res != nil {
			maskRows(in, inT, inL, b.in)
			y := make([]float32, inT*b.out)
			kern.Linear(y, in, inT, b.res.w, b.res.b, threads)
			kern.Add(x, y)
		}
		relu(x)
	}
	for i, d := range v.dec {
		y := make([]float32, T*d.w.Rows)
		kern.Linear(y, x, T, d.w, d.b, threads)
		x = y
		if i < len(v.dec)-1 {
			relu(x)
		}
	}
	return x, T, L
}

// Region is a stretch of speech, in seconds.
type Region struct{ Start, End float64 }

// SpeechParams tune how frame chances become regions.
type SpeechParams struct {
	Onset, Offset float64 // start speech at Onset or above, end below Offset
	MinSpeech     float64 // drop regions shorter than this (seconds)
	MinGap        float64 // join regions closer than this (seconds)
	PadOnset      float64 // widen each region by these (seconds)
	PadOffset     float64
}

// DefaultSpeech are NeMo's frame-VAD post-processing defaults
// (examples/asr/conf/vad/frame_vad_infer_postprocess.yaml).
var DefaultSpeech = SpeechParams{Onset: 0.3, Offset: 0.3, MinSpeech: 0.2, MinGap: 0.2, PadOnset: 0.2, PadOffset: 0.2}

// Speech turns frame chances into speech regions the way NeMo's
// binarization and filtering do (filter_speech_first): regions start
// at Onset and end below Offset, are widened by the pads, then regions
// shorter than MinSpeech are dropped and gaps shorter than MinGap
// closed.
func (v *VAD) Speech(probs []float32, p SpeechParams) []Region {
	step := v.FrameSeconds
	var regs []Region
	on := false
	var start float64
	for i, pr := range probs {
		t := float64(i) * step
		switch {
		case !on && float64(pr) >= p.Onset:
			on, start = true, t
		case on && float64(pr) < p.Offset:
			on = false
			regs = append(regs, Region{start, t})
		}
	}
	if on {
		regs = append(regs, Region{start, float64(len(probs)) * step})
	}
	end := float64(len(probs)) * step
	var out []Region
	for _, r := range regs {
		r.Start = math.Max(0, r.Start-p.PadOnset)
		r.End = math.Min(end, r.End+p.PadOffset)
		if n := len(out); n > 0 && r.Start <= out[n-1].End {
			out[n-1].End = math.Max(out[n-1].End, r.End)
			continue
		}
		out = append(out, r)
	}
	kept := out[:0]
	for _, r := range out {
		if r.End-r.Start >= p.MinSpeech {
			kept = append(kept, r)
		}
	}
	var joined []Region
	for _, r := range kept {
		if n := len(joined); n > 0 && r.Start-joined[n-1].End < p.MinGap {
			joined[n-1].End = r.End
			continue
		}
		joined = append(joined, r)
	}
	return joined
}
