package conformer

import (
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

var refModel *Model

// loadModel loads a test model, dropping the cached Nemotron first so
// two F32 models are never in memory together.
func loadModel(t *testing.T, path string) *Model {
	t.Helper()
	refModel = nil
	runtime.GC()
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func loadRef(t *testing.T) (*Model, string) {
	dir := refDir(t)
	if refModel == nil {
		runtime.GC()
		m, err := Load(filepath.Join(dir, "nemotron.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		refModel = m
	}
	return refModel, dir
}

// encTol allows for transcribe.cpp's own rounding: ggml's conv_2d
// rounds the pointwise subsampling convolutions' inputs and weights to
// float16 (about 2e-4 relative at the pre-encoder output), and the gap
// grows to a few 1e-3 by the last block. Rounding the same two
// convolutions to float16 in Go brings the pre-encoder gap to 1e-5.
const encTol = 5e-3

func TestEncoderMatchesReference(t *testing.T) {
	m, dir := loadRef(t)
	dump := cppDump(t, dir)
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	mel, frames, err := m.Mel.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{"pre_encode": true, "block.0.ff1": true, "block.0.attn": true, "block.0.conv": true,
		"block.0.ff2": true, "block.0.out": true, "block.1.out": true, "block.12.out": true, "block.23.out": true}
	start := time.Now()
	out, T := m.encode(mel, frames, m.Mel.Valid(len(pcm)), 2, func(name string, x []float32) {
		if !want[name] {
			return
		}
		ref, _ := loadDump(t, dump, "enc."+name+map[bool]string{true: ".out", false: ""}[name == "pre_encode"])
		compare(t, name, x, ref, encTol)
	})
	t.Logf("encoder: %d frames in %v", T, time.Since(start))
	ref, _ := loadDump(t, dump, "enc.final")
	compare(t, "final", out, ref, encTol)
}

func TestTranscribeJFK(t *testing.T) {
	m, dir := loadRef(t)
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	start := time.Now()
	mel, frames, err := m.Mel.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	enc, T := m.Encode(mel, frames, m.Mel.Valid(len(pcm)), 2)
	tenc := time.Since(start)
	toks := m.Decode(enc, T, nil, 0, 2)
	text := m.Text(ids(toks))
	el := time.Since(start)
	t.Logf("%q\nencode %v, total %v, %.1fx real time", text, tenc, el, float64(len(pcm))/16000/el.Seconds())
	const want = "And so, my fellow Americans ask not what your country can do for you. Ask what you can do for your country."
	if text != want {
		t.Fatalf("got %q\nwant %q", text, want)
	}
}

func ids(ts []Token) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.ID
	}
	return out
}

func TestDecoderFirstStep(t *testing.T) {
	m, dir := loadRef(t)
	dump := cppDump(t, dir)
	enc, shape := loadDump(t, dump, "dec.enc_out")
	st, next := m.NewDecoderState(), m.NewDecoderState()
	H, J := m.HP.PredHidden, m.HP.JointHidden
	pred := m.predict(st, next, make([]float32, 4*H), make([]float32, 4*H))
	for l := range m.lstm {
		h, _ := loadDump(t, dump, "dec.lstm."+itoa(l)+".h.0")
		c, _ := loadDump(t, dump, "dec.lstm."+itoa(l)+".c.0")
		compare(t, "lstm h", next.h[l], h, 1e-4)
		compare(t, "lstm c", next.c[l], c, 1e-4)
	}
	encProj := make([]float32, J)
	kern.Linear(encProj, enc[:shape[1]], 1, m.jEnc, m.jEncB, 1)
	logits := make([]float32, m.jOut.Rows)
	m.joint(encProj, pred, make([]float32, J), make([]float32, J), logits)
	mx := logits[argmax(logits)]
	var s float64
	for _, v := range logits {
		s += math.Exp(float64(v - mx))
	}
	for i := range logits {
		logits[i] -= mx + float32(math.Log(s))
	}
	want, _ := loadDump(t, dump, "dec.joint.0")
	compare(t, "joint", logits, want, 1e-4)
}
