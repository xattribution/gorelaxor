package conformer

import (
	"path/filepath"
	"testing"
)

func TestVADMatchesNeMo(t *testing.T) {
	dir := refDir(t)
	nd := nemoDir(t, dir, "vad")
	v, err := LoadVAD(filepath.Join(dir, "vad.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	pcm, _ := loadDump(t, nd, "pcm")
	mel, frames, err := v.Mel.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantMel, ms := loadDump(t, nd, "mel")
	cut := make([]float32, 0, len(mel))
	for j := 0; j < ms[0]; j++ {
		cut = append(cut, wantMel[j*ms[1]:j*ms[1]+frames]...)
	}
	compare(t, "mel", mel, cut, 1e-4)
	probs, err := v.Probs(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := loadDump(t, nd, "prob")
	compare(t, "speech chance", probs, want[:len(probs)], 1e-4)
	regs := v.Speech(probs, DefaultSpeech)
	t.Logf("%d frames of %.0f ms; speech %v", len(probs), v.FrameSeconds*1000, regs)
}

func TestVADLongMatchesWhole(t *testing.T) {
	dir := refDir(t)
	nd := nemoDir(t, dir, "vad")
	v, err := LoadVAD(filepath.Join(dir, "vad.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	pcm, _ := loadDump(t, nd, "pcm")
	whole, _ := v.Probs(pcm, 2)
	defer func(b, c float64) { vadBlockSeconds, vadContextSeconds = b, c }(vadBlockSeconds, vadContextSeconds)
	vadBlockSeconds, vadContextSeconds = 2.5, 3
	long, err := v.ProbsLong(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "pieces", long, whole, 1e-5)
}
