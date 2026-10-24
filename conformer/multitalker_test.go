package conformer

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMultiTalkerMatchesNeMo(t *testing.T) {
	dir := refDir(t)
	nd := nemoDir(t, dir, "multitalker")
	path := filepath.Join(dir, "multitalker.gguf")
	if _, err := os.Stat(path); err != nil {
		path = filepath.Join(dir, "multitalker-q8.gguf")
	}
	asr := loadModel(t, path)
	diar, err := LoadDiarizer(filepath.Join(dir, "sortformer.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	pcm := loadWAV(t, filepath.Join(dir, "multitalker-2spk-mix.wav"))
	mt, err := NewMultiTalker(asr, diar, 13, 2)
	if err != nil {
		t.Fatal(err)
	}
	var diarAll [][]float32
	mt.onDiar = func(p [][]float32) { diarAll = append(diarAll, p...) }
	start := time.Now()
	var toks []SpeakerToken
	for i := 0; i < len(pcm); i += 4000 {
		toks = append(toks, mt.Write(pcm[i:min(len(pcm), i+4000)])...)
	}
	toks = append(toks, mt.Close()...)
	t.Logf("%.1fx real time with %s", float64(len(pcm))/16000/time.Since(start).Seconds(), filepath.Base(path))
	got := mt.SpeakerText(toks)
	wantDiar, ds := loadDump(t, nd, "diar")
	flat := make([]float32, 0, len(diarAll)*4)
	for _, r := range diarAll {
		flat = append(flat, r...)
	}
	t.Logf("diarizer frames %d, NeMo %v", len(diarAll), ds)
	if len(flat) != len(wantDiar) {
		t.Fatalf("%d speaker chances, NeMo %d", len(flat), len(wantDiar))
	}
	compare(t, "speaker chances", flat, wantDiar, 1e-4)
	var ref struct {
		Texts map[string]string
	}
	readJSON(t, filepath.Join(nd, "result.json"), &ref)
	for k, want := range ref.Texts {
		spk := int(k[0] - '0')
		t.Logf("speaker %d: %q", spk, got[spk])
		if got[spk] != want {
			t.Errorf("speaker %d: got %q\nNeMo %q", spk, got[spk], want)
		}
	}
	if len(got) != len(ref.Texts) {
		t.Errorf("%d speakers, NeMo %d", len(got), len(ref.Texts))
	}
}
