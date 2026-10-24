package conformer

import (
	"path/filepath"
	"testing"
)

func nemotronMel() MelConfig {
	return MelConfig{SampleRate: 16000, NFFT: 512, WinLength: 400, Hop: 160, NumMels: 128,
		FMax: 8000, PreEmphasis: 0.97, Normalize: "none"}
}

func TestMelMatchesReference(t *testing.T) {
	dir := refDir(t)
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	m, err := NewMel(nemotronMel())
	if err != nil {
		t.Fatal(err)
	}
	got, frames, err := m.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, shape := loadDump(t, cppDump(t, dir), "enc.mel.in")
	if frames != shape[1] {
		t.Fatalf("%d frames, want %d", frames, shape[1])
	}
	compare(t, "mel", got, want, 1e-4)
}
