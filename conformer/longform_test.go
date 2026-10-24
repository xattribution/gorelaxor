package conformer

import (
	"math/rand"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSegmentsCutInPauses(t *testing.T) {
	const rate = 100
	// speech 0-12 s, 13-25 s, 26-38 s, 40-70 s (one long run)
	speech := []Region{{0, 12}, {13, 25}, {26, 38}, {40, 70}}
	got := Segments(71*rate, rate, speech, SegmentParams{MaxSeconds: 30, MinSeconds: 5, MinPause: 0.5})
	want := []Segment{{0, 2550}, {2550, 3900}, {3900, 6900}, {6900, 7100}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if s := Segments(20*rate, rate, speech, DefaultSegments); len(s) != 1 || s[0] != (Segment{0, 2000}) {
		t.Fatalf("a short recording should be one piece, got %v", s)
	}
}

func TestLongRecording(t *testing.T) {
	dir := refDir(t)
	nemoDir(t, dir, "parakeet")
	vad, err := LoadVAD(filepath.Join(dir, "vad.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	m := loadModel(t, filepath.Join(dir, "parakeet.gguf"))
	jfk := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	r := rand.New(rand.NewSource(3))
	var pcm []float32
	const copies = 6
	for i := 0; i < copies; i++ {
		pcm = append(pcm, jfk...)
		for j := 0; j < 11200; j++ {
			pcm = append(pcm, float32(r.NormFloat64()*0.002))
		}
	}
	// Pieces of at most 12 s put one copy in each, cut in the 0.7 s
	// pauses between copies. (With 30 s pieces both NeMo and this port
	// drop words when the same sentence repeats verbatim in one piece,
	// so a test made of copies can't use the default.)
	start := time.Now()
	toks, err := m.Transcribe(pcm, vad, SegmentParams{MaxSeconds: 12, MinSeconds: 3, MinPause: 0.6}, 2)
	if err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	text := m.Text(ids(toks))
	t.Logf("%.0f s in %v (%.1fx real time)", float64(len(pcm))/16000, el, float64(len(pcm))/16000/el.Seconds())
	if n := strings.Count(strings.ToLower(text), "what you can do for your country"); n != copies {
		t.Fatalf("found the sentence %d times, want %d: %q", n, copies, text)
	}
	words := m.Words(toks)
	period := float64(len(jfk)+11200) / 16000
	for _, w := range words {
		if strings.HasPrefix(w.Text, "And") {
			off := w.Start - period*float64(int(w.Start/period+0.5))
			if off < 0 || off > 0.6 {
				t.Errorf("%q at %.2f s, %.2f s into its copy", w.Text, w.Start, off)
			}
		}
	}
}
