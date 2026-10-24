package conformer

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestQ8Files checks the files apps actually ship: Q8_0 Nemotron and
// Parakeet give NeMo's words, and their encoders stay close to NeMo's
// float32 output.
func TestQ8Files(t *testing.T) {
	dir := refDir(t)
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	secs := float64(len(pcm)) / 16000
	for _, c := range []struct{ file, ref string }{{"nemotron-q8.gguf", "nemotron"}, {"parakeet-q8.gguf", "parakeet"}} {
		path := filepath.Join(dir, c.file)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skipping %s: %v", c.file, err)
			continue
		}
		nd := nemoDir(t, dir, c.ref)
		m := loadModel(t, path)
		start := time.Now()
		mel, frames, _ := m.Mel.Compute(pcm, 2)
		enc, T := m.Encode(mel, frames, m.Mel.Valid(len(pcm)), 2)
		toks := m.Decode(enc, T, nil, 0, 2)
		el := time.Since(start)
		want, _ := loadDump(t, nd, "enc")
		var num, den float64
		for i := range want {
			d := float64(enc[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		var ref struct{ Text string }
		readJSON(t, filepath.Join(nd, "text.json"), &ref)
		got := m.Text(ids(toks))
		t.Logf("%s: offline %.1fx real time, encoder error %.2g%% of NeMo's (RMS)", c.file, secs/el.Seconds(), 100*math.Sqrt(num/den))
		if got != ref.Text {
			t.Errorf("%s: got %q\nNeMo %q", c.file, got, ref.Text)
		}
		if m.HP.Causal() {
			s, _ := m.NewStream(13, 2)
			start := time.Now()
			toks := append(s.Write(pcm), s.Close()...)
			t.Logf("%s: streaming %.1fx real time, %q", c.file, secs/time.Since(start).Seconds(), m.Text(ids(toks)))
		}
	}
}

// TestQ8Sortformer checks a Q8_0 Sortformer decides who talks when as
// NeMo's float32 one does on nearly every frame.
func TestQ8Sortformer(t *testing.T) {
	dir := refDir(t)
	path := filepath.Join(dir, "sortformer-q8.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("no sortformer-q8.gguf")
	}
	nd := nemoDir(t, dir, "sortformer")
	d, err := LoadDiarizer(path)
	if err != nil {
		t.Fatal(err)
	}
	pcm, _ := loadDump(t, nd, "pcm")
	got, err := d.Diarize(pcm, DiarVeryHighLatency, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := loadDump(t, nd, "preds.high")
	agree, n := 0, 0
	for i, r := range got {
		for j, v := range r {
			n++
			if (v > 0.5) == (want[i*len(r)+j] > 0.5) {
				agree++
			}
		}
	}
	t.Logf("speaker decisions agree with NeMo on %.2f%% of frame-speaker pairs", 100*float64(agree)/float64(n))
	if float64(agree) < 0.99*float64(n) {
		t.Fatalf("only %d of %d agree", agree, n)
	}
}
