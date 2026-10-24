package conformer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// NeMo's own outputs (tools/nemo_reference.py), when present, are the
// tighter check: they come from NVIDIA's code in float32.
func nemoDir(t *testing.T, dir, name string) string {
	d := filepath.Join(dir, "nemo-"+name)
	if _, err := os.Stat(d); err != nil {
		t.Skipf("no NeMo reference in %s", d)
	}
	return d
}

func TestNemotronMatchesNeMo(t *testing.T) {
	m, dir := loadRef(t)
	nd := nemoDir(t, dir, "nemotron")
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	mel, frames, err := m.Mel.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantMel, shape := loadDump(t, nd, "mel")
	if shape[1] != frames {
		t.Fatalf("%d mel frames, NeMo %d", frames, shape[1])
	}
	compare(t, "mel", mel, wantMel, 1e-4)
	enc, _ := m.Encode(mel, frames, m.Mel.Valid(len(pcm)), 2)
	want, _ := loadDump(t, nd, "enc")
	compare(t, "encoder", enc, want, 1e-4)
}

func TestStreamingMatchesNeMo(t *testing.T) {
	m, dir := loadRef(t)
	nd := nemoDir(t, dir, "nemotron")
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	for _, R := range []int{13, 6, 1, 0} {
		want, shape := loadDump(t, nd, "stream"+itoa(R)+".enc")
		s, err := m.NewStream(R, 2)
		if err != nil {
			t.Fatal(err)
		}
		var enc []float32
		var toks []Token
		s.onEncoded = func(x []float32) { enc = append(enc, x...) }
		start := time.Now()
		// feed in uneven pieces, as a microphone would
		for i := 0; i < len(pcm); {
			n := min(len(pcm)-i, 1000+i%3777)
			toks = append(toks, s.Write(pcm[i:i+n])...)
			i += n
		}
		toks = append(toks, s.Close()...)
		el := time.Since(start)
		if s.frames != shape[0] {
			t.Errorf("R=%d: %d encoder frames, NeMo %d", R, s.frames, shape[0])
		}
		var meta struct{ Text string }
		readJSON(t, filepath.Join(nd, "stream"+itoa(R)+".json"), &meta)
		got := m.Text(ids(toks))
		t.Logf("R=%d: %q (%.1fx real time)", R, got, float64(len(pcm))/16000/el.Seconds())
		compare(t, "R="+itoa(R)+" encoder", enc, want, 1e-4)
		if got != meta.Text {
			t.Errorf("R=%d: got %q, NeMo %q", R, got, meta.Text)
		}
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func TestParakeetMatchesNeMo(t *testing.T) {
	dir := refDir(t)
	nd := nemoDir(t, dir, "parakeet")
	m := loadModel(t, filepath.Join(dir, "parakeet.gguf"))
	defer runtime.GC()
	pcm := loadWAV(t, filepath.Join(dir, "jfk.wav"))
	mel, frames, err := m.Mel.Compute(pcm, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantMel, shape := loadDump(t, nd, "mel")
	if shape[1] != frames {
		t.Fatalf("%d mel frames, NeMo %d", frames, shape[1])
	}
	compare(t, "mel", mel, wantMel, 1e-4)
	start := time.Now()
	enc, T := m.Encode(mel, frames, m.Mel.Valid(len(pcm)), 2)
	want, _ := loadDump(t, nd, "enc")
	compare(t, "encoder", enc, want, 1e-4)
	toks := m.Decode(enc, T, nil, 0, 2)
	t.Logf("%.1fx real time", float64(len(pcm))/16000/time.Since(start).Seconds())
	var ref struct {
		Text       string
		Tokens     []int
		Timestamps []int
	}
	readJSON(t, filepath.Join(nd, "text.json"), &ref)
	got := m.Text(ids(toks))
	if got != ref.Text {
		t.Fatalf("got %q\nNeMo %q", got, ref.Text)
	}
	for i, tk := range toks {
		if i < len(ref.Timestamps) && tk.Frame != ref.Timestamps[i] {
			t.Errorf("token %d at frame %d, NeMo %d", i, tk.Frame, ref.Timestamps[i])
		}
	}
}
