package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/xattribution/gorelaxor/internal/kern"
	"github.com/xattribution/gorelaxor/whisper"
)

// The parity tests compare against Kyutai's own PyTorch code, run by
// tools/pocket_tts_reference.py (its comment says what it saves). They
// skip unless GORELAXOR_POCKET_REF names the folder holding:
//
//	pocket-tts-f32.gguf, pocket-tts-q8.gguf    tools/convert_pocket_tts.py --type f32 / q8_0
//	voices/voice-<name>.gguf                    --voice-type f32: alba, marius, javert
//	voices-f16/voice-<name>.gguf                the default f16 voices
//	casual.wav                                  kyutai/tts-voices alba-mackenna/casual.wav
//	tokens.json                                 pocket_tts_reference.py tokens
//	alba-short/, marius-long/, javert-numbers/, clone-alba/   pocket_tts_reference.py speak
//
// The speech-recognition round trip also needs Parakeet (parakeet-q8.gguf
// from conformer's GORELAXOR_CONFORMER_REF, or in GORELAXOR_POCKET_REF).
func refDir(t *testing.T) string {
	d := os.Getenv("GORELAXOR_POCKET_REF")
	if d == "" {
		t.Skip("GORELAXOR_POCKET_REF is not set")
	}
	return d
}

var models sync.Map

func loadModel(t *testing.T, path string) *Model {
	t.Helper()
	if m, ok := models.Load(path); ok {
		return m.(*Model)
	}
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	models.Store(path, m)
	return m
}

func loadDump(t *testing.T, dir, name string) ([]float32, []int) {
	t.Helper()
	meta, err := os.ReadFile(filepath.Join(dir, name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var m struct{ Shape []int }
	if err := json.Unmarshal(meta, &m); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, name+".f32"))
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), kern.Float32s(b)...), m.Shape
}

func readWAV(t *testing.T, path string) ([]float32, int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	pcm, rate, err := whisper.DecodeWAVRate(f)
	if err != nil {
		t.Fatal(err)
	}
	return pcm, rate
}

// diff is the largest absolute difference relative to the reference's
// largest magnitude, and the RMS difference relative to its RMS.
type diff struct{ md, mx, se, ss float64 }

func (d *diff) add(got, want []float32) {
	for i := range want {
		w := float64(want[i])
		e := math.Abs(float64(got[i]) - w)
		if math.IsNaN(e) {
			e = math.Inf(1)
		}
		d.md, d.mx = max(d.md, e), max(d.mx, math.Abs(w))
		d.se += e * e
		d.ss += w * w
	}
}

func (d *diff) rel() float64 { return d.md / max(d.mx, 1e-12) }
func (d *diff) rms() float64 { return math.Sqrt(d.se / max(d.ss, 1e-30)) }

func (d *diff) check(t *testing.T, what string, tol float64) {
	t.Helper()
	t.Logf("%s: largest difference %.3g relative, RMS %.3g", what, d.rel(), d.rms())
	if tol > 0 && !(d.rel() <= tol) {
		t.Errorf("%s: relative difference %.3g above %.3g", what, d.rel(), tol)
	}
}

type refCase struct {
	Text   string
	Voice  string
	Chunks []struct {
		Text           string
		IDs            []int
		Frames         int
		FirstCall      int `json:"first_call"`
		FramesAfterEOS int `json:"frames_after_eos"`
	}
	Calls   int
	Samples int
}

type refRun struct {
	c                         refCase
	lmOut, noise, latent, eos []float32
	pcm                       []float32
}

func loadCase(t *testing.T, dir string) *refRun {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	r := &refRun{}
	if err := json.Unmarshal(b, &r.c); err != nil {
		t.Fatal(err)
	}
	r.lmOut, _ = loadDump(t, dir, "lm_out")
	r.noise, _ = loadDump(t, dir, "noise")
	r.latent, _ = loadDump(t, dir, "latent")
	r.eos, _ = loadDump(t, dir, "eos")
	r.pcm, _ = loadDump(t, dir, "pcm")
	return r
}

// result is what a Go run of a reference case made, and how far it was
// from Kyutai's.
type result struct {
	pcm                []float32
	lmOut, latent, eos diff
	first              diff // the first text prompt's output
	calls              int
}

// runCase speaks a case's text with the reference's noise and compares
// every flow head call on the way. With force, each frame's latent is
// replaced by the reference's before it's decoded and fed back (teacher
// forcing), so differences can't grow along the way.
func runCase(t *testing.T, m *Model, v *Voice, r *refRun, force bool) *result {
	t.Helper()
	d := m.lm.tf.d
	ld := m.lm.head.ldim
	res := &result{}
	call := 0
	opt := Options{Threads: 2}
	opt.noise = func(dst []float32) {
		if call < r.c.Calls {
			copy(dst, r.noise[call*ld:(call+1)*ld])
		}
	}
	opt.trace = func(out, noise, latent []float32, eos float32, prompt bool) {
		if call >= r.c.Calls {
			call++
			return
		}
		res.lmOut.add(out, r.lmOut[call*d:(call+1)*d])
		if call == 0 {
			res.first.add(out, r.lmOut[:d])
		}
		if !prompt {
			res.latent.add(latent, r.latent[call*ld:(call+1)*ld])
			res.eos.add([]float32{eos}, r.eos[call:call+1])
			if force {
				copy(latent, r.latent[call*ld:(call+1)*ld])
			}
		}
		call++
	}
	pcm, err := m.Synthesize(r.c.Text, v, opt)
	if err != nil {
		t.Fatal(err)
	}
	res.pcm, res.calls = pcm, call
	return res
}

func voiceFor(t *testing.T, m *Model, dir, name string) *Voice {
	t.Helper()
	v, err := m.LoadVoice(filepath.Join(dir, "voice-"+name+".gguf"))
	if err != nil {
		t.Fatal(err)
	}
	return v
}

var refCases = []string{"alba-short", "marius-long", "javert-numbers", "clone-alba"}

// voice is the case's voice: a preset from vdir, or casual.wav cloned.
func caseVoice(t *testing.T, m *Model, ref, vdir string, r *refRun, check bool) *Voice {
	t.Helper()
	if r.c.Voice != "clone:casual.wav" {
		return voiceFor(t, m, vdir, r.c.Voice)
	}
	pcm, rate := readWAV(t, filepath.Join(ref, "casual.wav"))
	if check {
		dir := filepath.Join(ref, "clone-alba")
		x := endOnPause(resamplePoly(pcm, 24000, rate), 24000)
		want, _ := loadDump(t, dir, "clone_pcm")
		if len(x) != len(want) {
			t.Fatalf("prepared clip: %d samples, want %d", len(x), len(want))
		}
		var dp diff
		dp.add(x, want)
		dp.check(t, "clone: resampled clip", 1e-5)
		cond, lat := m.cloneCond(want, 2)
		var dl, dc diff
		wl, _ := loadDump(t, dir, "clone_latent")
		wc, _ := loadDump(t, dir, "clone_cond")
		if len(lat) != len(wl) || len(cond) != len(wc) {
			t.Fatalf("clone: %d latents, want %d", len(lat)/32, len(wl)/32)
		}
		dl.add(lat, wl)
		dl.check(t, "clone: Mimi encoder latents", 1e-4)
		dc.add(cond, wc)
		dc.check(t, "clone: conditioning", 1e-4)
	}
	v, err := m.CloneVoice(pcm, rate, 2)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestReferenceF32(t *testing.T) {
	ref := refDir(t)
	m := loadModel(t, filepath.Join(ref, "pocket-tts-f32.gguf"))
	for _, name := range refCases {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(ref, name)
			r := loadCase(t, dir)
			chunks, err := m.chunks(r.c.Text, m.cfg.maxTokens)
			if err != nil {
				t.Fatal(err)
			}
			if len(chunks) != len(r.c.Chunks) {
				t.Fatalf("%d chunks, want %d", len(chunks), len(r.c.Chunks))
			}
			for i, c := range chunks {
				p, _, _ := m.cfg.prepareText(c)
				if p != r.c.Chunks[i].Text || !equalInts(m.tok.encode(p), r.c.Chunks[i].IDs) {
					t.Fatalf("chunk %d is %q, want %q", i, p, r.c.Chunks[i].Text)
				}
			}
			v := caseVoice(t, m, ref, filepath.Join(ref, "voices"), r, true)
			for _, force := range []bool{true, false} {
				res := runCase(t, m, v, r, force)
				if res.calls != r.c.Calls || len(res.pcm) != len(r.pcm) {
					t.Fatalf("%d flow calls and %d samples, want %d and %d", res.calls, len(res.pcm), r.c.Calls, len(r.pcm))
				}
				var dp diff
				dp.add(res.pcm, r.pcm)
				if force {
					res.first.check(t, "teacher forced: text prompt output", 1e-4)
					res.lmOut.check(t, "teacher forced: transformer output", 1e-4)
					res.latent.check(t, "teacher forced: latents", 1e-4)
					res.eos.check(t, "teacher forced: end-of-speech logit", 1e-4)
					dp.check(t, "teacher forced: audio", 1e-4)
					continue
				}
				res.latent.check(t, "free running: latents", 1e-2)
				dp.check(t, "free running: audio", 5e-2)
			}
		})
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestTokenizerReference(t *testing.T) {
	ref := refDir(t)
	m := loadModel(t, filepath.Join(ref, "pocket-tts-q8.gguf"))
	b, err := os.ReadFile(filepath.Join(ref, "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var want struct {
		Tokens []struct {
			Text, Decoded string
			IDs           []int
		}
		Chunks []struct {
			Text           string
			Chunks         []string
			Prepared       []string
			FramesAfterEOS []int `json:"frames_after_eos"`
			IDs            [][]int
		}
	}
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	for _, c := range want.Tokens {
		got := m.tok.encode(c.Text)
		if !equalInts(got, c.IDs) {
			t.Errorf("%q: ids %v, want %v", c.Text, got, c.IDs)
		}
		if d := m.tok.decode(c.IDs); d != c.Decoded {
			t.Errorf("%q: decoded %q, want %q", c.Text, d, c.Decoded)
		}
	}
	for _, c := range want.Chunks {
		got, err := m.chunks(c.Text, 50)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(c.Chunks) {
			t.Errorf("%q: %d chunks %q, want %q", c.Text, len(got), got, c.Chunks)
			continue
		}
		for i := range got {
			p, g, _ := m.cfg.prepareText(got[i])
			if got[i] != c.Chunks[i] || p != c.Prepared[i] || g+2 != c.FramesAfterEOS[i] || !equalInts(m.tok.encode(p), c.IDs[i]) {
				t.Errorf("chunk %q (%q, %d), want %q (%q, %d)", got[i], p, g+2, c.Chunks[i], c.Prepared[i], c.FramesAfterEOS[i])
			}
		}
	}
}
