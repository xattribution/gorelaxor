package whisper

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func modelDir() string {
	if d := os.Getenv("GORELAXOR_WHISPER_MODELS"); d != "" {
		return d
	}
	return "models" // ggml-<name>.bin files; tests needing one skip without it
}

var models = map[string]*Model{}

func testModel(t testing.TB, name string) *Model {
	t.Helper()
	if m := models[name]; m != nil {
		return m
	}
	p := filepath.Join(modelDir(), "ggml-"+name+".bin")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no model %s in %s", name, modelDir())
	}
	t0 := time.Now()
	m, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded %s in %v: %+v", name, time.Since(t0).Round(time.Millisecond), m.Info())
	models[name] = m
	return m
}

func jfk(t testing.TB) []float32 {
	t.Helper()
	f, err := os.Open(filepath.Join(modelDir(), "jfk.wav"))
	if err != nil {
		t.Skip("no jfk.wav")
	}
	defer f.Close()
	pcm, err := DecodeWAV(f)
	if err != nil {
		t.Fatal(err)
	}
	return pcm
}

var reNorm = regexp.MustCompile(`[^a-z0-9 ]+`)

func norm(s string) string {
	return strings.Join(strings.Fields(reNorm.ReplaceAllString(strings.ToLower(s), " ")), " ")
}

func joined(segs []Segment) string {
	var b []string
	for _, s := range segs {
		b = append(b, s.Text)
	}
	return strings.Join(b, " ")
}

func TestVectorMath(t *testing.T) {
	x := make([]float32, 1003)
	y := make([]float32, len(x))
	for i := range x {
		x[i] = float32(i%97)*0.37 - 20
		y[i] = x[i]
	}
	s1 := expSub(x, 3)
	s2 := expSubGo(y, 3)
	for i := range x {
		if d := float64(x[i] - y[i]); math.Abs(d) > 1e-5*float64(y[i])+1e-30 {
			t.Fatalf("exp %d: %g vs %g", i, x[i], y[i])
		}
	}
	if d := float64(s1 - s2); math.Abs(d) > 1e-4*float64(s2) {
		t.Fatalf("sum %g vs %g", s1, s2)
	}
	for i := range x {
		x[i] = float32(i%211)*0.05 - 5
		y[i] = x[i]
	}
	geluVec(x)
	gelu(y)
	for i := range x {
		if d := float64(x[i] - y[i]); math.Abs(d) > 1e-5 {
			t.Fatalf("gelu %d: %g vs %g", i, x[i], y[i])
		}
	}
	// expf against math.Exp.
	for v := float32(-87); v < 88; v += 0.731 {
		want := math.Exp(float64(v))
		if got := float64(expf(v)); math.Abs(got-want) > 1e-6*want {
			t.Fatalf("expf(%g) = %g, want %g", v, got, want)
		}
	}
}

// TestOtherModels reads quantized and multilingual files when they're
// present: base.en q5_1, tiny q8_0 and tiny (multilingual, with the
// language detected and given).
func TestOtherModels(t *testing.T) {
	pcm := jfk(t)
	for _, c := range []struct{ name, lang, want string }{
		{"base.en-q5_1", "", ""},
		{"tiny-q8_0", "", "en"},
		{"tiny", "", "en"},
		{"tiny", "en", "en"},
	} {
		t.Run(c.name+"/"+c.lang, func(t *testing.T) {
			m := testModel(t, c.name)
			s, err := m.newSession(Options{Language: c.lang})
			if err != nil {
				t.Fatal(err)
			}
			var segs []Segment
			seekEnd := frames(len(pcm))
			for seek := 0; seek+deltaMin < seekEnd; {
				next, sg, err := s.step(context.Background(), pcm, 0, seek, seekEnd)
				if err != nil {
					t.Fatal(err)
				}
				segs = append(segs, sg...)
				seek = max(next, seek+1)
			}
			if c.want != "" && s.language() != c.want {
				t.Errorf("language %s, want %s", s.language(), c.want)
			}
			if got := norm(joined(segs)); got != jfkText {
				t.Errorf("got %q", got)
			}
		})
	}
	if _, err := testModel(t, "tiny").Transcribe(context.Background(), pcm, Options{Language: "klingon"}); err == nil {
		t.Error("an unknown language should be refused")
	}
}

// The portable kernels (Apple silicon, older processors) give the same text.
func TestPortableKernels(t *testing.T) {
	if !useAVX {
		t.Skip("already portable")
	}
	m := testModel(t, "tiny.en")
	setAVX(false)
	defer setAVX(true)
	segs, err := m.Transcribe(context.Background(), jfk(t), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got := norm(joined(segs)); got != jfkText {
		t.Errorf("got %q", got)
	}
}

// wer is the word error rate of got against want, both normalized.
func wer(want, got string) float64 {
	a, b := strings.Fields(norm(want)), strings.Fields(norm(got))
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			c := prev[j-1]
			if a[i-1] != b[j-1] {
				c++
			}
			cur[j] = min(c, prev[j]+1, cur[j-1]+1)
		}
		prev, cur = cur, prev
	}
	return float64(prev[len(b)]) / float64(max(1, len(a)))
}

func wavFile(t testing.TB, name string) ([]float32, string) {
	t.Helper()
	f, err := os.Open(filepath.Join(modelDir(), name+".wav"))
	if err != nil {
		t.Skip("no " + name + ".wav")
	}
	defer f.Close()
	pcm, err := DecodeWAV(f)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := os.ReadFile(filepath.Join(modelDir(), name+".txt"))
	return pcm, string(ref)
}

// TestLongForm transcribes about two minutes of read speech (LibriSpeech)
// and checks the word error rate, the speed, and that a stream fed in
// small pieces says exactly the same.
func TestLongForm(t *testing.T) {
	pcm, ref := wavFile(t, "ls2min")
	for _, name := range []string{"tiny.en", "base.en"} {
		t.Run(name, func(t *testing.T) {
			m := testModel(t, name)
			t0 := time.Now()
			segs, err := m.Transcribe(context.Background(), pcm, Options{})
			if err != nil {
				t.Fatal(err)
			}
			el := time.Since(t0)
			audio := time.Duration(len(pcm)) * time.Second / SampleRate
			w := wer(ref, joined(segs))
			t.Logf("%s: %d segments, WER %.1f%%, %v for %v, %.1fx real time", name, len(segs), 100*w, el.Round(time.Millisecond), audio.Round(time.Second), audio.Seconds()/el.Seconds())
			if w > 0.12 {
				for _, s := range segs {
					t.Logf("[%v → %v] %s", s.Start, s.End, s.Text)
				}
				t.Errorf("word error rate %.1f%%", 100*w)
			}
			for i := 1; i < len(segs); i++ {
				if segs[i].Start < segs[i-1].Start || segs[i].End < segs[i].Start {
					t.Errorf("segment %d out of order: %+v after %+v", i, segs[i], segs[i-1])
				}
			}
			// The same audio as a live stream, in 0.37 s pieces.
			st := m.NewStream(context.Background(), Options{})
			var live []Segment
			for i := 0; i < len(pcm); i += 5920 {
				if err := st.Write(pcm[i:min(i+5920, len(pcm))]); err != nil {
					t.Fatal(err)
				}
			}
			all, err := st.Close()
			if err != nil {
				t.Fatal(err)
			}
			live = all
			if a, b := joined(segs), joined(live); a != b {
				t.Errorf("stream differs:\n%s\n%s", a, b)
			}
		})
	}
}

const jfkText = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

func TestJFK(t *testing.T) {
	for _, name := range []string{"tiny.en", "base.en"} {
		t.Run(name, func(t *testing.T) {
			m := testModel(t, name)
			pcm := jfk(t)
			t0 := time.Now()
			segs, err := m.Transcribe(context.Background(), pcm, Options{})
			if err != nil {
				t.Fatal(err)
			}
			el := time.Since(t0)
			for _, s := range segs {
				t.Logf("[%v → %v] %s", s.Start, s.End, s.Text)
			}
			audio := time.Duration(len(pcm)) * time.Second / SampleRate
			t.Logf("%s: %v for %v of audio, %.1fx real time", name, el.Round(time.Millisecond), audio, audio.Seconds()/el.Seconds())
			if got := norm(joined(segs)); got != jfkText {
				t.Errorf("got %q", got)
			}
		})
	}
}

// A short stream closed early (a recording paused after a few seconds)
// finishes with its text instead of failing, from either half of a clip,
// written a second at a time.
func TestShortStream(t *testing.T) {
	m := testModel(t, "tiny.en")
	audio := jfk(t)
	half := len(audio) / 2
	for _, part := range [][]float32{audio[:half], audio[half:]} {
		st := m.NewStream(context.Background(), Options{})
		for off := 0; off < len(part); off += SampleRate {
			st.Write(part[off:min(off+SampleRate, len(part))])
		}
		segs, err := st.Close()
		if err != nil || len(segs) == 0 {
			t.Fatalf("%v %q", err, joined(segs))
		}
	}
}

// Translate asks a multilingual model for English; an English-only model
// ignores it.
func TestTranslateTask(t *testing.T) {
	m := testModel(t, "tiny")
	s, err := m.newSession(Options{Language: "es", Translate: true})
	if err != nil || len(s.init) != 3 || s.init[2] != m.voc.translate {
		t.Fatalf("%v %v", err, s.init)
	}
	segs, err := m.Transcribe(context.Background(), jfk(t), Options{Language: "en", Translate: true})
	if err != nil || !strings.Contains(norm(joined(segs)), "country") {
		t.Fatalf("%v %q", err, joined(segs))
	}
	if e := testModel(t, "tiny.en"); e != nil {
		s, _ := e.newSession(Options{Translate: true})
		if s.task() != e.voc.transcribe {
			t.Fatal("English-only model asked to translate")
		}
	}
}
