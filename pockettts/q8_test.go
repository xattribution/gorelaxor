package pockettts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"github.com/xattribution/gorelaxor/conformer"
	"github.com/xattribution/gorelaxor/whisper"
)

// The Q8_0 file can't match Kyutai's float32 output exactly, and with
// sampling a small difference sends a frame down another path, so its
// quality gate is a speech-recognition round trip: Parakeet must hear in
// the Go Q8_0 audio what it hears in Kyutai's float32 audio, and the
// text itself where the text has no digits (Pocket TTS reads 25 as "two
// five", so a transcript can't give digits back).

func parakeet(t *testing.T, ref string) *conformer.Model {
	t.Helper()
	path := filepath.Join(ref, "parakeet-q8.gguf")
	if _, err := os.Stat(path); err != nil {
		if d := os.Getenv("GORELAXOR_CONFORMER_REF"); d != "" {
			path = filepath.Join(d, "parakeet-q8.gguf")
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Skip("no parakeet-q8.gguf for the round trip")
	}
	m, err := conformer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func transcribe(t *testing.T, asr *conformer.Model, pcm []float32) string {
	t.Helper()
	toks, err := asr.Transcribe(whisper.Resample(pcm, 24000, 16000), nil, conformer.DefaultSegments, 2)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int, len(toks))
	for i, k := range toks {
		ids[i] = k.ID
	}
	return asr.Text(ids)
}

// words lowercases s and keeps its letters and digits, as words.
func words(s string) []string {
	s = strings.Map(func(r rune) rune {
		switch {
		case r == '\'' || r == '’':
			return -1
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			return unicode.ToLower(r)
		}
		return ' '
	}, s)
	return strings.Fields(s)
}

// wordErrors is the word-level edit distance, a word written as two
// ("to day") counting as the same word.
func wordErrors(a, b []string) int {
	d := make([][]int, len(a)+1)
	for i := range d {
		d[i] = make([]int, len(b)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			c := d[i-1][j-1]
			if a[i-1] != b[j-1] {
				c++
			}
			c = min(c, d[i-1][j]+1, d[i][j-1]+1)
			if i > 1 && a[i-2]+a[i-1] == b[j-1] {
				c = min(c, d[i-2][j-1])
			}
			if j > 1 && b[j-2]+b[j-1] == a[i-1] {
				c = min(c, d[i-1][j-2])
			}
			d[i][j] = c
		}
	}
	return d[len(a)][len(b)]
}

func TestReferenceQ8(t *testing.T) {
	ref := refDir(t)
	m := loadModel(t, filepath.Join(ref, "pocket-tts-q8.gguf"))
	asr := parakeet(t, ref)
	for _, name := range refCases {
		t.Run(name, func(t *testing.T) {
			r := loadCase(t, filepath.Join(ref, name))
			v := caseVoice(t, m, ref, filepath.Join(ref, "voices-f16"), r, false)
			forced := runCase(t, m, v, r, true)
			forced.latent.check(t, "teacher forced: latents", 0)
			var fp diff
			fp.add(forced.pcm[:min(len(forced.pcm), len(r.pcm))], r.pcm[:min(len(forced.pcm), len(r.pcm))])
			fp.check(t, "teacher forced: audio", 0)
			res := runCase(t, m, v, r, false)
			t.Logf("free running: %.2f s of audio, Kyutai's %.2f s", float64(len(res.pcm))/24000, float64(len(r.pcm))/24000)
			want := words(r.c.Text)
			theirs := transcribe(t, asr, r.pcm)
			ours := transcribe(t, asr, res.pcm)
			t.Logf("Kyutai float32 heard as %q (%d word errors)", theirs, wordErrors(words(theirs), want))
			t.Logf("Go Q8_0 heard as %q (%d word errors)", ours, wordErrors(words(ours), want))
			if e := wordErrors(words(ours), words(theirs)); e > 1 {
				t.Errorf("Go's Q8_0 audio was heard %d words away from Kyutai's float32", e)
			}
			if !strings.ContainsAny(r.c.Text, "0123456789") {
				if e := wordErrors(words(ours), want); e > 1 {
					t.Errorf("Go's Q8_0 audio was heard with %d word errors", e)
				}
			}
		})
	}
}

// TestSeed checks that a seed repeats its output, whatever the threads.
func TestSeed(t *testing.T) {
	ref := refDir(t)
	m := loadModel(t, filepath.Join(ref, "pocket-tts-q8.gguf"))
	v := voiceFor(t, m, filepath.Join(ref, "voices-f16"), "alba")
	a, err := m.Synthesize("Same seed, same sound.", v, Options{Seed: 7, Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := m.Synthesize("Same seed, same sound.", v, Options{Seed: 7, Threads: 2})
	c, _ := m.Synthesize("Same seed, same sound.", v, Options{Seed: 8, Threads: 2})
	if len(a) != len(b) {
		t.Fatalf("%d and %d samples from one seed", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("sample %d differs between 1 and 2 threads", i)
		}
	}
	same := len(a) == len(c)
	for i := 0; same && i < len(a); i++ {
		same = a[i] == c[i]
	}
	if same {
		t.Fatal("seeds 7 and 8 made the same audio")
	}
}
