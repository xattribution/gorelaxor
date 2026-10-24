package conformer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// Nemotron 3.5 is checked on one clip per language, each with its
// locale and with auto, against NeMo in float32 (tools/nemo_reference.py
// makes nemo-nemotron35-<clip>/ and nemo-nemotron35-<clip>-auto/).
var n35Clips = []struct{ clip, lang string }{
	{"jfk", "en-US"}, {"fleurs-de", "de-DE"}, {"fleurs-es", "es-US"}, {"fleurs-fr", "fr-FR"},
	{"fleurs-ja", "ja-JP"}, {"fleurs-zh", "zh-CN"}, {"ko-short", "ko-KR"},
}

// n35Rights are the streaming right contexts the references hold.
var n35Rights = []int{13, 3}

// nemoTag is NeMo's own strip_lang_tags pattern (rnnt_decoding.py).
var nemoTag = regexp.MustCompile(`\s*<([a-z]{2}-[A-Z]{2})>`)

// nemoWords is NeMo's text with strip_lang_tags on, and the last tag.
func nemoWords(text string) (string, string) {
	lang := ""
	if all := nemoTag.FindAllStringSubmatch(text, -1); len(all) > 0 {
		lang = all[len(all)-1][1]
	}
	return strings.TrimSpace(nemoTag.ReplaceAllString(text, "")), lang
}

type n35Case struct {
	clip, lang, dir, name string
}

// n35Cases lists the reference folders present, or skips.
func n35Cases(t *testing.T, dir string) []n35Case {
	var out []n35Case
	for _, c := range n35Clips {
		for _, auto := range []bool{false, true} {
			name, lang := c.clip, c.lang
			if auto {
				name, lang = name+"-auto", "auto"
			}
			d := filepath.Join(dir, "nemo-nemotron35-"+name)
			if _, err := os.Stat(d); err != nil {
				t.Logf("no %s", d)
				continue
			}
			out = append(out, n35Case{c.clip, lang, d, name})
		}
	}
	if len(out) == 0 {
		t.Skip("no Nemotron 3.5 references")
	}
	return out
}

func n35Model(t *testing.T, dir, file string) *Model {
	path := filepath.Join(dir, file)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no %s", file)
	}
	return loadModel(t, path)
}

// streamAll feeds pcm in uneven pieces, as a microphone would, and
// returns the tokens and the encoder output before the prompt.
func streamAll(t *testing.T, m *Model, R int, pcm []float32) ([]Token, []float32, int) {
	s, err := m.NewStream(R, 2)
	if err != nil {
		t.Fatal(err)
	}
	var enc []float32
	s.onEncoded = func(x []float32) { enc = append(enc, x...) }
	var toks []Token
	for i := 0; i < len(pcm); {
		n := min(len(pcm)-i, 1000+i%3777)
		toks = append(toks, s.Write(pcm[i:i+n])...)
		i += n
	}
	return append(toks, s.Close()...), enc, s.frames
}

func TestNemotron35MatchesNeMo(t *testing.T) {
	dir := refDir(t)
	cases := n35Cases(t, dir)
	base := n35Model(t, dir, "nemotron35.gguf")
	if base.prompt == nil {
		t.Fatal("nemotron35.gguf has no language prompt")
	}
	var table []string
	for _, c := range cases {
		m, err := base.WithLanguage(c.lang)
		if err != nil {
			t.Fatal(err)
		}
		pcm := loadWAV(t, filepath.Join(dir, c.clip+".wav"))
		mel, frames, err := m.Mel.Compute(pcm, 2)
		if err != nil {
			t.Fatal(err)
		}
		wantMel, _ := loadDump(t, c.dir, "mel")
		compare(t, c.name+" mel", mel, wantMel, 1e-4)
		raw, T := m.encode(mel, frames, m.Mel.Valid(len(pcm)), 2, nil)
		want, _ := loadDump(t, c.dir, "enc")
		encRel := relDiff(raw, want)
		compare(t, c.name+" encoder", raw, want, 1e-4)
		prompted := m.applyPrompt(raw, T, 2)
		wantP, _ := loadDump(t, c.dir, "prompted")
		compare(t, c.name+" prompted", prompted, wantP, 1e-4)
		var ref struct {
			Text   string
			Tokens []int
		}
		readJSON(t, filepath.Join(c.dir, "text.json"), &ref)
		toks := m.Decode(prompted, T, nil, 0, 2)
		row := fmt.Sprintf("%-14s encoder %.1e, prompt out %.1e; offline %s", c.name, encRel, relDiff(prompted, wantP),
			checkWords(t, m, c.name+" offline", toks, ref.Text))
		if got := ids(toks); fmt.Sprint(got) != fmt.Sprint(ref.Tokens) {
			t.Errorf("%s: tokens %v\nNeMo %v", c.name, got, ref.Tokens)
		}
		for _, R := range n35Rights {
			wantS, shape := loadDump(t, c.dir, "stream"+itoa(R)+".enc")
			stoks, enc, nf := streamAll(t, m, R, pcm)
			if nf != shape[0] {
				t.Errorf("%s R=%d: %d encoder frames, NeMo %d", c.name, R, nf, shape[0])
			}
			compare(t, c.name+" R="+itoa(R)+" encoder", enc, wantS, 1e-4)
			var meta struct{ Text string }
			readJSON(t, filepath.Join(c.dir, "stream"+itoa(R)+".json"), &meta)
			row += fmt.Sprintf(", R=%d enc %.1e %s", R, relDiff(enc, wantS), checkWords(t, m, c.name+" R="+itoa(R), stoks, meta.Text))
		}
		table = append(table, row)
	}
	t.Log("\n" + strings.Join(table, "\n"))
}

// checkWords fails unless the words and the detected language are
// NeMo's, and says "same" or what differed.
func checkWords(t *testing.T, m *Model, what string, toks []Token, nemo string) string {
	t.Helper()
	want, wantLang := nemoWords(nemo)
	got, gotLang := m.Text(ids(toks)), m.DetectedLanguage(toks)
	if got != want || gotLang != wantLang {
		t.Errorf("%s: got %q [%s]\nNeMo %q [%s]", what, got, gotLang, want, wantLang)
		return "DIFFERENT"
	}
	t.Logf("%s: %q [%s]", what, got, gotLang)
	return "same [" + gotLang + "]"
}

// relDiff is the largest absolute difference over the reference's
// largest magnitude.
func relDiff(got, want []float32) float64 {
	if len(got) != len(want) {
		return math.Inf(1)
	}
	var md, mx float64
	for i := range want {
		mx = max(mx, math.Abs(float64(want[i])))
		md = max(md, math.Abs(float64(got[i]-want[i])))
	}
	return md / max(mx, 1e-12)
}

// units splits text into words, or into characters for scripts written
// without spaces (Chinese, Japanese), for counting differences.
func units(s string) []string {
	cjk := false
	for _, r := range s {
		if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana) {
			cjk = true
			break
		}
	}
	if !cjk {
		return strings.Fields(s)
	}
	var out []string
	for _, r := range s {
		if !unicode.IsSpace(r) {
			out = append(out, string(r))
		}
	}
	return out
}

// editDistance counts the insertions, deletions and substitutions
// between two lists.
func editDistance(a, b []string) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			c := prev[j-1]
			if a[i-1] != b[j-1] {
				c++
			}
			cur[j] = min(c, prev[j]+1, cur[j-1]+1)
		}
		prev = cur
	}
	return prev[len(b)]
}

// TestQ8Nemotron35 checks the Q8_0 file apps ship: words within two of
// NeMo's float32 ones on every clip, offline and streaming at R = 13.
func TestQ8Nemotron35(t *testing.T) {
	dir := refDir(t)
	cases := n35Cases(t, dir)
	base := n35Model(t, dir, "nemotron35-q8.gguf")
	var table []string
	for _, c := range cases {
		m, err := base.WithLanguage(c.lang)
		if err != nil {
			t.Fatal(err)
		}
		pcm := loadWAV(t, filepath.Join(dir, c.clip+".wav"))
		mel, frames, _ := m.Mel.Compute(pcm, 2)
		raw, T := m.encode(mel, frames, m.Mel.Valid(len(pcm)), 2, nil)
		want, _ := loadDump(t, c.dir, "enc")
		var num, den float64
		for i := range want {
			d := float64(raw[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		var ref struct{ Text string }
		readJSON(t, filepath.Join(c.dir, "text.json"), &ref)
		var meta struct{ Text string }
		readJSON(t, filepath.Join(c.dir, "stream13.json"), &meta)
		toks := m.Decode(m.applyPrompt(raw, T, 2), T, nil, 0, 2)
		stoks, _, _ := streamAll(t, m, 13, pcm)
		row := fmt.Sprintf("%-14s encoder RMS error %.2f%%", c.name, 100*math.Sqrt(num/den))
		for _, r := range []struct {
			what string
			toks []Token
			nemo string
		}{{"offline", toks, ref.Text}, {"R=13", stoks, meta.Text}} {
			want, wantLang := nemoWords(r.nemo)
			got := m.Text(ids(r.toks))
			d := editDistance(units(got), units(want))
			row += fmt.Sprintf(", %s %d off [%s]", r.what, d, m.DetectedLanguage(r.toks))
			if d > 0 {
				t.Logf("%s %s: %q\nNeMo float32 %q", c.name, r.what, got, want)
			}
			if d > 2 || m.DetectedLanguage(r.toks) != wantLang {
				t.Errorf("%s %s: got %q [%s]\nNeMo %q [%s]", c.name, r.what, got, m.DetectedLanguage(r.toks), want, wantLang)
			}
		}
		table = append(table, row)
	}
	t.Log("\n" + strings.Join(table, "\n"))
}

// TestLanguageChoice checks the language API without reference files.
func TestLanguageChoice(t *testing.T) {
	mel, err := NewMel(MelConfig{SampleRate: 16000, NFFT: 512, WinLength: 400, Hop: 160, NumMels: 80, FMax: 8000, PreEmphasis: 0.97, Normalize: "none"})
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Vocab: []string{"▁a", "<de-DE>", "<sl-SL>", "b", "▁"}, Mel: mel, HP: HParams{Blank: 5, Subsampling: 8}}
	if l := m.Languages(); l != nil {
		t.Fatalf("a single-language model lists %v", l)
	}
	if got, err := m.WithLanguage("auto"); err != nil || got != m {
		t.Fatalf("auto on a single-language model: %v", err)
	}
	if _, err := m.WithLanguage("de-DE"); err == nil {
		t.Fatal("a single-language model took de-DE")
	}
	if m.Text([]int{0, 1, 3}) != "a<de-DE>b" {
		t.Fatalf("a single-language model dropped a piece: %q", m.Text([]int{0, 1, 3}))
	}
	m.prompt = &languagePrompt{n: 4, index: map[string]int{"de-DE": 2, "de": 2, "auto": 3}, order: []string{"de-DE", "de", "auto"},
		auto: 3, tags: map[int]string{1: "de-DE", 2: "sl-SL"}}
	m.lang = 3
	de, err := m.WithLanguage("de")
	if err != nil || de.lang != 2 || m.lang != 3 {
		t.Fatalf("WithLanguage(de): slot %d, base %d, %v", de.lang, m.lang, err)
	}
	if _, err := m.WithLanguage("xx-XX"); err == nil {
		t.Fatal("took an unknown language")
	}
	toks := []Token{{ID: 0}, {ID: 3}, {ID: 4}, {ID: 1}, {ID: 0}, {ID: 2}}
	if got := m.Text(ids(toks)); got != "ab a" {
		t.Fatalf("text %q", got)
	}
	if got := m.DetectedLanguage(toks); got != "sl-SL" {
		t.Fatalf("language %q", got)
	}
	if w := m.Words(toks); len(w) != 2 || w[0].Text != "ab" || w[1].Text != "a" {
		t.Fatalf("words %+v", w)
	}
}
