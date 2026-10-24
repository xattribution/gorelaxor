package conformer

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// wordDistance is the word-level edit distance between two texts.
func wordDistance(a, b string) int {
	x, y := strings.Fields(a), strings.Fields(b)
	prev := make([]int, len(y)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(x); i++ {
		cur := make([]int, len(y)+1)
		cur[0] = i
		for j := 1; j <= len(y); j++ {
			c := 1
			if x[i-1] == y[j-1] {
				c = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+c)
		}
		prev = cur
	}
	return prev[len(y)]
}

// TestParakeetV3MatchesNeMo runs the multilingual Parakeet TDT v3 on
// English, German, Spanish and French, in float32 and Q8_0.
func TestParakeetV3MatchesNeMo(t *testing.T) {
	dir := refDir(t)
	for _, file := range []string{"parakeet3.gguf", "parakeet3-q8.gguf"} {
		path := filepath.Join(dir, file)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skipping %s", file)
			continue
		}
		m := loadModel(t, path)
		for _, clip := range []string{"jfk", "fleurs-de", "fleurs-es", "fleurs-fr"} {
			nd := nemoDir(t, dir, "parakeet3-"+clip)
			pcm := loadWAV(t, filepath.Join(dir, clip+".wav"))
			start := time.Now()
			mel, frames, err := m.Mel.Compute(pcm, 2)
			if err != nil {
				t.Fatal(err)
			}
			enc, T := m.Encode(mel, frames, m.Mel.Valid(len(pcm)), 2)
			toks := m.Decode(enc, T, nil, 0, 2)
			el := time.Since(start)
			want, _ := loadDump(t, nd, "enc")
			var num, den, md float64
			for i := range want {
				d := float64(enc[i] - want[i])
				num += d * d
				den += float64(want[i]) * float64(want[i])
				md = max(md, math.Abs(d))
			}
			var ref struct{ Text string }
			readJSON(t, filepath.Join(nd, "text.json"), &ref)
			got := m.Text(ids(toks))
			t.Logf("%s %s: %.1fx real time, encoder RMS error %.2g%%, max %.2g: %q", file, clip,
				float64(len(pcm))/16000/el.Seconds(), 100*math.Sqrt(num/den), md, got)
			// float32 must give NeMo's words exactly; Q8_0 may differ by
			// a word or two where NeMo's own choice was close (Q8_0 wrote "T-Rex"
			// for its "T Rex")
			allowed := 0
			if file != "parakeet3.gguf" {
				allowed = 2
			}
			if d := wordDistance(got, ref.Text); d > allowed {
				t.Errorf("%s %s: %d words differ\ngot  %q\nNeMo %q", file, clip, d, got, ref.Text)
			}
		}
	}
}
