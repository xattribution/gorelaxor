package conformer

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/xattribution/gorelaxor/internal/kern"
	"github.com/xattribution/gorelaxor/whisper"
)

// The parity tests compare against NVIDIA NeMo's own outputs, made by
// tools/nemo_reference.py (its comment lists the files), and against
// transcribe.cpp's tensor dumps where those are present. They skip
// unless GORELAXOR_CONFORMER_REF names the folder holding them:
//
//	nemotron.gguf, nemotron35.gguf, parakeet.gguf, parakeet3.gguf   tools/convert_nemo.py --type f32
//	sortformer.gguf, vad.gguf
//	nemotron-q8.gguf, nemotron35-q8.gguf, parakeet-q8.gguf, parakeet3-q8.gguf, multitalker-q8.gguf
//	jfk.wav, multitalker-2spk-mix.wav
//	fleurs-de.wav, fleurs-es.wav, fleurs-fr.wav, fleurs-ja.wav, fleurs-zh.wav, ko-short.wav
//	nemo-nemotron/, nemo-parakeet/, nemo-vad/, nemo-sortformer/, nemo-multitalker/
//	nemo-parakeet3-jfk/, nemo-parakeet3-fleurs-de/, -es/, -fr/
//	nemo-nemotron35-<clip>/, nemo-nemotron35-<clip>-auto/   for jfk and the six clips above
//	dump-nemotron-jfk/   optional: transcribe.cpp's TRANSCRIBE_DUMP_DIR output
func refDir(t *testing.T) string {
	d := os.Getenv("GORELAXOR_CONFORMER_REF")
	if d == "" {
		t.Skip("GORELAXOR_CONFORMER_REF is not set")
	}
	return d
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

func loadWAV(t *testing.T, path string) []float32 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	pcm, err := whisper.DecodeWAV(f)
	if err != nil {
		t.Fatal(err)
	}
	return pcm
}

// compare reports the largest absolute difference relative to the
// reference's largest magnitude, and fails above tol.
func compare(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	var md, mx float64
	at := 0
	for i := range want {
		mx = max(mx, math.Abs(float64(want[i])))
		if d := math.Abs(float64(got[i] - want[i])); d > md || math.IsNaN(d) {
			md, at = d, i
		}
	}
	rel := md / max(mx, 1e-12)
	t.Logf("%s: max abs diff %.3g at %d (got %g want %g), relative %.3g", what, md, at, got[at], want[at], rel)
	if rel > tol || math.IsNaN(md) {
		t.Fatalf("%s: relative diff %.3g above %.3g", what, rel, tol)
	}
}

// cppDump is the transcribe.cpp dump folder, or a skip.
func cppDump(t *testing.T, dir string) string {
	d := filepath.Join(dir, "dump-nemotron-jfk")
	if _, err := os.Stat(d); err != nil {
		t.Skip("no transcribe.cpp dumps")
	}
	return d
}
