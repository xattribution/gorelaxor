package conformer

import (
	"path/filepath"
	"testing"
	"time"
)

func TestSortformerMatchesNeMo(t *testing.T) {
	dir := refDir(t)
	nd := nemoDir(t, dir, "sortformer")
	refModel = nil
	d, err := LoadDiarizer(filepath.Join(dir, "sortformer.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	all, _ := loadDump(t, nd, "pcm")
	for _, c := range []struct {
		name string
		p    DiarParams
		secs float64 // audio to run; NeMo's output before its end doesn't depend on the rest
	}{{"high", DiarVeryHighLatency, 0}, {"low", DiarLowLatency, 45}} {
		pcm := all
		if c.secs > 0 {
			pcm = all[:int(c.secs*16000)]
		}
		start := time.Now()
		got, err := d.Diarize(pcm, c.p, 2)
		if err != nil {
			t.Fatal(err)
		}
		el := time.Since(start)
		want, shape := loadDump(t, nd, "preds."+c.name)
		n := shape[0]
		if c.secs > 0 {
			// leave out the last chunk and its right context
			n = len(got) - c.p.ChunkLen - c.p.RightContext
		} else if len(got) != n {
			t.Fatalf("%s: %d frames, NeMo %d", c.name, len(got), n)
		}
		flat := make([]float32, 0, n*4)
		for _, r := range got[:n] {
			flat = append(flat, r...)
		}
		t.Logf("%s: %d frames, %.1fx real time", c.name, len(got), float64(len(pcm))/16000/el.Seconds())
		compare(t, c.name, flat, want[:n*4], 1e-4)
	}
}
