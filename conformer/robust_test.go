package conformer

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

// Model files and recordings are outside input. These tests use the small
// random-weight models in synth_test.go.

// A tensor that doesn't fit its settings panics inside a kernel worker; the
// panic must reach the caller's recover, not stop the program.
func TestWorkerPanicReachesCaller(t *testing.T) {
	m := synthModel(2, false, "tdt")
	m.pre.w0 = m.pre.w0[: 9*m.HP.SubChannels-1 : 9*m.HP.SubChannels-1] // one value short
	err := checked(func() error {
		_, err := m.Transcribe(noise(16000, 1), nil, DefaultSegments, 2)
		return err
	})
	if err == nil {
		t.Fatal("a short weight transcribed without an error")
	}
}

// Settings that can't work fall back to the defaults instead of looping.
func TestSegmentsBadSettings(t *testing.T) {
	n := 16000 * 95
	for _, p := range []SegmentParams{{}, {MaxSeconds: -3}, {MaxSeconds: math.NaN()}, {MaxSeconds: 1e-6}, {MaxSeconds: math.Inf(1)}, {MaxSeconds: 30, MinSeconds: 40}} {
		done := make(chan []Segment, 1)
		go func() { done <- Segments(n, 16000, []Region{{0, 95}}, p) }()
		select {
		case segs := <-done:
			if len(segs) == 0 || segs[0].Start != 0 || segs[len(segs)-1].End != n {
				t.Errorf("%+v: segments %v don't cover the recording", p, segs)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("%+v: Segments didn't return", p)
		}
	}
}

// One NaN or infinite sample is heard as silence; it must not reach the
// stream's caches or decoder.
func TestNonFiniteSamples(t *testing.T) {
	m := synthModel(1, true, "rnnt")
	s, _ := m.NewStream(13, 2)
	s.Write(noise(16000*3, 5))
	bad := noise(1600, 6)
	bad[100], bad[900] = float32(math.NaN()), float32(math.Inf(-1))
	s.Write(bad)
	for i := 0; i < 5; i++ {
		s.Write(noise(16000, int64(10+i)))
	}
	for _, v := range append(append([]float32(nil), s.st.dec.h[0]...), s.st.kv[0]...) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("a non-finite sample reached the stream's state")
		}
	}
	if bad[100] == 0 {
		t.Fatal("Write changed the caller's samples")
	}

	f := synthModel(2, false, "tdt")
	pcm := noise(16000*20, 3)
	zero := append([]float32(nil), pcm...)
	pcm[16000*10], zero[16000*10] = float32(math.Inf(1)), 0
	a, _ := f.Transcribe(pcm, nil, DefaultSegments, 2)
	b, _ := f.Transcribe(zero, nil, DefaultSegments, 2)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("an infinite sample changed the transcript: %v against %v", a, b)
	}
}

// Per-feature normalization of one real frame has no spread to divide by.
func TestMelOneFrame(t *testing.T) {
	m := synthMel("per_feature")
	for _, n := range []int{160, 200, 319, 320} {
		mel, _, err := m.Compute(noise(n, 1), 1)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range mel {
			if math.IsNaN(float64(v)) {
				t.Fatalf("%d samples: NaN in the mel frames", n)
			}
		}
	}
}

// A long Write goes through in pieces with the same result as small writes.
func TestLongWriteMatchesSmallWrites(t *testing.T) {
	m := synthModel(1, true, "rnnt")
	pcm := noise(16000*25, 7)
	one, _ := m.NewStream(13, 2)
	a := append(one.Write(pcm), one.Close()...)
	small, _ := m.NewStream(13, 2)
	var b []Token
	for i := 0; i < len(pcm); i += 1600 {
		b = append(b, small.Write(pcm[i:min(i+1600, len(pcm))])...)
	}
	b = append(b, small.Close()...)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatal("one long write and many short ones gave different tokens")
	}
}

// Loaded models are read-only, so streams, final passes and the speaker
// model can share them (run with -race).
func TestSharedModels(t *testing.T) {
	ms := synthModel(1, true, "rnnt")
	mt := synthModel(2, false, "tdt")
	d := synthDiarizer(4)
	var wg sync.WaitGroup
	for g := 0; g < 3; g++ {
		wg.Add(3)
		go func() {
			defer wg.Done()
			s, _ := ms.NewStream(13, 2)
			pcm := noise(16000*4, int64(g))
			for i := 0; i < len(pcm); i += 1600 {
				s.Write(pcm[i : i+1600])
			}
			s.Close()
		}()
		go func() {
			defer wg.Done()
			mt.Transcribe(noise(16000*3, int64(g)), nil, DefaultSegments, 2)
		}()
		go func() {
			defer wg.Done()
			d.Diarize(noise(16000*12, int64(g)), DiarHighLatency, 2)
		}()
	}
	wg.Wait()
}
