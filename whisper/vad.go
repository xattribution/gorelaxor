package whisper

import (
	"math"
	"sort"
)

// An energy gate, so silence costs almost nothing and doesn't come back
// as invented text ("Thank you." over an empty room is Whisper's best-known
// habit). It works on 25 ms frames every 10 ms, aligned with the mel frames.

const (
	gateAbs      = 0.002 // RMS below this (about -54 dBFS) is silence whatever the room
	gateNoise    = 3.0   // or below 3x the window's noise floor (about +10 dB)
	gateMinVoice = 25    // a window needs 0.25 s of sound to be worth decoding
	gateRun      = 3     // sound starts at the first 30 ms run
	gateSteady   = 1.5   // loud frames under 1.5x the quiet ones: steady noise, not speech
)

// voiced marks each 10 ms frame in [from, to) of samples (frame indexes
// relative to the samples given) as sound or silence.
func voiced(samples []float32, from, to int) []bool {
	n := to - from
	if n <= 0 {
		return nil
	}
	rms := make([]float32, n)
	for f := 0; f < n; f++ {
		a := (from + f) * hop
		b := min(a+nFFTWin, len(samples))
		if a < 0 {
			a = 0
		}
		var s float64
		for _, v := range samples[a:max(a, b)] {
			s += float64(v) * float64(v)
		}
		if b > a {
			rms[f] = float32(math.Sqrt(s / float64(b-a)))
		}
	}
	thr := float32(gateAbs)
	steady := false
	if n >= 100 {
		sorted := append([]float32(nil), rms...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		p10, p90 := sorted[n/10], sorted[n*9/10]
		// Speech rises and falls by syllables; a fan or hiss holds steady.
		// The floor is capped at half the loud frames' level so continuous
		// speech, where even the quiet tenth is speech, still counts.
		steady = p90 < gateSteady*p10
		thr = max(thr, min(gateNoise*p10, 0.5*p90))
	}
	out := make([]bool, n)
	if steady {
		return out
	}
	for f, r := range rms {
		out[f] = r > thr
	}
	return out
}

// firstSound is the first frame starting a run of sound, or -1.
func firstSound(v []bool) int {
	run := 0
	for i, x := range v {
		if x {
			run++
			if run >= gateRun {
				return i - run + 1
			}
		} else {
			run = 0
		}
	}
	return -1
}

func countTrue(v []bool) int {
	n := 0
	for _, x := range v {
		if x {
			n++
		}
	}
	return n
}
