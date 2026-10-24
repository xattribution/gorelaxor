package whisper

import (
	"math"
	"sync"
)

const (
	SampleRate = 16000
	nFFTWin    = 400 // 25 ms
	hop        = 160 // 10 ms
	winFrames  = 3000
	winSamples = winFrames * hop // 30 s
)

var dftOnce sync.Once
var dftBasis []float32 // 402 x 400: cos rows then sin rows, Hann window folded in

func basis() []float32 {
	dftOnce.Do(func() {
		dftBasis = make([]float32, 402*nFFTWin)
		for n := 0; n < nFFTWin; n++ {
			w := 0.5 * (1 - math.Cos(2*math.Pi*float64(n)/nFFTWin)) // periodic Hann
			for k := 0; k <= 200; k++ {
				t := 2 * math.Pi * float64(k*n%nFFTWin) / nFFTWin
				dftBasis[k*nFFTWin+n] = float32(w * math.Cos(t))
				dftBasis[(201+k)*nFFTWin+n] = float32(w * math.Sin(t))
			}
		}
	})
	return dftBasis
}

// logMel is the normalized log-mel spectrogram of the 30-second window
// starting at sample start, as nMels rows of 3000 frames. Frames are
// centered as Whisper's training was: the audio is mirrored before its
// first sample and silent after its last.
func (m *Model) logMel(ws *workspace, samples []float32, start, threads int) []float32 {
	nMel := m.hp.nMels
	n := len(samples)
	sample := func(i int) float32 {
		if i < 0 {
			i = -i
		}
		if i >= n {
			return 0
		}
		return samples[i]
	}
	// Frames that touch audio get a transform; the rest are silent.
	live := 0
	for live < winFrames && start+live*hop-nFFTWin/2 < n {
		live++
	}
	out := ws.buf(sMelOut, nMel*winFrames, false)
	const floor = -10 // log10(1e-10)
	if live > 0 {
		frames := ws.buf(sMelFrames, live*nFFTWin, false)
		for f := 0; f < live; f++ {
			base := start + f*hop - nFFTWin/2
			row := frames[f*nFFTWin : (f+1)*nFFTWin]
			if base >= 0 && base+nFFTWin <= n {
				copy(row, samples[base:base+nFFTWin])
				continue
			}
			for j := range row {
				row[j] = sample(base + j)
			}
		}
		spec := ws.buf(sMelSpec, live*402, false)
		ws.matmul(spec, 402, frames, nFFTWin, live, 402, nFFTWin, fmat(basis(), nFFTWin), nil, threads)
		power := frames[:live*201] // reuse
		for f := 0; f < live; f++ {
			s := spec[f*402 : (f+1)*402]
			p := power[f*201 : (f+1)*201]
			for k := 0; k <= 200; k++ {
				p[k] = s[k]*s[k] + s[201+k]*s[201+k]
			}
		}
		mel := spec[:live*nMel]
		ws.matmul(mel, nMel, power, 201, live, nMel, 201, fmat(m.melF, 201), nil, threads)
		for f := 0; f < live; f++ {
			for j := 0; j < nMel; j++ {
				v := float64(mel[f*nMel+j])
				out[j*winFrames+f] = float32(math.Log10(max(v, 1e-10)))
			}
		}
	}
	for j := 0; j < nMel; j++ {
		for f := live; f < winFrames; f++ {
			out[j*winFrames+f] = floor
		}
	}
	mx := float32(floor)
	for _, v := range out {
		mx = max(mx, v)
	}
	mx -= 8
	for i, v := range out {
		out[i] = (max(v, mx) + 4) / 4
	}
	return out
}
