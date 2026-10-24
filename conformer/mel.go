package conformer

import (
	"errors"
	"math"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// MelConfig is the NeMo log-mel front end, read from the stt.frontend.*
// keys of a model file.
type MelConfig struct {
	SampleRate  int
	NFFT        int
	WinLength   int
	Hop         int
	NumMels     int
	FMin, FMax  float64
	PreEmphasis float64
	// Normalize is "none" (streaming models fold it into the weights)
	// or "per_feature" (mean and unbiased deviation per mel bin).
	Normalize string
	// SeqLenCeil counts frames as ceil(n/hop), as newer NeMo models
	// (Sortformer) do; the others use floor(n/hop)+1.
	SeqLenCeil bool
}

// Mel computes NeMo-style log-mel features: pre-emphasis, zero padding
// of n_fft/2 on each side, a symmetric Hann window centered in the FFT,
// the power spectrum, a Slaney mel filterbank and log(x + 2^-24).
type Mel struct {
	cfg    MelConfig
	window []float64
	fb     []float32 // [mels][freq]
	lo, hi []int     // nonzero filterbank span per mel
	cos    []float64
	sin    []float64
	rev    []int
}

const (
	logEps  = 1.0 / (1 << 24)
	normEps = 1e-5
)

// NewMel prepares a front end.
func NewMel(cfg MelConfig) (*Mel, error) {
	n := cfg.NFFT
	if n <= 0 || n&(n-1) != 0 {
		return nil, errors.New("conformer: n_fft must be a power of two")
	}
	if cfg.WinLength <= 1 || cfg.WinLength > n || cfg.Hop <= 0 || cfg.NumMels <= 0 {
		return nil, errors.New("conformer: front-end settings out of range")
	}
	m := &Mel{cfg: cfg}
	m.window = make([]float64, n)
	pad := (n - cfg.WinLength) / 2
	for k := 0; k < cfg.WinLength; k++ {
		m.window[pad+k] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(k)/float64(cfg.WinLength-1))
	}
	m.fb = slaneyFilterbank(cfg.SampleRate, n, cfg.NumMels, cfg.FMin, cfg.FMax)
	nf := n/2 + 1
	m.lo = make([]int, cfg.NumMels)
	m.hi = make([]int, cfg.NumMels)
	for j := 0; j < cfg.NumMels; j++ {
		row := m.fb[j*nf : (j+1)*nf]
		lo, hi := nf, 0
		for k, v := range row {
			if v != 0 {
				lo = min(lo, k)
				hi = k + 1
			}
		}
		if hi == 0 {
			lo = 0
		}
		m.lo[j], m.hi[j] = lo, hi
	}
	m.cos = make([]float64, n/2)
	m.sin = make([]float64, n/2)
	for i := range m.cos {
		a := -2 * math.Pi * float64(i) / float64(n)
		m.cos[i], m.sin[i] = math.Cos(a), math.Sin(a)
	}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	m.rev = make([]int, n)
	for i := range m.rev {
		r := 0
		for b := 0; b < bits; b++ {
			if i&(1<<b) != 0 {
				r |= 1 << (bits - 1 - b)
			}
		}
		m.rev[i] = r
	}
	return m, nil
}

func slaneyHzToMel(hz float64) float64 {
	const fsp, minLogHz = 200.0 / 3, 1000.0
	if hz < minLogHz {
		return hz / fsp
	}
	return minLogHz/fsp + math.Log(hz/minLogHz)/(math.Log(6.4)/27)
}

func slaneyMelToHz(mel float64) float64 {
	const fsp, minLogHz = 200.0 / 3, 1000.0
	minLogMel := minLogHz / fsp
	if mel < minLogMel {
		return mel * fsp
	}
	return minLogHz * math.Exp(math.Log(6.4)/27*(mel-minLogMel))
}

// slaneyFilterbank matches librosa.filters.mel(norm="slaney").
func slaneyFilterbank(sr, nfft, mels int, fmin, fmax float64) []float32 {
	nf := nfft/2 + 1
	out := make([]float32, mels*nf)
	lo, hi := slaneyHzToMel(fmin), slaneyHzToMel(fmax)
	hz := make([]float64, mels+2)
	for i := range hz {
		hz[i] = slaneyMelToHz(lo + (hi-lo)*float64(i)/float64(mels+1))
	}
	for j := 0; j < mels; j++ {
		enorm := 2 / (hz[j+2] - hz[j])
		for k := 0; k < nf; k++ {
			f := float64(sr) * float64(k) / float64(nfft)
			l := (f - hz[j]) / (hz[j+1] - hz[j])
			u := (hz[j+2] - f) / (hz[j+2] - hz[j+1])
			out[j*nf+k] = float32(max(0, min(l, u)) * enorm)
		}
	}
	return out
}

// Frames is how many feature frames n samples give.
func (m *Mel) Frames(n int) int {
	if m.cfg.SeqLenCeil {
		return (n + m.cfg.Hop - 1) / m.cfg.Hop
	}
	return n/m.cfg.Hop + 1
}

// Valid is how many of those frames NeMo counts as real (the rest are
// padding it masks): floor(n/hop), or all of them with SeqLenCeil.
func (m *Mel) Valid(n int) int {
	if m.cfg.SeqLenCeil {
		return m.Frames(n)
	}
	return n / m.cfg.Hop
}

// fft runs an in-place radix-2 FFT on re, im.
func (m *Mel) fft(re, im []float64) {
	n := len(re)
	for i, r := range m.rev {
		if i < r {
			re[i], re[r] = re[r], re[i]
			im[i], im[r] = im[r], im[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half, step := size/2, n/size
		for s := 0; s < n; s += size {
			for k := 0; k < half; k++ {
				c, sn := m.cos[k*step], m.sin[k*step]
				a, b := s+k, s+k+half
				tr := re[b]*c - im[b]*sn
				ti := re[b]*sn + im[b]*c
				re[b], im[b] = re[a]-tr, im[a]-ti
				re[a] += tr
				im[a] += ti
			}
		}
	}
}

type melScratch struct{ re, im, pow []float64 }

func (m *Mel) scratch() *melScratch {
	n := m.cfg.NFFT
	return &melScratch{re: make([]float64, n), im: make([]float64, n), pow: make([]float64, n/2+1)}
}

// frame computes one log-mel column from up to NFFT pre-emphasized,
// padded samples (missing ones count as zero).
func (m *Mel) frame(win []float64, out []float32, sc *melScratch) {
	c := m.cfg
	nf := c.NFFT/2 + 1
	re, im, pow := sc.re, sc.im, sc.pow
	for i := range re {
		x := 0.0
		if i < len(win) {
			x = win[i]
		}
		re[i] = x * m.window[i]
		im[i] = 0
	}
	m.fft(re, im)
	for k := 0; k < nf; k++ {
		pow[k] = float64(float32(re[k]*re[k] + im[k]*im[k]))
	}
	for j := 0; j < c.NumMels; j++ {
		row := m.fb[j*nf:]
		var s float64
		for k := m.lo[j]; k < m.hi[j]; k++ {
			s += float64(row[k]) * pow[k]
		}
		out[j] = float32(math.Log(s + logEps))
	}
}

// Compute returns log-mel features as [mels][frames], row-major.
func (m *Mel) Compute(pcm []float32, threads int) ([]float32, int, error) {
	c := m.cfg
	n := len(pcm)
	frames := m.Frames(n)
	if frames < 2 {
		return nil, 0, errors.New("conformer: audio too short")
	}
	pad := c.NFFT / 2
	padded := make([]float64, n+2*pad)
	if n > 0 {
		padded[pad] = float64(pcm[0])
		for i := 1; i < n; i++ {
			padded[pad+i] = float64(pcm[i]) - c.PreEmphasis*float64(pcm[i-1])
		}
	}
	out := make([]float32, c.NumMels*frames)
	kern.Parallel(frames, threads, func(lo, hi int) {
		sc := m.scratch()
		col := make([]float32, c.NumMels)
		for t := lo; t < hi; t++ {
			start := t * c.Hop
			end := min(start+c.NFFT, len(padded))
			m.frame(padded[start:end], col, sc)
			for j, v := range col {
				out[j*frames+t] = v
			}
		}
	})
	switch c.Normalize {
	case "none", "":
		valid := n / c.Hop
		for j := 0; j < c.NumMels; j++ {
			for t := valid; t < frames; t++ {
				out[j*frames+t] = 0
			}
		}
	case "per_feature":
		nn := frames - 1 // constant padding: the last frame is padding
		for j := 0; j < c.NumMels; j++ {
			row := out[j*frames : (j+1)*frames]
			var sum float64
			for _, v := range row[:nn] {
				sum += float64(v)
			}
			mean := sum / float64(nn)
			var sq float64
			for _, v := range row[:nn] {
				d := float64(v) - mean
				sq += d * d
			}
			inv := 1 / (math.Sqrt(sq/float64(max(nn-1, 1))) + normEps) // one frame: no spread to divide by
			for t := 0; t < nn; t++ {
				row[t] = float32((float64(row[t]) - mean) * inv)
			}
			row[nn] = 0
		}
	default:
		return nil, 0, errors.New("conformer: front-end normalize " + c.Normalize + " is not supported")
	}
	return out, frames, nil
}
