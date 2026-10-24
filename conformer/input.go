package conformer

import "math"

// finite returns pcm with any NaN or infinite sample made silent. One such
// sample would otherwise spread through normalization and the attention
// caches and fill minutes of output with junk words. pcm itself is only
// copied when it needs changing.
func finite(pcm []float32) []float32 {
	for i, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			out := append([]float32(nil), pcm...)
			for j := i; j < len(out); j++ {
				if f := float64(out[j]); math.IsNaN(f) || math.IsInf(f, 0) {
					out[j] = 0
				}
			}
			return out
		}
	}
	return pcm
}

// writePiece is the most audio a stream takes in at once: ten seconds, so
// a whole recording written in one call doesn't hold all of its samples
// and mel frames at the same time.
func writePiece(rate int) int { return 10 * rate }

// pieces cuts pcm into writePiece-sized pieces.
func pieces(pcm []float32, rate int) [][]float32 {
	n := writePiece(rate)
	if len(pcm) <= n {
		return [][]float32{pcm}
	}
	var out [][]float32
	for len(pcm) > 0 {
		k := min(n, len(pcm))
		out = append(out, pcm[:k])
		pcm = pcm[k:]
	}
	return out
}
