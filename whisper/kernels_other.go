//go:build !amd64 || purego

package whisper

const useAVX = false

func setAVX(bool) {}

func kernel6x16(k int, a []float32, lda int, b []float32, c []float32, ldc int) {
	kernel6x16Go(k, a, lda, b, c, ldc)
}

func dotF16x4(x []float32, w []uint16, ld, k int, out *[4]float32) {
	for r := 0; r < 4; r++ {
		out[r] = dotF16Go(x[:k], w[r*ld:r*ld+k])
	}
}

func dotF32x4(x []float32, w []float32, ld, k int, out *[4]float32) {
	for r := 0; r < 4; r++ {
		out[r] = dotF32Go(x[:k], w[r*ld:r*ld+k])
	}
}

func dotF16(x []float32, y []uint16) float32 { return dotF16Go(x, y) }
func dotF32(x, y []float32) float32          { return dotF32Go(x, y) }

func expSub(x []float32, mx float32) float32 { return expSubGo(x, mx) }
func geluVec(x []float32)                    { gelu(x) }
