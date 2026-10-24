//go:build !amd64 || purego

package kern

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

func dotQ8x4(x []float32, q []byte, ld, k int, out *[4]float32) { dotQ8x4Go(x, q, ld, k, out) }

func dequantQ8(dst []float32, q []byte) { dequantQ8Go(dst, q) }

// No int8 assembly here: products take the float path, and tests can
// run the portable int8 kernel.
var useInt = false

func intKind() int { return intNone }

func setInt(mode int) bool {
	if mode == intEVEX || mode == intVEX {
		return false
	}
	useInt = mode != intNone
	return true
}

func q8tile(nb int, a []byte, lda int, as []float32, w []byte, c []float32, ldc int) {
	q8tileGo(nb, a, lda, as, w, c, ldc)
}

func quantU8(dst []byte, sc []float32, x []float32, nb int) { quantU8Go(dst, sc, x, nb) }

func siluVec(x []float32)                { siluGo(x) }
func axpy(dst, src []float32, s float32) { axpyGo(dst, src, s) }
func reluVec(x []float32)                { reluGo(x) }
