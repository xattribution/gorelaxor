//go:build !purego

package whisper

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func k6x16avx(k int, a *float32, lda int, b *float32, c *float32, ldc int)

//go:noescape
func dot4f16avx(x *float32, w *uint16, ld int, k int, out *[4]float32)

//go:noescape
func dot4f32avx(x *float32, w *float32, ld int, k int, out *[4]float32)

func cpuF16C() bool

// useAVX is true on processors with AVX2, FMA and F16C (Intel Haswell and
// AMD Zen onward): the assembly kernels run there.
var useAVX = cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpuF16C()

// setAVX turns the assembly off or back on (tests only).
func setAVX(on bool) { useAVX = on && cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpuF16C() }

func kernel6x16(k int, a []float32, lda int, b []float32, c []float32, ldc int) {
	if !useAVX {
		kernel6x16Go(k, a, lda, b, c, ldc)
		return
	}
	if k == 0 {
		return
	}
	// The bounds the assembly relies on.
	_ = a[5*lda+k-1]
	_ = b[k*nr-1]
	_ = c[5*ldc+nr-1]
	k6x16avx(k, unsafe.SliceData(a), lda, unsafe.SliceData(b), unsafe.SliceData(c), ldc)
}

func dotF16x4(x []float32, w []uint16, ld, k int, out *[4]float32) {
	if !useAVX {
		for r := 0; r < 4; r++ {
			out[r] = dotF16Go(x[:k], w[r*ld:r*ld+k])
		}
		return
	}
	kk := k &^ 15
	_ = w[3*ld+k-1]
	_ = x[k-1]
	if kk > 0 {
		dot4f16avx(unsafe.SliceData(x), unsafe.SliceData(w), ld, kk, out)
	} else {
		*out = [4]float32{}
	}
	if kk < k {
		for r := 0; r < 4; r++ {
			out[r] += dotF16Go(x[kk:k], w[r*ld+kk:r*ld+k])
		}
	}
}

func dotF32x4(x []float32, w []float32, ld, k int, out *[4]float32) {
	if !useAVX {
		for r := 0; r < 4; r++ {
			out[r] = dotF32Go(x[:k], w[r*ld:r*ld+k])
		}
		return
	}
	kk := k &^ 15
	_ = w[3*ld+k-1]
	_ = x[k-1]
	if kk > 0 {
		dot4f32avx(unsafe.SliceData(x), unsafe.SliceData(w), ld, kk, out)
	} else {
		*out = [4]float32{}
	}
	if kk < k {
		for r := 0; r < 4; r++ {
			out[r] += dotF32Go(x[kk:k], w[r*ld+kk:r*ld+k])
		}
	}
}

func dotF16(x []float32, y []uint16) float32 { return dotF16Go(x, y) }
func dotF32(x, y []float32) float32          { return dotF32Go(x, y) }

//go:noescape
func expSubAVX(x *float32, n int, mx float32) float32

//go:noescape
func geluAVX(x *float32, n int)

// expSub sets x[i] = e^(x[i]-mx) and returns their sum.
func expSub(x []float32, mx float32) float32 {
	if !useAVX {
		return expSubGo(x, mx)
	}
	n := len(x) &^ 7
	var s float32
	if n > 0 {
		s = expSubAVX(unsafe.SliceData(x), n, mx)
	}
	return s + expSubGo(x[n:], mx)
}

func geluVec(x []float32) {
	if !useAVX {
		gelu(x)
		return
	}
	n := len(x) &^ 7
	if n > 0 {
		geluAVX(unsafe.SliceData(x), n)
	}
	gelu(x[n:])
}
