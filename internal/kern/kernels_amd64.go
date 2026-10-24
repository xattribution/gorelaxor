//go:build !purego

package kern

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

//go:noescape
func dot4q8avx(x *float32, w *byte, ld int, nb int, out *[4]float32)

//go:noescape
func dequantq8avx(dst *float32, q *byte, nb int)

func cpuF16C() bool

// useAVX is true on processors with AVX2, FMA and F16C (Intel Haswell and
// AMD Zen onward): the assembly kernels run there.
var useAVX = cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpuF16C()

// setAVX turns the assembly off or back on (tests only).
func setAVX(on bool) {
	useAVX = on && cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpuF16C()
	setInt(intKind())
}

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

func dotQ8x4(x []float32, q []byte, ld, k int, out *[4]float32) {
	nb := k / QK
	if !useAVX || nb == 0 {
		dotQ8x4Go(x, q, ld, k, out)
		return
	}
	_ = q[3*ld+nb*Q8Bytes-1]
	_ = x[k-1]
	dot4q8avx(unsafe.SliceData(x), unsafe.SliceData(q), ld, nb, out)
}

func dequantQ8(dst []float32, q []byte) {
	nb := len(dst) / QK
	if !useAVX || nb == 0 || len(dst)%QK != 0 {
		dequantQ8Go(dst, q)
		return
	}
	_ = q[nb*Q8Bytes-1]
	dequantq8avx(unsafe.SliceData(dst), unsafe.SliceData(q), nb)
}

//go:noescape
func q8tileEVEX(nb int, a *byte, lda int, as *float32, w *byte, c *float32, ldc int)

//go:noescape
func q8tileVEX(nb int, a *byte, lda int, as *float32, w *byte, c *float32, ldc int)

//go:noescape
func quantU8avx(dst *byte, sc *float32, x *float32, nb int)

func intKind() int {
	switch {
	case !useAVX:
		return intNone
	case cpu.X86.HasAVX512 && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512VNNI:
		return intEVEX
	case cpu.X86.HasAVXVNNI:
		return intVEX
	}
	return intNone
}

var intMode = intKind()

// useInt is true where Q8_0 products run on the int8 kernel.
var useInt = intMode != intNone

// canRun says whether this processor has a kernel's instructions. A
// processor with both (AMD Zen 5, Intel Xeon from Sapphire Rapids) runs
// the EVEX one, and tests can still try the other.
func canRun(mode int) bool {
	switch mode {
	case intEVEX:
		return useAVX && cpu.X86.HasAVX512 && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512VNNI
	case intVEX:
		return useAVX && cpu.X86.HasAVXVNNI
	}
	return true
}

// setInt picks a kernel (tests only); it reports false if this
// processor can't run it, leaving the choice as it was.
func setInt(mode int) bool {
	if !canRun(mode) {
		return false
	}
	intMode = mode
	useInt = mode != intNone
	return true
}

func q8tile(nb int, a []byte, lda int, as []float32, w []byte, c []float32, ldc int) {
	if nb == 0 {
		return
	}
	// The bounds the assembly relies on.
	_ = a[(tileRows-1)*lda+nb*QK-1]
	_ = as[tileRows*nb-1]
	_ = w[nb*panelBlock-1]
	_ = c[(tileRows-1)*ldc+panelCols-1]
	switch intMode {
	case intEVEX:
		q8tileEVEX(nb, unsafe.SliceData(a), lda, unsafe.SliceData(as), unsafe.SliceData(w), unsafe.SliceData(c), ldc)
	case intVEX:
		q8tileVEX(nb, unsafe.SliceData(a), lda, unsafe.SliceData(as), unsafe.SliceData(w), unsafe.SliceData(c), ldc)
	default:
		q8tileGo(nb, a, lda, as, w, c, ldc)
	}
}

func quantU8(dst []byte, sc []float32, x []float32, nb int) {
	if !useAVX || nb == 0 {
		quantU8Go(dst, sc, x, nb)
		return
	}
	_ = dst[nb*QK-1]
	_ = sc[nb-1]
	_ = x[nb*QK-1]
	quantU8avx(unsafe.SliceData(dst), unsafe.SliceData(sc), unsafe.SliceData(x), nb)
}

//go:noescape
func siluAVX(x *float32, n int)

//go:noescape
func axpyAVX(dst, src *float32, n int, s float32)

//go:noescape
func reluAVX(x *float32, n int)

func siluVec(x []float32) {
	n := 0
	if useAVX {
		n = len(x) &^ 7
		if n > 0 {
			siluAVX(unsafe.SliceData(x), n)
		}
	}
	siluGo(x[n:])
}

func axpy(dst, src []float32, s float32) {
	dst = dst[:len(src)]
	n := 0
	if useAVX {
		n = len(dst) &^ 7
		if n > 0 {
			axpyAVX(unsafe.SliceData(dst), unsafe.SliceData(src), n, s)
		}
	}
	axpyGo(dst[n:], src[n:], s)
}

func reluVec(x []float32) {
	n := 0
	if useAVX {
		n = len(x) &^ 7
		if n > 0 {
			reluAVX(unsafe.SliceData(x), n)
		}
	}
	reluGo(x[n:])
}
