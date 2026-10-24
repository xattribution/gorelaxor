// Package kern holds the matrix and vector kernels the local speech
// engines share: products with float32, float16 or Q8_0 weights, run in
// AVX2 assembly where the processor has it and portable Go elsewhere.
// On processors with VNNI, products with Q8_0 weights and two or more
// rows round the activations to Q8_0 too and run on int8 kernels
// (q8int.go), about five times the float path's speed.
//
// It started as a copy of the kernels in whisper/, generalized to Q8_0
// weights, so the FastConformer engine (conformer/) could use them
// without changing whisper/. whisper/ can move onto it later.
package kern

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// Type is how a weight matrix is stored.
type Type int

const (
	F32  Type = iota
	F16       // IEEE half precision
	Q8_0      // blocks of 32 int8 values with one float16 scale (34 bytes)
)

// QK is the number of values in a Q8_0 block, and Q8Bytes its size.
const (
	QK      = 32
	Q8Bytes = 34
)

// Weight is a matrix of Rows rows of Cols values, row-major: the layout
// of a PyTorch Linear weight ([out, in]) and of a GGUF tensor whose
// ne[0] is Cols.
type Weight struct {
	Rows, Cols int
	Type       Type
	f          []float32
	h          []uint16
	q          []byte

	panelOnce sync.Once
	pan       *q8Panels // Q8_0 rows packed for the int8 kernel, on first use
}

// NewF32 wraps rows×cols float32 values.
func NewF32(rows, cols int, v []float32) (*Weight, error) {
	if !sizeOK(rows, cols, len(v)) || len(v) < rows*cols {
		return nil, fmt.Errorf("kern: %d values for a %dx%d matrix", len(v), rows, cols)
	}
	return &Weight{Rows: rows, Cols: cols, Type: F32, f: v[:rows*cols]}, nil
}

// NewF16 wraps rows×cols halves.
func NewF16(rows, cols int, v []uint16) (*Weight, error) {
	if !sizeOK(rows, cols, len(v)) || len(v) < rows*cols {
		return nil, fmt.Errorf("kern: %d halves for a %dx%d matrix", len(v), rows, cols)
	}
	return &Weight{Rows: rows, Cols: cols, Type: F16, h: v[:rows*cols]}, nil
}

// sizeOK refuses negative sizes and ones whose product can't be held in
// n elements, so rows*cols can't overflow into a small number.
func sizeOK(rows, cols, n int) bool {
	return rows >= 0 && cols >= 0 && (cols == 0 || rows <= (n+QK*Q8Bytes)/cols+1)
}

// NewQ8 wraps rows×cols values stored as Q8_0 blocks; cols must be a
// multiple of 32.
func NewQ8(rows, cols int, b []byte) (*Weight, error) {
	if cols%QK != 0 {
		return nil, errors.New("kern: Q8_0 rows must be a multiple of 32 values")
	}
	if !sizeOK(rows, cols, len(b)) || len(b) < rows*cols/QK*Q8Bytes {
		return nil, fmt.Errorf("kern: %d bytes for a %dx%d Q8_0 matrix", len(b), rows, cols)
	}
	return &Weight{Rows: rows, Cols: cols, Type: Q8_0, q: b[:rows*cols/QK*Q8Bytes]}, nil
}

// Pack lays a Q8_0 weight out for the int8 kernel now, where the
// processor has one, so the first product doesn't wait. Without Pack,
// that happens on the first product of two or more rows.
func (w *Weight) Pack(threads int) {
	if w.Type == Q8_0 && useInt {
		w.panels(threads)
	}
}

// DropRows lets go of a packed weight's Q8_0 rows, which the panels
// repeat, so it costs its memory once. All of its products then take
// the int8 kernel, a single row too. For weights multiplied many rows at
// a time (an encoder's), where single rows are rare.
func (w *Weight) DropRows() {
	if w.pan != nil {
		w.q = nil
	}
}

func (w *Weight) mat() mat {
	switch w.Type {
	case F16:
		return mat{h: w.h, ld: w.Cols}
	case Q8_0:
		return mat{q: w.q, ld: w.Cols / QK * Q8Bytes}
	}
	return mat{f: w.f, ld: w.Cols}
}

// Row widens row j into dst (len Cols).
func (w *Weight) Row(dst []float32, j int) {
	if w.Type == Q8_0 && w.q == nil {
		w.pan.row(dst, j)
		return
	}
	w.mat().row(dst, j, w.Cols)
}

// Float32 returns the whole matrix widened to float32.
func (w *Weight) Float32() []float32 {
	if w.Type == F32 {
		return w.f
	}
	out := make([]float32, w.Rows*w.Cols)
	for j := 0; j < w.Rows; j++ {
		w.Row(out[j*w.Cols:(j+1)*w.Cols], j)
	}
	return out
}

// ToQ8 returns the weight rounded to Q8_0 (each block of 32 values
// scaled by its largest magnitude over 127), or w itself if it is Q8_0
// already or its rows aren't a multiple of 32 long.
func (w *Weight) ToQ8() *Weight {
	if w.Type == Q8_0 || w.Cols%QK != 0 {
		return w
	}
	nb := w.Cols / QK
	q := make([]byte, w.Rows*nb*Q8Bytes)
	parallel(w.Rows, runtime.GOMAXPROCS(0), func(lo, hi int) {
		row := make([]float32, w.Cols)
		for j := lo; j < hi; j++ {
			w.Row(row, j)
			for b := 0; b < nb; b++ {
				blk := q[(j*nb+b)*Q8Bytes : (j*nb+b+1)*Q8Bytes]
				xs := row[b*QK : (b+1)*QK]
				var amax float32
				for _, v := range xs {
					amax = max(amax, float32(math.Abs(float64(v))))
				}
				d := amax / 127
				id := float32(0)
				if d != 0 {
					id = 1 / d
				}
				binary.LittleEndian.PutUint16(blk, f32toF16(d))
				for i, v := range xs {
					blk[2+i] = byte(int8(math.Round(float64(v * id))))
				}
			}
		}
	})
	out, _ := NewQ8(w.Rows, w.Cols, q)
	return out
}

// Bytes is the memory the weight holds.
func (w *Weight) Bytes() int {
	n := len(w.f)*4 + len(w.h)*2 + len(w.q)
	if w.pan != nil {
		n += len(w.pan.b)
	}
	return n
}

var wsPool = sync.Pool{New: func() any { return &workspace{} }}

// MatMul sets dst[m][w.Rows] = x[m][w.Cols] · Wᵀ + bias (bias may be nil),
// with rows of x lda apart and rows of dst ldc apart.
func MatMul(dst []float32, ldc int, x []float32, lda, m int, w *Weight, bias []float32, threads int) {
	if m == 0 || w.Rows == 0 {
		return
	}
	if w.Type == Q8_0 && (w.q == nil || useInt && m >= intMinM) {
		matmulQ8Int(dst, ldc, x, lda, m, w, bias, threads)
		return
	}
	ws := wsPool.Get().(*workspace)
	ws.matmul(dst, ldc, x, lda, m, w.Rows, w.Cols, w.mat(), bias, threads)
	wsPool.Put(ws)
}

// Linear is MatMul with packed rows: dst[m][w.Rows] = x[m][w.Cols]·Wᵀ + bias.
func Linear(dst, x []float32, m int, w *Weight, bias []float32, threads int) {
	MatMul(dst, w.Rows, x, w.Cols, m, w, bias, threads)
}

// MatMulF32 sets c[m][n] = a[m][k] · b[n][k]ᵀ for float32 operands with
// row strides lda, ldb and ldc (attention scores and the like).
func MatMulF32(c []float32, ldc int, a []float32, lda int, b []float32, ldb int, m, n, k, threads int) {
	if m == 0 || n == 0 {
		return
	}
	ws := wsPool.Get().(*workspace)
	ws.matmul(c, ldc, a, lda, m, n, k, mat{f: b, ld: ldb}, nil, threads)
	wsPool.Put(ws)
}

// Dot is Σ x[i]·y[i].
func Dot(x, y []float32) float32 { return dotF32(x, y) }

// Parallel runs fn over [0,n) in chunks on up to threads goroutines.
func Parallel(n, threads int, fn func(lo, hi int)) { parallel(n, threads, fn) }

// Tasks runs fn(i) for i in [0,n) on up to threads goroutines.
func Tasks(n, threads int, fn func(i int)) { tasks(n, threads, fn) }

// F16ToF32 widens one half.
func F16ToF32(h uint16) float32 { return f16tab[h] }

// F32ToF16 rounds to the nearest half.
func F32ToF16(f float32) uint16 { return f32toF16(f) }

// Expf is a fast float32 e^x (about 2e-7 relative).
func Expf(x float32) float32 { return expf(x) }

// ExpSub sets x[i] = e^(x[i]-mx) and returns their sum.
func ExpSub(x []float32, mx float32) float32 { return expSub(x, mx) }

// dequantQ8Go widens one Q8_0 row of len(dst) values.
func dequantQ8Go(dst []float32, q []byte) {
	for b := 0; b*QK < len(dst); b++ {
		blk := q[b*Q8Bytes : (b+1)*Q8Bytes]
		d := f16tab[binary.LittleEndian.Uint16(blk)]
		out := dst[b*QK : (b+1)*QK]
		qs := blk[2:Q8Bytes]
		for i := range out {
			out[i] = d * float32(int8(qs[i]))
		}
	}
}

// dotQ8Go is Σ x[i]·w[i] for one Q8_0 row of len(x) values.
func dotQ8Go(x []float32, q []byte) float32 {
	var s float32
	for b := 0; b*QK < len(x); b++ {
		blk := q[b*Q8Bytes : (b+1)*Q8Bytes]
		d := f16tab[binary.LittleEndian.Uint16(blk)]
		xs := x[b*QK : (b+1)*QK]
		qs := blk[2:Q8Bytes:Q8Bytes]
		var a0, a1, a2, a3 float32
		for i := 0; i < QK; i += 4 {
			a0 += xs[i] * float32(int8(qs[i]))
			a1 += xs[i+1] * float32(int8(qs[i+1]))
			a2 += xs[i+2] * float32(int8(qs[i+2]))
			a3 += xs[i+3] * float32(int8(qs[i+3]))
		}
		s += d * (a0 + a1 + a2 + a3)
	}
	return s
}

func dotQ8x4Go(x []float32, q []byte, ld, k int, out *[4]float32) {
	for r := 0; r < 4; r++ {
		out[r] = dotQ8Go(x[:k], q[r*ld:])
	}
}

// Float32s reinterprets little-endian bytes as float32s without copying.
func Float32s(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
}

// Uint16s reinterprets little-endian bytes as uint16s without copying.
func Uint16s(b []byte) []uint16 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/2)
}

// LayerNorm normalizes rows of n values (eps 1e-5) with gain w and bias b.
func LayerNorm(dst, src []float32, rows, n int, w, b []float32) { layerNorm(dst, src, rows, n, w, b) }

// Softmax replaces x with its softmax.
func Softmax(x []float32) { softmax(x) }

// GELU applies the tanh GELU in place.
func GELU(x []float32) { geluVec(x) }

// SiLU sets x[i] = x[i]·sigmoid(x[i]).
func SiLU(x []float32) { siluVec(x) }

// ReLU sets x[i] = max(x[i], 0).
func ReLU(x []float32) { reluVec(x) }

// AddScaled sets dst[i] += s·src[i].
func AddScaled(dst, src []float32, s float32) { axpy(dst, src, s) }

// Add sets dst[i] += src[i].
func Add(dst, src []float32) { addTo(dst, src) }

// Scale sets x[i] *= s.
func Scale(x []float32, s float32) { scale(x, s) }
