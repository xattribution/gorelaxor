package kern

import (
	"encoding/binary"
	"math"
	"sync"
)

// Int8 products for Q8_0 weights on processors with VNNI (Intel 11th
// generation and Xeon from Cascade Lake with AVX-512, Intel 12th
// generation onward with AVX-VNNI, AMD Zen 4 onward). The activations are
// rounded to Q8_0 as well, the way ggml does, so each block of 32 values
// is one vpdpbusd per four columns instead of eight float multiplies.

// A panel holds 16 weight rows (output columns) for the int8 kernel,
// block after block. Each 640-byte block is
//
//	[8][16][4] int8  four consecutive values of each of the 16 rows, eight times
//	[16] float32     the rows' Q8_0 scales
//	[16] int32       128·Σ values, the offset the unsigned activations add
//
// Columns past the end of the matrix are zero.
const (
	panelCols  = 16
	panelBlock = panelCols*QK + panelCols*4 + panelCols*4
	tileRows   = 6
)

type q8Panels struct {
	b      []byte
	nb, np int
}

// packQ8Panels lays n Q8_0 rows of k values out as panels.
func packQ8Panels(q []byte, n, k, threads int) *q8Panels {
	nb := k / QK
	np := (n + panelCols - 1) / panelCols
	ld := nb * Q8Bytes
	p := &q8Panels{b: make([]byte, np*nb*panelBlock), nb: nb, np: np}
	parallel(np, threads, func(lo, hi int) {
		for pi := lo; pi < hi; pi++ {
			for b := 0; b < nb; b++ {
				dst := p.b[(pi*nb+b)*panelBlock : (pi*nb+b+1)*panelBlock]
				for j := 0; j < panelCols; j++ {
					col := pi*panelCols + j
					if col >= n {
						continue
					}
					blk := q[col*ld+b*Q8Bytes : col*ld+(b+1)*Q8Bytes]
					var s int32
					for g := 0; g < QK/4; g++ {
						v := blk[2+4*g : 6+4*g : 6+4*g]
						copy(dst[g*64+j*4:g*64+j*4+4], v)
						s += int32(int8(v[0])) + int32(int8(v[1])) + int32(int8(v[2])) + int32(int8(v[3]))
					}
					d := f16tab[binary.LittleEndian.Uint16(blk)]
					binary.LittleEndian.PutUint32(dst[512+j*4:], math.Float32bits(d))
					binary.LittleEndian.PutUint32(dst[576+j*4:], uint32(128*s))
				}
			}
		}
	})
	return p
}

// row widens row j of the matrix the panels hold into dst.
func (p *q8Panels) row(dst []float32, j int) {
	pi, lane := j/panelCols, j%panelCols
	for b := 0; b < p.nb; b++ {
		blk := p.b[(pi*p.nb+b)*panelBlock:]
		d := math.Float32frombits(binary.LittleEndian.Uint32(blk[512+lane*4:]))
		for g := 0; g < QK/4; g++ {
			for i := 0; i < 4; i++ {
				dst[b*QK+g*4+i] = float32(int8(blk[g*64+lane*4+i])) * d
			}
		}
	}
}

// panels returns w's panels, packing them on first use.
func (w *Weight) panels(threads int) *q8Panels {
	w.panelOnce.Do(func() { w.pan = packQ8Panels(w.q, w.Rows, w.Cols, threads) })
	return w.pan
}

// quantU8Go rounds nb blocks of 32 values of x to Q8_0, storing each
// value plus 128 as an unsigned byte and each block's scale in sc.
func quantU8Go(dst []byte, sc []float32, x []float32, nb int) {
	for b := 0; b < nb; b++ {
		xs := x[b*QK : (b+1)*QK]
		var amax float32
		for _, v := range xs {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		sc[b] = amax * (1.0 / 127)
		id := float32(0)
		if amax != 0 {
			id = 127 / amax
		}
		for i, v := range xs {
			q := int32(math.RoundToEven(float64(v * id)))
			dst[b*QK+i] = byte(min(127, max(-128, q)) + 128)
		}
	}
}

// q8tileGo is the portable form of the int8 tile kernel.
func q8tileGo(nb int, a []byte, lda int, as []float32, w []byte, c []float32, ldc int) {
	for b := 0; b < nb; b++ {
		blk := w[b*panelBlock : (b+1)*panelBlock]
		for r := 0; r < tileRows; r++ {
			ar := a[r*lda+b*QK : r*lda+(b+1)*QK]
			da := as[r*nb+b]
			for j := 0; j < panelCols; j++ {
				var s int32
				for i, u := range ar {
					s += int32(u) * int32(int8(blk[(i/4)*64+j*4+i%4]))
				}
				s -= int32(binary.LittleEndian.Uint32(blk[576+j*4:]))
				dw := math.Float32frombits(binary.LittleEndian.Uint32(blk[512+j*4:]))
				c[r*ldc+j] += float32(s) * dw * da
			}
		}
	}
}

// Which int8 kernel this processor runs, if any.
const (
	intNone = iota
	intEVEX // AVX512-VNNI with AVX512-VL
	intVEX  // AVX-VNNI
	intGo   // the portable kernel (tests only)
)

// intMinM is the fewest rows of activations that take the int8 kernel;
// a single row reads the Q8_0 rows directly.
var intMinM = 2

// intBuf holds a call's quantized activations.
type intBuf struct {
	q  []byte
	sc []float32
}

var intPool = sync.Pool{New: func() any { return &intBuf{} }}

// matmulQ8Int sets C[m][n] = A[m][k] · Wᵀ + bias with the int8 kernel.
func matmulQ8Int(c []float32, ldc int, a []float32, lda, m int, w *Weight, bias []float32, threads int) {
	n, k := w.Rows, w.Cols
	pan := w.panels(threads)
	nb := pan.nb
	mp := (m + tileRows - 1) / tileRows * tileRows
	buf := intPool.Get().(*intBuf)
	defer intPool.Put(buf)
	if cap(buf.q) < mp*k {
		buf.q = make([]byte, mp*k)
	}
	if cap(buf.sc) < mp*nb {
		buf.sc = make([]float32, mp*nb)
	}
	qa, sc := buf.q[:mp*k], buf.sc[:mp*nb]
	qt := threads
	if m*k < 1<<15 {
		qt = 1
	}
	parallel(m, qt, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			quantU8(qa[i*k:(i+1)*k], sc[i*nb:(i+1)*nb], a[i*lda:i*lda+k], nb)
		}
	})
	for i := m * k; i < mp*k; i++ {
		qa[i] = 128
	}
	clear(sc[m*nb:])
	for i := 0; i < m; i++ {
		ci := c[i*ldc : i*ldc+n]
		if bias != nil {
			copy(ci, bias[:n])
		} else {
			clear(ci)
		}
	}
	// Tasks are a group of row tiles against a run of panels, so each
	// panel stays in cache across the group.
	tiles := mp / tileRows
	const group = 8
	rowGroups := (tiles + group - 1) / group
	chunks := 1
	if threads > 1 {
		chunks = min(pan.np, max(1, (4*threads+rowGroups-1)/rowGroups))
	}
	if m*n*k < 1<<18 {
		threads = 1
	}
	tasks(rowGroups*chunks, threads, func(t int) {
		g, ch := t/chunks, t%chunks
		t0, t1 := g*group, min((g+1)*group, tiles)
		p0, p1 := pan.np*ch/chunks, pan.np*(ch+1)/chunks
		var tile [tileRows * panelCols]float32
		for p := p0; p < p1; p++ {
			wp := pan.b[p*nb*panelBlock : (p+1)*nb*panelBlock]
			col := p * panelCols
			cols := min(panelCols, n-col)
			for ti := t0; ti < t1; ti++ {
				r0 := ti * tileRows
				ar, as := qa[r0*k:], sc[r0*nb:]
				rows := min(tileRows, m-r0)
				if rows == tileRows && cols == panelCols {
					q8tile(nb, ar, k, as, wp, c[r0*ldc+col:], ldc)
					continue
				}
				clear(tile[:])
				for r := 0; r < rows; r++ {
					copy(tile[r*panelCols:r*panelCols+cols], c[(r0+r)*ldc+col:(r0+r)*ldc+col+cols])
				}
				q8tile(nb, ar, k, as, wp, tile[:], panelCols)
				for r := 0; r < rows; r++ {
					copy(c[(r0+r)*ldc+col:(r0+r)*ldc+col+cols], tile[r*panelCols:r*panelCols+cols])
				}
			}
		}
	})
}
