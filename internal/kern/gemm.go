package kern

import (
	"sync"
	"sync/atomic"
)

// mat is the right-hand side of a product: rows of length k, as float16,
// float32 or Q8_0 blocks (32 int8 values sharing one float16 scale, 34
// bytes a block), row j starting at element j*ld (for Q8_0, byte j*ld).
type mat struct {
	h  []uint16
	f  []float32
	q  []byte
	ld int
}

func fmat(f []float32, ld int) mat { return mat{f: f, ld: ld} }

// row widens row j's first k values into dst.
func (b mat) row(dst []float32, j, k int) {
	switch {
	case b.h != nil:
		widen(dst[:k], b.h[j*b.ld:j*b.ld+k])
	case b.q != nil:
		dequantQ8(dst[:k], b.q[j*b.ld:])
	default:
		copy(dst[:k], b.f[j*b.ld:j*b.ld+k])
	}
}

const (
	mr = 6   // rows per micro-kernel
	nr = 16  // columns per micro-kernel (one packed panel)
	kc = 256 // depth per pass, so a block of A stays in cache
	mc = 48  // rows per task
)

// workspace holds the packing buffer reused from call to call.
type workspace struct {
	pack []float32
}

func (w *workspace) packBuf(n int) []float32 {
	if cap(w.pack) < n {
		w.pack = make([]float32, n)
	}
	return w.pack[:n]
}

// parallel runs fn over [0,n) in chunks on up to threads goroutines.
func parallel(n, threads int, fn func(lo, hi int)) {
	if threads <= 1 || n <= 1 {
		fn(0, n)
		return
	}
	if threads > n {
		threads = n
	}
	var wg sync.WaitGroup
	var pe workerPanic
	for t := 0; t < threads; t++ {
		lo, hi := n*t/threads, n*(t+1)/threads
		if lo == hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer pe.catch()
			fn(lo, hi)
		}()
	}
	wg.Wait()
	pe.rethrow()
}

// tasks runs fn(i) for i in [0,n), handing them out to threads goroutines.
func tasks(n, threads int, fn func(i int)) {
	if threads <= 1 || n <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	if threads > n {
		threads = n
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	var pe workerPanic
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer pe.catch()
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
	pe.rethrow()
}

// smallM is the most rows of A that go through gemvRows instead of
// the packed, tiled kernel.
var smallM = 16

// matmul sets C[m][n] = A[m][k] · B[n][k]ᵀ + bias (bias may be nil).
// ldc and lda are the row strides of C and A.
func (w *workspace) matmul(c []float32, ldc int, a []float32, lda, m, n, k int, b mat, bias []float32, threads int) {
	if m <= smallM {
		gemvRows(c, ldc, a, lda, m, b, n, k, bias, threads)
		return
	}
	// Pack B into panels of nr rows, laid out [k][nr], widened to float32.
	np := (n + nr - 1) / nr
	packed := w.packBuf(np * k * nr)
	parallel(np, threads, func(lo, hi int) {
		row := make([]float32, k)
		for p := lo; p < hi; p++ {
			dst := packed[p*k*nr : (p+1)*k*nr]
			for j := 0; j < nr; j++ {
				col := p*nr + j
				if col >= n {
					for kk := 0; kk < k; kk++ {
						dst[kk*nr+j] = 0
					}
					continue
				}
				b.row(row, col, k)
				for kk, v := range row {
					dst[kk*nr+j] = v
				}
			}
		}
	})
	mFull := m / mr * mr
	nFull := n / nr * nr
	nBlocks := (mFull + mc - 1) / mc
	tasks(nBlocks, threads, func(t int) {
		i0, i1 := t*mc, min((t+1)*mc, mFull)
		for i := i0; i < i1; i++ {
			ci := c[i*ldc : i*ldc+n]
			if bias != nil {
				copy(ci, bias[:n])
			} else {
				clear(ci)
			}
		}
		var tile [mr * nr]float32
		for k0 := 0; k0 < k; k0 += kc {
			kn := min(kc, k-k0)
			for p := 0; p < np; p++ {
				bp := packed[p*k*nr+k0*nr:]
				if p*nr < nFull {
					for i := i0; i < i1; i += mr {
						kernel6x16(kn, a[i*lda+k0:], lda, bp, c[i*ldc+p*nr:], ldc)
					}
					continue
				}
				// The last, partial panel goes through a tile.
				w := n - p*nr
				for i := i0; i < i1; i += mr {
					for r := 0; r < mr; r++ {
						copy(tile[r*nr:r*nr+w], c[(i+r)*ldc+p*nr:(i+r)*ldc+n])
					}
					kernel6x16(kn, a[i*lda+k0:], lda, bp, tile[:], nr)
					for r := 0; r < mr; r++ {
						copy(c[(i+r)*ldc+p*nr:(i+r)*ldc+n], tile[r*nr:r*nr+w])
					}
				}
			}
		}
	})
	for i := mFull; i < m; i++ {
		gemv(c[i*ldc:i*ldc+n], a[i*lda:i*lda+k], b, n, k, bias, threads)
	}
}

// gemv sets y[n] = B[n][k] · x[k] + bias.
func gemv(y, x []float32, b mat, n, k int, bias []float32, threads int) {
	gemvRows(y, n, x, k, 1, b, n, k, bias, threads)
}

// gemvRows sets C[m][n] = A[m][k] · B[n][k]ᵀ + bias one group of four
// B rows at a time, running every row of A against the group while it
// is in cache. With few rows of A this beats packing B for the tiled
// kernel, since B is read from memory once and never copied.
func gemvRows(c []float32, ldc int, a []float32, lda, m int, b mat, n, k int, bias []float32, threads int) {
	if m*n*k < 1<<16 {
		threads = 1
	}
	// With several rows of A, Q8_0 rows are widened once per group
	// rather than once per row of A.
	widenQ8 := b.q != nil && m > 1
	parallel((n+3)/4, threads, func(lo, hi int) {
		var out [4]float32
		var wide []float32
		if widenQ8 {
			wide = make([]float32, 4*k)
		}
		for g := lo; g < hi; g++ {
			j := g * 4
			if widenQ8 && j+4 <= n {
				for r := 0; r < 4; r++ {
					dequantQ8(wide[r*k:(r+1)*k], b.q[(j+r)*b.ld:])
				}
				for i := 0; i < m; i++ {
					dotF32x4(a[i*lda:i*lda+k], wide, k, k, &out)
					y := c[i*ldc : i*ldc+n]
					copy(y[j:j+4], out[:])
					if bias != nil {
						for r := 0; r < 4; r++ {
							y[j+r] += bias[j+r]
						}
					}
				}
				continue
			}
			for i := 0; i < m; i++ {
				x := a[i*lda : i*lda+k]
				y := c[i*ldc : i*ldc+n]
				if j+4 <= n {
					switch {
					case b.h != nil:
						dotF16x4(x, b.h[j*b.ld:], b.ld, k, &out)
					case b.q != nil:
						dotQ8x4(x, b.q[j*b.ld:], b.ld, k, &out)
					default:
						dotF32x4(x, b.f[j*b.ld:], b.ld, k, &out)
					}
					copy(y[j:j+4], out[:])
				} else {
					for jj := j; jj < n; jj++ {
						switch {
						case b.h != nil:
							y[jj] = dotF16(x, b.h[jj*b.ld:jj*b.ld+k])
						case b.q != nil:
							y[jj] = dotQ8Go(x, b.q[jj*b.ld:])
						default:
							y[jj] = dotF32(x, b.f[jj*b.ld:jj*b.ld+k])
						}
					}
				}
				if bias != nil {
					for jj := j; jj < min(j+4, n); jj++ {
						y[jj] += bias[jj]
					}
				}
			}
		}
	})
}

// Portable kernels, used where there is no assembly version.

func kernel6x16Go(k int, a []float32, lda int, b []float32, c []float32, ldc int) {
	for r := 0; r < mr; r++ {
		var acc [nr]float32
		ar := a[r*lda : r*lda+k]
		bb := b[:k*nr]
		for kk, av := range ar {
			bk := bb[kk*nr : kk*nr+nr : kk*nr+nr]
			acc[0] += av * bk[0]
			acc[1] += av * bk[1]
			acc[2] += av * bk[2]
			acc[3] += av * bk[3]
			acc[4] += av * bk[4]
			acc[5] += av * bk[5]
			acc[6] += av * bk[6]
			acc[7] += av * bk[7]
			acc[8] += av * bk[8]
			acc[9] += av * bk[9]
			acc[10] += av * bk[10]
			acc[11] += av * bk[11]
			acc[12] += av * bk[12]
			acc[13] += av * bk[13]
			acc[14] += av * bk[14]
			acc[15] += av * bk[15]
		}
		cr := c[r*ldc : r*ldc+nr]
		for j := range cr {
			cr[j] += acc[j]
		}
	}
}

func dotF32Go(x, y []float32) float32 {
	y = y[:len(x)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(x); i += 4 {
		s0 += x[i] * y[i]
		s1 += x[i+1] * y[i+1]
		s2 += x[i+2] * y[i+2]
		s3 += x[i+3] * y[i+3]
	}
	for ; i < len(x); i++ {
		s0 += x[i] * y[i]
	}
	return s0 + s1 + s2 + s3
}

func dotF16Go(x []float32, y []uint16) float32 {
	y = y[:len(x)]
	t := f16tab
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(x); i += 4 {
		s0 += x[i] * t[y[i]]
		s1 += x[i+1] * t[y[i+1]]
		s2 += x[i+2] * t[y[i+2]]
		s3 += x[i+3] * t[y[i+3]]
	}
	for ; i < len(x); i++ {
		s0 += x[i] * t[y[i]]
	}
	return s0 + s1 + s2 + s3
}

// workerPanic carries a worker goroutine's panic back to the goroutine that
// started the work, where the caller's recover can handle it. Left on the
// worker, a panic (a model file whose tensors don't fit, say) would stop
// the whole program.
type workerPanic struct {
	mu  sync.Mutex
	v   any
	set bool
}

func (p *workerPanic) catch() {
	if r := recover(); r != nil {
		p.mu.Lock()
		if !p.set {
			p.v, p.set = r, true
		}
		p.mu.Unlock()
	}
}

func (p *workerPanic) rethrow() {
	if p.set {
		panic(p.v)
	}
}
