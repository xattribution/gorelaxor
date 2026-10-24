package whisper

import (
	"context"
	"math"
)

// encode runs the audio encoder on one window's log-mel spectrogram and
// fills every decoder layer's cross-attention keys and values.
func (m *Model) encode(ctx context.Context, ws *workspace, mel []float32, threads int) error {
	hp := m.hp
	st, nMel := hp.nAudioState, hp.nMels
	nCtx := hp.nAudioCtx // 1500

	// conv1: 3 taps, stride 1, padding 1, over 3000 frames.
	cols1 := nMel * 3
	x1 := ws.buf(sConv1, winFrames*cols1, true)
	parallel(winFrames, threads, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := x1[t*cols1 : (t+1)*cols1]
			for c := 0; c < nMel; c++ {
				src := mel[c*winFrames : (c+1)*winFrames]
				for k := 0; k < 3; k++ {
					if i := t + k - 1; i >= 0 && i < winFrames {
						row[c*3+k] = src[i]
					}
				}
			}
		}
	})
	h1 := ws.buf(sConvOut1, winFrames*st, false)
	ws.matmul(h1, st, x1, cols1, winFrames, st, cols1, hmat(m.conv1W), m.conv1B, threads)
	parGelu(h1, threads)
	if err := ctx.Err(); err != nil {
		return err
	}

	// conv2: 3 taps, stride 2, padding 1, down to 1500 positions.
	cols2 := st * 3
	x2 := ws.buf(sConv2, nCtx*cols2, true)
	parallel(nCtx, threads, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := x2[t*cols2 : (t+1)*cols2]
			for k := 0; k < 3; k++ {
				i := 2*t + k - 1
				if i < 0 || i >= winFrames {
					continue
				}
				src := h1[i*st : (i+1)*st]
				for c, v := range src {
					row[c*3+k] = v
				}
			}
		}
	})
	x := ws.buf(sX, nCtx*st, false)
	ws.matmul(x, st, x2, cols2, nCtx, st, cols2, hmat(m.conv2W), m.conv2B, threads)
	parGelu(x, threads)
	addTo(x, m.encPos[:nCtx*st])

	nh := hp.nAudioHead
	hd := st / nh
	sc := float32(1 / math.Sqrt(float64(hd)))
	cur := ws.buf(sCur, nCtx*st, false)
	q := ws.buf(sQ, nCtx*st, false)
	k := ws.buf(sK, nCtx*st, false)
	v := ws.buf(sV, nCtx*st, false)
	att := ws.buf(sAtt, nCtx*st, false)
	ff := ws.buf(sFF, nCtx*4*st, false)
	qh := ws.buf(sQH, nCtx*hd, false)
	kh := ws.buf(sKH, nCtx*hd, false)
	vt := ws.buf(sVT, hd*nCtx, false)
	oh := ws.buf(sOH, nCtx*hd, false)
	scores := ws.buf(sScores, nCtx*nCtx, false)

	for li := range m.enc {
		L := &m.enc[li]
		parLN(cur, x, nCtx, st, L.attnLnW, L.attnLnB, threads)
		ws.matmul(q, st, cur, st, nCtx, st, st, hmat(L.q), L.qB, threads)
		ws.matmul(k, st, cur, st, nCtx, st, st, hmat(L.k), nil, threads)
		ws.matmul(v, st, cur, st, nCtx, st, st, hmat(L.v), L.vB, threads)
		for h := 0; h < nh; h++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			for t := 0; t < nCtx; t++ {
				copy(qh[t*hd:(t+1)*hd], q[t*st+h*hd:t*st+(h+1)*hd])
				copy(kh[t*hd:(t+1)*hd], k[t*st+h*hd:t*st+(h+1)*hd])
				for d := 0; d < hd; d++ {
					vt[d*nCtx+t] = v[t*st+h*hd+d]
				}
			}
			scale(qh, sc)
			ws.matmul(scores, nCtx, qh, hd, nCtx, nCtx, hd, fmat(kh, hd), nil, threads)
			parallel(nCtx, threads, func(lo, hi int) {
				for t := lo; t < hi; t++ {
					softmax(scores[t*nCtx : (t+1)*nCtx])
				}
			})
			ws.matmul(oh, hd, scores, nCtx, nCtx, hd, nCtx, fmat(vt, nCtx), nil, threads)
			for t := 0; t < nCtx; t++ {
				copy(att[t*st+h*hd:t*st+(h+1)*hd], oh[t*hd:(t+1)*hd])
			}
		}
		ws.matmul(cur, st, att, st, nCtx, st, st, hmat(L.o), L.oB, threads)
		addTo(x, cur)

		parLN(cur, x, nCtx, st, L.mlpLnW, L.mlpLnB, threads)
		ws.matmul(ff, 4*st, cur, st, nCtx, 4*st, st, hmat(L.fc1), L.fc1B, threads)
		parGelu(ff, threads)
		ws.matmul(cur, st, ff, 4*st, nCtx, st, 4*st, hmat(L.fc2), L.fc2B, threads)
		addTo(x, cur)
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	enc := cur
	parLN(enc, x, nCtx, st, m.encLnW, m.encLnB, threads)

	// Cross-attention keys (scaled) and values for every decoder layer,
	// kept per head: keys as rows, values transposed.
	nth := hp.nTextHead
	thd := st / nth
	tsc := float32(1 / math.Sqrt(float64(thd)))
	for li := range m.dec {
		L := &m.dec[li]
		ws.matmul(k, st, enc, st, nCtx, st, st, hmat(L.xk), nil, threads)
		ws.matmul(v, st, enc, st, nCtx, st, st, hmat(L.xv), L.xvB, threads)
		if len(L.crossK) != nCtx*st {
			L.crossK = make([]float32, nCtx*st)
			L.crossV = make([]float32, nCtx*st)
		}
		for h := 0; h < nth; h++ {
			kd := L.crossK[h*nCtx*thd : (h+1)*nCtx*thd]
			vd := L.crossV[h*nCtx*thd : (h+1)*nCtx*thd]
			for t := 0; t < nCtx; t++ {
				for d := 0; d < thd; d++ {
					kd[t*thd+d] = k[t*st+h*thd+d] * tsc
					vd[d*nCtx+t] = v[t*st+h*thd+d]
				}
			}
		}
	}
	return ctx.Err()
}

func parGelu(x []float32, threads int) {
	const chunk = 1 << 14
	n := (len(x) + chunk - 1) / chunk
	tasks(n, threads, func(i int) { geluVec(x[i*chunk : min((i+1)*chunk, len(x))]) })
}

func parLN(dst, src []float32, rows, n int, w, b []float32, threads int) {
	parallel(rows, threads, func(lo, hi int) {
		layerNorm(dst[lo*n:hi*n], src[lo*n:hi*n], hi-lo, n, w, b)
	})
}
