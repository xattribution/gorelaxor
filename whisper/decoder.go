package whisper

import (
	"math"
)

// kvCache is one decoding sequence's self-attention keys and values.
type kvCache struct {
	k, v [][]float32 // per layer: nTextCtx x state
	n    int         // positions filled
}

func (m *Model) newCache() *kvCache {
	c := &kvCache{}
	for range m.dec {
		c.k = append(c.k, make([]float32, m.hp.nTextCtx*m.hp.nTextState))
		c.v = append(c.v, make([]float32, m.hp.nTextCtx*m.hp.nTextState))
	}
	return c
}

func (c *kvCache) copyFrom(o *kvCache, st int) {
	for l := range c.k {
		copy(c.k[l][:o.n*st], o.k[l][:o.n*st])
		copy(c.v[l][:o.n*st], o.v[l][:o.n*st])
	}
	c.n = o.n
}

// decode runs the text decoder over tokens placed after the cache's
// positions and returns the logits at the positions listed in want
// (indexes into tokens).
func (m *Model) decode(ws *workspace, cache *kvCache, tokens []int, want []int, threads int) [][]float32 {
	hp := m.hp
	st := hp.nTextState
	nt := len(tokens)
	pos0 := cache.n
	if pos0+nt > hp.nTextCtx {
		return nil
	}
	x := ws.buf(sDecX, nt*st, false)
	for i, t := range tokens {
		row := x[i*st : (i+1)*st]
		widen(row, m.tokEmb.row(t))
		addTo(row, m.decPos[(pos0+i)*st:(pos0+i+1)*st])
	}
	nh := hp.nTextHead
	hd := st / nh
	sc := float32(1 / math.Sqrt(float64(hd)))
	nCtx := hp.nAudioCtx
	cur := ws.buf(sDecCur, nt*st, false)
	q := ws.buf(sDecQ, nt*st, false)
	att := ws.buf(sDecAtt, nt*st, false)
	ff := ws.buf(sDecFF, nt*4*st, false)
	kn := ws.buf(sDecK, nt*st, false)
	vn := ws.buf(sDecV, nt*st, false)
	sw := max(nCtx, pos0+nt) // a scores row per (token, head) job
	sbuf := ws.buf(sDecS, nt*nh*sw, false)

	for li := range m.dec {
		L := &m.dec[li]
		// Self-attention, causal.
		parLN(cur, x, nt, st, L.attnLnW, L.attnLnB, threads)
		ws.matmul(q, st, cur, st, nt, st, st, hmat(L.q), L.qB, threads)
		ws.matmul(kn, st, cur, st, nt, st, st, hmat(L.k), nil, threads)
		ws.matmul(vn, st, cur, st, nt, st, st, hmat(L.v), L.vB, threads)
		ck, cv := cache.k[li], cache.v[li]
		copy(ck[pos0*st:(pos0+nt)*st], kn)
		copy(cv[pos0*st:(pos0+nt)*st], vn)
		tasks(nt*nh, threads, func(job int) {
			i, h := job/nh, job%nh
			n := pos0 + i + 1
			s := sbuf[job*sw : job*sw+n]
			qi := q[i*st+h*hd : i*st+(h+1)*hd]
			for j := 0; j < n; j++ {
				s[j] = dotF32(qi, ck[j*st+h*hd:j*st+(h+1)*hd]) * sc
			}
			softmax(s)
			out := att[i*st+h*hd : i*st+(h+1)*hd]
			clear(out)
			for j, p := range s {
				vj := cv[j*st+h*hd : j*st+(h+1)*hd]
				for d := range out {
					out[d] += p * vj[d]
				}
			}
		})
		ws.matmul(cur, st, att, st, nt, st, st, hmat(L.o), L.oB, threads)
		addTo(x, cur)

		// Cross-attention over the encoded audio.
		parLN(cur, x, nt, st, L.xLnW, L.xLnB, threads)
		ws.matmul(q, st, cur, st, nt, st, st, hmat(L.xq), L.xqB, threads)
		tasks(nt*nh, threads, func(job int) {
			i, h := job/nh, job%nh
			kd := L.crossK[h*nCtx*hd : (h+1)*nCtx*hd]
			vd := L.crossV[h*nCtx*hd : (h+1)*nCtx*hd]
			qi := q[i*st+h*hd : i*st+(h+1)*hd]
			s := sbuf[job*sw : job*sw+nCtx]
			var o4 [4]float32
			j := 0
			for ; j+4 <= nCtx; j += 4 {
				dotF32x4(qi, kd[j*hd:], hd, hd, &o4)
				copy(s[j:j+4], o4[:])
			}
			for ; j < nCtx; j++ {
				s[j] = dotF32(qi, kd[j*hd:(j+1)*hd])
			}
			softmax(s)
			out := att[i*st+h*hd : i*st+(h+1)*hd]
			d := 0
			for ; d+4 <= hd; d += 4 {
				dotF32x4(s, vd[d*nCtx:], nCtx, nCtx, &o4)
				copy(out[d:d+4], o4[:])
			}
			for ; d < hd; d++ {
				out[d] = dotF32(s, vd[d*nCtx:(d+1)*nCtx])
			}
		})
		ws.matmul(cur, st, att, st, nt, st, st, hmat(L.xo), L.xoB, threads)
		addTo(x, cur)

		// Feed-forward.
		parLN(cur, x, nt, st, L.mlpLnW, L.mlpLnB, threads)
		ws.matmul(ff, 4*st, cur, st, nt, 4*st, st, hmat(L.fc1), L.fc1B, threads)
		parGelu(ff, threads)
		ws.matmul(cur, st, ff, 4*st, nt, st, 4*st, hmat(L.fc2), L.fc2B, threads)
		addTo(x, cur)
	}
	cache.n = pos0 + nt
	out := make([][]float32, len(want))
	row := make([]float32, st)
	for wi, i := range want {
		layerNorm(row, x[i*st:(i+1)*st], 1, st, m.decLnW, m.decLnB)
		lg := ws.buf(sLogits+min(wi, 3), hp.nVocab, false)
		gemv(lg, row, hmat(m.tokEmb), hp.nVocab, st, nil, threads)
		out[wi] = lg
	}
	return out
}
