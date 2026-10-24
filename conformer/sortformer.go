package conformer

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// Diarizer is NVIDIA's streaming Sortformer
// (diar_streaming_sortformer_4spk-v2.1): a FastConformer encoder, a
// small Transformer and a head giving each of four speakers' chance of
// talking every 80 ms. Speakers keep their numbers through a recording
// thanks to the speaker cache (NeMo's arrival-order speaker cache,
// AOSC). This is a translation of NeMo's SortformerEncLabelModel
// forward_streaming and SortformerModules.streaming_update.
type Diarizer struct {
	Mel      *Mel
	Speakers int
	enc      *Model
	proj     vadLinear
	tf       []tfBlock
	tfHeads  int
	fc1      vadLinear
	head     vadLinear
	// Defaults are the streaming settings stored in the model file.
	Defaults DiarParams
}

type tfBlock struct {
	q, k, v, out vadLinear
	ln1w, ln1b   []float32
	ffIn, ffOut  vadLinear
	ln2w, ln2b   []float32
}

// DiarParams are Sortformer's streaming settings, in 80 ms frames.
type DiarParams struct {
	ChunkLen, LeftContext, RightContext int
	FIFOLen, UpdatePeriod, CacheLen     int
	SilFramesPerSpk                     int
	PredScoreThreshold                  float64
	ScoresBoostLatest                   float64
	SilThreshold                        float64
	StrongBoostRate, WeakBoostRate      float64
	MinPosScoresRate                    float64
}

// Presets from NVIDIA's model cards (chunk, right context, FIFO, update
// period, speaker cache; HighLatency is on the v2 card). HighLatency is
// the one a CPU keeps up with live: each step covers 10 s, and
// LowLatency's 0.48 s steps cost about 20 times more per second of audio.
var (
	DiarVeryHighLatency = DiarParams{ChunkLen: 340, RightContext: 40, FIFOLen: 40, UpdatePeriod: 300, CacheLen: 188}
	DiarHighLatency     = DiarParams{ChunkLen: 124, RightContext: 1, FIFOLen: 124, UpdatePeriod: 124, CacheLen: 188}
	DiarLowLatency      = DiarParams{ChunkLen: 6, RightContext: 7, FIFOLen: 188, UpdatePeriod: 144, CacheLen: 188}
)

// LoadDiarizer opens a converted Sortformer file.
func LoadDiarizer(path string) (*Diarizer, error) {
	var d *Diarizer
	err := checked(func() error {
		var err error
		if d, err = loadDiarizer(path); err != nil {
			return err
		}
		_, err = d.Diarize(make([]float32, d.Mel.cfg.SampleRate), DiarHighLatency, 1)
		return err
	})
	if err != nil {
		return nil, err
	}
	return d, nil
}

func loadDiarizer(path string) (*Diarizer, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	if a := f.String("general.architecture", ""); a != "sortformer" {
		return nil, fmt.Errorf("conformer: %q is not a Sortformer file", a)
	}
	m := &Model{}
	if err := readEncoder(f, "stt.sortformer.encoder", &m.HP); err != nil {
		return nil, err
	}
	mc := melConfig(f)
	if m.Mel, err = NewMel(mc); err != nil {
		return nil, err
	}
	l := &loader{f: f}
	m.loadEncoder(l)
	lin := func(n string) vadLinear { return vadLinear{w: l.mat(n+".weight", false), b: l.vec(n+".bias", false)} }
	d := &Diarizer{
		Mel: m.Mel, enc: m,
		Speakers: f.Int("stt.sortformer.max_speakers", 4),
		proj:     lin("diar.encoder_proj"),
		fc1:      lin("diar.fc1"),
		head:     lin("diar.single_spk_head"),
		tfHeads:  f.Int("stt.sortformer.transformer.n_heads", 8),
	}
	if f.Bool("stt.sortformer.transformer.pre_ln", false) {
		return nil, errors.New("conformer: pre-LN Sortformer transformers are not supported")
	}
	if a := f.String("stt.sortformer.transformer.activation", "relu"); a != "relu" {
		return nil, fmt.Errorf("conformer: Sortformer activation %q is not supported", a)
	}
	for i := 0; i < f.Int("stt.sortformer.transformer.n_layers", 0); i++ {
		p := fmt.Sprintf("tf.blocks.%d.", i)
		d.tf = append(d.tf, tfBlock{
			q: lin(p + "attn.q"), k: lin(p + "attn.k"), v: lin(p + "attn.v"), out: lin(p + "attn.out"),
			ln1w: l.vec(p+"norm_1.weight", false), ln1b: l.vec(p+"norm_1.bias", false),
			ffIn: lin(p + "ff.in"), ffOut: lin(p + "ff.out"),
			ln2w: l.vec(p+"norm_2.weight", false), ln2b: l.vec(p+"norm_2.bias", false),
		})
	}
	if l.err != nil {
		return nil, l.err
	}
	s := "stt.sortformer.stream."
	d.Defaults = DiarParams{
		ChunkLen: f.Int(s+"chunk_len", 188), LeftContext: f.Int(s+"chunk_left_context", 1), RightContext: f.Int(s+"chunk_right_context", 1),
		FIFOLen: f.Int(s+"fifo_len", 0), UpdatePeriod: f.Int(s+"spkcache_update_period", 188), CacheLen: f.Int(s+"spkcache_len", 188),
		SilFramesPerSpk:    f.Int(s+"spkcache_sil_frames_per_spk", 3),
		PredScoreThreshold: f.Float(s+"pred_score_threshold", 0.25),
		ScoresBoostLatest:  f.Float(s+"scores_boost_latest", 0.05),
		SilThreshold:       f.Float(s+"sil_threshold", 0.2),
		StrongBoostRate:    f.Float(s+"strong_boost_rate", 0.75),
		WeakBoostRate:      f.Float(s+"weak_boost_rate", 1.5),
		MinPosScoresRate:   f.Float(s+"min_pos_scores_rate", 0.5),
	}
	return d, nil
}

// complete fills the settings a preset leaves at zero from the file.
func (d *Diarizer) complete(p DiarParams) (DiarParams, error) {
	def := d.Defaults
	if p.LeftContext == 0 {
		p.LeftContext = def.LeftContext
	}
	if p.SilFramesPerSpk == 0 {
		p.SilFramesPerSpk = def.SilFramesPerSpk
	}
	for _, pair := range []struct {
		v *float64
		d float64
	}{
		{&p.PredScoreThreshold, def.PredScoreThreshold}, {&p.ScoresBoostLatest, def.ScoresBoostLatest},
		{&p.SilThreshold, def.SilThreshold}, {&p.StrongBoostRate, def.StrongBoostRate},
		{&p.WeakBoostRate, def.WeakBoostRate}, {&p.MinPosScoresRate, def.MinPosScoresRate},
	} {
		if *pair.v == 0 {
			*pair.v = pair.d
		}
	}
	if p.ChunkLen < 1 || p.UpdatePeriod < 1 || p.FIFOLen < 0 || p.CacheLen < (1+p.SilFramesPerSpk)*d.Speakers {
		return p, errors.New("conformer: Sortformer streaming settings out of range")
	}
	return p, nil
}

// transformer runs the post-norm Transformer over x [T][d] in place;
// rows from L on are padding.
func (d *Diarizer) transformer(x []float32, T, L, threads int) {
	if len(d.tf) == 0 {
		return
	}
	D := d.tf[0].q.w.Cols
	H := d.tfHeads
	hd := D / H
	q, k, v := make([]float32, T*D), make([]float32, T*D), make([]float32, T*D)
	ctx := make([]float32, T*D)
	tmp := make([]float32, T*D)
	vt := make([]float32, D*max(L, 1))
	scale := float32(1 / math.Sqrt(float64(hd)))
	for _, b := range d.tf {
		kern.Linear(q, x, T, b.q.w, b.q.b, threads)
		kern.Linear(k, x, T, b.k.w, b.k.b, threads)
		kern.Linear(v, x, T, b.v.w, b.v.b, threads)
		if L == 0 {
			clear(ctx)
		} else {
			// v transposed per head, so both products run on the tiled kernel
			for j := 0; j < L; j++ {
				for c := 0; c < D; c++ {
					vt[c*L+j] = v[j*D+c]
				}
			}
			kern.Tasks(H, threads, func(h int) {
				sc := make([]float32, T*L)
				kern.MatMulF32(sc, L, q[h*hd:], D, k[h*hd:], D, T, L, hd, 1)
				for i := 0; i < T; i++ {
					row := sc[i*L : (i+1)*L]
					kern.Scale(row, scale)
					kern.Softmax(row)
				}
				kern.MatMulF32(ctx[h*hd:], D, sc, L, vt[h*hd*L:], L, T, hd, L, 1)
			})
		}
		kern.Linear(tmp, ctx, T, b.out.w, b.out.b, threads)
		kern.Add(x[:T*D], tmp)
		kern.LayerNorm(x, x, T, D, b.ln1w, b.ln1b)
		inner := make([]float32, T*b.ffIn.w.Rows)
		kern.Linear(inner, x, T, b.ffIn.w, b.ffIn.b, threads)
		relu(inner)
		kern.Linear(tmp, inner, T, b.ffOut.w, b.ffOut.b, threads)
		kern.Add(x[:T*D], tmp)
		kern.LayerNorm(x, x, T, D, b.ln2w, b.ln2b)
	}
}

// predict runs encoder, Transformer and head over pre-encoder
// embeddings [T][512] (L real) and returns speaker chances [T][spk].
func (d *Diarizer) predict(emb []float32, T, L, threads int) []float32 {
	enc := append([]float32(nil), emb...)
	d.enc.encodeEmbeddings(enc, T, L, threads, nil) // works in place on all T rows
	pd := d.proj.w.Rows
	h := make([]float32, T*pd)
	kern.Linear(h, enc, T, d.proj.w, d.proj.b, threads)
	d.transformer(h, T, L, threads)
	relu(h)
	h2 := make([]float32, T*pd)
	kern.Linear(h2, h, T, d.fc1.w, d.fc1.b, threads)
	relu(h2)
	out := make([]float32, T*d.Speakers)
	kern.Linear(out, h2, T, d.head.w, d.head.b, threads)
	for i := range out {
		if i/d.Speakers >= L {
			out[i] = 0
			continue
		}
		out[i] = sigmoid(out[i])
	}
	return out
}

// DiarStream follows speakers through audio as it arrives.
type DiarStream struct {
	d       *Diarizer
	p       DiarParams
	threads int
	sub     int

	// audio and features, as in Stream
	emph    []float64
	emphBas int
	nIn     int
	lastIn  float32
	mel     [][]float32
	melBase int
	stt     int // next chunk's first mel frame
	scratch *melScratch
	closed  bool

	// NeMo's StreamingSortformerState
	cache, cachePreds [][]float32 // cachePreds nil until the cache first fills
	cacheSet          bool
	fifo, fifoPreds   [][]float32
	meanSil           []float64
	nSil              int

	frames int
}

// NewStream starts following speakers with the given settings.
func (d *Diarizer) NewStream(p DiarParams, threads int) (*DiarStream, error) {
	p, err := d.complete(p)
	if err != nil {
		return nil, err
	}
	return &DiarStream{d: d, p: p, threads: threads, sub: d.enc.HP.Subsampling, scratch: d.Mel.scratch(),
		meanSil: make([]float64, d.enc.HP.DModel)}, nil
}

// Diarize runs a whole recording and returns each 80 ms frame's
// speaker chances, [frames][speakers].
func (d *Diarizer) Diarize(pcm []float32, p DiarParams, threads int) ([][]float32, error) {
	s, err := d.NewStream(p, threads)
	if err != nil {
		return nil, err
	}
	out := s.Write(pcm)
	return append(out, s.Close()...), nil
}

// Write adds 16 kHz mono samples and returns the speaker chances of the
// frames that became final.
func (s *DiarStream) Write(pcm []float32) [][]float32 {
	if s.closed || len(pcm) == 0 {
		return nil
	}
	// A long recording goes through in pieces, so its samples and mel
	// frames aren't all held at once.
	if piece := writePiece(s.d.Mel.cfg.SampleRate); len(pcm) > piece {
		var out [][]float32
		for len(pcm) > 0 {
			n := min(piece, len(pcm))
			out = append(out, s.Write(pcm[:n])...)
			pcm = pcm[n:]
		}
		return out
	}
	pcm = finite(pcm)
	cfg := s.d.Mel.cfg
	if s.nIn == 0 {
		s.emph = make([]float64, cfg.NFFT/2)
		s.emphBas = -cfg.NFFT / 2
		s.emph = append(s.emph, float64(pcm[0]))
		for i := 1; i < len(pcm); i++ {
			s.emph = append(s.emph, float64(pcm[i])-cfg.PreEmphasis*float64(pcm[i-1]))
		}
	} else {
		prev := s.lastIn
		for _, v := range pcm {
			s.emph = append(s.emph, float64(v)-cfg.PreEmphasis*float64(prev))
			prev = v
		}
	}
	s.nIn += len(pcm)
	s.lastIn = pcm[len(pcm)-1]
	s.makeFrames()
	return s.run(false)
}

// Close finishes the recording.
func (s *DiarStream) Close() [][]float32 {
	if s.closed {
		return nil
	}
	s.closed = true
	if s.nIn == 0 {
		return nil
	}
	s.makeFrames()
	return s.run(true)
}

// makeFrames computes the mel frames NeMo counts as real (floor(n/hop))
// as soon as their audio is in.
func (s *DiarStream) makeFrames() {
	c := s.d.Mel.cfg
	for {
		t := s.melBase + len(s.mel)
		if t >= s.nIn/c.Hop || (!s.closed && t*c.Hop+c.NFFT/2 > s.nIn) {
			break
		}
		lo := t*c.Hop - c.NFFT/2 - s.emphBas
		hi := min(lo+c.NFFT, len(s.emph))
		col := make([]float32, c.NumMels)
		s.d.Mel.frame(s.emph[lo:hi], col, s.scratch)
		s.mel = append(s.mel, col)
	}
	next := (s.melBase+len(s.mel))*c.Hop - c.NFFT/2
	if cut := next - s.emphBas; cut > 0 && cut <= len(s.emph) {
		s.emph = append(s.emph[:0], s.emph[cut:]...)
		s.emphBas = next
	}
}

// run takes every chunk whose right context is in, as NeMo's
// streaming_feat_loader would hand them out.
func (s *DiarStream) run(final bool) [][]float32 {
	p := s.p
	var out [][]float32
	avail := s.melBase + len(s.mel)
	for s.stt < avail {
		left := min(p.LeftContext*s.sub, s.stt)
		end := s.stt + p.ChunkLen*s.sub
		right := p.RightContext * s.sub
		if final {
			end = min(end, avail)
			right = min(right, avail-end)
		} else if end+right > avail {
			break
		}
		n := end + right - (s.stt - left)
		mel := make([]float32, s.d.Mel.cfg.NumMels*n)
		for t := 0; t < n; t++ {
			for j, v := range s.mel[s.stt-left+t-s.melBase] {
				mel[j*n+t] = v
			}
		}
		lc := int(math.Round(float64(left) / float64(s.sub)))
		rc := (right + s.sub - 1) / s.sub
		out = append(out, s.step(mel, n, n, 0, lc, rc)...)
		s.stt = end
	}
	keep := s.stt - p.LeftContext*s.sub
	if keep > s.melBase {
		k := min(keep-s.melBase, len(s.mel))
		s.mel = append(s.mel[:0], s.mel[k:]...)
		s.melBase += k
	}
	return out
}

func rowsOf(x []float32, T, D int) [][]float32 {
	out := make([][]float32, T)
	for i := range out {
		out[i] = append([]float32(nil), x[i*D:(i+1)*D]...)
	}
	return out
}

// step is NeMo's forward_streaming_step for one chunk of n mel frames
// (length real): drop leading pre-encoder frames are discarded, and of
// the rest the first lc and last rc are context, not output.
func (s *DiarStream) step(mel []float32, n, length, drop, lc, rc int) [][]float32 {
	d := s.d
	m := d.enc
	D := m.HP.DModel
	ns := d.Speakers
	emb, Tc, Lc := m.subsample(mel, n, length, s.threads)
	if drop > 0 {
		emb, Tc, Lc = emb[min(drop, Tc)*D:], max(0, Tc-drop), max(0, Lc-drop)
	}
	nc, nf := len(s.cache), len(s.fifo)
	T := nc + nf + Tc
	x := make([]float32, 0, T*D)
	for _, r := range s.cache {
		x = append(x, r...)
	}
	for _, r := range s.fifo {
		x = append(x, r...)
	}
	x = append(x, emb...)
	preds := d.predict(x, T, nc+nf+Lc, s.threads)
	row := func(i int) []float32 { return append([]float32(nil), preds[i*ns:(i+1)*ns]...) }

	// streaming_update
	p := s.p
	chunkLen := Tc - lc - rc
	fifoPreds := make([][]float32, 0, nf+chunkLen)
	for i := nc; i < nc+nf; i++ {
		fifoPreds = append(fifoPreds, row(i))
	}
	chunkRows := rowsOf(emb[lc*D:], chunkLen, D)
	var chunkPreds [][]float32
	for i := 0; i < chunkLen; i++ {
		chunkPreds = append(chunkPreds, row(nc+nf+lc+i))
	}
	s.fifo = append(s.fifo, chunkRows...)
	s.fifoPreds = append(fifoPreds, chunkPreds...)
	if nf+chunkLen > p.FIFOLen {
		pop := p.UpdatePeriod
		pop = max(pop, chunkLen-p.FIFOLen+nf)
		pop = min(pop, nf+chunkLen)
		popEmb, popPreds := s.fifo[:pop], s.fifoPreds[:pop]
		s.silenceProfile(popEmb, popPreds)
		s.fifo = append([][]float32(nil), s.fifo[pop:]...)
		s.fifoPreds = append([][]float32(nil), s.fifoPreds[pop:]...)
		s.cache = append(s.cache, popEmb...)
		if s.cacheSet {
			s.cachePreds = append(s.cachePreds, popPreds...)
		}
		if len(s.cache) > p.CacheLen {
			if !s.cacheSet {
				s.cachePreds = nil
				for i := 0; i < nc; i++ {
					s.cachePreds = append(s.cachePreds, row(i))
				}
				s.cachePreds = append(s.cachePreds, popPreds...)
				s.cacheSet = true
			}
			s.compress()
		}
	}
	s.frames += len(chunkPreds)
	return chunkPreds
}

func (s *DiarStream) silenceProfile(emb, preds [][]float32) {
	var cnt int
	sum := make([]float64, len(s.meanSil))
	for i, pr := range preds {
		var t float64
		for _, v := range pr {
			t += float64(v)
		}
		if t < s.p.SilThreshold {
			cnt++
			for c, v := range emb[i] {
				sum[c] += float64(v)
			}
		}
	}
	if cnt == 0 {
		return
	}
	n := s.nSil + cnt
	for c := range s.meanSil {
		s.meanSil[c] = (s.meanSil[c]*float64(s.nSil) + sum[c]) / float64(max(n, 1))
	}
	s.nSil = n
}

// compress keeps CacheLen frames of the speaker cache: for each speaker
// its most telling frames (confident, single-speaker speech), boosted
// toward the newest, in time order, plus a few silence frames.
func (s *DiarStream) compress() {
	p := s.p
	ns := s.d.Speakers
	n := len(s.cache)
	perSpk := p.CacheLen/ns - p.SilFramesPerSpk
	strong := int(math.Floor(float64(perSpk) * p.StrongBoostRate))
	weak := int(math.Floor(float64(perSpk) * p.WeakBoostRate))
	minPos := int(math.Floor(float64(perSpk) * p.MinPosScoresRate))
	thr := p.PredScoreThreshold
	inf := math.Inf(1)
	// scores [frame][spk] in float32, as NeMo computes them
	sc := make([][]float32, n)
	for t := 0; t < n; t++ {
		pr := s.cachePreds[t]
		sc[t] = make([]float32, ns)
		var l1sum float32
		for _, v := range pr {
			l1sum += float32(math.Log(float64(max(1-v, float32(thr)))))
		}
		for j, v := range pr {
			lp := float32(math.Log(float64(max(v, float32(thr)))))
			l1 := float32(math.Log(float64(max(1-v, float32(thr)))))
			sc[t][j] = lp - l1 + l1sum - float32(math.Log(0.5))
		}
	}
	// _disable_low_scores
	pos := make([]int, ns)
	for t := 0; t < n; t++ {
		for j := 0; j < ns; j++ {
			if s.cachePreds[t][j] <= 0.5 {
				sc[t][j] = float32(math.Inf(-1))
			} else if sc[t][j] > 0 {
				pos[j]++
			}
		}
	}
	for t := 0; t < n; t++ {
		for j := 0; j < ns; j++ {
			if s.cachePreds[t][j] > 0.5 && !(sc[t][j] > 0) && pos[j] >= minPos {
				sc[t][j] = float32(math.Inf(-1))
			}
		}
	}
	if p.ScoresBoostLatest > 0 {
		for t := p.CacheLen; t < n; t++ {
			for j := range sc[t] {
				sc[t][j] += float32(p.ScoresBoostLatest)
			}
		}
	}
	boost := func(k int, scale float64) {
		add := float32(-scale * math.Log(0.5))
		for j := 0; j < ns; j++ {
			idx := topK(n, k, func(t int) float32 { return sc[t][j] })
			for _, t := range idx {
				sc[t][j] += add
			}
		}
	}
	boost(strong, 2)
	boost(weak, 1)
	// flatten [spk][frame + silence pads], pads score +inf
	nAll := n + p.SilFramesPerSpk
	flat := func(i int) float32 {
		j, t := i/nAll, i%nAll
		if t >= n {
			return float32(inf)
		}
		return sc[t][j]
	}
	idx := topK(ns*nAll, p.CacheLen, flat)
	const disabled = math.MaxInt
	keys := make([]int, len(idx))
	for i, k := range idx {
		if math.IsInf(float64(flat(k)), -1) {
			keys[i] = disabled
		} else {
			keys[i] = k
		}
	}
	sort.Ints(keys)
	cache := make([][]float32, p.CacheLen)
	preds := make([][]float32, p.CacheLen)
	sil := make([]float32, len(s.meanSil))
	for c, v := range s.meanSil {
		sil[c] = float32(v)
	}
	for i, k := range keys {
		t := k % nAll
		if k == disabled || t >= n {
			cache[i] = append([]float32(nil), sil...)
			preds[i] = make([]float32, ns)
			continue
		}
		cache[i] = s.cache[t]
		preds[i] = s.cachePreds[t]
	}
	s.cache, s.cachePreds = cache, preds
}

// topK returns the indices of the k largest of n values (ties go to
// the lower index).
func topK(n, k int, val func(int) float32) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return val(idx[a]) > val(idx[b]) })
	return idx[:min(k, n)]
}
