package conformer

import (
	"errors"
	"fmt"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// streamGeom is a cache-aware model's chunking for one right context:
// NeMo's streaming_cfg. Mel frames go to the encoder in chunks of
// 8·(R+1) frames (the first 1+8·R), each led by the 9 mel frames before
// it, whose 2 encoder frames are then dropped.
type streamGeom struct {
	right, chunk, left int // attention context in encoder frames
	leftChunks         int
	melFirst, melNext  int // mel frames per chunk
	melCache, drop     int
	sampleFirst        int
	posProj            [][]float32 // per layer: positions left+chunk-1 down to -(chunk-1), projected
}

func (m *Model) newGeom(right, threads int) (*streamGeom, error) {
	hp := &m.HP
	if !hp.Causal() {
		return nil, errors.New("conformer: this model can't stream")
	}
	left := -1
	for _, c := range hp.ContextChoices {
		if c[1] == right {
			left = c[0]
		}
	}
	if len(hp.ContextChoices) == 0 && right == hp.AttRight {
		left = hp.AttLeft
	}
	if left < 0 {
		return nil, fmt.Errorf("conformer: the model wasn't trained for a right context of %d", right)
	}
	sub := hp.Subsampling
	cache := hp.PreEncodeCache
	if cache < 0 {
		cache = sub + 1
	}
	drop := hp.DropExtra
	if drop < 0 {
		drop = 1 + (cache-1)/sub
	}
	first := hp.SamplingFramesFirst
	if first < 0 {
		first = 1
	}
	chunk := right + 1
	g := &streamGeom{
		right: right, chunk: chunk, left: left, leftChunks: left / chunk,
		melFirst: first + sub*right, melNext: sub * chunk,
		melCache: cache, drop: drop, sampleFirst: first,
	}
	D := hp.DModel
	maxFirst := left + chunk - 1
	pos := relPosRange(maxFirst, left+2*chunk-1, D)
	for i := range m.blocks {
		pp := make([]float32, len(pos))
		kern.Linear(pp, pos, len(pos)/D, m.blocks[i].pos, nil, threads)
		g.posProj = append(g.posProj, pp)
	}
	return g, nil
}

// encState is one stream's encoder caches and decoder memory: NeMo's
// cache_last_channel, cache_last_time and previous hypothesis.
type encState struct {
	kv    [][]float32 // per layer: up to left rows of normalized attention input
	kvLen int
	conv  [][]float32 // per layer: the last padL gated frames
	dec   *DecoderState
}

func (m *Model) newEncState() *encState {
	st := &encState{dec: m.NewDecoderState()}
	padL := m.convPadLeft()
	for range m.blocks {
		st.kv = append(st.kv, nil)
		st.conv = append(st.conv, make([]float32, padL*m.HP.DModel))
	}
	return st
}

// melChunk is one chunk of mel frames for the encoder.
type melChunk struct {
	frames [][]float32 // the cache frames, then the chunk
	length int         // real frames, cache included
	first  bool
	index  int
}

// melChunker computes mel frames as audio arrives and cuts them into
// chunks the way NeMo's CacheAwareStreamingAudioBuffer does.
type melChunker struct {
	mel     *Mel
	g       *streamGeom
	sub     int
	emph    []float64 // pre-emphasized samples from emphBas on
	emphBas int
	nIn     int
	lastIn  float32
	closed  bool
	frames  [][]float32 // from base on
	base    int
	bufIdx  int // next chunk's first mel frame
	nChunks int
	scratch *melScratch
}

func newMelChunker(mel *Mel, g *streamGeom, sub int) *melChunker {
	return &melChunker{mel: mel, g: g, sub: sub, scratch: mel.scratch()}
}

func (c *melChunker) write(pcm []float32) []melChunk {
	if c.closed || len(pcm) == 0 {
		return nil
	}
	cfg := c.mel.cfg
	if c.nIn == 0 {
		// the zero padding NeMo's centered STFT puts before the audio
		c.emph = make([]float64, cfg.NFFT/2, cfg.NFFT/2+len(pcm))
		c.emphBas = -cfg.NFFT / 2
		c.emph = append(c.emph, float64(pcm[0]))
		for i := 1; i < len(pcm); i++ {
			c.emph = append(c.emph, float64(pcm[i])-cfg.PreEmphasis*float64(pcm[i-1]))
		}
	} else {
		prev := c.lastIn
		for _, v := range pcm {
			c.emph = append(c.emph, float64(v)-cfg.PreEmphasis*float64(prev))
			prev = v
		}
	}
	c.nIn += len(pcm)
	c.lastIn = pcm[len(pcm)-1]
	c.makeFrames(false)
	return c.chunks(false)
}

func (c *melChunker) close() []melChunk {
	if c.closed {
		return nil
	}
	c.closed = true
	if c.nIn == 0 {
		return nil
	}
	c.makeFrames(true)
	return c.chunks(true)
}

// makeFrames computes every mel frame the audio so far allows. Frame t
// covers samples t·hop-n_fft/2 to t·hop+n_fft/2; at the end NeMo zeroes
// the frames past floor(n/hop).
func (c *melChunker) makeFrames(final bool) {
	cfg := c.mel.cfg
	total := c.nIn/cfg.Hop + 1
	for {
		t := c.base + len(c.frames)
		if t >= total {
			break
		}
		start := t*cfg.Hop - cfg.NFFT/2
		if !final && start+cfg.NFFT > c.nIn {
			break
		}
		col := make([]float32, cfg.NumMels)
		if !(final && t >= c.nIn/cfg.Hop) {
			lo := start - c.emphBas
			hi := min(lo+cfg.NFFT, len(c.emph))
			c.mel.frame(c.emph[lo:hi], col, c.scratch)
		}
		c.frames = append(c.frames, col)
	}
	next := (c.base+len(c.frames))*cfg.Hop - cfg.NFFT/2
	if cut := next - c.emphBas; cut > 0 && cut <= len(c.emph) {
		c.emph = append(c.emph[:0], c.emph[cut:]...)
		c.emphBas = next
	}
}

func (c *melChunker) chunks(final bool) []melChunk {
	g := c.g
	var out []melChunk
	avail := c.base + len(c.frames)
	zero := make([]float32, c.mel.cfg.NumMels)
	for c.bufIdx < avail {
		size, need := g.melNext, c.sub // NeMo drops a remainder shorter than its sampling frames
		if c.nChunks == 0 {
			size, need = g.melFirst, g.sampleFirst
		}
		if !final && c.bufIdx+size > avail {
			break
		}
		n := min(size, avail-c.bufIdx)
		if n < need {
			break
		}
		cache := 0
		if c.nChunks > 0 {
			cache = g.melCache
		}
		ch := melChunk{first: c.nChunks == 0, index: c.nChunks, length: cache + n}
		for t := c.bufIdx - cache; t < c.bufIdx+n; t++ {
			if t < 0 {
				ch.frames = append(ch.frames, zero)
				continue
			}
			ch.frames = append(ch.frames, c.frames[t-c.base])
		}
		// NeMo's buffer counts every frame of the feature tensor as real,
		// the zeroed one at the end too (streams_length is its size)
		out = append(out, ch)
		c.bufIdx += size
		c.nChunks++
	}
	if keep := c.bufIdx - g.melCache; keep > c.base {
		k := min(keep-c.base, len(c.frames))
		c.frames = append(c.frames[:0], c.frames[k:]...)
		c.base += k
	}
	return out
}

// chunkMel lays a chunk's frames out as [mels][frames].
func chunkMel(frames [][]float32, nm int) []float32 {
	nf := len(frames)
	mel := make([]float32, nm*nf)
	for t, col := range frames {
		for j, v := range col {
			mel[j*nf+t] = v
		}
	}
	return mel
}

// streamStep runs one chunk through the encoder with st's caches
// (NeMo's cache_aware_stream_step) and returns its encoder frames.
func (m *Model) streamStep(g *streamGeom, st *encState, mel []float32, nf, length int, first bool, threads int) ([]float32, int) {
	x, T, L := m.subsample(mel, nf, length, threads)
	return m.streamEncode(g, st, x, T, L, first, threads)
}

// streamEncode is streamStep after the pre-encoder: x [T][d] with L
// real rows, before NeMo drops the cache frames.
func (m *Model) streamEncode(g *streamGeom, st *encState, x []float32, T, L int, first bool, threads int) ([]float32, int) {
	D := m.HP.DModel
	if !first {
		x, T = x[min(g.drop, T)*D:], max(0, T-g.drop)
		L = max(0, L-g.drop)
	}
	L = min(L, T)
	if T <= 0 {
		return nil, 0
	}
	m.xscale(x)
	ls := newLayerState(T, D, m.HP.DFF, threads)
	ls.lnConv = m.HP.ConvNorm == "layer_norm"
	ls.valid = L
	ls.blockLen = T
	c := st.kvLen
	slot0 := g.left - c // NeMo's slot of our key 0 in its fixed-size cache
	chunkOf := func(slot int) int { return slot / g.chunk }
	ls.keys = func(q int) (int, int) {
		if q-c >= L {
			return 0, 0
		}
		qc := chunkOf(slot0 + q)
		lo := 0
		for lo < c+T && qc-chunkOf(slot0+lo) > g.leftChunks {
			lo++
		}
		hi := c + L
		for hi > lo && chunkOf(slot0+hi-1) > qc {
			hi--
		}
		return lo, hi
	}
	Tk := c + T
	maxFirst := g.left + g.chunk - 1
	padL := m.convPadLeft()
	nop := func(string, []float32) {}
	for i := range m.blocks {
		m.injectSpeaker(i, x, T, threads)
		pp := g.posProj[i][(maxFirst-(Tk-1))*D:]
		before := make([]float32, c*D, Tk*D)
		copy(before, st.kv[i])
		kv := m.runBlock(ls, x, &m.blocks[i], pp, before, st.conv[i], nop)
		if keep := min(len(kv)/D, g.left); keep > 0 {
			st.kv[i] = append(st.kv[i][:0], kv[len(kv)-keep*D:]...)
		}
		if padL > 0 {
			all := append(append([]float32(nil), st.conv[i]...), ls.glu[:T*D]...)
			copy(st.conv[i], all[len(all)-padL*D:])
		}
	}
	st.kvLen = min(c+T, g.left)
	return x[:L*D], L
}

// Stream transcribes audio as it arrives with a cache-aware streaming
// model (Nemotron Speech Streaming). It follows NeMo's
// CacheAwareStreamingAudioBuffer, cache_aware_stream_step and greedy
// RNN-T decoding across chunks.
type Stream struct {
	m       *Model
	threads int
	g       *streamGeom
	c       *melChunker
	st      *encState
	frames  int // encoder frames decoded so far

	onEncoded func(x []float32) // tests: each chunk's encoder output, before the prompt
}

// NewStream starts a stream with right context R encoder frames, one
// of the model's trained choices (Nemotron: 13, 6, 1 or 0; Nemotron
// 3.5: 13, 6, 3 or 0; 80 ms each, so 13 is 1.12 s of lookahead). A
// multilingual model streams in the language WithLanguage set.
func (m *Model) NewStream(right, threads int) (*Stream, error) {
	g, err := m.newGeom(right, threads)
	if err != nil {
		return nil, err
	}
	return &Stream{m: m, threads: threads, g: g, c: newMelChunker(m.Mel, g, m.HP.Subsampling), st: m.newEncState()}, nil
}

// Latency is the audio, in seconds, a word waits for before it can be
// final: the chunk plus the right context.
func (s *Stream) Latency() float64 {
	return float64(s.g.melNext*s.m.Mel.cfg.Hop) / float64(s.m.Mel.cfg.SampleRate)
}

// Write adds 16 kHz mono samples and returns the tokens that became
// final.
func (s *Stream) Write(pcm []float32) []Token {
	var out []Token
	for _, p := range pieces(finite(pcm), s.m.Mel.cfg.SampleRate) {
		out = append(out, s.run(s.c.write(p))...)
	}
	return out
}

// Close finishes the stream: the last frames are decoded and returned.
func (s *Stream) Close() []Token { return s.run(s.c.close()) }

func (s *Stream) run(chunks []melChunk) []Token {
	var out []Token
	for _, ch := range chunks {
		mel := chunkMel(ch.frames, s.m.Mel.cfg.NumMels)
		x, L := s.m.streamStep(s.g, s.st, mel, len(ch.frames), ch.length, ch.first, s.threads)
		if L == 0 {
			continue
		}
		if s.onEncoded != nil {
			s.onEncoded(x)
		}
		x = s.m.applyPrompt(x, L, s.threads) // NeMo applies it to each chunk's output
		out = append(out, s.m.Decode(x, L, s.st.dec, s.frames, s.threads)...)
		s.frames += L
	}
	return out
}
