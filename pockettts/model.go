package pockettts

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// Model is a loaded Pocket TTS model. It can be shared by goroutines.
type Model struct {
	cfg  config
	tok  *tokenizer
	lm   flowLM
	dec  mimiDecoder
	enc  *mimiEncoder // nil without voice cloning
	sha  string
	name string
}

type config struct {
	sampleRate, frameSamples int
	temperature              float32
	decodeSteps              int
	eosThreshold             float32
	maxTokens                int
	minFramesBeforeEOS       int
	framesAfterEOS           int // -1: guessed per chunk
	appendPunct, capitalize  bool
	removeSemicolons         bool
	padShort                 bool
}

// SampleRate is the audio's sample rate, 24000.
func (m *Model) SampleRate() int { return m.cfg.sampleRate }

// FrameSamples is how many samples each frame of audio has, 1920 (80 ms).
func (m *Model) FrameSamples() int { return m.cfg.frameSamples }

// CanClone reports whether the file has Mimi's encoder, which CloneVoice needs.
func (m *Model) CanClone() bool { return m.enc != nil }

// Name is the model's name from its file.
func (m *Model) Name() string { return m.name }

// loader reads tensors, checking their shapes; the first problem sticks.
type loader struct {
	f   *gguf.File
	err error
}

func (l *loader) fail(format string, a ...any) {
	if l.err == nil {
		l.err = fmt.Errorf("pockettts: "+format, a...)
	}
}

func (l *loader) mat(name string, rows, cols int) *kern.Weight {
	if l.err != nil {
		return nil
	}
	w, err := l.f.Weight(name)
	if err != nil {
		l.fail("%v", err)
		return nil
	}
	if w.Rows != rows || w.Cols != cols {
		l.fail("%s is %dx%d, want %dx%d", name, w.Rows, w.Cols, rows, cols)
		return nil
	}
	return w
}

func (l *loader) vec(name string, n int) []float32 {
	if l.err != nil {
		return nil
	}
	t, ok := l.f.Tensors[name]
	if !ok || t.Elems() != n {
		l.fail("no %d-value %s", n, name)
		return nil
	}
	v, err := l.f.Float32(name)
	if err != nil {
		l.fail("%v", err)
	}
	return v
}

func (l *loader) num(key string) int {
	n := l.f.Int(key, -1)
	if n < 1 || n > 1<<16 {
		l.fail("the setting %s is %d", key, n)
		return 1
	}
	return n
}

func (l *loader) transformer(prefix string, d, heads, ff, context int, period float64, scale bool) transformer {
	t := transformer{d: d, heads: heads, ff: ff, context: context}
	if d%heads != 0 || (d/heads)%2 != 0 {
		l.fail("%d values don't split into %d heads", d, heads)
		return t
	}
	t.freqs = ropeFreqs(d/heads, period)
	for i := 0; ; i++ {
		p := fmt.Sprintf("%s.%d.", prefix, i)
		if !l.f.Has(p + "norm1.weight") {
			break
		}
		ly := layer{
			norm1w: l.vec(p+"norm1.weight", d), norm1b: l.vec(p+"norm1.bias", d),
			norm2w: l.vec(p+"norm2.weight", d), norm2b: l.vec(p+"norm2.bias", d),
			qkv: l.mat(p+"attn_qkv.weight", 3*d, d), out: l.mat(p+"attn_out.weight", d, d),
			ff1: l.mat(p+"ff1.weight", ff, d), ff2: l.mat(p+"ff2.weight", d, ff),
		}
		if scale {
			ly.ls1, ly.ls2 = l.vec(p+"ls1", d), l.vec(p+"ls2", d)
		}
		t.layers = append(t.layers, ly)
	}
	if len(t.layers) == 0 {
		l.fail("no %s layers", prefix)
	}
	return t
}

func (l *loader) conv(name string, cin, cout, k, stride, dil int) conv {
	return conv{w: l.mat(name+".weight", cout, k*cin), b: l.vec(name+".bias", cout), cin: cin, cout: cout, k: k, stride: stride, dil: dil}
}

func (l *loader) convtr(name string, cin, cout, k, stride int) *convtr {
	return &convtr{w: l.mat(name+".weight", k*cout, cin), b: l.vec(name+".bias", cout), cin: cin, cout: cout, k: k, stride: stride}
}

type seanetCfg struct {
	ratios                             []int
	nf, dim, k, resK, lastK, dil, comp int
	nres                               int
}

func (l *loader) resBlocks(prefix string, idx *int, dim int, s seanetCfg) []seaStep {
	var out []seaStep
	for j := 0; j < s.nres; j++ {
		p := fmt.Sprintf("%s.%d.", prefix, *idx)
		d := 1
		for i := 0; i < j; i++ {
			d *= s.dil
		}
		r := &resBlock{a: l.conv(p+"1", dim, dim/s.comp, s.resK, 1, d), b: l.conv(p+"3", dim/s.comp, dim, 1, 1, 1)}
		out = append(out, seaStep{r: r})
		*idx++
	}
	return out
}

// decoder is SEANetDecoder: conv, then for each ratio ELU, transposed
// conv and residual blocks, then ELU and conv to one channel.
func (l *loader) decoder(s seanetCfg) []seaStep {
	mult := 1 << len(s.ratios)
	c := l.conv("mimi.dec.0", s.dim, mult*s.nf, s.k, 1, 1)
	steps := []seaStep{{c: &c}}
	idx := 1
	for _, r := range s.ratios {
		idx++ // ELU
		t := l.convtr(fmt.Sprintf("mimi.dec.%d", idx), mult*s.nf, mult*s.nf/2, 2*r, r)
		steps = append(steps, seaStep{t: t, elu: true})
		idx++
		steps = append(steps, l.resBlocks("mimi.dec", &idx, mult*s.nf/2, s)...)
		mult /= 2
	}
	idx++
	last := l.conv(fmt.Sprintf("mimi.dec.%d", idx), s.nf, 1, s.lastK, 1, 1)
	return append(steps, seaStep{c: &last, elu: true})
}

// encoder is SEANetEncoder: conv, then for each ratio (smallest first)
// residual blocks, ELU and a strided conv, then ELU and conv.
func (l *loader) encoder(s seanetCfg) []seaStep {
	c := l.conv("mimi.enc.0", 1, s.nf, s.k, 1, 1)
	steps := []seaStep{{c: &c}}
	idx := 1
	mult := 1
	for i := len(s.ratios) - 1; i >= 0; i-- {
		r := s.ratios[i]
		steps = append(steps, l.resBlocks("mimi.enc", &idx, mult*s.nf, s)...)
		idx++ // ELU
		cv := l.conv(fmt.Sprintf("mimi.enc.%d", idx), mult*s.nf, 2*mult*s.nf, 2*r, r, 1)
		steps = append(steps, seaStep{c: &cv, elu: true})
		idx++
		mult *= 2
	}
	idx++
	last := l.conv(fmt.Sprintf("mimi.enc.%d", idx), mult*s.nf, s.dim, s.lastK, 1, 1)
	return append(steps, seaStep{c: &last, elu: true})
}

// Load reads a model file made by tools/convert_pocket_tts.py.
func Load(path string) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	if f.String("general.architecture", "") != "pockettts" {
		return nil, fmt.Errorf("pockettts: %s isn't a Pocket TTS model", path)
	}
	m := &Model{sha: f.String("general.source.sha256", ""), name: f.String("general.name", "")}
	if m.sha == "" {
		return nil, errors.New("pockettts: the file doesn't name its source")
	}
	if m.tok, err = loadTokenizer(f); err != nil {
		return nil, err
	}
	l := &loader{f: f}
	ldim := l.num("pockettts.latent_dim")
	d := l.num("pockettts.lm.d_model")
	sr := l.num("pockettts.sample_rate")
	fr := f.Float("pockettts.frame_rate", 0)
	if fr <= 0 || math.Mod(float64(sr), fr) != 0 {
		l.fail("a frame rate of %g", fr)
	}
	m.cfg = config{
		sampleRate:         sr,
		frameSamples:       int(float64(sr) / fr),
		temperature:        float32(f.Float("pockettts.temperature", 0.3)),
		decodeSteps:        l.num("pockettts.decode_steps"),
		eosThreshold:       float32(f.Float("pockettts.eos_threshold", -4)),
		maxTokens:          l.num("pockettts.max_tokens_per_chunk"),
		minFramesBeforeEOS: f.Int("pockettts.min_frames_before_eos", 6),
		framesAfterEOS:     f.Int("pockettts.frames_after_eos", -1),
		appendPunct:        f.Bool("pockettts.text.append_terminal_punctuation", true),
		capitalize:         f.Bool("pockettts.text.capitalize_first_letter", true),
		removeSemicolons:   f.Bool("pockettts.text.remove_semicolons", false),
		padShort:           f.Bool("pockettts.text.pad_with_spaces_for_short_inputs", false),
	}
	nTok := len(m.tok.pieces)
	lm := &m.lm
	lm.tf = l.transformer("lm.blk", d, l.num("pockettts.lm.heads"), l.num("pockettts.lm.ff"), 0, f.Float("pockettts.lm.max_period", 1e4), false)
	if l.err == nil && len(lm.tf.layers) != l.num("pockettts.lm.layers") {
		l.fail("the flow LM has %d layers", len(lm.tf.layers))
	}
	lm.embed = l.mat("lm.embed", nTok+1, d)
	lm.input = l.mat("lm.input.weight", d, ldim)
	lm.bos = l.vec("lm.bos", ldim)
	lm.bosVoice = l.vec("lm.bos_voice", d)
	lm.outW, lm.outB = l.vec("lm.out_norm.weight", d), l.vec("lm.out_norm.bias", d)
	lm.eosW = l.mat("lm.eos.weight", 1, d)
	if b := l.vec("lm.eos.bias", 1); b != nil {
		lm.eosB = b[0]
	}
	lm.mean, lm.std = l.vec("lm.emb_mean", ldim), l.vec("lm.emb_std", ldim)
	fd := l.num("pockettts.flow.dim")
	h := &lm.head
	h.dim, h.ldim = fd, ldim
	h.cond, h.condB = l.mat("flow.cond.weight", fd, d), l.vec("flow.cond.bias", fd)
	h.input, h.inputB = l.mat("flow.input.weight", fd, ldim), l.vec("flow.input.bias", fd)
	for i := range h.time {
		p := fmt.Sprintf("flow.time.%d.", i)
		t := &h.time[i]
		t.freqs = l.vec(p+"freqs", 128)
		t.fc1, t.b1 = l.mat(p+"fc1.weight", fd, 256), l.vec(p+"fc1.bias", fd)
		t.fc2, t.b2 = l.mat(p+"fc2.weight", fd, fd), l.vec(p+"fc2.bias", fd)
		t.alpha = l.vec(p+"norm.alpha", fd)
	}
	for i := 0; i < l.num("pockettts.flow.depth"); i++ {
		p := fmt.Sprintf("flow.res.%d.", i)
		h.res = append(h.res, resMLP{
			normW: l.vec(p+"norm.weight", fd), normB: l.vec(p+"norm.bias", fd),
			fc1: l.mat(p+"fc1.weight", fd, fd), b1: l.vec(p+"fc1.bias", fd),
			fc2: l.mat(p+"fc2.weight", fd, fd), b2: l.vec(p+"fc2.bias", fd),
			ada: l.mat(p+"ada.weight", 3*fd, fd), adaB: l.vec(p+"ada.bias", 3*fd),
		})
	}
	h.finalAda, h.finalAdaB = l.mat("flow.final.ada.weight", 2*fd, fd), l.vec("flow.final.ada.bias", 2*fd)
	h.out, h.outB = l.mat("flow.final.out.weight", ldim, fd), l.vec("flow.final.out.bias", ldim)

	md := l.num("pockettts.mimi.d_model")
	mh := l.num("pockettts.mimi.heads")
	mff := l.num("pockettts.mimi.ff")
	mctx := l.num("pockettts.mimi.context")
	mper := f.Float("pockettts.mimi.max_period", 1e4)
	up := l.num("pockettts.mimi.upsample")
	s := seanetCfg{
		ratios: f.Ints("pockettts.seanet.ratios"),
		nf:     l.num("pockettts.seanet.n_filters"), dim: l.num("pockettts.seanet.dimension"),
		k: l.num("pockettts.seanet.kernel_size"), resK: l.num("pockettts.seanet.residual_kernel_size"),
		lastK: l.num("pockettts.seanet.last_kernel_size"), dil: l.num("pockettts.seanet.dilation_base"),
		comp: l.num("pockettts.seanet.compress"), nres: l.num("pockettts.seanet.n_residual_layers"),
	}
	hop := 1
	for _, r := range s.ratios {
		if r < 1 || r > 64 {
			l.fail("a SEANet ratio of %d", r)
		}
		hop *= r
	}
	if l.err == nil && (len(s.ratios) == 0 || s.dim != md || hop*up != m.cfg.frameSamples) {
		l.fail("SEANet's ratios %v don't make %d-sample frames", s.ratios, m.cfg.frameSamples)
	}
	if l.err != nil {
		return nil, l.err
	}
	m.dec = mimiDecoder{quant: l.mat("mimi.quant", md, ldim), up: l.vec("mimi.up", md*2*up), upS: up}
	m.dec.tf = l.transformer("mimi.dec_tf", md, mh, mff, mctx, mper, true)
	m.dec.steps = l.decoder(s)
	if f.Bool("pockettts.voice_cloning", false) {
		e := &mimiEncoder{steps: l.encoder(s)}
		e.tf = l.transformer("mimi.enc_tf", md, mh, mff, mctx, mper, true)
		e.down = conv{w: l.mat("mimi.down", ldim, 2*up*md), cin: md, cout: ldim, k: 2 * up, stride: up, dil: 1}
		e.down.b = make([]float32, ldim)
		lm.speaker = l.mat("lm.speaker_proj", d, ldim)
		m.enc = e
	}
	if l.err != nil {
		return nil, l.err
	}
	m.pack()
	return m, nil
}

// pack lays the transformers' and convolutions' Q8_0 weights out for the
// int8 kernels at load, where the processor has them, so the first
// sentence doesn't wait for it, and lets go of their rows. The flow LM's
// transformer runs one row at a time while speaking, and with its rows
// dropped that row takes the int8 kernel too, which was faster (and
// starts sooner) than the float path. The flow head and the token table
// stay as they are.
func (m *Model) pack() {
	var ws []*kern.Weight
	tf := func(t *transformer) {
		for i := range t.layers {
			l := &t.layers[i]
			ws = append(ws, l.qkv, l.out, l.ff1, l.ff2)
		}
	}
	sea := func(steps []seaStep) {
		for _, s := range steps {
			switch {
			case s.c != nil:
				ws = append(ws, s.c.w)
			case s.t != nil:
				ws = append(ws, s.t.w)
			default:
				ws = append(ws, s.r.a.w, s.r.b.w)
			}
		}
	}
	tf(&m.lm.tf)
	tf(&m.dec.tf)
	sea(m.dec.steps)
	if m.enc != nil {
		tf(&m.enc.tf)
		sea(m.enc.steps)
		ws = append(ws, m.enc.down.w)
	}
	n := runtime.GOMAXPROCS(0)
	for _, w := range ws {
		if w.Type == kern.Q8_0 {
			w.Pack(n)
			w.DropRows()
		}
	}
}
