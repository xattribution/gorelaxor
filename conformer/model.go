package conformer

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// HParams are a model's shape and settings, read from stt.parakeet.*.
type HParams struct {
	DModel, Heads, Layers, DFF int
	ConvKernel                 int
	SubChannels, Subsampling   int
	XScaling                   bool
	// AttStyle is "regular" or "chunked_limited" (cache-aware streaming).
	AttStyle          string
	AttLeft, AttRight int // -1 means unlimited
	// ConvLeft and ConvRight are the depthwise convolution's padding;
	// both -1 means centered.
	ConvLeft, ConvRight int
	ConvNorm            string // "batch_norm" or "layer_norm"
	CausalSub           bool   // the pre-encoder pads causally (NeMo causal_downsampling)
	ContextChoices      [][2]int

	// Streaming settings.
	PreEncodeCache, DropExtra, SamplingFramesFirst int

	HeadKind     string // "rnnt", "tdt" or "ctc"
	PredHidden   int
	PredLayers   int
	PredVocab    int // tokens plus blank
	JointHidden  int
	JointAct     string
	ExtraOutputs int
	Durations    []int
	MaxSymbols   int
	Blank        int
}

// Causal reports whether the model is cache-aware streaming.
func (h *HParams) Causal() bool { return h.AttStyle == "chunked_limited" }

type ffWeights struct {
	normW, normB []float32
	w1, w2       *kern.Weight
	b1, b2       []float32
}

type block struct {
	ff1, ff2           ffWeights
	normAttnW, normAtt []float32
	q, k, v, out, pos  *kern.Weight
	qb, kb, vb, outb   []float32
	posU, posV         []float32 // [heads][headDim]
	normConvW, normCon []float32
	pw1, pw2           *kern.Weight
	pw1b, pw2b         []float32
	dw                 []float32 // [kernel][d] (transposed for the inner loop)
	dwb                []float32
	cnW, cnB           []float32 // layer norm, or batch norm folded to scale and bias
	normOutW, normOutB []float32
}

type preEncode struct {
	w0, w2, w5 []float32 // [9][c]
	b0, b2, b5 []float32
	w3, w6     *kern.Weight
	b3, b6     []float32
	out        *kern.Weight
	outb       []float32
}

type lstmLayer struct {
	wx, wh *kern.Weight
	b      []float32
}

// Model is a loaded FastConformer transducer.
type Model struct {
	HP     HParams
	Mel    *Mel
	Vocab  []string
	pre    preEncode
	blocks []block

	embed     []float32 // [vocab][pred hidden]
	lstm      []lstmLayer
	jEnc      *kern.Weight
	jEncB     []float32
	jPred     *kern.Weight
	jPredB    []float32
	jOut      *kern.Weight
	jOutB     []float32
	spk       map[int]*spkKernel // multitalker speaker kernels by layer
	ctc       *kern.Weight
	ctcB      []float32
	maxPosEmb int

	prompt *languagePrompt // multilingual models (Nemotron 3.5)
	lang   int             // the prompt's one-hot slot (WithLanguage)
}

type loader struct {
	f   *gguf.File
	err error
}

func (l *loader) vec(name string, optional bool) []float32 {
	if l.err != nil {
		return nil
	}
	if !l.f.Has(name) {
		if !optional {
			l.err = fmt.Errorf("conformer: the model has no %s", name)
		}
		return nil
	}
	v, err := l.f.Float32(name)
	if err != nil {
		l.err = err
	}
	return v
}

func (l *loader) mat(name string, optional bool) *kern.Weight {
	if l.err != nil {
		return nil
	}
	if !l.f.Has(name) {
		if !optional {
			l.err = fmt.Errorf("conformer: the model has no %s", name)
		}
		return nil
	}
	w, err := l.f.Weight(name)
	if err != nil {
		l.err = err
	}
	return w
}

// transpose9 turns [c][taps] into [taps][c].
func transposeTaps(w []float32, c, taps int) []float32 {
	if w == nil {
		return nil
	}
	out := make([]float32, len(w))
	for i := 0; i < c; i++ {
		for j := 0; j < taps; j++ {
			out[j*c+i] = w[i*taps+j]
		}
	}
	return out
}

// readEncoder reads the FastConformer encoder settings under ns
// (stt.parakeet.encoder or stt.sortformer.encoder).
func readEncoder(f *gguf.File, ns string, h *HParams) error {
	p := ns + "."
	h.DModel = f.Int(p+"d_model", 0)
	h.Heads = f.Int(p+"n_heads", 0)
	h.Layers = f.Int(p+"n_layers", 0)
	h.DFF = f.Int(p+"d_ff", 0)
	h.ConvKernel = f.Int(p+"conv_kernel", 0)
	h.SubChannels = f.Int(p+"subsampling_channels", 0)
	h.Subsampling = f.Int(p+"subsampling_factor", 8)
	h.XScaling = f.Bool(p+"xscaling", false)
	h.AttStyle = f.String(p+"att_context_style", "regular")
	h.AttLeft = f.Int(p+"att_context_left", -1)
	h.AttRight = f.Int(p+"att_context_right", -1)
	h.ConvLeft = f.Int(p+"conv_context_left", -1)
	h.ConvRight = f.Int(p+"conv_context_right", -1)
	h.ConvNorm = f.String(p+"conv_norm_type", "batch_norm")
	h.CausalSub = f.Bool(p+"causal_downsampling", h.AttStyle == "chunked_limited")
	if h.DModel <= 0 || h.Heads <= 0 || h.DModel%h.Heads != 0 || h.Layers <= 0 || h.ConvKernel <= 0 {
		return errors.New("conformer: the model file is missing its encoder settings")
	}
	if h.Subsampling != 8 {
		return fmt.Errorf("conformer: subsampling %d is not supported (only 8)", h.Subsampling)
	}
	if h.ConvNorm == "" {
		h.ConvNorm = "batch_norm"
	}
	return nil
}

func readHParams(f *gguf.File) (HParams, error) {
	p := "stt.parakeet."
	h := HParams{
		PreEncodeCache:      f.Int(p+"encoder.streaming.pre_encode_cache_size", -1),
		DropExtra:           f.Int(p+"encoder.streaming.drop_extra_pre_encoded", -1),
		SamplingFramesFirst: f.Int(p+"encoder.streaming.sampling_frames_first", -1),

		HeadKind:     f.String(p+"head_kind", "rnnt"),
		PredHidden:   f.Int(p+"predictor.hidden", 0),
		PredLayers:   f.Int(p+"predictor.n_layers", 0),
		PredVocab:    f.Int(p+"predictor.vocab", 0),
		JointHidden:  f.Int(p+"joint.hidden", 0),
		JointAct:     f.String(p+"joint.activation", "relu"),
		ExtraOutputs: f.Int(p+"joint.num_extra_outputs", 0),
		Durations:    f.Ints(p + "tdt.durations"),
		MaxSymbols:   f.Int(p+"tdt.max_symbols", 10),
		Blank:        f.Int("tokenizer.ggml.blank_token_id", -1),
	}
	if err := readEncoder(f, p+"encoder", &h); err != nil {
		return h, err
	}
	if c := f.Ints(p + "encoder.att_context_size_choices"); len(c) >= 2 {
		for i := 0; i+1 < len(c); i += 2 {
			h.ContextChoices = append(h.ContextChoices, [2]int{c[i], c[i+1]})
		}
	}
	switch h.HeadKind {
	case "rnnt":
		h.Durations, h.MaxSymbols = nil, 10
	case "tdt":
		if len(h.Durations) == 0 {
			return h, errors.New("conformer: a TDT model needs its durations")
		}
		if h.MaxSymbols < 1 || h.MaxSymbols > 100 {
			return h, fmt.Errorf("conformer: tdt.max_symbols %d is out of range", h.MaxSymbols)
		}
	default:
		return h, fmt.Errorf("conformer: %q models are not supported (only RNN-T and TDT)", h.HeadKind)
	}
	if h.PredVocab < 2 || h.PredHidden <= 0 || h.PredLayers <= 0 || h.JointHidden <= 0 {
		return h, errors.New("conformer: the model file is missing its decoder settings")
	}
	return h, nil
}

func melConfig(f *gguf.File) MelConfig {
	p := "stt.frontend."
	return MelConfig{
		SampleRate:  f.Int(p+"sample_rate", 16000),
		NFFT:        f.Int(p+"n_fft", 512),
		WinLength:   f.Int(p+"win_length", 400),
		Hop:         f.Int(p+"hop_length", 160),
		NumMels:     f.Int(p+"num_mels", 80),
		FMin:        f.Float(p+"f_min", 0),
		FMax:        f.Float(p+"f_max", 8000),
		PreEmphasis: f.Float(p+"pre_emphasis", 0.97),
		Normalize:   f.String(p+"normalize", "per_feature"),
	}
}

// Load opens a converted model file.
func Load(path string) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	return FromGGUF(f)
}

// FromGGUF builds a model from an opened file. It runs a second of silence
// through the model, so a file whose tensors don't fit its settings fails
// here, with an error, rather than mid-recording.
func FromGGUF(f *gguf.File) (*Model, error) {
	var m *Model
	err := checked(func() error {
		var err error
		if m, err = fromGGUF(f); err != nil {
			return err
		}
		_, err = m.Transcribe(make([]float32, m.Mel.cfg.SampleRate), nil, DefaultSegments, 1)
		return err
	})
	if err != nil {
		return nil, err
	}
	return m, nil
}

// checked runs fn, turning a panic into an error: a model file is
// untrusted input, and a bad one must fail to load, not stop the program.
func checked(fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("conformer: the model file doesn't fit its own settings (%v)", r)
		}
	}()
	return fn()
}

func fromGGUF(f *gguf.File) (*Model, error) {
	if a := f.String("general.architecture", ""); a != "parakeet" {
		return nil, fmt.Errorf("conformer: architecture %q is not a FastConformer transducer", a)
	}
	hp, err := readHParams(f)
	if err != nil {
		return nil, err
	}
	m := &Model{HP: hp, Vocab: f.Strings("tokenizer.ggml.tokens")}
	m.maxPosEmb = f.Int("stt.parakeet.encoder.pos_emb_max_len", 5000)
	if m.Mel, err = NewMel(melConfig(f)); err != nil {
		return nil, err
	}
	l := &loader{f: f}
	m.loadEncoder(l)
	if hp.HeadKind == "rnnt" || hp.HeadKind == "tdt" {
		m.embed = l.vec("pred.embed.weight", false)
		for i := 0; i < hp.PredLayers; i++ {
			p := fmt.Sprintf("pred.lstm.%d.", i)
			m.lstm = append(m.lstm, lstmLayer{wx: l.mat(p+"Wx", false), wh: l.mat(p+"Wh", false), b: l.vec(p+"bias", false)})
		}
		m.jEnc, m.jEncB = l.mat("joint.enc.weight", false), l.vec("joint.enc.bias", false)
		m.jPred, m.jPredB = l.mat("joint.pred.weight", false), l.vec("joint.pred.bias", false)
		m.jOut, m.jOutB = l.mat("joint.out.weight", false), l.vec("joint.out.bias", false)
	}
	for _, layer := range f.Ints("stt.parakeet.encoder.spk_kernel_layers") {
		if m.spk == nil {
			m.spk = map[int]*spkKernel{}
		}
		p := fmt.Sprintf("enc.spk_kernel.%d.", layer)
		k := &spkKernel{w0: l.mat(p+"0.weight", false), b0: l.vec(p+"0.bias", false), w3: l.mat(p+"3.weight", false), b3: l.vec(p+"3.bias", false)}
		if f.Bool("stt.parakeet.encoder.add_bg_spk_kernel", false) {
			p = fmt.Sprintf("enc.bg_spk_kernel.%d.", layer)
			k.bw0, k.bb0, k.bw3, k.bb3 = l.mat(p+"0.weight", false), l.vec(p+"0.bias", false), l.mat(p+"3.weight", false), l.vec(p+"3.bias", false)
		}
		m.spk[layer] = k
	}
	if f.Has("head.ctc.weight") {
		m.ctc, m.ctcB = l.mat("head.ctc.weight", false), l.vec("head.ctc.bias", true)
	}
	if l.err != nil {
		return nil, l.err
	}
	if err := m.loadPrompt(f, l); err != nil {
		return nil, err
	}
	if hp.Blank < 0 {
		m.HP.Blank = hp.PredVocab - 1
	}
	return m, nil
}

// packed lays an encoder's Q8_0 weights out for the int8 kernel now, at
// load, where the processor has one, and lets go of their rows: the
// encoder multiplies many frames at a time, so keeping the rows too would
// double the model's memory. (A 1-frame streaming chunk still works; it
// takes the int8 kernel too.)
func packed(ws ...*kern.Weight) {
	for _, w := range ws {
		if w != nil {
			w.Pack(runtime.GOMAXPROCS(0))
			w.DropRows()
		}
	}
}

// loadEncoder reads the pre-encoder and Conformer blocks.
func (m *Model) loadEncoder(l *loader) {
	c := m.HP.SubChannels
	pe := "enc.pre_encode."
	m.pre = preEncode{
		w0: transposeTaps(l.vec(pe+"conv.0.weight", false), c, 9), b0: l.vec(pe+"conv.0.bias", false),
		w2: transposeTaps(l.vec(pe+"conv.2.weight", false), c, 9), b2: l.vec(pe+"conv.2.bias", false),
		w3: l.mat(pe+"conv.3.weight", false), b3: l.vec(pe+"conv.3.bias", false),
		w5: transposeTaps(l.vec(pe+"conv.5.weight", false), c, 9), b5: l.vec(pe+"conv.5.bias", false),
		w6: l.mat(pe+"conv.6.weight", false), b6: l.vec(pe+"conv.6.bias", false),
		out: l.mat(pe+"out.weight", false), outb: l.vec(pe+"out.bias", false),
	}
	packed(m.pre.w3, m.pre.w6, m.pre.out)
	d := m.HP.DModel
	for i := 0; i < m.HP.Layers && l.err == nil; i++ {
		p := fmt.Sprintf("enc.blocks.%d.", i)
		ff := func(n string) ffWeights {
			return ffWeights{
				normW: l.vec(p+"norm_"+n+".weight", false), normB: l.vec(p+"norm_"+n+".bias", false),
				w1: l.mat(p+n+".linear1.weight", false), b1: l.vec(p+n+".linear1.bias", true),
				w2: l.mat(p+n+".linear2.weight", false), b2: l.vec(p+n+".linear2.bias", true),
			}
		}
		b := block{
			ff1: ff("ff1"), ff2: ff("ff2"),
			normAttnW: l.vec(p+"norm_attn.weight", false), normAtt: l.vec(p+"norm_attn.bias", false),
			q: l.mat(p+"attn.linear_q.weight", false), qb: l.vec(p+"attn.linear_q.bias", true),
			k: l.mat(p+"attn.linear_k.weight", false), kb: l.vec(p+"attn.linear_k.bias", true),
			v: l.mat(p+"attn.linear_v.weight", false), vb: l.vec(p+"attn.linear_v.bias", true),
			out: l.mat(p+"attn.linear_out.weight", false), outb: l.vec(p+"attn.linear_out.bias", true),
			pos:  l.mat(p+"attn.linear_pos.weight", false),
			posU: l.vec(p+"attn.pos_bias_u", false), posV: l.vec(p+"attn.pos_bias_v", false),
			normConvW: l.vec(p+"norm_conv.weight", false), normCon: l.vec(p+"norm_conv.bias", false),
			pw1: l.mat(p+"conv.pointwise1.weight", false), pw1b: l.vec(p+"conv.pointwise1.bias", true),
			pw2: l.mat(p+"conv.pointwise2.weight", false), pw2b: l.vec(p+"conv.pointwise2.bias", true),
			dw:       transposeTaps(l.vec(p+"conv.depthwise.weight", false), d, m.HP.ConvKernel),
			dwb:      l.vec(p+"conv.depthwise.bias", true),
			normOutW: l.vec(p+"norm_out.weight", false), normOutB: l.vec(p+"norm_out.bias", false),
		}
		// Files rounded to Q8_0 keep the 1x1 convolutions in float16 so
		// other programs can read them; they run as fast as the rest
		// rounded to Q8_0 too.
		if l.err == nil && b.ff1.w1.Type == kern.Q8_0 {
			b.pw1, b.pw2 = b.pw1.ToQ8(), b.pw2.ToQ8()
		}
		if l.err == nil {
			packed(b.ff1.w1, b.ff1.w2, b.ff2.w1, b.ff2.w2, b.q, b.k, b.v, b.out, b.pos, b.pw1, b.pw2)
		}
		gw, gb := l.vec(p+"conv.bn.weight", false), l.vec(p+"conv.bn.bias", false)
		if m.HP.ConvNorm == "layer_norm" {
			b.cnW, b.cnB = gw, gb
		} else if l.err == nil {
			mean, vr := l.vec(p+"conv.bn.running_mean", false), l.vec(p+"conv.bn.running_var", false)
			if l.err == nil {
				b.cnW, b.cnB = make([]float32, d), make([]float32, d)
				for j := 0; j < d; j++ {
					s := float32(float64(gw[j]) / math.Sqrt(float64(vr[j])+1e-5))
					b.cnW[j] = s
					b.cnB[j] = gb[j] - mean[j]*s
				}
			}
		}
		m.blocks = append(m.blocks, b)
	}
}

// TokenText is Text for tokens from Transcribe, a Stream or Decode.
func (m *Model) TokenText(toks []Token) string {
	ids := make([]int, len(toks))
	for i, t := range toks {
		ids[i] = t.ID
	}
	return m.Text(ids)
}

// Text joins SentencePiece tokens into words, leaving out language
// tags (DetectedLanguage) and the space before each, as NeMo's
// strip_lang_tags does.
func (m *Model) Text(ids []int) string {
	var b []byte
	for _, id := range ids {
		if id < 0 || id >= len(m.Vocab) || id == m.HP.Blank {
			continue
		}
		if m.isLangTag(id) {
			b = []byte(strings.TrimRight(string(b), " ▁"))
			continue
		}
		b = append(b, m.Vocab[id]...)
	}
	return strings.TrimSpace(strings.ReplaceAll(string(b), "▁", " "))
}
