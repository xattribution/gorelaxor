package whisper

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime/debug"
	"strings"
	"sync"
)

// hparams are the model's dimensions, as the file header gives them.
type hparams struct {
	nVocab, nAudioCtx, nAudioState, nAudioHead, nAudioLayer int
	nTextCtx, nTextState, nTextHead, nTextLayer, nMels      int
	ftype                                                   int
}

// Info describes a loaded model.
type Info struct {
	Name         string // "tiny", "base.en", "small", ...
	Multilingual bool
	Params       int64 // number of weights
	Bytes        int64 // memory the weights take
}

// half is a matrix of IEEE half-precision values, rows by cols, row-major.
// Every large weight is kept this way: it halves the memory of float32 and
// the kernels widen it as they go.
type half struct {
	rows, cols int
	d          []uint16
}

func (h half) row(i int) []uint16 { return h.d[i*h.cols : (i+1)*h.cols] }

type encLayer struct {
	attnLnW, attnLnB []float32
	q, k, v, o       half
	qB, vB, oB       []float32
	mlpLnW, mlpLnB   []float32
	fc1, fc2         half
	fc1B, fc2B       []float32
}

type decLayer struct {
	attnLnW, attnLnB  []float32
	q, k, v, o        half
	qB, vB, oB        []float32
	xLnW, xLnB        []float32
	xq, xk, xv, xo    half
	xqB, xvB, xoB     []float32
	mlpLnW, mlpLnB    []float32
	fc1, fc2          half
	fc1B, fc2B        []float32
	selfK, selfV      []float32 // cache, nTextCtx x state
	crossK, crossV    []float32 // per window: heads x audioCtx x headDim (V transposed per head)
	crossKV, crossVTr bool
}

// Model is a loaded Whisper model. One transcription runs at a time; calls
// from several goroutines wait their turn.
type Model struct {
	mu   sync.Mutex
	hp   hparams
	name string
	melF []float32 // nMels x nFFT (201)
	nFFT int
	voc  *vocab

	encPos           []float32 // nAudioCtx x state
	conv1W, conv2W   half      // out x (in*3)
	conv1B, conv2B   []float32
	encLnW, encLnB   []float32
	enc              []encLayer
	decPos           []float32 // nTextCtx x state
	tokEmb           half      // vocab x state
	decLnW, decLnB   []float32
	dec              []decLayer
	params, memBytes int64
	closed           bool
}

const fileMagic = 0x67676d6c // "ggml"

// Load reads a whisper.cpp model file (ggml format) with float32, float16
// or quantized (q4_0, q4_1, q5_0, q5_1, q8_0) weights. Quantized weights
// are widened to float16 as they load, so a quantized file saves disk
// space, not memory.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, err := load(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

func rd32(r io.Reader) (int32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return int32(binary.LittleEndian.Uint32(b[:])), nil
}

var errFormat = errors.New("not a Whisper model file in whisper.cpp (ggml) format")

func load(r io.Reader) (*Model, error) {
	magic, err := rd32(r)
	if err != nil || uint32(magic) != fileMagic {
		return nil, errFormat
	}
	var hv [11]int32
	for i := range hv {
		if hv[i], err = rd32(r); err != nil {
			return nil, errFormat
		}
	}
	hp := hparams{int(hv[0]), int(hv[1]), int(hv[2]), int(hv[3]), int(hv[4]), int(hv[5]), int(hv[6]), int(hv[7]), int(hv[8]), int(hv[9]), int(hv[10]) % 1000}
	if hp.nAudioState != hp.nTextState || hp.nAudioState <= 0 || hp.nAudioHead <= 0 || hp.nAudioState%hp.nAudioHead != 0 ||
		hp.nTextHead <= 0 || hp.nTextState%hp.nTextHead != 0 || hp.nMels <= 0 || hp.nMels > 256 || hp.nAudioCtx != 1500 ||
		hp.nVocab < 51864 || hp.nVocab > 60000 || hp.nAudioLayer <= 0 || hp.nAudioLayer > 64 || hp.nTextLayer <= 0 || hp.nTextLayer > 64 ||
		hp.nTextCtx <= 0 || hp.nTextCtx > 4096 {
		return nil, errors.New("the model's dimensions aren't ones this program knows")
	}
	m := &Model{hp: hp}

	// Mel filters.
	nMel, err := rd32(r)
	if err != nil {
		return nil, errFormat
	}
	nFFT, err := rd32(r)
	if err != nil || int(nMel) != hp.nMels || nFFT != 201 {
		return nil, errFormat
	}
	m.nFFT = int(nFFT)
	m.melF = make([]float32, int(nMel)*int(nFFT))
	if err := binary.Read(r, binary.LittleEndian, m.melF); err != nil {
		return nil, errFormat
	}

	// Vocabulary.
	nv, err := rd32(r)
	if err != nil || nv <= 0 || int(nv) > hp.nVocab {
		return nil, errFormat
	}
	words := make([]string, nv)
	buf := make([]byte, 256)
	for i := range words {
		l, err := rd32(r)
		if err != nil || l < 0 || l > 1<<16 {
			return nil, errFormat
		}
		if int(l) > len(buf) {
			buf = make([]byte, l)
		}
		if _, err := io.ReadFull(r, buf[:l]); err != nil {
			return nil, errFormat
		}
		words[i] = string(buf[:l])
	}
	m.voc = newVocab(words, hp.nVocab)

	// Tensors.
	tensors := map[string]*rawTensor{}
	for {
		nd, err := rd32(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errFormat
		}
		ln, err1 := rd32(r)
		tt, err2 := rd32(r)
		if err1 != nil || err2 != nil || nd < 1 || nd > 4 || ln <= 0 || ln > 256 {
			return nil, errFormat
		}
		ne := make([]int, nd)
		n := 1
		for i := range ne {
			v, err := rd32(r)
			if err != nil || v <= 0 || int(v) > (1<<31)/n {
				return nil, errFormat // a dimension no real model has, or a size that overflows
			}
			ne[i] = int(v)
			n *= int(v)
		}
		nb := make([]byte, ln)
		if _, err := io.ReadFull(r, nb); err != nil {
			return nil, errFormat
		}
		size, ok := typeBytes(int(tt), n)
		if !ok {
			return nil, fmt.Errorf("tensor %s has a weight type (%d) this program can't read", nb, tt)
		}
		data := make([]byte, size)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, errors.New("the model file is cut short")
		}
		// Convert as it's read, so the file's bytes don't stay in memory.
		t := &rawTensor{typ: int(tt), ne: ne, n: n, data: data}
		if name := string(nb); nd == 1 || ne[0] == 1 || strings.HasSuffix(name, "positional_embedding") {
			t.f32 = toF32(t)
		} else {
			t.f16 = toF16(t)
		}
		t.data = nil
		tensors[string(nb)] = t
	}
	if err := m.bind(tensors); err != nil {
		return nil, err
	}
	m.name = modelName(hp)
	return m, nil
}

func modelName(hp hparams) string {
	n := map[int]string{4: "tiny", 6: "base", 12: "small", 24: "medium", 32: "large"}[hp.nAudioLayer]
	if n == "" {
		n = fmt.Sprintf("custom-%d", hp.nAudioLayer)
	}
	if n == "large" && hp.nTextLayer == 4 {
		n = "large-v3-turbo"
	} else if n == "large" && hp.nVocab == 51866 {
		n = "large-v3"
	}
	if hp.nVocab < 51865 {
		n += ".en"
	}
	return n
}

// Info says which model this is and what it costs in memory.
func (m *Model) Info() Info {
	return Info{Name: m.name, Multilingual: m.voc.multilingual, Params: m.params, Bytes: m.memBytes}
}

// Close drops the weights so their memory can be returned to the system.
// The model can't be used afterwards.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	m.closed = true
	m.enc, m.dec = nil, nil
	m.tokEmb, m.conv1W, m.conv2W = half{}, half{}, half{}
	m.encPos, m.decPos = nil, nil
	debug.FreeOSMemory()
}

type rawTensor struct {
	typ  int
	ne   []int
	n    int
	data []byte
	f32  []float32
	f16  []uint16
	used bool
}

const (
	tF32  = 0
	tF16  = 1
	tQ4_0 = 2
	tQ4_1 = 3
	tQ5_0 = 6
	tQ5_1 = 7
	tQ8_0 = 8
)

func typeBytes(t, n int) (int, bool) {
	switch t {
	case tF32:
		return n * 4, true
	case tF16:
		return n * 2, true
	}
	bs := map[int]int{tQ4_0: 18, tQ4_1: 20, tQ5_0: 22, tQ5_1: 24, tQ8_0: 34}[t]
	if bs == 0 || n%32 != 0 {
		return 0, false
	}
	return n / 32 * bs, true
}

// bind checks every tensor's shape and converts it to the form the
// kernels use: vectors as float32, matrices as float16.
func (m *Model) bind(ts map[string]*rawTensor) error {
	hp := m.hp
	var bad error
	get := func(name string, ne ...int) *rawTensor {
		t := ts[name]
		if t == nil {
			if bad == nil {
				bad = fmt.Errorf("the model file has no tensor %s", name)
			}
			return nil
		}
		n := 1
		for _, v := range ne {
			n *= v
		}
		if t.n != n || (len(t.ne) == len(ne) && t.ne[0] != ne[0]) {
			if bad == nil {
				bad = fmt.Errorf("tensor %s has the wrong shape", name)
			}
			return nil
		}
		t.used = true
		m.params += int64(n)
		return t
	}
	vec := func(name string, n int) []float32 {
		t := get(name, n)
		if t == nil {
			return make([]float32, n)
		}
		m.memBytes += int64(n) * 4
		if t.f32 != nil {
			return t.f32
		}
		out := make([]float32, n)
		widen(out, t.f16)
		return out
	}
	mat := func(name string, rows, cols int) half {
		t := get(name, cols, rows)
		if t == nil {
			return half{rows, cols, make([]uint16, rows*cols)}
		}
		m.memBytes += int64(rows*cols) * 2
		if t.f16 == nil {
			t.f16 = make([]uint16, len(t.f32))
			for i, v := range t.f32 {
				t.f16[i] = f32toF16(v)
			}
		}
		return half{rows, cols, t.f16}
	}
	st := hp.nAudioState
	m.encPos = vec("encoder.positional_embedding", hp.nAudioCtx*st)
	m.conv1W = mat("encoder.conv1.weight", st, hp.nMels*3)
	m.conv1B = vec("encoder.conv1.bias", st)
	m.conv2W = mat("encoder.conv2.weight", st, st*3)
	m.conv2B = vec("encoder.conv2.bias", st)
	m.encLnW = vec("encoder.ln_post.weight", st)
	m.encLnB = vec("encoder.ln_post.bias", st)
	for i := 0; i < hp.nAudioLayer; i++ {
		p := fmt.Sprintf("encoder.blocks.%d.", i)
		m.enc = append(m.enc, encLayer{
			attnLnW: vec(p+"attn_ln.weight", st), attnLnB: vec(p+"attn_ln.bias", st),
			q: mat(p+"attn.query.weight", st, st), qB: vec(p+"attn.query.bias", st),
			k: mat(p+"attn.key.weight", st, st),
			v: mat(p+"attn.value.weight", st, st), vB: vec(p+"attn.value.bias", st),
			o: mat(p+"attn.out.weight", st, st), oB: vec(p+"attn.out.bias", st),
			mlpLnW: vec(p+"mlp_ln.weight", st), mlpLnB: vec(p+"mlp_ln.bias", st),
			fc1: mat(p+"mlp.0.weight", 4*st, st), fc1B: vec(p+"mlp.0.bias", 4*st),
			fc2: mat(p+"mlp.2.weight", st, 4*st), fc2B: vec(p+"mlp.2.bias", st),
		})
	}
	m.decPos = vec("decoder.positional_embedding", hp.nTextCtx*st)
	m.tokEmb = mat("decoder.token_embedding.weight", hp.nVocab, st)
	m.decLnW = vec("decoder.ln.weight", st)
	m.decLnB = vec("decoder.ln.bias", st)
	for i := 0; i < hp.nTextLayer; i++ {
		p := fmt.Sprintf("decoder.blocks.%d.", i)
		m.dec = append(m.dec, decLayer{
			attnLnW: vec(p+"attn_ln.weight", st), attnLnB: vec(p+"attn_ln.bias", st),
			q: mat(p+"attn.query.weight", st, st), qB: vec(p+"attn.query.bias", st),
			k: mat(p+"attn.key.weight", st, st),
			v: mat(p+"attn.value.weight", st, st), vB: vec(p+"attn.value.bias", st),
			o: mat(p+"attn.out.weight", st, st), oB: vec(p+"attn.out.bias", st),
			xLnW: vec(p+"cross_attn_ln.weight", st), xLnB: vec(p+"cross_attn_ln.bias", st),
			xq: mat(p+"cross_attn.query.weight", st, st), xqB: vec(p+"cross_attn.query.bias", st),
			xk: mat(p+"cross_attn.key.weight", st, st),
			xv: mat(p+"cross_attn.value.weight", st, st), xvB: vec(p+"cross_attn.value.bias", st),
			xo: mat(p+"cross_attn.out.weight", st, st), xoB: vec(p+"cross_attn.out.bias", st),
			mlpLnW: vec(p+"mlp_ln.weight", st), mlpLnB: vec(p+"mlp_ln.bias", st),
			fc1: mat(p+"mlp.0.weight", 4*st, st), fc1B: vec(p+"mlp.0.bias", 4*st),
			fc2: mat(p+"mlp.2.weight", st, 4*st), fc2B: vec(p+"mlp.2.bias", st),
		})
	}
	if bad != nil {
		return bad
	}
	for name, t := range ts {
		if !t.used {
			return fmt.Errorf("the model file has a tensor this program doesn't know: %s", name)
		}
	}
	return nil
}

// toF32 widens any tensor to float32.
func toF32(t *rawTensor) []float32 {
	out := make([]float32, t.n)
	switch t.typ {
	case tF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.data[i*4:]))
		}
	default:
		h := toF16(t)
		for i, v := range h {
			out[i] = f16tab[v]
		}
	}
	return out
}

// toF16 converts any tensor to float16, dequantizing ggml's block formats.
func toF16(t *rawTensor) []uint16 {
	out := make([]uint16, t.n)
	d := t.data
	switch t.typ {
	case tF16:
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(d[i*2:])
		}
	case tF32:
		for i := range out {
			out[i] = f32toF16(math.Float32frombits(binary.LittleEndian.Uint32(d[i*4:])))
		}
	default:
		var blk [32]float32
		bs, _ := typeBytes(t.typ, 32)
		for b := 0; b < t.n/32; b++ {
			dequantBlock(t.typ, d[b*bs:(b+1)*bs], &blk)
			for j, v := range blk {
				out[b*32+j] = f32toF16(v)
			}
		}
	}
	return out
}

func dequantBlock(typ int, b []byte, out *[32]float32) {
	h := func(o int) float32 { return f16tab[binary.LittleEndian.Uint16(b[o:])] }
	switch typ {
	case tQ4_0:
		dd := h(0)
		qs := b[2:18]
		for j := 0; j < 16; j++ {
			out[j] = float32(int(qs[j]&0xF)-8) * dd
			out[j+16] = float32(int(qs[j]>>4)-8) * dd
		}
	case tQ4_1:
		dd, mm := h(0), h(2)
		qs := b[4:20]
		for j := 0; j < 16; j++ {
			out[j] = float32(qs[j]&0xF)*dd + mm
			out[j+16] = float32(qs[j]>>4)*dd + mm
		}
	case tQ5_0:
		dd := h(0)
		qh := binary.LittleEndian.Uint32(b[2:6])
		qs := b[6:22]
		for j := 0; j < 16; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			out[j] = float32(int(qs[j]&0xF|xh0)-16) * dd
			out[j+16] = float32(int(qs[j]>>4|xh1)-16) * dd
		}
	case tQ5_1:
		dd, mm := h(0), h(2)
		qh := binary.LittleEndian.Uint32(b[4:8])
		qs := b[8:24]
		for j := 0; j < 16; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			out[j] = float32(qs[j]&0xF|xh0)*dd + mm
			out[j+16] = float32(qs[j]>>4|xh1)*dd + mm
		}
	case tQ8_0:
		dd := h(0)
		for j := 0; j < 32; j++ {
			out[j] = float32(int8(b[2+j])) * dd
		}
	}
}
