// Package gguf reads GGUF model files: the key-value header and the
// tensors, which come back as kern weights or float32 slices.
//
// It reads the subset the local speech engines use: format versions 2
// and 3, and tensors stored as F32, F16, BF16 or Q8_0.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/xattribution/gorelaxor/internal/kern"
)

// ggml tensor types this reader understands.
const (
	TypeF32  = 0
	TypeF16  = 1
	TypeQ8_0 = 8
	TypeBF16 = 30
)

// Tensor describes one tensor. Dims is in GGUF order: Dims[0] is the
// fastest-moving (the row length).
type Tensor struct {
	Name   string
	Dims   []int
	Type   int
	offset uint64
	data   []byte
}

// Elems is the number of values.
func (t *Tensor) Elems() int {
	n := 1
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// File is an opened GGUF file. Its tensors point into one buffer that
// lives as long as the File.
type File struct {
	Version int
	KV      map[string]any
	Tensors map[string]*Tensor
	order   []string
	buf     []byte
}

// MaxFile is the largest file Open reads. The biggest speech model is
// about 0.75 GB at Q8_0 and 2.5 GB at float32.
const MaxFile = 4 << 30

// Open reads a GGUF file into memory, each tensor in memory of its own,
// so a caller that repacks a tensor (kern's int8 panels) can let go of
// the original.
func Open(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() > MaxFile {
		return nil, fmt.Errorf("gguf: %s is %d MB, more than a model can be", path, st.Size()>>20)
	}
	size := st.Size()
	var f *File
	var base uint64
	for n := int64(1 << 20); ; n *= 8 { // the header, read until it's whole
		b := make([]byte, min(n, size))
		if _, err := fh.ReadAt(b, 0); err != nil && err != io.EOF {
			return nil, err
		}
		f, base, err = parseHeader(b, uint64(size))
		if err == errShort && int64(len(b)) < size {
			continue
		}
		if err != nil {
			return nil, err
		}
		break
	}
	// Tensors that overlap would each cost their full size here, so a
	// crafted file could ask for many times its length.
	var total uint64
	for _, name := range f.order {
		n, _ := byteSize(f.Tensors[name])
		total += uint64(n)
	}
	if total > uint64(size) {
		return nil, errors.New("gguf: the tensors overlap")
	}
	for _, name := range f.order {
		t := f.Tensors[name]
		n, _ := byteSize(t)
		t.data = make([]byte, n)
		if _, err := fh.ReadAt(t.data, int64(base+t.offset)); err != nil {
			return nil, fmt.Errorf("gguf: reading %s: %v", t.Name, err)
		}
	}
	return f, nil
}

var errShort = errors.New("gguf: file is cut short")

type reader struct {
	b   []byte
	off int
	err error
}

func (r *reader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || r.off+n > len(r.b) {
		r.err = errShort
		return nil
	}
	s := r.b[r.off : r.off+n]
	r.off += n
	return s
}

func (r *reader) u8() uint8 {
	if s := r.take(1); s != nil {
		return s[0]
	}
	return 0
}
func (r *reader) u16() uint16 {
	if s := r.take(2); s != nil {
		return binary.LittleEndian.Uint16(s)
	}
	return 0
}
func (r *reader) u32() uint32 {
	if s := r.take(4); s != nil {
		return binary.LittleEndian.Uint32(s)
	}
	return 0
}
func (r *reader) u64() uint64 {
	if s := r.take(8); s != nil {
		return binary.LittleEndian.Uint64(s)
	}
	return 0
}

func (r *reader) str() string {
	n := r.u64()
	if n > 1<<30 {
		r.err = errors.New("gguf: string too long")
		return ""
	}
	return string(r.take(int(n)))
}

func (r *reader) value(t uint32, depth int) any {
	switch t {
	case 0:
		return r.u8()
	case 1:
		return int8(r.u8())
	case 2:
		return r.u16()
	case 3:
		return int16(r.u16())
	case 4:
		return r.u32()
	case 5:
		return int32(r.u32())
	case 6:
		return math.Float32frombits(r.u32())
	case 7:
		return r.u8() != 0
	case 8:
		return r.str()
	case 9:
		if depth > 2 {
			r.err = errors.New("gguf: arrays nested too deep")
			return nil
		}
		et := r.u32()
		n := r.u64()
		// Every element takes at least one byte, so a count can't pass the
		// bytes left; nothing is reserved on the file's word.
		if n > uint64(len(r.b)-r.off) || n > 1<<24 {
			r.err = errors.New("gguf: array too long")
			return nil
		}
		out := make([]any, 0, min(n, 1024))
		for i := uint64(0); i < n && r.err == nil; i++ {
			out = append(out, r.value(et, depth+1))
		}
		return out
	case 10:
		return r.u64()
	case 11:
		return int64(r.u64())
	case 12:
		return math.Float64frombits(r.u64())
	}
	r.err = fmt.Errorf("gguf: unknown value type %d", t)
	return nil
}

// Parse reads a GGUF file held in b. The File keeps b.
// metaMax is the most of a file ReadMeta reads: the settings and the
// tokenizer's word list come first and fit well inside it.
const metaMax = 16 << 20

// ReadMeta reads only a file's settings (its key-value section), without
// its weights, so a list of model files can say what each one is. It
// reads the start of the file, and more only when the settings run on.
func ReadMeta(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	for n := 256 << 10; ; n *= 8 {
		b := make([]byte, min(n, metaMax))
		got, err := io.ReadFull(fh, b)
		if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
			return nil, err
		}
		f, err := parseMeta(b[:got])
		if err == errShort && got == len(b) && len(b) < metaMax {
			if _, err := fh.Seek(0, io.SeekStart); err != nil {
				return nil, err
			}
			continue
		}
		return f, err
	}
}

func parseMeta(b []byte) (*File, error) {
	r := &reader{b: b}
	if string(r.take(4)) != "GGUF" {
		return nil, errors.New("gguf: not a GGUF file")
	}
	f := &File{Version: int(r.u32()), KV: map[string]any{}, Tensors: map[string]*Tensor{}}
	if f.Version < 2 || f.Version > 3 {
		return nil, fmt.Errorf("gguf: version %d is not supported", f.Version)
	}
	r.u64()
	nkv := r.u64()
	if nkv > 1<<20 {
		return nil, errors.New("gguf: header counts out of range")
	}
	for i := uint64(0); i < nkv && r.err == nil; i++ {
		k := r.str()
		f.KV[k] = r.value(r.u32(), 0)
	}
	if r.err != nil {
		return nil, r.err
	}
	return f, nil
}

// Parse reads a GGUF file held in b; tensors point into b.
func Parse(b []byte) (*File, error) {
	f, base, err := parseHeader(b, uint64(len(b)))
	if err != nil {
		return nil, err
	}
	f.buf = b
	for _, t := range f.Tensors {
		n, _ := byteSize(t)
		start := base + t.offset
		t.data = b[start : start+uint64(n)]
	}
	return f, nil
}

// parseHeader reads the settings and tensor table from b, the start of a
// file of size bytes, and checks that every tensor fits in the file. It
// returns where the tensor data starts.
func parseHeader(b []byte, size uint64) (*File, uint64, error) {
	r := &reader{b: b}
	if string(r.take(4)) != "GGUF" {
		return nil, 0, errors.New("gguf: not a GGUF file")
	}
	f := &File{Version: int(r.u32()), KV: map[string]any{}, Tensors: map[string]*Tensor{}}
	if f.Version < 2 || f.Version > 3 {
		return nil, 0, fmt.Errorf("gguf: version %d is not supported", f.Version)
	}
	nt, nkv := r.u64(), r.u64()
	if nt > 1<<20 || nkv > 1<<20 {
		return nil, 0, errors.New("gguf: header counts out of range")
	}
	for i := uint64(0); i < nkv && r.err == nil; i++ {
		k := r.str()
		f.KV[k] = r.value(r.u32(), 0)
	}
	for i := uint64(0); i < nt && r.err == nil; i++ {
		t := &Tensor{Name: r.str()}
		nd := r.u32()
		if nd > 4 {
			return nil, 0, fmt.Errorf("gguf: tensor %s has %d dimensions", t.Name, nd)
		}
		// Each dimension, and their product, must fit in the file: a
		// crafted size can't overflow into a negative or tiny length.
		elems := uint64(1)
		for d := uint32(0); d < nd; d++ {
			v := r.u64()
			elems *= v
			if v == 0 || v > size || elems > size {
				return nil, 0, fmt.Errorf("gguf: tensor %s has an impossible size", t.Name)
			}
			t.Dims = append(t.Dims, int(v))
		}
		t.Type = int(r.u32())
		t.offset = r.u64()
		f.Tensors[t.Name] = t
		f.order = append(f.order, t.Name)
	}
	if r.err != nil {
		return nil, 0, r.err
	}
	align := 32
	if a, ok := f.KV["general.alignment"]; ok {
		v, ok := a.(uint32)
		if !ok || v < 8 || v > 4096 || v&(v-1) != 0 {
			return nil, 0, errors.New("gguf: general.alignment must be a power of two from 8 to 4096")
		}
		align = int(v)
	}
	base := uint64((r.off + align - 1) / align * align)
	for _, t := range f.Tensors {
		n, err := byteSize(t)
		if err != nil {
			return nil, 0, err
		}
		// In this order, so no sum can wrap around.
		if base > size || t.offset > size-base || uint64(n) > size-base-t.offset {
			return nil, 0, fmt.Errorf("gguf: tensor %s runs past the end of the file", t.Name)
		}
	}
	return f, base, nil
}

func byteSize(t *Tensor) (int, error) {
	n := t.Elems()
	switch t.Type {
	case TypeF32:
		return n * 4, nil
	case TypeF16, TypeBF16:
		return n * 2, nil
	case TypeQ8_0:
		if len(t.Dims) == 0 || t.Dims[0]%kern.QK != 0 {
			return 0, fmt.Errorf("gguf: Q8_0 tensor %s rows are not a multiple of 32", t.Name)
		}
		return n / kern.QK * kern.Q8Bytes, nil
	}
	return 0, fmt.Errorf("gguf: tensor %s has type %d, which this reader doesn't handle", t.Name, t.Type)
}

// Names lists the tensors in file order.
func (f *File) Names() []string { return append([]string(nil), f.order...) }

// Keys lists the header keys, sorted.
func (f *File) Keys() []string {
	k := make([]string, 0, len(f.KV))
	for s := range f.KV {
		k = append(k, s)
	}
	sort.Strings(k)
	return k
}

func toInt(v any) (int, bool) {
	switch x := v.(type) {
	case uint8:
		return int(x), true
	case int8:
		return int(x), true
	case uint16:
		return int(x), true
	case int16:
		return int(x), true
	case uint32:
		return int(x), true
	case int32:
		return int(x), true
	case uint64:
		return int(x), true
	case int64:
		return int(x), true
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// Int returns an integer key, or def when it is missing.
func (f *File) Int(key string, def int) int {
	if v, ok := toInt(f.KV[key]); ok {
		return v
	}
	return def
}

// Float returns a number key, or def when it is missing.
func (f *File) Float(key string, def float64) float64 {
	switch x := f.KV[key].(type) {
	case float32:
		return float64(x)
	case float64:
		return x
	}
	if v, ok := toInt(f.KV[key]); ok {
		return float64(v)
	}
	return def
}

// Bool returns a boolean key, or def when it is missing.
func (f *File) Bool(key string, def bool) bool {
	if x, ok := f.KV[key].(bool); ok {
		return x
	}
	if v, ok := toInt(f.KV[key]); ok {
		return v != 0
	}
	return def
}

// String returns a string key, or def when it is missing.
func (f *File) String(key, def string) string {
	if x, ok := f.KV[key].(string); ok {
		return x
	}
	return def
}

// Strings returns a string-array key.
func (f *File) Strings(key string) []string {
	a, _ := f.KV[key].([]any)
	out := make([]string, 0, len(a))
	for _, v := range a {
		s, _ := v.(string)
		out = append(out, s)
	}
	return out
}

// Floats returns a number-array key.
func (f *File) Floats(key string) []float32 {
	a, _ := f.KV[key].([]any)
	out := make([]float32, 0, len(a))
	for _, v := range a {
		switch x := v.(type) {
		case float32:
			out = append(out, x)
		case float64:
			out = append(out, float32(x))
		default:
			n, _ := toInt(v)
			out = append(out, float32(n))
		}
	}
	return out
}

// Ints returns an integer-array key.
func (f *File) Ints(key string) []int {
	a, _ := f.KV[key].([]any)
	out := make([]int, 0, len(a))
	for _, v := range a {
		n, _ := toInt(v)
		out = append(out, n)
	}
	return out
}

// Has reports whether a tensor exists.
func (f *File) Has(name string) bool { _, ok := f.Tensors[name]; return ok }

// Float32 returns a tensor's values as float32, widening F16, BF16 and
// Q8_0. An F32 tensor's slice points into the file buffer.
func (f *File) Float32(name string) ([]float32, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %s", name)
	}
	n := t.Elems()
	switch t.Type {
	case TypeF32:
		return aligned32(t.data, n), nil
	case TypeF16:
		out := make([]float32, n)
		for i := range out {
			out[i] = kern.F16ToF32(binary.LittleEndian.Uint16(t.data[2*i:]))
		}
		return out, nil
	case TypeBF16:
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(t.data[2*i:])) << 16)
		}
		return out, nil
	}
	w, err := f.Weight(name)
	if err != nil {
		return nil, err
	}
	return w.Float32(), nil
}

// aligned32 views b as float32s, copying when b is not 4-byte aligned.
func aligned32(b []byte, n int) []float32 {
	v := kern.Float32s(b)
	if len(v) == n && len(b) > 0 && uintptrOf(b)%4 == 0 {
		return v
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// Weight returns a tensor as a matrix whose rows are its first
// dimension longer than one (so a 1x1 convolution kernel [1,1,in,out]
// is an out-by-in matrix), the rest flattened into rows. BF16 is
// widened to F32.
func (f *File) Weight(name string) (*kern.Weight, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %s", name)
	}
	cols := 1
	for _, d := range t.Dims {
		if cols = d; d > 1 {
			break
		}
	}
	rows := t.Elems() / max(cols, 1)
	if t.Type == TypeQ8_0 && cols != t.Dims[0] {
		return nil, fmt.Errorf("gguf: Q8_0 tensor %s has leading unit dimensions", name)
	}
	switch t.Type {
	case TypeF32:
		return kern.NewF32(rows, cols, aligned32(t.data, rows*cols))
	case TypeF16:
		h := kern.Uint16s(t.data)
		if uintptrOf(t.data)%2 != 0 {
			h = make([]uint16, rows*cols)
			for i := range h {
				h[i] = binary.LittleEndian.Uint16(t.data[2*i:])
			}
		}
		return kern.NewF16(rows, cols, h)
	case TypeQ8_0:
		return kern.NewQ8(rows, cols, t.data)
	case TypeBF16:
		v, err := f.Float32(name)
		if err != nil {
			return nil, err
		}
		return kern.NewF32(rows, cols, v)
	}
	return nil, fmt.Errorf("gguf: tensor %s has type %d", name, t.Type)
}
