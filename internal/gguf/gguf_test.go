package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/xattribution/gorelaxor/internal/kern"
)

type tensorSpec struct {
	name string
	dims []uint64
	typ  uint32
	data []byte
}

func build(kvs func(b []byte) ([]byte, int), ts []tensorSpec) []byte {
	le := binary.LittleEndian
	str := func(b []byte, s string) []byte { return append(le.AppendUint64(b, uint64(len(s))), s...) }
	b := []byte("GGUF")
	b = le.AppendUint32(b, 3)
	b = le.AppendUint64(b, uint64(len(ts)))
	kvb, n := kvs(nil)
	b = le.AppendUint64(b, uint64(n))
	b = append(b, kvb...)
	off := uint64(0)
	for _, t := range ts {
		b = str(b, t.name)
		b = le.AppendUint32(b, uint32(len(t.dims)))
		for _, d := range t.dims {
			b = le.AppendUint64(b, d)
		}
		b = le.AppendUint32(b, t.typ)
		b = le.AppendUint64(b, off)
		off += uint64(len(t.data))
		off += (32 - off%32) % 32
	}
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	for _, t := range ts {
		b = append(b, t.data...)
		for len(b)%32 != 0 {
			b = append(b, 0)
		}
	}
	return b
}

func TestParse(t *testing.T) {
	le := binary.LittleEndian
	kvs := func(b []byte) ([]byte, int) {
		str := func(b []byte, s string) []byte { return append(le.AppendUint64(b, uint64(len(s))), s...) }
		b = str(b, "general.architecture")
		b = le.AppendUint32(b, 8)
		b = str(b, "parakeet")
		b = str(b, "n")
		b = le.AppendUint32(b, 4)
		b = le.AppendUint32(b, 7)
		b = str(b, "toks")
		b = le.AppendUint32(b, 9)
		b = le.AppendUint32(b, 8)
		b = le.AppendUint64(b, 2)
		b = str(b, "a")
		b = str(b, "▁b")
		return b, 3
	}
	f32 := make([]byte, 0, 4*6)
	for i := 0; i < 6; i++ {
		f32 = le.AppendUint32(f32, math.Float32bits(float32(i)+0.5))
	}
	f16 := make([]byte, 0, 2*32)
	q8 := make([]byte, 0, 34)
	q8 = le.AppendUint16(q8, kern.F32ToF16(0.25))
	for i := 0; i < 32; i++ {
		f16 = le.AppendUint16(f16, kern.F32ToF16(float32(i)))
		q8 = append(q8, byte(int8(i-16)))
	}
	b := build(kvs, []tensorSpec{
		{"a", []uint64{3, 2}, TypeF32, f32},
		{"pw", []uint64{1, 4, 8}, TypeF16, f16}, // a 1x1 convolution kernel: 8 rows of 4
		{"q", []uint64{32, 1}, TypeQ8_0, q8},
	})
	f, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if f.String("general.architecture", "") != "parakeet" || f.Int("n", 0) != 7 {
		t.Fatalf("settings: %v", f.KV)
	}
	if s := f.Strings("toks"); len(s) != 2 || s[1] != "▁b" {
		t.Fatalf("strings: %v", s)
	}
	w, err := f.Weight("a")
	if err != nil || w.Rows != 2 || w.Cols != 3 {
		t.Fatalf("a: %v %+v", err, w)
	}
	if v := w.Float32(); v[4] != 4.5 {
		t.Fatalf("a values %v", v)
	}
	w, err = f.Weight("pw")
	if err != nil || w.Rows != 8 || w.Cols != 4 {
		t.Fatalf("pw: %v %+v", err, w)
	}
	q, err := f.Float32("q")
	if err != nil || q[0] != -4 || q[31] != 3.75 {
		t.Fatalf("q: %v %v", err, q)
	}
	if _, err := Parse(b[:len(b)-40]); err == nil {
		t.Fatal("a cut-short file parsed")
	}
}
