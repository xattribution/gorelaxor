package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
)

// one builds a file with no KVs (or the given kv bytes) and a single tensor
// whose dims/type/offset are raw, followed by `pad` bytes of data.
func one(kv []byte, nkv int, dims []uint64, typ uint32, off uint64, pad int) []byte {
	le := binary.LittleEndian
	b := []byte("GGUF")
	b = le.AppendUint32(b, 3)
	b = le.AppendUint64(b, 1)
	b = le.AppendUint64(b, uint64(nkv))
	b = append(b, kv...)
	b = le.AppendUint64(b, 1)
	b = append(b, 't')
	b = le.AppendUint32(b, uint32(len(dims)))
	for _, d := range dims {
		b = le.AppendUint64(b, d)
	}
	b = le.AppendUint32(b, typ)
	b = le.AppendUint64(b, off)
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	return append(b, make([]byte, pad)...)
}

// A model file comes from outside, so a crafted one must give an
// error: no panic, no huge allocation, no tensor reading the header.
func TestHostileFiles(t *testing.T) {
	b0 := one(nil, 0, []uint64{2}, TypeF32, 0, 64)
	base := uint64(len(b0) - 64)
	le := binary.LittleEndian
	align := func(v uint64, typ uint32) []byte {
		kv := le.AppendUint64(nil, 17)
		kv = append(kv, "general.alignment"...)
		kv = le.AppendUint32(kv, typ)
		if typ == 4 {
			return le.AppendUint32(kv, uint32(v))
		}
		return le.AppendUint64(kv, v)
	}
	cases := []struct {
		name string
		b    []byte
	}{
		{"a dimension that reads as negative", one(nil, 0, []uint64{math.MaxUint64}, TypeF32, 8, 64)},
		{"an offset that wraps", one(nil, 0, []uint64{2}, TypeF32, math.MaxUint64-3, 64)},
		{"an offset that wraps onto the header", one(nil, 0, []uint64{2}, TypeF32, -base-4, 64)},
		{"dimensions whose product overflows", one(nil, 0, []uint64{1 << 32, 1 << 31}, TypeF16, 0, 0)},
		{"a dimension past the file", one(nil, 0, []uint64{1<<62 + 1}, TypeF32, 0, 32)},
		{"two negative dimensions", one(nil, 0, []uint64{math.MaxUint64 - 1, math.MaxUint64 - 2}, TypeF32, 0, 64)},
		{"negative Q8_0 dimensions", one(nil, 0, []uint64{uint64(1<<64 - 32), uint64(1<<64 - 32)}, TypeQ8_0, 0, 40000)},
		{"a zero dimension", one(nil, 0, []uint64{0, 4}, TypeF32, 0, 64)},
		{"an alignment near the largest integer", one(align(math.MaxInt64-5, 10), 1, []uint64{4}, TypeF32, 0, 0)},
		{"an alignment that isn't a power of two", one(align(48, 4), 1, []uint64{4}, TypeF32, 0, 64)},
	}
	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panic %v", c.name, r)
				}
			}()
			if _, err := Parse(c.b); err == nil {
				t.Errorf("%s: parsed without an error", c.name)
			}
		}()
	}
}

// An array count larger than the file is refused before anything is
// reserved for it.
func TestArrayCountAllocatesNothing(t *testing.T) {
	le := binary.LittleEndian
	big := 8 << 20
	kv := le.AppendUint64(nil, 1)
	kv = append(kv, 'a')
	kv = le.AppendUint32(kv, 9)
	kv = le.AppendUint32(kv, 0)
	kv = le.AppendUint64(kv, uint64(big))
	b := []byte("GGUF")
	b = le.AppendUint32(b, 3)
	b = le.AppendUint64(b, 0)
	b = le.AppendUint64(b, 1)
	b = append(b, kv...)
	b = append(b, make([]byte, big-len(b))...)
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	_, err := Parse(b)
	runtime.ReadMemStats(&m1)
	if err == nil {
		t.Fatal("an array longer than the bytes left parsed")
	}
	if got := (m1.TotalAlloc - m0.TotalAlloc) >> 20; got > 4 {
		t.Fatalf("parsing allocated %d MB for a count it refused", got)
	}
}

func FuzzParse(f *testing.F) {
	f.Add(one(nil, 0, []uint64{4, 2}, TypeF32, 0, 64))
	f.Add(one(nil, 0, []uint64{32, 2}, TypeQ8_0, 0, 96))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<16 {
			return
		}
		g, err := Parse(b)
		if err != nil {
			return
		}
		for _, n := range g.Names() {
			g.Float32(n)
			g.Weight(n)
		}
	})
}

// ReadMeta reads the settings of a real file without its weights, and a
// cut-off or foreign file gives an error.
func TestReadMeta(t *testing.T) {
	dir := t.TempDir()
	le := binary.LittleEndian
	kv := le.AppendUint64(nil, 20)
	kv = append(kv, "general.architecture"...)
	kv = le.AppendUint32(kv, 8)
	kv = le.AppendUint64(kv, 8)
	kv = append(kv, "parakeet"...)
	b := one(kv, 1, []uint64{2}, TypeF32, 0, 8)
	p := dir + "/m.gguf"
	os.WriteFile(p, b, 0o600)
	f, err := ReadMeta(p)
	if err != nil || f.String("general.architecture", "") != "parakeet" {
		t.Fatalf("%v %v", err, f)
	}
	os.WriteFile(p, b[:30], 0o600)
	if _, err := ReadMeta(p); err == nil {
		t.Fatal("a cut-off header was read")
	}
	os.WriteFile(p, []byte("ggml not gguf"), 0o600)
	if _, err := ReadMeta(p); err == nil {
		t.Fatal("a foreign file was read")
	}
}

// Open gives each tensor memory of its own with the same bytes Parse
// sees, reads a header longer than its first read, and refuses tensors
// that overlap, which would cost more memory than the file holds.
func TestOpenOwnMemory(t *testing.T) {
	le := binary.LittleEndian
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i)
	}
	long := func(b []byte) ([]byte, int) { // a 2 MB string key, past Open's first 1 MB read
		b = le.AppendUint64(b, 3)
		b = append(b, "pad"...)
		b = le.AppendUint32(b, 8)
		b = le.AppendUint64(b, 2<<20)
		return append(b, make([]byte, 2<<20)...), 1
	}
	b := build(long, []tensorSpec{{"a", []uint64{1024}, TypeF32, data}, {"b", []uint64{16}, TypeF32, data[:64]}})
	p := t.TempDir() + "/m.gguf"
	os.WriteFile(p, b, 0o600)
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := Parse(b)
	for _, name := range []string{"a", "b"} {
		got, want := f.Tensors[name].data, ref.Tensors[name].data
		if string(got) != string(want) || &got[0] == &want[0] {
			t.Fatalf("%s: not its own copy of the same bytes", name)
		}
	}
	// Two tensors at the same offset, each most of the file.
	two := []byte("GGUF")
	two = le.AppendUint32(two, 3)
	two = le.AppendUint64(two, 2)
	two = le.AppendUint64(two, 0)
	for _, name := range []string{"x", "y"} {
		two = le.AppendUint64(two, 1)
		two = append(two, name...)
		two = le.AppendUint32(two, 1)
		two = le.AppendUint64(two, 1024)
		two = le.AppendUint32(two, TypeF32)
		two = le.AppendUint64(two, 0)
	}
	for len(two)%32 != 0 {
		two = append(two, 0)
	}
	two = append(two, data...)
	os.WriteFile(p, two, 0o600)
	if _, err := Parse(two); err != nil {
		t.Fatalf("overlapping tensors are a valid file to Parse: %v", err)
	}
	if _, err := Open(p); err == nil {
		t.Fatal("Open read overlapping tensors")
	}
}
