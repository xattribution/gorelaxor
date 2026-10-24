package whisper

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
	"time"
)

// wav builds a WAV file of a tone in the given format.
func wav(rate, channels, bits int, float, extensible bool, secs float64, freq float64) []byte {
	n := int(float64(rate) * secs)
	var data bytes.Buffer
	for i := 0; i < n; i++ {
		v := 0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate))
		for c := 0; c < channels; c++ {
			x := v
			if c == 1 {
				x = -v * 0 // a silent second channel halves the level
			}
			switch {
			case float && bits == 32:
				binary.Write(&data, binary.LittleEndian, float32(x))
			case float:
				binary.Write(&data, binary.LittleEndian, x)
			case bits == 8:
				data.WriteByte(byte(int(x*127) + 128))
			case bits == 16:
				binary.Write(&data, binary.LittleEndian, int16(x*32767))
			case bits == 24:
				s := int32(x * 8388607)
				data.Write([]byte{byte(s), byte(s >> 8), byte(s >> 16)})
			default:
				binary.Write(&data, binary.LittleEndian, int32(x*2147483647))
			}
		}
	}
	var b bytes.Buffer
	fmtLen := 16
	if extensible {
		fmtLen = 40
	}
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(4+8+fmtLen+8+4+8+data.Len()))
	b.WriteString("WAVE")
	b.WriteString("LIST")
	binary.Write(&b, binary.LittleEndian, uint32(4))
	b.WriteString("INFO")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(fmtLen))
	tag := 1
	if float {
		tag = 3
	}
	if extensible {
		binary.Write(&b, binary.LittleEndian, uint16(0xFFFE))
	} else {
		binary.Write(&b, binary.LittleEndian, uint16(tag))
	}
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(rate))
	binary.Write(&b, binary.LittleEndian, uint32(rate*channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(bits))
	if extensible {
		binary.Write(&b, binary.LittleEndian, uint16(22))
		binary.Write(&b, binary.LittleEndian, uint16(bits))
		binary.Write(&b, binary.LittleEndian, uint32(0))
		binary.Write(&b, binary.LittleEndian, uint16(tag))
		b.Write([]byte{0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71})
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(data.Len()))
	b.Write(data.Bytes())
	return b.Bytes()
}

// toneLevel is the RMS of x and its power at freq (Hz, at 16 kHz).
func toneLevel(x []float32, freq float64) (rms, at float64) {
	var re, im, s float64
	for i, v := range x {
		s += float64(v) * float64(v)
		a := 2 * math.Pi * freq * float64(i) / SampleRate
		re += float64(v) * math.Cos(a)
		im += float64(v) * math.Sin(a)
	}
	n := float64(len(x))
	return math.Sqrt(s / n), 2 * math.Hypot(re, im) / n
}

func TestDecodeWAVFormats(t *testing.T) {
	for _, c := range []struct {
		rate, ch, bits int
		float, ext     bool
	}{
		{16000, 1, 16, false, false},
		{16000, 1, 8, false, false},
		{48000, 1, 24, false, false},
		{44100, 2, 16, false, false},
		{44100, 1, 32, false, false},
		{48000, 1, 32, true, false},
		{22050, 1, 64, true, false},
		{48000, 2, 32, true, true},
		{8000, 1, 16, false, true},
	} {
		data := wav(c.rate, c.ch, c.bits, c.float, c.ext, 1.0, 440)
		pcm, err := DecodeWAV(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		if d := len(pcm) - SampleRate; d < -2 || d > 2 {
			t.Errorf("%+v: %d samples, want about 16000", c, len(pcm))
		}
		want := 0.5 / float64(c.ch)
		_, amp := toneLevel(pcm[400:len(pcm)-400], 440)
		tol := 0.01
		if c.bits == 8 {
			tol = 0.02
		}
		if math.Abs(amp-want) > tol {
			t.Errorf("%+v: tone amplitude %.3f, want %.3f", c, amp, want)
		}
	}
	if _, err := DecodeWAV(bytes.NewReader([]byte("RIFF\x00\x00\x00\x00WAVEjunk"))); err == nil {
		t.Error("a broken file should fail")
	}
}

func TestResample(t *testing.T) {
	// A 1 kHz tone survives 48 kHz -> 16 kHz; a 10 kHz one (above the new
	// Nyquist frequency) is filtered out instead of folding back to 6 kHz.
	mk := func(freq float64) []float32 {
		x := make([]float32, 48000)
		for i := range x {
			x[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/48000))
		}
		return x
	}
	y := Resample(mk(1000), 48000, 16000)
	if len(y) != 16000 {
		t.Fatalf("%d samples", len(y))
	}
	if _, a := toneLevel(y[200:len(y)-200], 1000); math.Abs(a-0.5) > 0.005 {
		t.Errorf("1 kHz amplitude %.4f", a)
	}
	z := Resample(mk(10000), 48000, 16000)
	if r, _ := toneLevel(z[200:len(z)-200], 6000); r > 0.005 {
		t.Errorf("10 kHz leaked through at RMS %.4f", r)
	}
	// Upsampling 8 kHz -> 16 kHz keeps a 1 kHz tone.
	x := make([]float32, 8000)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(2*math.Pi*1000*float64(i)/8000))
	}
	if _, a := toneLevel(Resample(x, 8000, 16000)[400:15600], 1000); math.Abs(a-0.5) > 0.01 {
		t.Errorf("upsampled amplitude %.4f", a)
	}
}

// Silence and steady noise come back empty, and quickly.
func TestSilence(t *testing.T) {
	m := testModel(t, "tiny.en")
	n := 90 * SampleRate
	quiet := make([]float32, n)
	noise := make([]float32, n)
	r := rand.New(rand.NewPCG(3, 4))
	for i := range noise {
		quiet[i] = float32(r.NormFloat64() * 0.0005) // about -66 dBFS
		noise[i] = float32(r.NormFloat64() * 0.01)
	}
	for name, pcm := range map[string][]float32{"zeros": make([]float32, n), "quiet": quiet, "noise": noise} {
		t0 := time.Now()
		segs, err := m.Transcribe(context.Background(), pcm, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if len(segs) > 0 {
			t.Errorf("%s: %q", name, joined(segs))
		}
		if el := time.Since(t0); el > 2*time.Second {
			t.Errorf("%s: took %v", name, el)
		}
	}
	// Speech after a long silence keeps its true time.
	pcm := append(make([]float32, 50*SampleRate), jfk(t)...)
	segs, err := m.Transcribe(context.Background(), pcm, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) == 0 || norm(joined(segs)) != jfkText {
		t.Fatalf("got %q", joined(segs))
	}
	if s := segs[0].Start; s < 49500*time.Millisecond || s > 51*time.Second {
		t.Errorf("speech starts at %v, want about 50 s", s)
	}
}

// Speech squashed flat by automatic gain control, and speech in a noisy
// room, still get through the gate.
func TestGateKeepsSpeech(t *testing.T) {
	m := testModel(t, "base.en")
	src := jfk(t)
	// AGC: every 100 ms scaled to the same loudness, pauses included.
	agc := make([]float32, len(src))
	const blk = 1600
	for i := 0; i < len(src); i += blk {
		e := min(i+blk, len(src))
		var s float64
		for _, v := range src[i:e] {
			s += float64(v) * float64(v)
		}
		g := 0.1 / (math.Sqrt(s/float64(e-i)) + 1e-3)
		for j := i; j < e; j++ {
			agc[j] = float32(float64(src[j]) * g)
		}
	}
	noisy := make([]float32, len(src))
	r := rand.New(rand.NewPCG(5, 6))
	var s float64
	for _, v := range src {
		s += float64(v) * float64(v)
	}
	sd := math.Sqrt(s/float64(len(src))) / 2 // 6 dB below the speech
	for i, v := range src {
		noisy[i] = v + float32(r.NormFloat64()*sd)
	}
	for name, pcm := range map[string][]float32{"agc": agc, "noisy": noisy} {
		segs, err := m.Transcribe(context.Background(), pcm, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if w := wer(jfkText, joined(segs)); w > 0.15 {
			t.Errorf("%s: WER %.0f%%: %q", name, 100*w, joined(segs))
		}
	}
}

func TestCancel(t *testing.T) {
	m := testModel(t, "base.en")
	pcm, _ := wavFile(t, "ls2min")
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(700*time.Millisecond, cancel)
	t0 := time.Now()
	_, err := m.Transcribe(ctx, pcm, Options{})
	if err == nil {
		t.Fatal("no error after cancel")
	}
	if el := time.Since(t0); el > 2*time.Second {
		t.Errorf("stopped %v after starting", el)
	}
	// The model still works afterwards.
	segs, err := m.Transcribe(context.Background(), jfk(t), Options{})
	if err != nil || norm(joined(segs)) != jfkText {
		t.Fatalf("after cancel: %v %q", err, joined(segs))
	}
	m2 := testModel(t, "tiny.en")
	st := m2.NewStream(ctx, Options{})
	if _, err := st.Close(); err == nil {
		t.Error("a stream on a canceled context should report it")
	}
}

func BenchmarkTranscribe(b *testing.B) {
	for _, name := range []string{"tiny.en", "base.en"} {
		b.Run(name, func(b *testing.B) {
			m := testModel(b, name)
			pcm, _ := wavFile(b, "ls2min")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.Transcribe(context.Background(), pcm, Options{}); err != nil {
					b.Fatal(err)
				}
			}
			audio := float64(len(pcm)) / SampleRate
			b.ReportMetric(audio*float64(b.N)/b.Elapsed().Seconds(), "x-realtime")
		})
	}
}
