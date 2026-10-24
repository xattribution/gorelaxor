package whisper

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

var errWAV = errors.New("not a WAV file this program can read")

// MaxSeconds is the longest recording the decoders read: three hours. A
// longer one is refused before it can use gigabytes of memory.
const MaxSeconds = 3 * 3600

var errTooLong = errors.New("the recording is longer than three hours; split it first")

// DecodeWAV reads a WAV file (8, 16, 24 or 32-bit PCM, or 32 or 64-bit
// float; any sample rate; any number of channels) and returns it as 16 kHz
// mono, the channels averaged.
func DecodeWAV(r io.Reader) ([]float32, error) {
	mono, rate, err := DecodeWAVRate(r)
	if err != nil {
		return nil, err
	}
	return Resample(mono, rate, SampleRate), nil
}

// DecodeWAVRate is DecodeWAV without the resampling: mono at the file's
// own sample rate, which it returns.
func DecodeWAVRate(r io.Reader) ([]float32, int, error) {
	br := bufio.NewReaderSize(r, 1<<16)
	var hdr [12]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil || string(hdr[0:4]) != "RIFF" && string(hdr[0:4]) != "RF64" || string(hdr[8:12]) != "WAVE" {
		return nil, 0, errWAV
	}
	var format, channels, bits int
	var rate int
	haveFmt := false
	for {
		var ch [8]byte
		if _, err := io.ReadFull(br, ch[:]); err != nil {
			return nil, 0, errWAV
		}
		id := string(ch[0:4])
		size := int64(binary.LittleEndian.Uint32(ch[4:8]))
		switch id {
		case "fmt ":
			if size < 16 || size > 1<<16 {
				return nil, 0, errWAV
			}
			b := make([]byte, size+size%2)
			if _, err := io.ReadFull(br, b); err != nil {
				return nil, 0, errWAV
			}
			format = int(binary.LittleEndian.Uint16(b[0:]))
			channels = int(binary.LittleEndian.Uint16(b[2:]))
			rate = int(binary.LittleEndian.Uint32(b[4:]))
			bits = int(binary.LittleEndian.Uint16(b[14:]))
			if format == 0xFFFE && size >= 40 {
				format = int(binary.LittleEndian.Uint16(b[24:])) // the subformat GUID's first two bytes
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, 0, errWAV
			}
			if channels < 1 || channels > 32 || rate < 1000 || rate > 768000 {
				return nil, 0, errWAV
			}
			var lr io.Reader = br
			if size > 0 && size != 0xFFFFFFFF {
				lr = io.LimitReader(br, size)
			}
			mono, err := readPCM(lr, format, bits, channels, rate)
			if err != nil {
				return nil, 0, err
			}
			return mono, rate, nil
		default:
			if _, err := io.CopyN(io.Discard, br, size+size%2); err != nil {
				return nil, 0, errWAV
			}
		}
	}
}

func readPCM(r io.Reader, format, bits, channels, rate int) ([]float32, error) {
	bps := bits / 8
	switch {
	case format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32):
	case format == 3 && (bits == 32 || bits == 64):
	default:
		return nil, errors.New("the WAV file's sample format isn't plain PCM or float")
	}
	frame := bps * channels
	buf := make([]byte, frame*4096)
	var out []float32
	carry := 0
	for {
		n, err := io.ReadFull(r, buf[carry:])
		n += carry
		whole := n - n%frame
		for o := 0; o < whole; o += frame {
			var s float64
			for c := 0; c < channels; c++ {
				p := buf[o+c*bps:]
				switch {
				case format == 3 && bits == 32:
					s += float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
				case format == 3:
					s += math.Float64frombits(binary.LittleEndian.Uint64(p))
				case bits == 8:
					s += (float64(p[0]) - 128) / 128
				case bits == 16:
					s += float64(int16(binary.LittleEndian.Uint16(p))) / 32768
				case bits == 24:
					v := int32(uint32(p[0])|uint32(p[1])<<8|uint32(p[2])<<16) << 8 >> 8
					s += float64(v) / 8388608
				default:
					s += float64(int32(binary.LittleEndian.Uint32(p))) / 2147483648
				}
			}
			out = append(out, float32(s/float64(channels)))
		}
		carry = copy(buf, buf[whole:n])
		if len(out) > MaxSeconds*rate {
			return nil, errTooLong
		}
		if err != nil {
			break
		}
	}
	return out, nil
}

// Resample converts audio between sample rates with a windowed-sinc filter
// (Blackman-Harris, 24 zero crossings), low-passed below the lower rate's
// Nyquist frequency.
func Resample(samples []float32, from, to int) []float32 {
	if from == to || len(samples) == 0 {
		return samples
	}
	ratio := float64(to) / float64(from)
	cut := 0.95 * min(1, ratio) // cutoff as a fraction of the input Nyquist
	const zc = 12               // zero crossings each side
	half := float64(zc) / cut   // filter half-width in input samples
	nOut := int(math.Floor(float64(len(samples)) * ratio))
	out := make([]float32, nOut)
	// A table of the filter at 512 steps per input sample.
	const steps = 512
	tl := int(math.Ceil(half*steps)) + 2
	table := make([]float32, tl)
	for i := range table {
		x := float64(i) / steps // input samples from the center
		if x > half {
			continue
		}
		t := x * cut
		s := 1.0
		if t != 0 {
			s = math.Sin(math.Pi*t) / (math.Pi * t)
		}
		w := (x/half + 1) / 2 // 0.5 at the center, 1 at the edge
		bh := 0.35875 - 0.48829*math.Cos(2*math.Pi*w) + 0.14128*math.Cos(4*math.Pi*w) - 0.01168*math.Cos(6*math.Pi*w)
		table[i] = float32(cut * s * bh)
	}
	tap := func(x float64) float32 {
		x = math.Abs(x) * steps
		i := int(x)
		if i+1 >= tl {
			return 0
		}
		f := float32(x - float64(i))
		return table[i]*(1-f) + table[i+1]*f
	}
	n := len(samples)
	for j := range out {
		c := float64(j) / ratio
		lo := max(0, int(math.Ceil(c-half)))
		hi := min(n-1, int(math.Floor(c+half)))
		var s float32
		for i := lo; i <= hi; i++ {
			s += samples[i] * tap(float64(i)-c)
		}
		out[j] = s
	}
	return out
}

// CanDecode reports whether DecodeFile can read the file, by its name.
func CanDecode(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".wav" || ext == ".wave" {
		return true
	}
	return platformCanDecode(ext)
}

// DecodeFile reads an audio file as 16 kHz mono. WAV works everywhere; on
// Windows, Media Foundation also reads MP3, M4A, MP4, AAC, WMA and the rest
// of what Windows can play.
func DecodeFile(path string) ([]float32, error) {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".wav" || ext == ".wave" {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		pcm, err := DecodeWAV(f)
		if err == nil || !platformCanDecode(ext) {
			return pcm, err
		}
	}
	if !platformCanDecode(ext) {
		return nil, errors.New("this computer can't read " + ext + " audio; save it as WAV")
	}
	pcm, rate, err := platformDecode(path)
	if err != nil {
		return nil, err
	}
	return Resample(pcm, rate, SampleRate), nil
}
