// Command speak runs the Pocket TTS engine on a line of text and writes
// a WAV file, for testing and timing:
//
//	speak -m pocket-tts-english_2026-09-Q8_0.gguf -voice alba -o out.wav "Text to read."
//	speak -m model.gguf -voice voices/voice-marius.gguf -o out.wav "Text to read."
//	speak -m model.gguf -voice clip.wav -o out.wav "Text in the clip's voice."
//
// A -voice name is looked up as voice-<name>.gguf beside the model; a
// .wav clip is cloned.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/xattribution/gorelaxor/pockettts"
	"github.com/xattribution/gorelaxor/whisper"
)

func main() {
	model := flag.String("m", "", "model file")
	voice := flag.String("voice", "alba", "preset voice name, voice file, or a .wav clip to clone")
	out := flag.String("o", "out.wav", "WAV file to write")
	threads := flag.Int("t", 0, "threads (0: all cores)")
	seed := flag.Uint64("seed", 0, "random seed")
	temp := flag.Float64("temp", 0, "temperature (0: the model's, 0.3)")
	profile := flag.String("cpuprofile", "", "write a CPU profile of the run to this file")
	flag.Parse()
	if *model == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: speak -m model.gguf [-voice alba] [-o out.wav] text")
		os.Exit(2)
	}
	if *threads <= 0 {
		*threads = runtime.NumCPU()
	}
	text := strings.Join(flag.Args(), " ")
	t0 := time.Now()
	m, err := pockettts.Load(*model)
	check(err)
	fmt.Fprintf(os.Stderr, "loaded %s in %v\n", *model, time.Since(t0).Round(time.Millisecond))

	t0 = time.Now()
	var v *pockettts.Voice
	switch {
	case strings.HasSuffix(strings.ToLower(*voice), ".wav"):
		f, err := os.Open(*voice)
		check(err)
		pcm, rate, err := whisper.DecodeWAVRate(f)
		f.Close()
		check(err)
		v, err = m.CloneVoice(pcm, rate, *threads)
		check(err)
		fmt.Fprintf(os.Stderr, "cloned %.1f s of %s in %v\n", float64(len(pcm))/float64(rate), *voice, time.Since(t0).Round(time.Millisecond))
	case strings.HasSuffix(*voice, ".gguf"):
		v, err = m.LoadVoice(*voice)
		check(err)
	default:
		v, err = m.LoadVoice(filepath.Join(filepath.Dir(*model), "voice-"+*voice+".gguf"))
		check(err)
	}

	if *profile != "" {
		f, err := os.Create(*profile)
		check(err)
		check(pprof.StartCPUProfile(f))
		defer pprof.StopCPUProfile()
	}
	var pcm []float32
	var first time.Duration
	t0 = time.Now()
	err = m.Speak(text, v, pockettts.Options{Seed: *seed, Temperature: float32(*temp), Threads: *threads}, func(p []float32) error {
		if pcm == nil {
			first = time.Since(t0)
		}
		pcm = append(pcm, p...)
		return nil
	})
	check(err)
	el := time.Since(t0)
	secs := float64(len(pcm)) / float64(m.SampleRate())
	fmt.Fprintf(os.Stderr, "%.2f s of audio in %v: %.2fx real time, first audio after %v\n",
		secs, el.Round(time.Millisecond), secs/el.Seconds(), first.Round(time.Millisecond))
	check(writeWAV(*out, pcm, m.SampleRate()))
}

func writeWAV(path string, pcm []float32, rate int) error {
	b := make([]byte, 44+2*len(pcm))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*len(pcm)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1) // PCM
	binary.LittleEndian.PutUint16(b[22:], 1) // mono
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(2*rate))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*len(pcm)))
	for i, v := range pcm {
		s := math.Round(float64(max(-1, min(1, v))) * 32767)
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(int16(s)))
	}
	return os.WriteFile(path, b, 0o644)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
