// Command transcribe runs the local transcription engine on an audio file,
// for testing and timing: transcribe -m ggml-base.en.bin recording.wav
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/xattribution/gorelaxor/whisper"
)

func main() {
	model := flag.String("m", "", "model file (whisper.cpp ggml format)")
	threads := flag.Int("t", 0, "threads (0: all cores but one)")
	lang := flag.String("l", "", "language code; empty detects it for multilingual models")
	prompt := flag.String("p", "", "words to expect")
	decodeOnly := flag.Bool("decode", false, "only decode the audio and report its length")
	flag.Parse()
	if flag.NArg() != 1 || (*model == "" && !*decodeOnly) {
		fmt.Fprintln(os.Stderr, "usage: transcribe -m model.bin [-t threads] [-l lang] audio-file")
		os.Exit(2)
	}
	t0 := time.Now()
	pcm, err := whisper.DecodeFile(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(1)
	}
	audio := time.Duration(len(pcm)) * time.Second / whisper.SampleRate
	var peak float32
	for _, v := range pcm {
		peak = max(peak, v, -v)
	}
	fmt.Fprintf(os.Stderr, "decoded %v of audio (peak %.2f) in %v\n", audio.Round(time.Millisecond), peak, time.Since(t0).Round(time.Millisecond))
	if *decodeOnly {
		return
	}
	t0 = time.Now()
	m, err := whisper.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Fprintf(os.Stderr, "loaded %+v in %v, heap %d MB\n", m.Info(), time.Since(t0).Round(time.Millisecond), ms.HeapAlloc>>20)
	t0 = time.Now()
	segs, err := m.Transcribe(context.Background(), pcm, whisper.Options{Threads: *threads, Language: *lang, Prompt: *prompt,
		OnSegment: func(s whisper.Segment) {
			fmt.Printf("[%s → %s] %s\n", stamp(s.Start), stamp(s.End), s.Text)
		}})
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}
	el := time.Since(t0)
	runtime.ReadMemStats(&ms)
	fmt.Fprintf(os.Stderr, "%d segments in %v, %.1fx real time, peak heap %d MB\n", len(segs), el.Round(time.Millisecond), audio.Seconds()/el.Seconds(), ms.Sys>>20)
}

func stamp(d time.Duration) string {
	s := int(d / time.Second)
	return fmt.Sprintf("%02d:%02d:%02d.%02d", s/3600, s/60%60, s%60, int(d%time.Second/(10*time.Millisecond)))
}
