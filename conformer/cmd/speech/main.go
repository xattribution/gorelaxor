// Command speech runs the FastConformer engine on an audio file, for
// testing and timing:
//
//	speech -m parakeet-tdt-0.6b-v2-Q8_0.gguf -vad vad.gguf meeting.wav
//	speech -m nemotron-speech-streaming-en-0.6b-Q8_0.gguf -stream 13 meeting.wav
//	speech -m nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf -lang de-DE -stream 13 meeting.wav
//	speech -diar diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf meeting.wav
//	speech -m multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf -diar sortformer.gguf -stream 13 meeting.wav
//	speech -vad vad.gguf meeting.wav
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"sort"
	"time"

	"github.com/xattribution/gorelaxor/conformer"
	"github.com/xattribution/gorelaxor/whisper"
)

func main() {
	model := flag.String("m", "", "Parakeet, Nemotron or multitalker model file")
	vadPath := flag.String("vad", "", "MarbleNet silence detector file")
	diarPath := flag.String("diar", "", "Sortformer speaker file")
	stream := flag.Int("stream", -1, "stream with this right context (Nemotron: 13, 6, 1 or 0; Nemotron 3.5: 13, 6, 3 or 0) instead of one pass")
	preset := flag.String("preset", "high", "Sortformer settings: very-high, high or low latency")
	threads := flag.Int("t", 0, "threads (0: all cores but one)")
	words := flag.Bool("words", false, "print each word with its time")
	profile := flag.String("cpuprofile", "", "write a CPU profile of the run to this file")
	lang := flag.String("lang", "auto", "multilingual models: a locale such as de-DE, or auto")
	flag.Parse()
	if flag.NArg() != 1 || (*model == "" && *diarPath == "" && *vadPath == "") {
		fmt.Fprintln(os.Stderr, "usage: speech [-m model.gguf] [-vad vad.gguf] [-diar sortformer.gguf] [-stream R] audio-file")
		os.Exit(2)
	}
	if *threads <= 0 {
		*threads = max(1, runtime.NumCPU()-1)
	}
	pcm, err := whisper.DecodeFile(flag.Arg(0))
	check(err)
	secs := float64(len(pcm)) / 16000
	fmt.Fprintf(os.Stderr, "%.1f s of audio\n", secs)

	var vad *conformer.VAD
	if *vadPath != "" {
		vad, err = conformer.LoadVAD(*vadPath)
		check(err)
	}
	var diar *conformer.Diarizer
	if *diarPath != "" {
		diar, err = conformer.LoadDiarizer(*diarPath)
		check(err)
	}
	var m *conformer.Model
	if *model != "" {
		t0 := time.Now()
		m, err = conformer.Load(*model)
		check(err)
		fmt.Fprintf(os.Stderr, "loaded %s in %v\n", *model, time.Since(t0).Round(time.Millisecond))
		m, err = m.WithLanguage(*lang)
		check(err)
	}
	if *profile != "" {
		f, err := os.Create(*profile)
		check(err)
		check(pprof.StartCPUProfile(f))
		defer pprof.StopCPUProfile()
	}
	t0 := time.Now()
	report := func(what string) {
		el := time.Since(t0)
		fmt.Fprintf(os.Stderr, "%s: %v, %.1fx real time\n", what, el.Round(time.Millisecond), secs/el.Seconds())
	}
	switch {
	case m != nil && diar != nil:
		look := *stream
		if look < 0 {
			look = 13 // the model card's default lookahead
		}
		mt, err := conformer.NewMultiTalker(m, diar, look, *threads)
		check(err)
		toks := append(mt.Write(pcm), mt.Close()...)
		report("multitalker")
		text := mt.SpeakerText(toks)
		var spk []int
		for s := range text {
			spk = append(spk, s)
		}
		sort.Ints(spk)
		for _, s := range spk {
			fmt.Printf("speaker %d: %s\n", s+1, text[s])
		}
	case m != nil:
		var toks []conformer.Token
		if *stream >= 0 {
			s, err := m.NewStream(*stream, *threads)
			check(err)
			for i := 0; i < len(pcm); i += 1600 { // 100 ms at a time, as a microphone would
				toks = append(toks, s.Write(pcm[i:min(len(pcm), i+1600)])...)
			}
			toks = append(toks, s.Close()...)
			report(fmt.Sprintf("streaming (%.2f s lookahead)", s.Latency()))
		} else {
			toks, err = m.Transcribe(pcm, vad, conformer.DefaultSegments, *threads)
			check(err)
			report("one pass")
		}
		if *words {
			for _, w := range m.Words(toks) {
				fmt.Printf("%7.2f %7.2f  %s\n", w.Start, w.End, w.Text)
			}
		} else {
			fmt.Println(m.Text(idsOf(toks)))
		}
		if l := m.DetectedLanguage(toks); l != "" {
			fmt.Fprintf(os.Stderr, "language: %s\n", l)
		}
	case diar != nil:
		p := map[string]conformer.DiarParams{"very-high": conformer.DiarVeryHighLatency, "high": conformer.DiarHighLatency, "low": conformer.DiarLowLatency}[*preset]
		probs, err := diar.Diarize(pcm, p, *threads)
		check(err)
		report("speakers")
		printTurns(probs)
	default:
		probs, err := vad.ProbsLong(pcm, *threads)
		check(err)
		report("silence detector")
		for _, r := range vad.Speech(probs, conformer.DefaultSpeech) {
			fmt.Printf("speech %7.2f %7.2f\n", r.Start, r.End)
		}
	}
}

// printTurns prints when each speaker's chance is above one half.
func printTurns(probs [][]float32) {
	const frame = 0.08
	for s := 0; len(probs) > 0 && s < len(probs[0]); s++ {
		start := -1
		for t := 0; t <= len(probs); t++ {
			on := t < len(probs) && probs[t][s] > 0.5
			switch {
			case on && start < 0:
				start = t
			case !on && start >= 0:
				fmt.Printf("speaker %d %7.2f %7.2f\n", s+1, float64(start)*frame, float64(t)*frame)
				start = -1
			}
		}
	}
}

func idsOf(ts []conformer.Token) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.ID
	}
	return out
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
