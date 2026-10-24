// Command bench runs every engine on one machine and reports load time,
// speed, memory and accuracy, so results from different computers can be
// compared:
//
//	bench -models ~/models -audio talk.wav -threads 2,8 -json results.json
//	bench -models ~/models -audio talk.wav -engines tts,whisper -whisper small.en
//
// It looks in -models for the files tools/ makes and whisper.cpp
// publishes, and runs what it finds:
//
//	ggml-*.bin                                  Whisper, one pass
//	parakeet-tdt-*.gguf                         Parakeet, final pass (with the VAD if present)
//	nemotron-speech-streaming-*.gguf            Nemotron, streaming at R = 13 and 6
//	nemotron-3.5-asr-streaming-*.gguf           Nemotron 3.5, streaming at R = 13
//	diar_streaming_sortformer_*.gguf            Sortformer, high and very high latency
//	multitalker-parakeet-*.gguf                 multitalker (needs Sortformer too)
//	frame_vad_multilingual_marblenet_*.gguf     MarbleNet silence detection
//	pocket-tts-*.gguf, voice-alba.gguf          Pocket TTS: speed, first audio, voice cloning
//
// Accuracy needs no labeled data: Pocket TTS reads a fixed passage aloud,
// and every speech-to-text model transcribes that recording. The word
// error rate against the passage is reported for each. -ref adds a word
// error rate on -audio against a transcript you supply.
//
// Speed is x real time: seconds of audio handled per second of work.
// Memory is the Go heap in use with only that model loaded, after it runs
// and a collection (runtime.MemStats.HeapInuse): the weights plus what the
// model keeps between calls, not the run's peak.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/xattribution/gorelaxor/conformer"
	"github.com/xattribution/gorelaxor/pockettts"
	"github.com/xattribution/gorelaxor/whisper"
)

// passage is what Pocket TTS reads for the round trip: no digits or
// abbreviations, so every model has one right answer.
const passage = "The lighthouse keeper climbed the narrow stairs before sunrise. " +
	"He trimmed the wick, polished the great lens, and watched the fishing boats leave the harbor. " +
	"By noon the fog had lifted, and the whole coast lay bright and quiet under a clear blue sky. " +
	"In the evening he wrote a short letter to his sister, folded it twice, and set it by the window."

type result struct {
	Engine   string   `json:"engine"`
	File     string   `json:"file"`
	Mode     string   `json:"mode"`
	Threads  int      `json:"threads"`
	LoadSec  float64  `json:"load_s"`
	AudioSec float64  `json:"audio_s"`
	Speed    float64  `json:"x_real_time"`
	FirstSec float64  `json:"first_audio_s,omitempty"`
	MemMB    int      `json:"mem_mb"`
	WER      *float64 `json:"round_trip_wer,omitempty"`
	RefWER   float64  `json:"ref_wer,omitempty"`
	Text     string   `json:"text,omitempty"`
}

type env struct {
	dir      string
	audio    []float32 // -audio, 16 kHz
	refText  string
	trip     []float32 // Pocket TTS's reading of passage, 16 kHz
	vad      *conformer.VAD
	threads  []int
	results  []result
	whispers map[string]bool
}

func main() {
	dir := flag.String("models", ".", "folder with the model files")
	audioPath := flag.String("audio", "", "recording to time on (16 kHz or any WAV/FLAC/MP3 the decoder reads)")
	refPath := flag.String("ref", "", "transcript of -audio, for its word error rate")
	threadList := flag.String("threads", "2,8", "comma-separated thread counts")
	jsonOut := flag.String("json", "", "also write the results here")
	engines := flag.String("engines", "tts,vad,parakeet,nemotron,speakers,whisper", "comma-separated engines to run")
	only := flag.String("whisper", "", "comma-separated Whisper names to run (base.en, small.en, large-v3-turbo...); empty runs all")
	flag.Parse()

	e := &env{dir: *dir, whispers: map[string]bool{}}
	for _, s := range strings.Split(*threadList, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		check(err)
		e.threads = append(e.threads, n)
	}
	for _, w := range strings.Split(*only, ",") {
		if w = strings.TrimSpace(w); w != "" {
			e.whispers[w] = true
		}
	}
	if *audioPath != "" {
		pcm, err := whisper.DecodeFile(*audioPath)
		check(err)
		e.audio = pcm
	}
	if *refPath != "" {
		b, err := os.ReadFile(*refPath)
		check(err)
		e.refText = string(b)
	}
	fmt.Printf("%s/%s, %d logical CPUs, Go %s\n\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version())

	on := map[string]bool{}
	for _, s := range strings.Split(*engines, ",") {
		on[strings.TrimSpace(s)] = true
	}
	if on["tts"] {
		e.tts()
	}
	if on["vad"] || on["parakeet"] {
		e.vadRun()
	}
	if on["parakeet"] {
		for _, p := range e.glob("parakeet-tdt-*.gguf") {
			e.asr(p, "final pass", -1)
		}
	}
	if on["nemotron"] {
		for _, p := range e.glob("nemotron-speech-streaming-*.gguf") {
			e.asr(p, "streaming R=13", 13)
			e.asr(p, "streaming R=6", 6)
		}
		for _, p := range e.glob("nemotron-3.5-asr-streaming-*.gguf") {
			e.asr(p, "streaming R=13", 13)
		}
	}
	if on["speakers"] {
		e.diar()
	}
	if on["whisper"] {
		for _, p := range e.glob("ggml-*.bin") {
			name := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), "ggml-"), ".bin")
			if len(e.whispers) == 0 || e.whispers[name] {
				e.whisperRun(p)
			}
		}
	}

	if *jsonOut != "" {
		b, _ := json.MarshalIndent(e.results, "", " ")
		check(os.WriteFile(*jsonOut, b, 0o644))
	}
}

func (e *env) glob(pat string) []string {
	m, _ := filepath.Glob(filepath.Join(e.dir, pat))
	sort.Strings(m)
	return m
}

// free drops what the last model held, so each memory figure is one model's.
func free() {
	runtime.GC()
	debug.FreeOSMemory()
}

func memMB() int {
	runtime.GC() // count what the model keeps, not garbage from the run
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return int(ms.HeapInuse >> 20)
}

func (e *env) add(r result) {
	e.results = append(e.results, r)
	extra := ""
	if r.FirstSec > 0 {
		extra = fmt.Sprintf("  first audio %.2f s", r.FirstSec)
	}
	if r.RefWER > 0 {
		extra += fmt.Sprintf("  ref WER %.1f%%", 100*r.RefWER)
	}
	trip := "    -"
	if r.WER != nil {
		trip = fmt.Sprintf("%4.1f%%", 100**r.WER)
	}
	fmt.Printf("%-10s %-48s %-28s t=%-2d load %5.2f s %7.1fx %5d MB  round-trip WER %s%s\n",
		r.Engine, r.File, r.Mode, r.Threads, r.LoadSec, r.Speed, r.MemMB, trip, extra)
}

func (e *env) tts() {
	ms := e.glob("pocket-tts-*.gguf")
	if len(ms) == 0 {
		fmt.Println("no Pocket TTS model: the round trip uses -audio only, without a word error rate")
		return
	}
	free()
	t0 := time.Now()
	m, err := pockettts.Load(ms[0])
	check(err)
	load := time.Since(t0).Seconds()
	v, err := m.LoadVoice(filepath.Join(e.dir, "voice-alba.gguf"))
	check(err)
	for _, th := range e.threads {
		var pcm []float32
		var first time.Duration
		t0 := time.Now()
		check(m.Speak(passage, v, pockettts.Options{Seed: 1, Threads: th}, func(p []float32) error {
			if pcm == nil {
				first = time.Since(t0)
			}
			pcm = append(pcm, p...)
			return nil
		}))
		el := time.Since(t0).Seconds()
		secs := float64(len(pcm)) / float64(m.SampleRate())
		e.add(result{Engine: "pockettts", File: filepath.Base(ms[0]), Mode: "speak (alba)", Threads: th, LoadSec: load,
			AudioSec: secs, Speed: secs / el, FirstSec: first.Seconds(), MemMB: memMB()})
		if e.trip == nil {
			e.trip = whisper.Resample(pcm, m.SampleRate(), 16000)
		}
	}
	if m.CanClone() && e.audio != nil {
		clip := e.audio[:min(len(e.audio), 10*16000)]
		t0 := time.Now()
		_, err := m.CloneVoice(clip, 16000, e.threads[len(e.threads)-1])
		check(err)
		el := time.Since(t0).Seconds()
		e.add(result{Engine: "pockettts", File: filepath.Base(ms[0]), Mode: "clone 10 s voice", Threads: e.threads[len(e.threads)-1],
			LoadSec: load, AudioSec: float64(len(clip)) / 16000, Speed: float64(len(clip)) / 16000 / el, MemMB: memMB()})
	}
}

func (e *env) vadRun() {
	ps := e.glob("frame_vad_multilingual_marblenet_*.gguf")
	if len(ps) == 0 || e.audio == nil {
		return
	}
	t0 := time.Now()
	v, err := conformer.LoadVAD(ps[0])
	check(err)
	load := time.Since(t0).Seconds()
	e.vad = v
	for _, th := range e.threads {
		t0 := time.Now()
		_, err := v.ProbsLong(e.audio, th)
		check(err)
		secs := float64(len(e.audio)) / 16000
		e.add(result{Engine: "conformer", File: filepath.Base(ps[0]), Mode: "silence detection", Threads: th, LoadSec: load,
			AudioSec: secs, Speed: secs / time.Since(t0).Seconds(), MemMB: memMB()})
	}
}

func idsOf(ts []conformer.Token) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.ID
	}
	return out
}

func (e *env) asr(path, mode string, right int) {
	free()
	t0 := time.Now()
	m, err := conformer.Load(path)
	check(err)
	load := time.Since(t0).Seconds()
	if len(m.Languages()) > 0 {
		m, err = m.WithLanguage("en-US")
		check(err)
	}
	run := func(pcm []float32, th int) (string, float64) {
		t0 := time.Now()
		var toks []conformer.Token
		if right >= 0 {
			s, err := m.NewStream(right, th)
			check(err)
			for i := 0; i < len(pcm); i += 1600 { // 100 ms at a time, as a microphone would
				toks = append(toks, s.Write(pcm[i:min(len(pcm), i+1600)])...)
			}
			toks = append(toks, s.Close()...)
		} else {
			toks, err = m.Transcribe(pcm, e.vad, conformer.DefaultSegments, th)
			check(err)
		}
		return m.Text(idsOf(toks)), time.Since(t0).Seconds()
	}
	tripWER, tripText := e.tripWER(func(p []float32) string { t, _ := run(p, e.threads[len(e.threads)-1]); return t })
	for _, th := range e.threads {
		r := result{Engine: "conformer", File: filepath.Base(path), Mode: mode, Threads: th, LoadSec: load, WER: tripWER, Text: tripText}
		if e.audio != nil {
			text, el := run(e.audio, th)
			r.AudioSec = float64(len(e.audio)) / 16000
			r.Speed = r.AudioSec / el
			if e.refText != "" {
				r.RefWER = wer(e.refText, text)
			}
		}
		r.MemMB = memMB()
		e.add(r)
	}
}

func (e *env) diar() {
	ds := e.glob("diar_streaming_sortformer_*.gguf")
	if len(ds) == 0 || e.audio == nil {
		return
	}
	free()
	t0 := time.Now()
	d, err := conformer.LoadDiarizer(ds[0])
	check(err)
	load := time.Since(t0).Seconds()
	secs := float64(len(e.audio)) / 16000
	for _, p := range []struct {
		name string
		p    conformer.DiarParams
	}{{"speakers, very high latency", conformer.DiarVeryHighLatency}, {"speakers, high latency", conformer.DiarHighLatency}} {
		for _, th := range e.threads {
			t0 := time.Now()
			_, err := d.Diarize(e.audio, p.p, th)
			check(err)
			e.add(result{Engine: "conformer", File: filepath.Base(ds[0]), Mode: p.name, Threads: th, LoadSec: load,
				AudioSec: secs, Speed: secs / time.Since(t0).Seconds(), MemMB: memMB()})
		}
	}
	for _, mp := range e.glob("multitalker-parakeet-*.gguf") {
		t0 := time.Now()
		m, err := conformer.Load(mp)
		check(err)
		mload := time.Since(t0).Seconds()
		for _, th := range e.threads {
			mt, err := conformer.NewMultiTalker(m, d, 13, th)
			check(err)
			t0 := time.Now()
			mt.Write(e.audio)
			mt.Close()
			e.add(result{Engine: "conformer", File: filepath.Base(mp), Mode: "multitalker R=13", Threads: th, LoadSec: mload,
				AudioSec: secs, Speed: secs / time.Since(t0).Seconds(), MemMB: memMB()})
		}
	}
}

func (e *env) whisperRun(path string) {
	free()
	t0 := time.Now()
	m, err := whisper.Load(path)
	check(err)
	load := time.Since(t0).Seconds()
	defer m.Close()
	lang := ""
	if !strings.Contains(filepath.Base(path), ".en") {
		lang = "en"
	}
	run := func(pcm []float32, th int) (string, float64) {
		t0 := time.Now()
		segs, err := m.Transcribe(context.Background(), pcm, whisper.Options{Threads: th, Language: lang})
		check(err)
		var b strings.Builder
		for _, s := range segs {
			b.WriteString(s.Text)
			b.WriteByte(' ')
		}
		return b.String(), time.Since(t0).Seconds()
	}
	tripWER, tripText := e.tripWER(func(p []float32) string { t, _ := run(p, e.threads[len(e.threads)-1]); return t })
	for _, th := range e.threads {
		r := result{Engine: "whisper", File: filepath.Base(path), Mode: "one pass", Threads: th, LoadSec: load, WER: tripWER, Text: tripText}
		if e.audio != nil {
			text, el := run(e.audio, th)
			r.AudioSec = float64(len(e.audio)) / 16000
			r.Speed = r.AudioSec / el
			if e.refText != "" {
				r.RefWER = wer(e.refText, text)
			}
		}
		r.MemMB = memMB()
		e.add(r)
	}
}

func (e *env) tripWER(transcribe func([]float32) string) (*float64, string) {
	if e.trip == nil {
		return nil, ""
	}
	text := transcribe(e.trip)
	w := wer(passage, text)
	return &w, strings.TrimSpace(text)
}

// words lowercases and drops punctuation, so "Harbor," and "harbor" match.
func words(s string) []string {
	f := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
	return f
}

// wer is the word-level edit distance over the reference's length.
func wer(ref, hyp string) float64 {
	a, b := words(ref), words(hyp)
	if len(a) == 0 {
		return 0
	}
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			c := prev[j-1]
			if a[i-1] != b[j-1] {
				c++
			}
			cur[j] = min(c, prev[j]+1, cur[j-1]+1)
		}
		prev, cur = cur, prev
	}
	return float64(prev[len(b)]) / float64(len(a))
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
