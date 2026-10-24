package conformer_test

import (
	"fmt"
	"log"

	"github.com/xattribution/gorelaxor/conformer"
	"github.com/xattribution/gorelaxor/whisper"
)

// A final pass with Parakeet, cut at the silence detector's pauses.
func ExampleModel_Transcribe() {
	pcm, err := whisper.DecodeFile("talk.wav") // 16 kHz mono
	if err != nil {
		log.Fatal(err)
	}
	m, err := conformer.Load("parakeet-tdt-0.6b-v2-Q8_0.gguf")
	if err != nil {
		log.Fatal(err)
	}
	vad, err := conformer.LoadVAD("frame_vad_multilingual_marblenet_v2.0-F32.gguf")
	if err != nil {
		log.Fatal(err)
	}
	toks, err := m.Transcribe(pcm, vad, conformer.DefaultSegments, 4)
	if err != nil {
		log.Fatal(err)
	}
	for _, w := range m.Words(toks) {
		fmt.Printf("%6.2f %s\n", w.Start, w.Text)
	}
}

// Live captions: feed audio as it arrives and print text as it comes.
func ExampleModel_NewStream() {
	m, err := conformer.Load("nemotron-speech-streaming-en-0.6b-Q8_0.gguf")
	if err != nil {
		log.Fatal(err)
	}
	s, err := m.NewStream(13, 2) // 1.12 s lookahead, 2 threads
	if err != nil {
		log.Fatal(err)
	}
	var mic chan []float32 // 16 kHz mono pieces from a microphone
	for pcm := range mic {
		fmt.Print(m.TokenText(s.Write(pcm)))
	}
	fmt.Println(m.TokenText(s.Close()))
}

// Who spoke when: four speakers' chances every 80 ms.
func ExampleDiarizer_Diarize() {
	pcm, _ := whisper.DecodeFile("meeting.wav")
	d, err := conformer.LoadDiarizer("diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf")
	if err != nil {
		log.Fatal(err)
	}
	probs, err := d.Diarize(pcm, conformer.DiarHighLatency, 4)
	if err != nil {
		log.Fatal(err)
	}
	for t, p := range probs {
		for spk, x := range p {
			if x > 0.5 {
				fmt.Printf("%.2f s: speaker %d\n", float64(t)*0.08, spk+1)
			}
		}
	}
}
