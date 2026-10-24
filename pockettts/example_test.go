package pockettts_test

import (
	"log"

	"github.com/xattribution/gorelaxor/pockettts"
)

// Read text aloud, 80 ms of 24 kHz audio at a time as it is made.
func ExampleModel_Speak() {
	m, err := pockettts.Load("pocket-tts-english_2026-09-Q8_0.gguf")
	if err != nil {
		log.Fatal(err)
	}
	v, err := m.LoadVoice("voice-alba.gguf")
	if err != nil {
		log.Fatal(err)
	}
	err = m.Speak("The fog lifted by noon.", v, pockettts.Options{}, func(pcm []float32) error {
		// play or store pcm (1920 samples)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
