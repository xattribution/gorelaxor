package whisper_test

import (
	"context"
	"fmt"
	"log"

	"github.com/xattribution/gorelaxor/whisper"
)

// Transcribe a file with a whisper.cpp model.
func ExampleModel_Transcribe() {
	pcm, err := whisper.DecodeFile("talk.wav")
	if err != nil {
		log.Fatal(err)
	}
	m, err := whisper.Load("ggml-base.en.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	segs, err := m.Transcribe(context.Background(), pcm, whisper.Options{})
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range segs {
		fmt.Printf("[%v] %s\n", s.Start, s.Text)
	}
}
