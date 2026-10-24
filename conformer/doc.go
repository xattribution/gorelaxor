// Package conformer turns speech into text and tells speakers apart on
// this computer with NVIDIA's FastConformer models, in pure Go: no
// Python, no cgo, no helper programs or DLLs. Audio and text never
// leave the machine.
//
// # Models
//
// Each model is a GGUF file made from NVIDIA's original .nemo download by
// tools/convert_nemo.py, which needs PyTorch but not NeMo:
//
//	role               NVIDIA original                                 Q8_0 file   API
//	live captions      nvidia/nemotron-speech-streaming-en-0.6b        732 MB      Load, Model.NewStream
//	32 languages       nvidia/nemotron-3.5-asr-streaming-0.6b          776 MB      Load, Model.WithLanguage
//	final pass         nvidia/parakeet-tdt-0.6b-v2                     732 MB      Load, Model.Transcribe
//	final, 25 languages nvidia/parakeet-tdt-0.6b-v3                    755 MB      Load, Model.Transcribe
//	speakers           nvidia/diar_streaming_sortformer_4spk-v2.1      140 MB      LoadDiarizer
//	crosstalk          nvidia/multitalker-parakeet-streaming-0.6b-v1   736 MB      NewMultiTalker
//	silence detection  nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0    0.4 MB      LoadVAD
//
// A file is read into memory whole, each tensor in memory of its own, so
// a model costs about its file size plus working space. On processors
// with VNNI the encoder's Q8_0 weights are repacked at load for the int8
// kernels and their rows let go, so that stays true (Parakeet and Nemotron
// hold about 750 MB each). With a Go memory limit of 900 MiB
// (runtime/debug.SetMemoryLimit), Nemotron streamed a 5-minute talk in at
// most 0.9 GB. Without a limit Go lets the heap grow to about twice that
// before collecting.
//
// Weights may be float32, float16 or Q8_0; Q8_0 keeps every NeMo word on
// the test recordings at under a third of float32's size.
//
// # How it works
//
// Audio is 16 kHz mono. The front end is NeMo's log-mel spectrogram
// (pre-emphasis, a centered 512-point FFT with a 400-sample Hann window,
// Slaney mel bands, log(x + 2^-24)). The encoder is NeMo's
// ConformerEncoder: 8x depthwise-striding subsampling to 80 ms frames,
// then blocks of feed-forward, relative-position self-attention,
// convolution and feed-forward. Transducers decode greedily: RNN-T for
// Nemotron and multitalker, TDT with its duration head for Parakeet.
// Nemotron 3.5 is Nemotron with a language prompt: each encoder frame,
// joined with a one-hot language vector, goes through a small
// Linear-ReLU-Linear network before the decoder (NeMo
// PromptStreamingMixin). Model.WithLanguage picks the locale, "auto" by
// default, and the model then writes a language tag such as <de-DE>
// after each sentence; Text and Words leave tags out and
// Model.DetectedLanguage reads them.
//
//   - Model.Transcribe is the final pass. Recordings longer than 30 s are
//     cut in the silence detector's pauses (Segments) and each piece is
//     encoded with full attention.
//   - Model.NewStream is cache-aware streaming, as NeMo's
//     CacheAwareStreamingAudioBuffer and cache_aware_stream_step do it:
//     chunks of R+1 encoder frames with R frames of lookahead (Nemotron:
//     13, 6, 1 or 0, so 1.12 s down to 0.08 s; Nemotron 3.5: 13, 6, 3
//     or 0), each layer keeping its last 70 attention inputs (56 for
//     Nemotron 3.5) and its convolution's last kernel-1 frames (8).
//   - Diarizer runs Sortformer's streaming loop with its speaker cache
//     (NeMo forward_streaming, streaming_update and the cache compression),
//     giving each of four speakers' chance of talking every 80 ms.
//     DiarHighLatency (10 s steps) keeps up live on a CPU.
//   - MultiTalker is NVIDIA's multitalker pipeline (SpeakerTaggedASR,
//     parallel streaming, the model card's defaults): Sortformer runs on
//     each ASR chunk and every active speaker gets its own stream, fed the
//     chunk with that speaker's silent frames masked.
//   - VAD is MarbleNet: speech chances every 20 ms, turned into regions
//     with NeMo's onset, offset, padding and minimum-duration rules.
//
// On amd64 processors with AVX2, FMA and F16C the matrix and dot products
// run in assembly (internal/kern), including Q8_0 rows, and on int8
// kernels where the processor has VNNI.
//
// # Outside input
//
// Model files and recordings come from outside, so:
//
//   - Load, LoadDiarizer and LoadVAD run a second of silence through the
//     model before returning it, so a file whose tensors don't fit its
//     settings fails to load with an error; internal/gguf refuses sizes
//     and offsets that don't fit the file.
//   - A panic inside a kernel worker is carried back to the calling
//     goroutine, where the caller's recover sees it.
//   - NaN and infinite samples are heard as silence, and a Write of a
//     long recording goes through in 10 s pieces.
//
// A loaded Model, Diarizer or VAD can be shared by goroutines; a Stream,
// DiarStream or MultiTalker belongs to one goroutine at a time. The calls
// run on the caller's goroutine and take no context: a caller that needs
// to stop a long final pass feeds it in pieces.
//
// # Checked against NeMo
//
// tools/nemo_reference.py runs NVIDIA NeMo in float32 and saves its
// outputs; the tests in this package compare against them when
// GORELAXOR_CONFORMER_REF names the folder (ref_test.go lists the files):
//
//	model, mode                                   largest difference from NeMo
//	Nemotron, one pass (JFK, 11 s)                encoder 1.2e-5 relative; same words
//	Nemotron, streaming R = 13, 6, 1, 0           encoder 2e-5 or less; same words at each R
//	Nemotron 3.5, one pass (7 languages, 2 modes) encoder 5.3e-6, prompt 5.1e-6; same tokens
//	Nemotron 3.5, streaming R = 13 and 3          encoder 3.1e-5 or less; same words and language
//	Parakeet TDT v2 (JFK)                         encoder 2.3e-6; same words and token frames
//	Parakeet TDT v3 (English, German, Spanish, French) encoder 3e-6; same words
//	MarbleNet (JFK in noise)                      speech chance 1.1e-6
//	Sortformer, very high and low latency (110 s) speaker chance 1e-6
//	Multitalker + Sortformer (2 speakers, 14 s)   speaker chances 6.6e-7; same words per speaker
//
// With Q8_0 files the encoders differ from NeMo's float32 by 1 to 4% RMS
// and the words stay the same. On processors with VNNI the activations
// are rounded to Q8_0 too (internal/kern), which accounts for the upper
// end of that range; 84 s and 306 s talks still match the float32 model
// word for word. Nemotron 3.5's differ by 0.5 to 3.3% and its words stay
// the same except one Korean word, which the Q8_0 file writes as two
// ("성인 수컷과" for NeMo's "성인수꺼과").
//
// # Speed
//
// Measured on 2 cores of a 2.8 GHz Xeon, Q8_0 files, 2 threads. The VNNI
// column is what Intel 11th generation and newer and AMD Zen 4 and newer
// run (the products round the activations to 8 bits as well and run on
// int8 kernels, internal/kern q8int.go); older processors run the AVX2
// column.
//
//	                                              AVX2     VNNI   (x real time)
//	Parakeet final pass, 84 s talk                3.7      8.1
//	Nemotron streaming, R = 13                    1.9      4.7
//	Nemotron streaming, R = 6                     1.2      3.3
//	Nemotron streaming, R = 0                     -        0.5   too slow live: every 80 ms rereads the model
//	Sortformer, DiarVeryHighLatency               8.4     14.1
//	Sortformer, DiarHighLatency                   5.1      8.6
//	Sortformer, DiarLowLatency                    -        0.6   too slow live on a CPU
//	MultiTalker, 2 speakers talking               0.8      2.1   (one ASR stream per speaker)
//	MarbleNet                                     over 200x
//
// On a 16-core Ryzen 9 9950X3D (Zen 5, VNNI) with a 345-second recording,
// Parakeet ran at 31.6x real time on 2 threads and 67.2x on 8, Nemotron
// streaming at R = 13 at 19.8x on 2, and Sortformer (very high latency)
// at 38.8x on 2. On LibriSpeech's first 8.7 minutes Parakeet made 2.35%
// word errors on the float path and 2.26% on int8.
//
// Loading a Q8_0 model takes under 2 s; with VNNI that includes repacking
// its weights for the int8 kernels.
package conformer
