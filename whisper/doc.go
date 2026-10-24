// Package whisper turns speech into text on this computer, with OpenAI's
// Whisper models and nothing else: no Python, no cgo, no helper programs.
// WAV is read in Go; on Windows, DecodeFile reads other formats through
// the system's own Media Foundation. Audio and text never leave the
// machine.
//
// # Models
//
// Load reads a model file in whisper.cpp's ggml format (ggml-base.en.bin and
// the like, from huggingface.co/ggerganov/whisper.cpp). Weights may be
// float32, float16 or quantized (q4_0, q4_1, q5_0, q5_1, q8_0). Every
// weight matrix is held as float16 in memory, so a quantized file is smaller
// on disk but costs the same memory as the float16 one. The ".en" models are
// English only and a little more accurate in English; the others detect the
// language from the first window with speech, or take Options.Language.
//
// # How it works
//
// Audio is 16 kHz mono (DecodeWAV and DecodeFile convert). It's cut into
// 30-second windows: each becomes an 80-band log-mel spectrogram (128 for large-v3), the
// encoder turns that into 1,500 audio positions, and the decoder writes
// text with timestamps, greedily, with whisper.cpp's rules for timestamp
// pairs, suppressed tokens, a fallback to sampling when the text loops or
// comes out unlikely, and a no-speech check. The next window starts at the
// last timestamp, and earlier text is passed along as context. An energy
// gate skips silence and steady noise before any of that runs, and drops
// segments that land on silence, so an empty room doesn't turn into "Thank
// you." Stream decodes each window as soon as the audio runs past it and
// gives exactly the text Transcribe would.
//
// On amd64 processors with AVX2, FMA and F16C (Intel since 2013, AMD Zen)
// the matrix products, dot products, exp and GELU run in assembly; elsewhere
// in portable Go, several times slower.
//
// # Speed and memory
//
// Measured on a 2-core 2.8 GHz Xeon (AVX2), LibriSpeech read speech, Options
// with 2 threads. The default is every core but one; on a typical 4- to
// 8-core laptop expect these speeds or better with the default. Word error
// rates count "Mr." against "MISTER" and the like as errors.
//
//	model         file     memory   speed (x real time)   word errors
//	tiny.en        78 MB   220 MB   17.9                   9.9% (2 min)
//	base.en       148 MB   344 MB    9.6                   9.1% (2 min), 7.9% (8.7 min)
//	base.en q5_1   60 MB   332 MB    9.1                   9.9% (2 min)
//	small.en      488 MB   810 MB    2.8                   8.0% (2 min)
//
// Memory is the process's peak while transcribing. Loading takes 0.2 s to
// 0.7 s from a warm disk cache. Model.Close drops the weights and returns
// the memory to the system.
//
// whisper.cpp itself (AVX2 build, same machine, greedy) took 14.5 s for the
// two minutes with base.en, where this package takes 12.7 s with 2 threads,
// and scored 10.6% and 27.7% for tiny.en and base.en (it looped on one
// passage), against 9.9% and 9.1% here.
package whisper
