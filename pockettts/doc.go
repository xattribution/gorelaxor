// Package pockettts reads text aloud on this computer with Kyutai's
// Pocket TTS (100 million parameters, English), in pure Go: no Python,
// no cgo, no helper programs or DLLs. Text and audio never leave the
// machine.
//
// # Files
//
// tools/convert_pocket_tts.py makes the files from Kyutai's original
// download (kyutai/pocket-tts, languages/english_2026-09, the build with
// voice cloning) and needs only Python and numpy:
//
//	file                                       size     what
//	pocket-tts-english_2026-09-Q8_0.gguf       119 MB   the model: tokenizer, flow LM, Mimi decoder and encoder
//	pocket-tts-english_2026-09-F32.gguf        438 MB   the same in float32 (Kyutai's bfloat16 weights, exactly)
//	voice-<name>.gguf                          2-4 MB   one of Kyutai's 21 preset English voices, float16
//
// A voice is the flow LM's state after hearing a few seconds of the
// speaker (each layer's keys and values), so voices are separate files an
// app picks from rather than part of the model. A voice file names the
// model it was made for, and LoadVoice refuses it with another. In
// float16 a voice changes the latents by 5e-5 RMS, about a hundredth of what
// Q8_0 weights do. CloneVoice makes a voice from a clip instead.
//
// # API
//
//	m, err := pockettts.Load("pocket-tts-english_2026-09-Q8_0.gguf")
//	v, err := m.LoadVoice("voice-alba.gguf")         // or m.CloneVoice(pcm, rate, threads)
//	err = m.Speak(text, v, pockettts.Options{}, func(pcm []float32) error {
//		// 1920 samples (80 ms at 24 kHz), as soon as each is made
//		return nil
//	})
//
// Synthesize returns the whole recording instead. Options.Seed makes
// output repeatable (the same samples for the same text, voice, options
// and seed, whatever the thread count); the noise comes from Go's PCG
// generator, so it is not PyTorch's random stream. The zero Options take
// the model's settings: temperature 0.3, one LSD step, end of speech at a
// logit above −4, Kyutai's guess of frames after it.
//
// # How it works
//
// It is Kyutai's pocket_tts package step for step:
//
//   - Text is tidied (spaces, a capital first letter, a final period) and
//     split at sentence ends, and long sentences at commas, into chunks of
//     at most 50 tokens (text_chunking.py). Each chunk is spoken from the
//     voice's state on its own.
//   - The tokenizer is tokenizer.json's SentencePiece unigram model as
//     Hugging Face tokenizers runs it: the Prepend and Metaspace steps,
//     the best-scoring path, unknown characters as UTF-8 bytes.
//   - The flow LM is a 6-layer, 1024-wide causal transformer with rotary
//     positions. It reads the voice prompt, the chunk's token embeddings,
//     then one latent per 80 ms frame; its output conditions a 6-block
//     MLP head (adaLN, two time conditions) that turns Gaussian noise into
//     the next 32-value latent in one LSD step, and an end-of-speech
//     logit. Generation stops a few frames after that logit crosses −4,
//     or at 3 tokens a second plus 2 s.
//   - Mimi decodes each latent: a projection to 512 channels, a
//     transposed convolution up to 200 Hz, a 2-layer transformer seeing
//     250 steps (1.25 s) back, then SEANet's transposed convolutions and
//     residual blocks up to 24 kHz. Everything is causal and keeps its
//     state between frames, so audio comes out one frame at a time.
//   - The flow LM runs on its own goroutine while the caller's goroutine
//     decodes, as Kyutai runs its decoder on a thread of its own.
//   - CloneVoice resamples the clip to 24 kHz (scipy's resample_poly,
//     ported), cuts trailing silence to 80 ms (end_on_pause), runs Mimi's
//     encoder down to 12.5 Hz and projects the latents into the flow LM,
//     after a learned start position.
//
// # Writing for speech
//
// Like Kyutai's, the engine reads text as written: digits one by one
// ("25" is "two five") and abbreviations with periods broken at each
// period ("a.m." becomes "a. m. ,"). Text should be rewritten for the
// ear before Speak, by the model when it writes a script to be read
// aloud and by code for anything read as it stands (documents, web pages):
//
//	write                                not
//	twenty-five                          25
//	three point five percent             3.5%
//	twenty twenty-six                    2026
//	October sixteenth                    Oct. 16
//	seven forty-one p m                  7:41 p.m.
//	Doctor Lee, Professor Kim            Dr. Lee, Prof. Kim
//	five kilometers                      5 km
//	R two D two (digit by digit)         R twenty-two D two, for R2D2
//	and, or, per                         &, /
//	the link text alone                  a URL
//	plain sentences                      Markdown, bullets, tables, emoji
//
// Also:
//
//   - End every sentence with a period, question mark or exclamation
//     point; commas are the short pauses.
//   - Keep sentences short: a chunk is at most 50 tokens, and a long
//     sentence is cut where the splitter finds room, not where a
//     speaker would breathe.
//   - Write acronyms the way they are said: NASA and UNESCO as words if
//     people say them as words, letters spaced out (H T M L) if they say
//     the letters. Check each new one by ear.
//
// On amd64 processors with AVX2 the products run in internal/kern's
// assembly, and on its int8 kernels where the processor has VNNI. Load
// packs the transformers' and convolutions' Q8_0 weights for those
// kernels and lets their rows go; the flow LM's one row per frame runs on
// them too, at about the memory's read speed.
//
// # Outside input
//
// Model, voice files and clips come from outside, so: internal/gguf
// refuses sizes and offsets that don't fit the file; Load checks every
// tensor's shape against the settings; LoadVoice refuses voices for
// another model and values that aren't numbers; CloneVoice hears silence
// for NaN and infinite samples and at most MaxCloneSeconds. A panic in
// the flow LM's goroutine comes back as Speak's error.
//
// A Model and a Voice can be shared by goroutines; each Speak call has
// its own state. An emit function that returns an error stops Speak
// (ErrStop to stop without one).
//
// # Checked against Kyutai
//
// tools/pocket_tts_reference.py runs Kyutai's PyTorch code in float32 and
// saves its token ids, every flow head call (transformer output, noise,
// latent, end-of-speech logit) and the audio. The tests in this package
// compare against them when GORELAXOR_POCKET_REF names the folder (ref_test.go
// lists the files), feeding Kyutai's noise back. Tokens and chunks match
// exactly on 20 awkward strings and 4 chunking texts. With teacher
// forcing (each frame fed Kyutai's latent) every step is compared on its
// own; free running feeds Go's latents back, so differences grow along
// the way. Largest difference relative to the reference's largest value,
// float32 file and voices:
//
//	case                                         teacher forced: transformer, latent, audio   free running: audio
//	alba, 6 words                                9.2e-7, 5.7e-7, 2.3e-6                       5.9e-4
//	marius, 32 words in two chunks               1.1e-6, 7.8e-7, 2.9e-6                       1.1e-4
//	javert, numbers and abbreviations            1.1e-6, 1.6e-6, 7.1e-6                       2.1e-3
//	casual.wav cloned (48 kHz, 10 s)             1.1e-6, 2.7e-6, 1.6e-6                       9.9e-3
//
// Cloning itself: the resampled clip 3.5e-7, Mimi's encoder 2.2e-6, the
// conditioning 1.6e-6. Every case makes the same number of frames as
// Kyutai's.
//
// With the Q8_0 file and float16 voices, teacher-forced latents differ
// from Kyutai's float32 by 0.7 to 2.3% RMS and the audio by 1.9 to 3.8%.
// Free running, sampling then takes other paths, so the quality gate is
// Parakeet (conformer/): it hears the same words in the Go Q8_0 audio as
// in Kyutai's float32 audio, and the text itself where the text has no
// digits.
//
// # Speed
//
// Measured on 2 cores of a 2.8 GHz Xeon, 2 threads, a 32-word passage
// (about 10.5 s of audio). VNNI is what Intel 11th generation and newer
// and AMD Zen 4 and newer have; older processors run the AVX2 column.
//
//	                         float32   Q8_0 AVX2   Q8_0 VNNI   Kyutai's PyTorch, float32
//	times real time          2.1       3.1         3.9         1.5
//	first audio after        0.31 s    0.28 s      0.06 s      0.20 s
//	cloning a 10 s clip                1.6 s       1.1 s
//
// The Q8_0 model loads in about 0.2 s and holds about 280 MB while it
// speaks.
package pockettts
