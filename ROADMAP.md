# Roadmap

The aim is for small AI models to run natively in any Go program, on any device, with no Python or native libraries to install. Speech is where it started. The kernels and file readers are general, and the next steps make them faster on more hardware and open them to more kinds of models.

## 1. Fast kernels on ARM

ARM is the gap that matters most. Apple silicon, Android phones, Windows on ARM and ARM servers all build and run today, but on the portable Go kernels, which are about 30 times slower than the x86 int8 path.

- **NEON assembly** for the float32 6×16 micro-kernel, the Q8_0 dequantizing product and the vector operations (exp, GELU, softmax, layer norm).
- **The dot-product instructions** (`SDOT` and `UDOT`, on Apple M1 and newer and most phones since 2019) for the int8 path. These are the ARM counterpart of VNNI, and they should bring ARM close to the x86 numbers.
- **Benchmarks on real ARM hardware** (Apple silicon, a recent phone, an ARM server) with `cmd/bench`, before and after the NEON kernels. CI already runs the tests on Apple silicon and Linux arm64.

## 2. One kernel package

- Move `whisper/` onto `internal/kern`. Whisper would then keep Q8_0 weights in memory (about half the size of today's float16) and use the int8 path, which is about five times the float path.
- Make the kernels and the GGUF reader a public, documented API (`tensor` and `gguf`), so other projects can build on them. Before then, the API settles against at least one non-speech model.

## 3. More models

Models are picked when their job is common in apps, they are small enough for a CPU, and their weights are openly licensed.

- **Text encoders:** BERT-family and ModernBERT (Ettin) classifiers, rerankers and sentence embeddings. They are small, run in one pass, and are useful everywhere: search, sorting, scoring, deduplication.
- **Whisper in GGUF,** with Q8_0 and its int8 path, and **distil-whisper.**
- **More TTS:** Pocket TTS's other languages (French, German, Portuguese, Italian, Spanish), and Kokoro.
- **Silero VAD** as a second silence detector.
- **Audio decoding everywhere:** pure-Go FLAC, MP3 and Opus readers, so non-Windows platforms read more than WAV.

## 4. GPUs, still without cgo

- **Apple Metal and Vulkan** compute through dynamic loading (as `ebitengine/purego` does), so a GPU is optional at run time and the build stays pure Go.
- **WebGPU** for the WebAssembly build, so models run in the browser on the GPU.
- The CPU path stays the reference: GPU results are tested against it the same way the CPU path is tested against PyTorch.

## 5. Platforms and packaging

- **Mobile:** `gomobile bind` wrappers for iOS and Android apps, with microphone capture examples.
- **WebAssembly:** a browser demo (live captions from the microphone), with WASM SIMD kernels.
- **Model fetching:** an optional helper that downloads and converts a model once and checks its SHA-256.
