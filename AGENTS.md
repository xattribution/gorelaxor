# gorelaxor

Pure-Go inference for speech models: `conformer/` (NVIDIA FastConformer family), `whisper/`, `pockettts/`, shared kernels in `internal/kern` and the GGUF reader in `internal/gguf`. No cgo, ever.

## Every change

1. `gofmt -l .` prints nothing; `go vet ./...` and `GOOS=windows go vet ./...` pass.
2. `go test ./...` passes, and `go test -tags purego ./internal/kern` when kernels change.
3. A change to a model's numerics passes its reference tests (`GORELAXOR_CONFORMER_REF`, `GORELAXOR_POCKET_REF`, `GORELAXOR_WHISPER_MODELS`; see docs/ARCHITECTURE.md, "How it's checked").
4. A speed change gives before and after numbers from `cmd/bench` or the kernel benchmarks, and updates docs/BENCHMARKS.md.
5. Every target in the CI cross-build still builds.

## Rules

- Model files are untrusted input: check sizes and offsets, fail at load rather than mid-run, and carry worker panics back to the caller.
- No weights, recordings or reference dumps in the repository (`.gitignore`).
- New models come with a converter in `tools/`, a reference script, and tests against it (docs/ARCHITECTURE.md, "Adding a model").
- Write docs and comments in clear, plain English (American spelling).
