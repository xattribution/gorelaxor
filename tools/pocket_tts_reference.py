#!/usr/bin/env python3
"""Make Kyutai Pocket TTS reference outputs for the pockettts/ parity tests.

This is a developer tool, not part of the engine: it needs Kyutai's
pocket_tts package (pip install --no-deps -e <pocket-tts source>, with
torch, numpy, scipy, pydantic, pyyaml, safetensors, tokenizers and
sentencepiece) and the original download's language folder (model.safetensors,
tokenizer.json, embeddings/). It runs Kyutai's PyTorch code in float32 on
the CPU, reading the local files rather than Hugging Face, and saves what
the Go tests compare against into the folder GORELAXOR_POCKET_REF names:

    L=languages/english_2026-09
    python -I pocket_tts_reference.py tokens $L REF
    python -I pocket_tts_reference.py speak $L REF/alba-short --voice alba \
        --text "Hello there, how are you today?"
    python -I pocket_tts_reference.py speak $L REF/marius-long --voice marius \
        --text "The harbor ferry now leaves at noon on Thursdays from the old north pier. Please bring your printed ticket, a warm jacket, and any questions you have for the captain before boarding."
    python -I pocket_tts_reference.py speak $L REF/javert-numbers --voice javert \
        --text "On Friday at 9 a.m., Dr. Lee will meet 3 teams about the 25 new laptops."
    python -I pocket_tts_reference.py speak $L REF/clone-alba --clone casual.wav \
        --text "Voice cloning makes this sentence sound like the person in the clip."

casual.wav is kyutai/tts-voices alba-mackenna/casual.wav, the clip behind
the alba voice. The model and voice files the tests load go in REF too
(ref_test.go in pockettts/ lists them).

For each speak run it saves, as NAME.f32 (raw little-endian float32,
row-major) with NAME.json giving the shape:

    lm_out    [calls, 1024]  the transformer's output (after out_norm) the flow head took
    noise     [calls, 32]    the noise it started from, which the Go test feeds back
    latent    [calls, 32]    the latent it made
    eos       [calls]        the end-of-speech logit
    pcm       [samples]      the 24 kHz audio, every chunk one after another

Calls run in order; each text chunk's first call is its text prompt (its
latent is thrown away, as in Kyutai's code), and case.json lists the
chunks, their token ids and frame counts. With --clone it also saves
clone_pcm (the clip after resampling and end_on_pause), clone_latent
(Mimi's encoding) and clone_cond (the conditioning the flow LM hears).

The tokens run tokenizes a fixed list of awkward strings and splits a few
long texts into chunks as generate_audio_stream does, into tokens.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

TOKEN_CASES = [
    "Hello world.",
    "Hello world",
    "  leading and trailing spaces  ",
    "Two  spaces and   three.",
    "Numbers: 3.14, 1,000,000 and 42nd.",
    "Dr. Smith met Prof. Jones at 9:30 on Oct. 16, 2026.",
    "Email me at someone@example.com, or call 555-0100!",
    "Quotes \"straight\" and “curly”, apostrophes ' and ’.",
    "Café naïve résumé über å",
    "Emoji \U0001F600 and CJK 中文 and Cyrillic привет.",
    "Tabs\tand\nnewlines\r\nhere.",
    "<s> special </s> tokens <unk> <pad> <0x41>",
    "▁already▁metaspace",
    "ALL CAPS ABBREVIATIONS: NASA, UNESCO, HTML, R2D2.",
    "a",
    ".",
    "...",
    "!?",
    "Mr. and Mrs. O'Neil's 2nd-floor apartment – it's 50% off — costs $1,250.99.",
    "Ellipsis… and en dash – and em dash — too.",
]

CHUNK_CASES = [
    "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    "the meeting moved to 14:30 on 3.5 floors, so bring the 2.1 draft. version 3.2.1 is out! does it work? yes",
    "This is one very long sentence that goes on and on, with many clauses separated by commas, "
    "semicolons; and colons: so that the splitter must fall back to the weaker punctuation marks, "
    "because otherwise the chunk would hold far more than fifty tokens, which the model does not like, "
    "and it might skip some words at the end of the chunk",
    "Short. Also short. Another one. And one more. Then a longer sentence that brings the total "
    "count of tokens in this chunk past the limit of fifty tokens per chunk, so a new chunk starts. "
    "Then the end",
]


def save(out: Path, name: str, arr) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    arr.tofile(out / f"{name}.f32")
    (out / f"{name}.json").write_text(json.dumps({"name": name, "shape": list(arr.shape), "source": {"kind": "pocket_tts"}}))


def load(lang_dir: Path):
    """TTSModel from the local files: Kyutai's english_2026-09.yaml with its
    hf:// paths pointed at the folder."""
    import yaml
    from pocket_tts import TTSModel
    from pocket_tts.utils.config import CONFIGS_DIR

    cfg = yaml.safe_load((CONFIGS_DIR / "english_2026-09.yaml").read_text())
    cfg["weights_path"] = str((lang_dir / "model.safetensors").resolve())
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str((lang_dir / "tokenizer.json").resolve())
    tmp = Path(tempfile.mkdtemp()) / "english_2026-09-local.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    m = TTSModel.load_model(config=str(tmp))
    m.eval()
    return m


def tokens(model, out: Path) -> None:
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences

    tok = model.flow_lm.conditioner.tokenizer
    cases = []
    for s in TOKEN_CASES:
        ids = tok.encode(s)
        cases.append({"text": s, "ids": ids, "decoded": tok.decode(ids)})
    chunks = []
    for s in CHUNK_CASES:
        parts = split_into_best_sentences(tok, s, 50, False, False, True, True, {})
        prepared = [prepare_text_prompt(p, False, False, True, True, {}) for p in parts]
        chunks.append({"text": s, "chunks": parts, "prepared": [p for p, _ in prepared],
                       "frames_after_eos": [g + 2 for _, g in prepared],
                       "ids": [tok.encode(p) for p, _ in prepared]})
    (out / "tokens.json").write_text(json.dumps({"tokens": cases, "chunks": chunks}, ensure_ascii=False, indent=1))


def speak(model, out: Path, text: str, voice: Path | None, clone: Path | None, seed: int) -> None:
    import pocket_tts.models.flow_lm as flow_lm
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences

    rec = {"lm_out": [], "noise": [], "latent": [], "eos": []}
    prompts: list[int] = []  # the calls that were a chunk's text prompt
    frames: list[int] = []  # latents decoded per chunk
    orig = flow_lm.lsd_decode
    in_prompt = [False]

    def lsd(v_t, x_0, num_steps=1):
        y = orig(v_t, x_0, num_steps)
        if in_prompt[0]:
            prompts.append(len(rec["lm_out"]))
        rec["lm_out"].append(v_t.args[0][0].clone())
        rec["noise"].append(x_0[0].clone())
        rec["latent"].append(y[0].clone())
        return y

    flow_lm.lsd_decode = lsd
    model.flow_lm.out_eos.register_forward_hook(lambda m, i, o: rec["eos"].append(o.reshape(-1)[-1].clone()))
    run = model._run_flow_lm

    def run_flow_lm(model_state, text_tokens, backbone_input_latents, audio_conditioning):
        in_prompt[0] = text_tokens.shape[1] > 0
        try:
            return run(model_state, text_tokens, backbone_input_latents, audio_conditioning)
        finally:
            in_prompt[0] = False

    model._run_flow_lm = run_flow_lm
    dec = model.mimi.decode_from_latent

    def decode(latent, state):
        while len(frames) < len(prompts):
            frames.append(0)
        frames[-1] += latent.shape[1]
        return dec(latent, state)

    model.mimi.decode_from_latent = decode
    meta = {"text": text, "seed": seed, "temperature": model.temp}
    if clone is not None:
        got = {}
        enc = model.mimi.encode_to_latent
        mimi_enc = lambda x: got.setdefault("latent", enc(x))  # noqa: E731
        model.mimi.encode_to_latent = mimi_enc
        ea = model._encode_audio

        def encode_audio(audio):
            got["pcm"] = audio.clone()
            c = ea(audio)
            got["cond"] = c.clone()
            return c

        model._encode_audio = encode_audio
        state = model.get_state_for_audio_prompt(clone)
        save(out, "clone_pcm", got["pcm"].reshape(-1))
        save(out, "clone_latent", got["latent"][0])
        save(out, "clone_cond", got["cond"][0])
        for k in rec:
            rec[k].clear()  # the voice prompt's own call
        meta["voice"] = "clone:" + clone.name
    else:
        state = model.get_state_for_audio_prompt(voice)
        meta["voice"] = voice.stem
    tok = model.flow_lm.conditioner.tokenizer
    parts = split_into_best_sentences(tok, text, 50, False, False, True, True, {})
    meta["chunks"] = []
    for p in parts:
        prep, guess = prepare_text_prompt(p, False, False, True, True, {})
        meta["chunks"].append({"text": prep, "ids": tok.encode(prep), "frames_after_eos": guess + 2})
    torch.manual_seed(seed)
    pcm = model.generate_audio(state, text)
    calls = len(rec["lm_out"])
    if len(prompts) != len(meta["chunks"]):
        raise SystemExit(f"{len(prompts)} text prompts for {len(meta['chunks'])} chunks")
    for c, first, n in zip(meta["chunks"], prompts, frames + [0] * (len(prompts) - len(frames))):
        c["first_call"], c["frames"] = first, n
    meta["calls"] = calls
    meta["samples"] = int(pcm.shape[-1])
    for k, v in rec.items():
        save(out, k, torch.stack(v).numpy() if v else np.zeros((0,)))
    save(out, "pcm", pcm.reshape(-1).numpy())
    (out / "case.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{out}: {len(parts)} chunks, {calls} flow calls, {pcm.shape[-1] / 24000:.2f} s")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tokens", "speak"])
    ap.add_argument("lang_dir", type=Path, help="the downloaded languages/english_2026-09 folder")
    ap.add_argument("out", type=Path)
    ap.add_argument("--text")
    ap.add_argument("--voice", default="alba", help="a preset voice in embeddings/")
    ap.add_argument("--clone", type=Path, help="a WAV clip to clone instead of a preset voice")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    model = load(a.lang_dir)
    if a.mode == "tokens":
        tokens(model, a.out)
        return 0
    if not a.text:
        raise SystemExit("speak needs --text")
    speak(model, a.out, a.text, a.lang_dir / "embeddings" / f"{a.voice}.safetensors", a.clone, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
