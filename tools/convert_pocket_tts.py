#!/usr/bin/env python3
"""Convert Kyutai's Pocket TTS into GGUF files for gorelaxor.

gorelaxor's text-to-speech engine (pockettts/) reads GGUF files. This script
makes them from Kyutai's original download, so the only things to trust
are Kyutai's files and this
script. It needs only Python 3.10+ and numpy: it reads the safetensors
files itself (a JSON header and raw numbers, nothing that can run code).

It converts the English model with voice cloning (kyutai/pocket-tts,
languages/english_2026-09): model.safetensors, tokenizer.json beside it,
and the preset voices in embeddings/*.safetensors.

Usage:

    python convert_pocket_tts.py english_2026-09/model.safetensors
    python convert_pocket_tts.py english_2026-09/model.safetensors --type f32 --voices alba,marius

It writes the model (pocket-tts-english_2026-09-Q8_0.gguf beside the
safetensors unless --out says otherwise) and one small file per preset
voice (voice-<name>.gguf in --voice-dir, by default beside the model).
--voices picks them: "all" (the default), a comma-separated list, or
"none". A voice is the model's state after hearing a short clip of the
speaker, about 3 MB at f16 (the default --voice-type) or 6 MB at f32,
which is why voices are separate files: an app ships the ones it wants.
A voice file names the model file it was made for and the engine
refuses it with another model.

--type is q8_0 (the default) or f32. q8_0 stores the transformer, the
flow head's larger matrices and the audio decoder's convolutions as
8-bit blocks, about a quarter of f32's size; norms, biases and small
matrices stay f32. Kyutai's weights are bfloat16, so f32 holds them
exactly. The script prints each output file's SHA-256 for
checking downloads, and the sources' SHA-256s.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("convert_nemo", Path(__file__).with_name("convert_nemo.py"))
cn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cn)
GGUF, GGML_F32, GGML_F16, GGML_Q8_0, FILE_TYPE = cn.GGUF, cn.GGML_F32, cn.GGML_F16, cn.GGML_Q8_0, cn.FILE_TYPE

# english_2026-09.yaml in Kyutai's pocket_tts/config, the settings the
# weights don't show by their shapes. Shapes are checked against the rest.
CONFIG = {
    "name": "english_2026-09",
    "lm_heads": 16,
    "lm_max_period": 10000.0,
    "mimi_heads": 8,
    "mimi_context": 250,
    "mimi_max_period": 10000.0,
    "sample_rate": 24000,
    "frame_rate": 12.5,
    "ratios": [6, 5, 4],
    "n_filters": 64,
    "n_residual_layers": 1,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "dilation_base": 2,
    "compress": 2,
    # default_parameters.py and tts_model.py
    "temperature": 0.3,
    "decode_steps": 1,
    "eos_threshold": -4.0,
    "max_tokens_per_chunk": 50,
    "min_frames_before_eos": 6,
    "frames_after_eos": -1,  # None: guessed from the chunk's length
    # Config defaults english_2026-09 keeps
    "append_terminal_punctuation": True,
    "capitalize_first_letter": True,
    "remove_semicolons": False,
    "pad_with_spaces_for_short_inputs": False,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Reads a safetensors file into numpy arrays: bfloat16 and float16
    widened to float32, int64 kept."""
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8 or n > 100 << 20:
        raise SystemExit(f"{path}: not a safetensors file")
    head = json.loads(data[8 : 8 + n])
    base = 8 + n
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        if not (0 <= a <= b <= len(data) - base):
            raise SystemExit(f"{path}: {k} lies outside the file")
        raw = data[base + a : base + b]
        dt = v["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, "<f2").astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, "<f4").astype(np.float32)
        elif dt == "I64":
            arr = np.frombuffer(raw, "<i8").copy()
        else:
            raise SystemExit(f"{path}: {k} has type {dt}")
        shape = v["shape"]
        if arr.size != int(np.prod(shape, dtype=np.int64)):
            raise SystemExit(f"{path}: {k} has the wrong size")
        out[k] = arr.reshape(shape)
    return out


class Emitter:
    """Moves tensors into the GGUF file under their new names and checks
    that every weight was used."""

    def __init__(self, g: GGUF, sd: dict, qtype: str):
        self.g, self.sd, self.qtype = g, sd, qtype
        self.used: set[str] = set()

    def arr(self, key: str, shape: tuple | None = None) -> np.ndarray:
        if key not in self.sd:
            raise SystemExit(f"the model has no {key}")
        a = self.sd[key]
        if a.dtype != np.float32:
            raise SystemExit(f"{key} is not a float tensor")
        if shape is not None and tuple(a.shape) != tuple(shape):
            raise SystemExit(f"{key} is {tuple(a.shape)}, want {tuple(shape)}")
        self.used.add(key)
        return np.ascontiguousarray(a)

    def put(self, name: str, arr: np.ndarray) -> None:
        """Matrices of at least 64 by 64 whose rows are whole Q8_0 blocks
        go to Q8_0 in a q8_0 file; everything else stays f32."""
        gtype = GGML_F32
        if self.qtype == "q8_0" and arr.ndim == 2 and arr.shape[0] >= 64 and arr.shape[1] >= 64 and arr.shape[1] % 32 == 0:
            gtype = GGML_Q8_0
        self.g.tensor(name, np.ascontiguousarray(arr, dtype=np.float32), gtype)

    def move(self, key: str, name: str, shape: tuple | None = None) -> np.ndarray:
        a = self.arr(key, shape)
        self.put(name, a)
        return a

    def conv(self, key: str, name: str) -> tuple[int, int, int]:
        """A Conv1d kernel [out, in, k] stored as [out, k*in], so one row
        multiplies k consecutive time steps of a time-major input."""
        w = self.arr(key)
        co, ci, k = w.shape
        self.put(name, w.transpose(0, 2, 1).reshape(co, k * ci))
        return co, ci, k

    def convtr(self, key: str, name: str) -> tuple[int, int, int]:
        """A ConvTranspose1d kernel [in, out, k] stored as [k*out, in]: one
        product gives every output tap of an input step."""
        w = self.arr(key)
        ci, co, k = w.shape
        self.put(name, w.transpose(2, 1, 0).reshape(k * co, ci))
        return ci, co, k

    def check_all_used(self) -> None:
        left = sorted(k for k in self.sd if k not in self.used)
        if left:
            raise SystemExit(f"{len(left)} weights were not converted, for example {left[:5]}")


def transformer(e: Emitter, src: str, dst: str, d: int, scale: bool) -> tuple[int, int]:
    n = 0
    while f"{src}.layers.{n}.norm1.weight" in e.sd:
        s, o = f"{src}.layers.{n}.", f"{dst}.{n}."
        for part in ("norm1", "norm2"):
            e.move(s + part + ".weight", o + part + ".weight", (d,))
            e.move(s + part + ".bias", o + part + ".bias", (d,))
        e.move(s + "self_attn.in_proj.weight", o + "attn_qkv.weight", (3 * d, d))
        e.move(s + "self_attn.out_proj.weight", o + "attn_out.weight", (d, d))
        ff = e.move(s + "linear1.weight", o + "ff1.weight").shape[0]
        e.move(s + "linear2.weight", o + "ff2.weight", (d, ff))
        if scale:
            e.move(s + "layer_scale_1.scale", o + "ls1", (d,))
            e.move(s + "layer_scale_2.scale", o + "ls2", (d,))
        n += 1
    if n == 0:
        raise SystemExit(f"the model has no {src} layers")
    return n, ff


def seanet(e: Emitter, src: str, dst: str) -> None:
    """Copies SEANet's convolutions under their module numbers: mimi.dec.0
    is decoder.model.0, mimi.dec.3.1 is decoder.model.3.block.1."""
    mods = sorted({int(k[len(src) + 7 :].split(".")[0]) for k in e.sd if k.startswith(f"{src}.model.")})
    if not mods:
        raise SystemExit(f"the model has no {src}")
    for j in mods:
        p = f"{src}.model.{j}."
        if p + "conv.weight" in e.sd:
            e.conv(p + "conv.weight", f"{dst}.{j}.weight")
            e.move(p + "conv.bias", f"{dst}.{j}.bias")
        elif p + "convtr.weight" in e.sd:
            e.convtr(p + "convtr.weight", f"{dst}.{j}.weight")
            e.move(p + "convtr.bias", f"{dst}.{j}.bias")
        elif p + "block.1.conv.weight" in e.sd:
            for b in (1, 3):
                e.conv(f"{p}block.{b}.conv.weight", f"{dst}.{j}.{b}.weight")
                e.move(f"{p}block.{b}.conv.bias", f"{dst}.{j}.{b}.bias")


def tokenizer(g: GGUF, tj: dict, n_bins: int) -> None:
    """Pocket TTS's tokenizer.json is a SentencePiece unigram model in
    Hugging Face's format. The engine implements exactly this pipeline,
    so anything else is refused rather than tokenized differently."""
    want = {
        "normalizer": {"type": "Prepend", "prepend": "▁"},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
    }
    for k, v in want.items():
        if tj.get(k) != v:
            raise SystemExit(f"tokenizer.json: {k} is {tj.get(k)!r}, which the engine doesn't implement")
    m = tj["model"]
    if m.get("type") != "Unigram" or m.get("byte_fallback") is not True or m.get("unk_id") != 0:
        raise SystemExit("tokenizer.json: the model isn't a byte-fallback unigram with <unk> first")
    vocab = m["vocab"]
    if len(vocab) != n_bins:
        raise SystemExit(f"tokenizer.json has {len(vocab)} pieces, the model's table {n_bins}")
    special = []
    for t in tj.get("added_tokens", []):
        if t["content"] != vocab[t["id"]][0] or t["normalized"] or t["single_word"] or t["lstrip"] or t["rstrip"] or not t["special"]:
            raise SystemExit(f"tokenizer.json: added token {t!r} isn't a plain special token")
        special.append(int(t["id"]))
    g.str("tokenizer.ggml.model", "unigram")
    g.strs("tokenizer.ggml.tokens", [p for p, _ in vocab])
    g.add("tokenizer.ggml.scores", cn.T_ARR, [float(s) for _, s in vocab], cn.T_F64)
    g.u32("tokenizer.ggml.unknown_token_id", 0)
    g.bool("tokenizer.ggml.byte_fallback", True)
    g.i32s("tokenizer.ggml.special_ids", special)


def convert_model(st: Path, tok: Path, qtype: str, sha: str) -> GGUF:
    sd = read_safetensors(st)
    c = CONFIG
    g = GGUF("pockettts")
    g.str("general.name", f"Pocket TTS {c['name']}")
    g.str("general.basename", "pocket-tts")
    g.str("general.author", "Kyutai")
    g.str("general.organization", "kyutai")
    g.u32("general.file_type", FILE_TYPE[qtype])
    g.str("general.source.file", st.name)
    g.str("general.source.sha256", sha)
    g.str("general.source.tokenizer.sha256", sha256(tok))
    e = Emitter(g, sd, qtype)

    # flow LM
    d = sd["flow_lm.out_norm.weight"].shape[0]
    ldim = sd["flow_lm.bos_emb"].shape[0]
    n_bins = sd["flow_lm.conditioner.embed.weight"].shape[0] - 1
    g.str("pockettts.config", c["name"])
    g.u32("pockettts.latent_dim", ldim)
    g.u32("pockettts.lm.d_model", d)
    g.u32("pockettts.lm.heads", c["lm_heads"])
    g.f32("pockettts.lm.max_period", c["lm_max_period"])
    e.put("lm.bos_voice", e.arr("flow_lm.bos_before_voice", (1, 1, d)).reshape(d))
    e.move("flow_lm.bos_emb", "lm.bos", (ldim,))
    e.move("flow_lm.conditioner.embed.weight", "lm.embed", (n_bins + 1, d))
    e.move("flow_lm.emb_mean", "lm.emb_mean", (ldim,))
    e.move("flow_lm.emb_std", "lm.emb_std", (ldim,))
    e.move("flow_lm.input_linear.weight", "lm.input.weight", (d, ldim))
    e.move("flow_lm.out_norm.weight", "lm.out_norm.weight", (d,))
    e.move("flow_lm.out_norm.bias", "lm.out_norm.bias", (d,))
    e.move("flow_lm.out_eos.weight", "lm.eos.weight", (1, d))
    e.move("flow_lm.out_eos.bias", "lm.eos.bias", (1,))
    has_clone = "flow_lm.speaker_proj_weight" in sd and any(k.startswith("mimi.encoder.") for k in sd)
    layers, ff = transformer(e, "flow_lm.transformer", "lm.blk", d, False)
    g.u32("pockettts.lm.layers", layers)
    g.u32("pockettts.lm.ff", ff)

    # flow head (SimpleMLPAdaLN, LSD with two time conditions)
    f = "flow_lm.flow_net."
    fd = sd[f + "input_proj.weight"].shape[0]
    e.move(f + "cond_embed.weight", "flow.cond.weight", (fd, d))
    e.move(f + "cond_embed.bias", "flow.cond.bias", (fd,))
    e.move(f + "input_proj.weight", "flow.input.weight", (fd, ldim))
    e.move(f + "input_proj.bias", "flow.input.bias", (fd,))
    nt = 0
    while f"{f}time_embed.{nt}.freqs" in sd:
        s, o = f"{f}time_embed.{nt}.", f"flow.time.{nt}."
        e.move(s + "freqs", o + "freqs")
        e.move(s + "mlp.0.weight", o + "fc1.weight")
        e.move(s + "mlp.0.bias", o + "fc1.bias", (fd,))
        e.move(s + "mlp.2.weight", o + "fc2.weight", (fd, fd))
        e.move(s + "mlp.2.bias", o + "fc2.bias", (fd,))
        e.move(s + "mlp.3.alpha", o + "norm.alpha", (fd,))
        nt += 1
    if nt != 2:
        raise SystemExit(f"the flow head has {nt} time conditions; only LSD's two are supported")
    depth = 0
    while f"{f}res_blocks.{depth}.in_ln.weight" in sd:
        s, o = f"{f}res_blocks.{depth}.", f"flow.res.{depth}."
        e.move(s + "in_ln.weight", o + "norm.weight", (fd,))
        e.move(s + "in_ln.bias", o + "norm.bias", (fd,))
        e.move(s + "mlp.0.weight", o + "fc1.weight", (fd, fd))
        e.move(s + "mlp.0.bias", o + "fc1.bias", (fd,))
        e.move(s + "mlp.2.weight", o + "fc2.weight", (fd, fd))
        e.move(s + "mlp.2.bias", o + "fc2.bias", (fd,))
        e.move(s + "adaLN_modulation.1.weight", o + "ada.weight", (3 * fd, fd))
        e.move(s + "adaLN_modulation.1.bias", o + "ada.bias", (3 * fd,))
        depth += 1
    e.move(f + "final_layer.adaLN_modulation.1.weight", "flow.final.ada.weight", (2 * fd, fd))
    e.move(f + "final_layer.adaLN_modulation.1.bias", "flow.final.ada.bias", (2 * fd,))
    e.move(f + "final_layer.linear.weight", "flow.final.out.weight", (ldim, fd))
    e.move(f + "final_layer.linear.bias", "flow.final.out.bias", (ldim,))
    g.u32("pockettts.flow.dim", fd)
    g.u32("pockettts.flow.depth", depth)

    # Mimi
    md = sd["mimi.decoder_transformer.transformer.layers.0.norm1.weight"].shape[0]
    e.put("mimi.quant", e.arr("mimi.quantizer.output_proj.weight", (md, ldim, 1)).reshape(md, ldim))
    stride = int(round(c["sample_rate"] / int(np.prod(c["ratios"])) / c["frame_rate"]))
    e.put("mimi.up", e.arr("mimi.upsample.convtr.convtr.weight", (md, 1, 2 * stride)).reshape(md, 2 * stride))
    mlayers, mff = transformer(e, "mimi.decoder_transformer.transformer", "mimi.dec_tf", md, True)
    seanet(e, "mimi.decoder", "mimi.dec")
    if has_clone:
        e.move("flow_lm.speaker_proj_weight", "lm.speaker_proj", (d, ldim))
        transformer(e, "mimi.encoder_transformer.transformer", "mimi.enc_tf", md, True)
        seanet(e, "mimi.encoder", "mimi.enc")
        e.conv("mimi.downsample.conv.conv.weight", "mimi.down")
    g.bool("pockettts.voice_cloning", has_clone)
    g.u32("pockettts.mimi.d_model", md)
    g.u32("pockettts.mimi.heads", c["mimi_heads"])
    g.u32("pockettts.mimi.layers", mlayers)
    g.u32("pockettts.mimi.ff", mff)
    g.u32("pockettts.mimi.context", c["mimi_context"])
    g.f32("pockettts.mimi.max_period", c["mimi_max_period"])
    g.u32("pockettts.mimi.upsample", stride)
    g.u32("pockettts.sample_rate", c["sample_rate"])
    g.f32("pockettts.frame_rate", c["frame_rate"])
    g.i32s("pockettts.seanet.ratios", c["ratios"])
    g.u32("pockettts.seanet.dimension", md)
    for k in ("n_filters", "n_residual_layers", "kernel_size", "residual_kernel_size", "last_kernel_size", "dilation_base", "compress"):
        g.u32("pockettts.seanet." + k, c[k])
    g.f32("pockettts.temperature", c["temperature"])
    g.u32("pockettts.decode_steps", c["decode_steps"])
    g.f32("pockettts.eos_threshold", c["eos_threshold"])
    g.u32("pockettts.max_tokens_per_chunk", c["max_tokens_per_chunk"])
    g.u32("pockettts.min_frames_before_eos", c["min_frames_before_eos"])
    g.i32("pockettts.frames_after_eos", c["frames_after_eos"])
    for k in ("append_terminal_punctuation", "capitalize_first_letter", "remove_semicolons", "pad_with_spaces_for_short_inputs"):
        g.bool("pockettts.text." + k, c[k])
    e.check_all_used()
    tokenizer(g, json.loads(tok.read_text(encoding="utf-8")), n_bins)
    return g


def convert_voice(path: Path, vtype: str, model_sha: str, d: int) -> GGUF:
    """A preset voice: each transformer layer's keys and values after the
    voice prompt (pocket_tts export_model_state), [positions, d] each."""
    sd = read_safetensors(path)
    g = GGUF("pockettts-voice")
    g.str("general.name", path.stem)
    g.str("general.source.file", path.name)
    g.str("general.source.sha256", sha256(path))
    g.str("pockettts.voice.model.sha256", model_sha)
    n, positions = 0, None
    while f"transformer.layers.{n}.self_attn/cache" in sd:
        p = f"transformer.layers.{n}.self_attn/"
        cache = sd[p + "cache"]
        off = int(sd[p + "offset"].reshape(-1)[0])
        if cache.dtype != np.float32 or cache.ndim != 5 or cache.shape[:2] != (2, 1) or cache.shape[3] * cache.shape[4] != d:
            raise SystemExit(f"{path}: layer {n}'s cache is {cache.shape}")
        if p + "pad" in sd and int(sd[p + "pad"].reshape(-1)[0]) != 0:
            raise SystemExit(f"{path}: padded voice states aren't supported")
        if positions is None:
            positions = off
        if off != positions or off > cache.shape[2]:
            raise SystemExit(f"{path}: layer {n} holds {off} positions")
        kv = cache[:, 0, :off].reshape(2, off, d)
        if not np.isfinite(kv).all():
            raise SystemExit(f"{path}: layer {n} has values that aren't numbers")
        gt = GGML_F16 if vtype == "f16" else GGML_F32
        g.tensor(f"voice.{n}.k", kv[0], gt)
        g.tensor(f"voice.{n}.v", kv[1], gt)
        n += 1
    if n == 0:
        raise SystemExit(f"{path}: not a Pocket TTS voice state")
    g.u32("pockettts.voice.layers", n)
    g.u32("pockettts.voice.positions", positions)
    return g


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="Kyutai's model.safetensors")
    ap.add_argument("--tokenizer", type=Path, help="tokenizer.json (default: beside the model)")
    ap.add_argument("--type", choices=["f32", "q8_0"], default="q8_0")
    ap.add_argument("--out", type=Path, help="output file (default: beside the model)")
    ap.add_argument("--voices", default="all", help='"all", "none" or a comma-separated list of preset voices')
    ap.add_argument("--voice-src", type=Path, help="the preset voices' folder (default: embeddings/ beside the model)")
    ap.add_argument("--voice-dir", type=Path, help="where voice files go (default: beside the output)")
    ap.add_argument("--voice-type", choices=["f16", "f32"], default="f16")
    a = ap.parse_args(argv)
    tok = a.tokenizer or a.model.with_name("tokenizer.json")
    sha = sha256(a.model)
    out = a.out or a.model.with_name(f"pocket-tts-{CONFIG['name']}-{a.type.upper()}.gguf")
    g = convert_model(a.model, tok, a.type, sha)
    digest = g.write(out)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    print(f"source {a.model.name} sha256 {sha}")
    print(f"source {tok.name} sha256 {sha256(tok)}")
    if a.voices != "none":
        src = a.voice_src or a.model.with_name("embeddings")
        names = sorted(p.stem for p in src.glob("*.safetensors")) if a.voices == "all" else a.voices.split(",")
        vdir = a.voice_dir or out.parent
        vdir.mkdir(parents=True, exist_ok=True)
        d = next(t[1].shape[0] for t in g.tensors if t[0] == "lm.out_norm.weight")
        for n in names:
            vp = src / f"{n}.safetensors"
            if not vp.exists():
                raise SystemExit(f"no preset voice {vp}")
            vo = vdir / f"voice-{n}.gguf"
            vd = convert_voice(vp, a.voice_type, sha, d).write(vo)
            print(f"{vo}  {vo.stat().st_size / 1e6:.1f} MB  sha256 {vd}  source {vp.name} sha256 {sha256(vp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
