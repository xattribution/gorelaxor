#!/usr/bin/env python3
"""Convert NVIDIA's .nemo speech models into GGUF files for gorelaxor.

gorelaxor's FastConformer engine (conformer/) reads GGUF files. This script
makes them from NVIDIA's original .nemo checkpoints, so the only things to
trust are NVIDIA's download and this script. It does
not need NVIDIA NeMo: it opens the .nemo archive itself, reads the weights
with PyTorch's safe loader (weights_only=True, which refuses pickled code)
and writes the file.

Models it converts:

    nvidia/nemotron-speech-streaming-en-0.6b        live captions
    nvidia/nemotron-3.5-asr-streaming-0.6b          live captions, 32 languages
    nvidia/parakeet-tdt-0.6b-v2                     final pass, English
    nvidia/parakeet-tdt-0.6b-v3                     final pass, 25 languages
    nvidia/multitalker-parakeet-streaming-0.6b-v1   crosstalk
    nvidia/diar_streaming_sortformer_4spk-v2.1      speakers
    nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0    silence detection

Usage:

    python convert_nemo.py nemotron-speech-streaming-en-0.6b.nemo --type q8_0
    python convert_nemo.py parakeet-tdt-0.6b-v2.nemo --type q8_0 --out final.gguf

--type is f32, f16 or q8_0 (the default). q8_0 stores the large matrices as
8-bit blocks (the convolution modules' 1x1 kernels as f16), under a third of
f32's size, with no measurable loss in accuracy; norms, biases and small
convolution kernels stay f32. The script prints the output file's SHA-256
so a download can be checked against it.

Needs Python 3.10+ with torch 2.6 or later (earlier versions can run code
hidden in a checkpoint even with weights_only=True: CVE-2025-32434),
numpy, pyyaml and sentencepiece:

    pip install "torch>=2.6" numpy pyyaml sentencepiece

Tensor and setting names follow transcribe.cpp's GGUF layout, so that
program can read these files too; its tensor dumps are an optional extra
check in conformer/ref_test.go.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7}  # llama.cpp's LlamaFileType numbers

# GGUF value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

# llama.cpp token types
TOK_NORMAL, TOK_UNKNOWN, TOK_CONTROL, TOK_USER, TOK_UNUSED, TOK_BYTE = 1, 2, 3, 4, 5, 6

ALIGN = 32


# ---------------------------------------------------------------- GGUF writer


class GGUF:
    """A small GGUF v3 writer: settings first, then tensors, 32-byte aligned."""

    def __init__(self, arch: str):
        self.kv: list[tuple[str, int, object, int]] = []
        self.tensors: list[tuple[str, np.ndarray, int]] = []
        self.add("general.architecture", T_STR, arch)

    def add(self, key: str, vtype: int, value, etype: int = -1) -> None:
        if any(k == key for k, *_ in self.kv):
            raise ValueError(f"setting {key} written twice")
        self.kv.append((key, vtype, value, etype))

    def u32(self, k, v): self.add(k, T_U32, int(v))
    def i32(self, k, v): self.add(k, T_I32, int(v))
    def f32(self, k, v): self.add(k, T_F32, float(v))
    def bool(self, k, v): self.add(k, T_BOOL, bool(v))
    def str(self, k, v): self.add(k, T_STR, str(v))
    def strs(self, k, v): self.add(k, T_ARR, [str(x) for x in v], T_STR)
    def i32s(self, k, v): self.add(k, T_ARR, [int(x) for x in v], T_I32)
    def f32s(self, k, v): self.add(k, T_ARR, [float(x) for x in v], T_F32)

    def tensor(self, name: str, arr: np.ndarray, gtype: int) -> None:
        if any(n == name for n, *_ in self.tensors):
            raise ValueError(f"tensor {name} written twice")
        self.tensors.append((name, arr, gtype))

    @staticmethod
    def _s(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, vtype: int, v, etype: int) -> bytes:
        if vtype == T_STR:
            return self._s(v)
        if vtype == T_ARR:
            out = struct.pack("<IQ", etype, len(v))
            if etype == T_STR:
                return out + b"".join(self._s(x) for x in v)
            fmt = {T_I32: "<i", T_U32: "<I", T_F32: "<f", T_F64: "<d"}[etype]
            return out + b"".join(struct.pack(fmt, x) for x in v)
        fmt = {T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_F64: "<d"}[vtype]
        return struct.pack(fmt, v)

    def write(self, path: Path) -> str:
        infos = []
        offset = 0
        for name, arr, gtype in self.tensors:
            dims = list(reversed(arr.shape)) or [1]
            info = self._s(name) + struct.pack("<I", len(dims))
            info += b"".join(struct.pack("<Q", d) for d in dims)
            info += struct.pack("<IQ", gtype, offset)
            infos.append(info)
            offset += encoded_size(arr, gtype)
            offset += (-offset) % ALIGN
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        for k, vtype, v, etype in self.kv:
            head += self._s(k) + struct.pack("<I", vtype) + self._value(vtype, v, etype)
        head += b"".join(infos)
        head += b"\0" * ((-len(head)) % ALIGN)
        h = hashlib.sha256()
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            f.write(head)
            h.update(head)
            # one tensor's bytes at a time, so the file never sits in memory whole
            for _, arr, gtype in self.tensors:
                d = encode(arr, gtype)
                if len(d) != encoded_size(arr, gtype):
                    raise ValueError("encoded size mismatch")
                pad = b"\0" * ((-len(d)) % ALIGN)
                f.write(d)
                f.write(pad)
                h.update(d)
                h.update(pad)
        tmp.replace(path)
        return h.hexdigest()


def f16(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).astype("<f2").tobytes()


def q8_0(arr: np.ndarray) -> bytes:
    """ggml's Q8_0: each run of 32 values becomes a float16 scale and 32
    int8s, rounded half away from zero as ggml's roundf does."""
    x = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    v = x * inv
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def encoded_size(arr: np.ndarray, gtype: int) -> int:
    n = int(arr.size)
    if gtype == GGML_Q8_0:
        if n % 32:
            raise ValueError("Q8_0 needs whole blocks of 32")
        return n // 32 * 34
    return n * {GGML_F32: 4, GGML_F16: 2}[gtype]


def encode(arr: np.ndarray, gtype: int) -> bytes:
    if gtype == GGML_F32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if gtype == GGML_F16:
        return f16(arr)
    if gtype == GGML_Q8_0:
        return q8_0(arr)
    raise ValueError(gtype)


# ----------------------------------------------------------- .nemo archives


def read_member(tar: tarfile.TarFile, m: tarfile.TarInfo) -> bytearray:
    """Reads one archive file into a single buffer of its size: a gzipped
    archive's read() would hold the weights twice while joining chunks."""
    buf = bytearray(m.size)
    view = memoryview(buf)
    src = tar.extractfile(m)
    pos = 0
    while pos < m.size:
        n = src.readinto(view[pos:pos + (16 << 20)])
        if not n:
            raise SystemExit(f"{m.name} ends early")
        pos += n
    return buf


class MemoryFile(io.RawIOBase):
    """A read-only file over a buffer, without the copy io.BytesIO makes,
    so the weights are in memory once while PyTorch reads them."""

    def __init__(self, buf: bytearray):
        self.view, self.pos = memoryview(buf), 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = max(0, {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: len(self.view)}[whence] + off)
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), len(self.view) - self.pos))
        b[:n] = self.view[self.pos:self.pos + n]
        self.pos += n
        return n


class Nemo:
    """Reads a .nemo archive in memory: the config, the weights and any
    tokenizer files. Nothing is extracted to disk."""

    def __init__(self, path: Path):
        import torch
        import yaml

        major, minor = (int(x) for x in re.findall(r"\d+", torch.__version__)[:2])
        if (major, minor) < (2, 6):
            raise SystemExit(f"torch {torch.__version__} can run code hidden in a checkpoint (CVE-2025-32434); install torch 2.6 or later")
        self.path = path
        self.files: dict[str, bytes] = {}
        # One open file for the hash and the archive, so what's converted is
        # exactly what the hash covers, even on a share that changes.
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            self.sha256 = h.hexdigest()
            f.seek(0)
            # Only NeMo's own top-level files are read, each once, with a
            # size cap; nothing is written to disk.
            caps = {"model_config.yaml": 1 << 20, "model_weights.ckpt": 8 << 30}
            # NeMo writes plain tar; some downloads (Nemotron 3.5) are
            # gzipped, which NeMo's _tar_open also accepts.
            mode = "r:gz" if f.read(2) == b"\x1f\x8b" else "r:"
            f.seek(0)
            with tarfile.open(fileobj=f, mode=mode) as tar:
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name[2:] if m.name.startswith("./") else m.name
                    if "/" in name:
                        continue
                    cap = caps.get(name) or (4 << 20 if name.endswith("tokenizer.model") else 0)
                    if not cap:
                        continue
                    if name in self.files:
                        raise SystemExit(f"{path} has {name} twice")
                    if m.size > cap:
                        raise SystemExit(f"{path}: {name} is larger than a NeMo model's")
                    data = read_member(tar, m)
                    self.files[name] = data if name == "model_weights.ckpt" else bytes(data)
        if "model_config.yaml" not in self.files or "model_weights.ckpt" not in self.files:
            raise SystemExit(f"{path} is not a NeMo model archive")
        self.cfg = yaml.safe_load(self.files["model_config.yaml"])
        sd = torch.load(io.BufferedReader(MemoryFile(self.files.pop("model_weights.ckpt"))), map_location="cpu",
                        weights_only=True)
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        self.sd = sd

    def file(self, suffix: str) -> bytes | None:
        hits = [k for k in self.files if k.endswith(suffix)]
        return self.files[hits[0]] if hits else None


# ------------------------------------------------------------ tensor policy


class Emitter:
    """Moves state-dict tensors into the GGUF file under their new names and
    checks that every weight was used."""

    def __init__(self, g: GGUF, sd: dict, qtype: str):
        self.g, self.sd, self.qtype = g, sd, qtype
        self.used: set[str] = set()

    def arr(self, key: str) -> np.ndarray:
        import torch

        if key not in self.sd:
            raise SystemExit(f"the checkpoint has no {key}")
        t = self.sd[key]
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
            raise SystemExit(f"{key} is not a float32 tensor")
        self.used.add(key)
        return np.ascontiguousarray(t.detach().cpu().numpy())

    def put(self, name: str, arr: np.ndarray, matrix: bool = False) -> None:
        """matrix marks weights used as matrix products, which may be
        stored as F16 or Q8_0. Everything else stays F32."""
        gtype = GGML_F32
        if matrix and self.qtype != "f32":
            gtype = GGML_F16
            # Q8_0 blocks run along a tensor's first GGUF dimension (the
            # last numpy one). A 1x1 convolution kernel [out, in, 1] has 1
            # there, so it stays F16 in its own shape, as transcribe.cpp's
            # quantizer leaves it, and transcribe.cpp can still load the file.
            if self.qtype == "q8_0" and arr.ndim == 2 and arr.shape[-1] % 32 == 0:
                gtype = GGML_Q8_0
        self.g.tensor(name, arr, gtype)

    def move(self, key: str, name: str, matrix: bool = False) -> None:
        self.put(name, self.arr(key), matrix)

    def has(self, key: str) -> bool:
        return key in self.sd

    def check_all_used(self, ignore_prefixes=("preprocessor.",)) -> None:
        left = sorted(
            k for k in self.sd
            if k not in self.used and not k.startswith(ignore_prefixes) and not k.endswith(".num_batches_tracked")
        )
        if left:
            raise SystemExit(f"{len(left)} weights were not converted, for example {left[:5]}")


# ------------------------------------------------------------ shared parts


def canonical_normalize(v) -> str:
    v = "none" if v is None else str(v)
    m = {"": "none", "NA": "none", "none": "none", "None": "none", "per_feature": "per_feature",
         "all_features": "global"}
    if v not in m:
        raise SystemExit(f"front-end normalize {v!r} is not supported")
    return m[v]


# NeMo's AudioToMelSpectrogramPreprocessor defaults for the settings the Go
# front end implements one way only; any other value is refused.
FRONTEND_FIXED = {"window": "hann", "log": True, "log_zero_guard_type": "add", "mag_power": 2.0,
                  "exact_pad": False, "frame_splicing": 1, "mel_norm": "slaney"}


def frontend(g: GGUF, pre: dict) -> None:
    if "MelSpectrogram" not in str(pre.get("_target_", "")):
        raise SystemExit("only mel-spectrogram front ends are supported")
    for k, want in FRONTEND_FIXED.items():
        got = pre.get(k, want)
        if got != want and not (isinstance(want, float) and isinstance(got, (int, float)) and float(got) == want):
            raise SystemExit(f"front-end {k} {got!r} is not supported (only {want!r})")
    guard = pre.get("log_zero_guard_value", 2 ** -24)
    if not isinstance(guard, (int, float)) or float(guard) != 2 ** -24:
        raise SystemExit(f"front-end log_zero_guard_value {guard!r} is not supported (only 2**-24)")
    sr = int(pre["sample_rate"])
    g.str("stt.frontend.type", "mel")
    g.u32("stt.frontend.num_mels", int(pre["features"]))
    g.u32("stt.frontend.sample_rate", sr)
    g.u32("stt.frontend.n_fft", int(pre["n_fft"]))
    g.u32("stt.frontend.win_length", int(round(float(pre["window_size"]) * sr)))
    g.u32("stt.frontend.hop_length", int(round(float(pre["window_stride"]) * sr)))
    g.str("stt.frontend.window", "hann")
    # NeMo's defaults where a key is missing: per-feature normalization,
    # pre-emphasis 0.97, padding to 16 frames. An explicit null turns
    # pre-emphasis off.
    g.str("stt.frontend.normalize", canonical_normalize(pre.get("normalize", "per_feature")))
    g.f32("stt.frontend.dither", float(pre.get("dither", 0.0) or 0.0))
    g.f32("stt.frontend.pre_emphasis", float(pre["preemph"] or 0.0) if "preemph" in pre else 0.97)
    g.f32("stt.frontend.f_min", float(pre.get("lowfreq") or 0.0))
    g.f32("stt.frontend.f_max", float(pre.get("highfreq") or sr / 2))
    pad = int(pre["pad_to"] or 0) if "pad_to" in pre else 16
    if pad > 1:
        g.u32("stt.frontend.pad_to", pad)


def identity(g: GGUF, nemo: Nemo, name: str, basename: str, qtype: str) -> None:
    g.str("general.name", name)
    g.str("general.basename", basename)
    g.str("general.author", "NVIDIA")
    g.str("general.organization", "nvidia")
    g.u32("general.file_type", FILE_TYPE[qtype])
    g.str("general.source.file", nemo.path.name)
    g.str("general.source.sha256", nemo.sha256)


PRE_ENCODE = ["conv.0", "conv.2", "conv.3", "conv.5", "conv.6", "out"]

BLOCK = [  # NeMo name, GGUF name, used as a matrix product
    ("norm_feed_forward1.weight", "norm_ff1.weight", False),
    ("norm_feed_forward1.bias", "norm_ff1.bias", False),
    ("feed_forward1.linear1.weight", "ff1.linear1.weight", True),
    ("feed_forward1.linear2.weight", "ff1.linear2.weight", True),
    ("norm_self_att.weight", "norm_attn.weight", False),
    ("norm_self_att.bias", "norm_attn.bias", False),
    ("self_attn.linear_q.weight", "attn.linear_q.weight", True),
    ("self_attn.linear_k.weight", "attn.linear_k.weight", True),
    ("self_attn.linear_v.weight", "attn.linear_v.weight", True),
    ("self_attn.linear_out.weight", "attn.linear_out.weight", True),
    ("self_attn.linear_pos.weight", "attn.linear_pos.weight", True),
    ("self_attn.pos_bias_u", "attn.pos_bias_u", False),
    ("self_attn.pos_bias_v", "attn.pos_bias_v", False),
    ("norm_conv.weight", "norm_conv.weight", False),
    ("norm_conv.bias", "norm_conv.bias", False),
    ("conv.pointwise_conv1.weight", "conv.pointwise1.weight", True),
    ("conv.depthwise_conv.weight", "conv.depthwise.weight", False),
    ("conv.pointwise_conv2.weight", "conv.pointwise2.weight", True),
    ("conv.batch_norm.weight", "conv.bn.weight", False),
    ("conv.batch_norm.bias", "conv.bn.bias", False),
    ("norm_feed_forward2.weight", "norm_ff2.weight", False),
    ("norm_feed_forward2.bias", "norm_ff2.bias", False),
    ("feed_forward2.linear1.weight", "ff2.linear1.weight", True),
    ("feed_forward2.linear2.weight", "ff2.linear2.weight", True),
    ("norm_out.weight", "norm_out.weight", False),
    ("norm_out.bias", "norm_out.bias", False),
]

BLOCK_BIAS = [
    ("feed_forward1.linear1.bias", "ff1.linear1.bias"),
    ("feed_forward1.linear2.bias", "ff1.linear2.bias"),
    ("self_attn.linear_q.bias", "attn.linear_q.bias"),
    ("self_attn.linear_k.bias", "attn.linear_k.bias"),
    ("self_attn.linear_v.bias", "attn.linear_v.bias"),
    ("self_attn.linear_out.bias", "attn.linear_out.bias"),
    ("conv.pointwise_conv1.bias", "conv.pointwise1.bias"),
    ("conv.depthwise_conv.bias", "conv.depthwise.bias"),
    ("conv.pointwise_conv2.bias", "conv.pointwise2.bias"),
    ("feed_forward2.linear1.bias", "ff2.linear1.bias"),
    ("feed_forward2.linear2.bias", "ff2.linear2.bias"),
]


def conformer_encoder(e: Emitter, enc: dict, prefix: str) -> bool:
    """Writes the FastConformer encoder's tensors; returns use_bias."""
    for p in PRE_ENCODE:
        e.move(f"encoder.pre_encode.{p}.weight", f"enc.pre_encode.{p}.weight", matrix=(p == "out"))
        e.move(f"encoder.pre_encode.{p}.bias", f"enc.pre_encode.{p}.bias")
    n = int(enc["n_layers"])
    with_bias = [e.has(f"encoder.layers.{i}.feed_forward1.linear1.bias") for i in range(n)]
    if any(with_bias) != all(with_bias):
        raise SystemExit("some encoder layers have biases and some don't")
    bn = str(enc.get("conv_norm_type", "batch_norm")) == "batch_norm"
    for i in range(n):
        src, dst = f"encoder.layers.{i}.", f"{prefix}.{i}."
        for a, b, mat in BLOCK:
            e.move(src + a, dst + b, matrix=mat)
        if bn:
            e.move(src + "conv.batch_norm.running_mean", dst + "conv.bn.running_mean")
            e.move(src + "conv.batch_norm.running_var", dst + "conv.bn.running_var")
        if with_bias[0]:
            for a, b in BLOCK_BIAS:
                e.move(src + a, dst + b)
    return with_bias[0]


def att_context(raw) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """NeMo's att_context_size is [L, R] or, for cache-aware streaming
    models, a list of [L, R] choices. Returns the main one and all of
    them in NeMo's order: NeMo's encoder runs a whole recording with the
    first choice (ConformerEncoder._calc_context_sizes), which is not
    always the largest (Nemotron 3.5 lists [56, 3] first)."""
    if not raw:
        return (-1, -1), []
    if isinstance(raw[0], (list, tuple)):
        ch = [(int(a), int(b)) for a, b in raw]
        if len(set(ch)) != len(ch):
            raise SystemExit(f"att_context_size {raw} lists a choice twice")
        return ch[0], ch
    return (int(raw[0]), int(raw[1])), [(int(raw[0]), int(raw[1]))]


def conv_context(raw, k: int) -> tuple[int, int]:
    if raw is None:
        return (k - 1) // 2, (k - 1) // 2
    if raw == "causal":
        return k - 1, 0
    a, b = int(raw[0]), int(raw[1])
    if a + b + 1 != k:
        raise SystemExit(f"conv_context_size {raw} doesn't fit kernel {k}")
    return a, b


def encoder_settings(g: GGUF, enc: dict, ns: str) -> None:
    """The ConformerEncoder settings common to Parakeet and Sortformer."""
    d = int(enc["d_model"])
    k = int(enc["conv_kernel_size"])
    g.u32(f"{ns}.n_layers", int(enc["n_layers"]))
    g.u32(f"{ns}.d_model", d)
    g.u32(f"{ns}.n_heads", int(enc["n_heads"]))
    g.u32(f"{ns}.d_ff", d * int(enc["ff_expansion_factor"]))
    g.u32(f"{ns}.conv_kernel", k)
    g.u32(f"{ns}.subsampling_factor", int(enc["subsampling_factor"]))
    g.u32(f"{ns}.subsampling_channels", int(enc["subsampling_conv_channels"]))
    g.u32(f"{ns}.pos_emb_max_len", int(enc.get("pos_emb_max_len", 5000)))
    g.str(f"{ns}.conv_norm_type", str(enc.get("conv_norm_type", "batch_norm")))
    if str(enc.get("subsampling", "dw_striding")) != "dw_striding":
        raise SystemExit(f"subsampling {enc.get('subsampling')} is not supported")
    if str(enc.get("self_attention_model", "rel_pos")) != "rel_pos":
        raise SystemExit(f"attention {enc.get('self_attention_model')} is not supported")


# ------------------------------------------------------- Parakeet family


LANG_TAG = re.compile(r"^<[a-z]{2,3}-[A-Za-z]{2,4}>$")


def tokenizer(g: GGUF, nemo: Nemo, vocab: int) -> None:
    import sentencepiece as spm

    blob = nemo.file("tokenizer.model")
    if blob is None:
        raise SystemExit("the archive has no SentencePiece tokenizer")
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(blob)
    if sp.vocab_size() != vocab:
        raise SystemExit(f"tokenizer has {sp.vocab_size()} pieces, the model {vocab}")
    toks, scores, types = [], [], []
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        t = TOK_NORMAL
        if sp.is_unknown(i):
            t = TOK_UNKNOWN
        elif sp.is_control(i) or LANG_TAG.match(piece):
            t = TOK_CONTROL
        elif sp.is_unused(i):
            t = TOK_UNUSED
        elif sp.is_byte(i):
            t = TOK_BYTE
        toks.append(piece)
        scores.append(sp.get_score(i))
        types.append(t)
    toks.append("<blank>")
    scores.append(0.0)
    types.append(TOK_CONTROL)
    g.str("tokenizer.ggml.model", "bpe")
    g.strs("tokenizer.ggml.tokens", toks)
    g.f32s("tokenizer.ggml.scores", scores)
    g.i32s("tokenizer.ggml.token_type", types)
    for key, fn in (("unknown", sp.unk_id), ("bos", sp.bos_id), ("eos", sp.eos_id)):
        if fn() >= 0:
            g.u32(f"tokenizer.ggml.{key}_token_id", fn())
    g.u32("tokenizer.ggml.blank_token_id", vocab)


def prompt_settings(cfg: dict, sd: dict, d_model: int) -> dict | None:
    """The language prompt of NeMo's EncDecRNNTBPEModelWithPrompt
    (Nemotron 3.5): PromptStreamingMixin's prompt_kernel, Linear(D + P,
    2D), ReLU, Linear(2D, D), fed the encoder output and a one-hot
    language vector. Returns None for models without one."""
    md = cfg.get("model_defaults") or {}
    has = any(k.startswith("prompt_kernel.") for k in sd)
    if not has:
        if "Prompt" in str(cfg.get("target", "")) or md.get("initialize_prompt_feature"):
            raise SystemExit("the model asks for a language prompt but the checkpoint has no prompt_kernel")
        return None
    if not md.get("initialize_prompt_feature"):
        raise SystemExit("the checkpoint has a prompt_kernel that NeMo would not use (initialize_prompt_feature is off)")
    if "Hybrid" in str(cfg.get("target", "")):
        raise SystemExit("hybrid RNN-T and CTC prompt models are not supported")
    if int(md.get("enc_hidden", d_model)) != d_model:
        raise SystemExit(f"model_defaults.enc_hidden {md.get('enc_hidden')} is not the encoder's d_model {d_model}")
    n = int(md.get("num_prompts", 128))  # NeMo copies it over the top-level num_prompts
    want = {"prompt_kernel.0.weight": (2 * d_model, d_model + n), "prompt_kernel.0.bias": (2 * d_model,),
            "prompt_kernel.2.weight": (d_model, 2 * d_model), "prompt_kernel.2.bias": (d_model,)}
    for k, shape in want.items():
        if k not in sd or tuple(sd[k].shape) != shape:
            raise SystemExit(f"{k} should be {shape}, the checkpoint has {tuple(sd[k].shape) if k in sd else 'none'}")
    extra = sorted(k for k in sd if k.startswith("prompt_kernel.") and k not in want)
    if extra:
        raise SystemExit(f"the prompt network has layers this converter doesn't know: {extra}")
    table = md.get("prompt_dictionary") or {}
    if not table:
        raise SystemExit("the model has a language prompt but no prompt_dictionary")
    locales, indices = [], []
    for k, v in table.items():
        if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n:
            raise SystemExit(f"prompt_dictionary entry {k!r}: {v!r} is not a locale and an index under {n}")
        locales.append(k)
        indices.append(v)
    field = str((cfg.get("train_ds") or {}).get("prompt_field", "target_lang"))
    return {"num_prompts": n, "hidden": 2 * d_model, "field": field, "locales": locales, "indices": indices,
            "auto": table.get("auto")}


def convert_parakeet(nemo: Nemo, qtype: str, name: str) -> GGUF:
    cfg = nemo.cfg
    enc, dec, pre = cfg["encoder"], cfg["decoder"], cfg["preprocessor"]
    joint = cfg.get("joint") or {}
    jn = joint.get("jointnet") or {}
    pn = dec.get("prednet") or {}
    decoding = cfg.get("decoding") or {}
    target = str(cfg.get("target", ""))
    if "CTC" in target and "RNNT" not in target:
        raise SystemExit("CTC-only models are not supported")
    vocab = int(dec.get("vocab_size") or dec.get("num_classes"))
    durations = decoding.get("durations")
    head = "tdt" if durations else "rnnt"
    spk_layers = [int(x) for x in (cfg.get("spk_kernel_layers") or [])]

    g = GGUF("parakeet")
    identity(g, nemo, name, "parakeet-" + head, qtype)
    g.str("stt.variant", nemo.path.stem)
    (left, right), choices = att_context(enc.get("att_context_size"))
    style = str(enc.get("att_context_style", "regular"))
    if style not in ("regular", "chunked_limited"):
        raise SystemExit(f"attention style {style} is not supported")
    streaming = style == "chunked_limited" and left >= 0 and right >= 0
    prompt = prompt_settings(cfg, nemo.sd, int(enc["d_model"]))
    g.bool("stt.capability.translate", False)
    g.bool("stt.capability.lang_detect", bool(prompt and prompt["auto"] is not None))
    g.bool("stt.capability.streaming", streaming)
    g.str("stt.parakeet.head_kind", head)
    tokenizer(g, nemo, vocab)

    ns = "stt.parakeet.encoder"
    encoder_settings(g, enc, ns)
    k = int(enc["conv_kernel_size"])
    cl, cr = conv_context(enc.get("conv_context_size"), k)
    g.bool(f"{ns}.xscaling", bool(enc.get("xscaling", True)))
    g.i32(f"{ns}.att_context_left", left)
    g.i32(f"{ns}.att_context_right", right)
    g.str(f"{ns}.att_context_style", style)
    g.i32(f"{ns}.conv_context_left", cl)
    g.i32(f"{ns}.conv_context_right", cr)
    g.bool(f"{ns}.causal_downsampling", bool(enc.get("causal_downsampling", False)))
    if streaming:
        g.i32s(f"{ns}.att_context_size_choices", [v for lr in choices for v in lr])
        # NeMo's ConvSubsampling streaming settings: a cache of factor+1
        # mel frames, the first chunk one frame, and that cache's encoder
        # frames dropped from each later chunk.
        sub = int(enc["subsampling_factor"])
        cache = sub + 1
        g.i32(f"{ns}.streaming.pre_encode_cache_size", cache)
        g.i32(f"{ns}.streaming.drop_extra_pre_encoded", 1 + (cache - 1) // sub)
        g.i32(f"{ns}.streaming.sampling_frames_first", 1)

    e = Emitter(g, nemo.sd, qtype)
    sd = nemo.sd
    pred_hidden = int(pn.get("pred_hidden") or sd["decoder.prediction.embed.weight"].shape[1])
    n_lstm = int(pn.get("pred_rnn_layers") or 0)
    if not n_lstm:
        while f"decoder.prediction.dec_rnn.lstm.weight_ih_l{n_lstm}" in sd:
            n_lstm += 1
    extra = int(joint.get("num_extra_outputs") or jn.get("num_extra_outputs") or 0)
    if head == "tdt" and extra != len(durations):
        raise SystemExit(f"the joint has {extra} extra outputs for {len(durations)} durations")
    if head == "rnnt" and extra:
        raise SystemExit(f"the joint has {extra} extra outputs but the model lists no TDT durations")
    if spk_layers:
        g.str(f"{ns}.spk_kernel_type", str(cfg.get("spk_kernel_type", "ff")))
        g.i32s(f"{ns}.spk_kernel_layers", spk_layers)
        g.bool(f"{ns}.add_bg_spk_kernel", bool(cfg.get("add_bg_spk_kernel", False)))
    g.u32("stt.parakeet.predictor.hidden", pred_hidden)
    g.u32("stt.parakeet.predictor.n_layers", n_lstm)
    g.u32("stt.parakeet.predictor.vocab", vocab + 1)
    g.u32("stt.parakeet.joint.hidden", int(jn.get("joint_hidden") or joint.get("joint_hidden")))
    g.u32("stt.parakeet.joint.num_extra_outputs", extra)
    g.str("stt.parakeet.joint.activation", str(jn.get("activation") or joint.get("activation") or "relu").lower())
    if head == "tdt":
        g.i32s("stt.parakeet.tdt.durations", [int(x) for x in durations])
        g.u32("stt.parakeet.tdt.max_symbols", int((decoding.get("greedy") or {}).get("max_symbols") or 10))
    if prompt:
        p = "stt.parakeet.prompt."
        g.u32(p + "num_prompts", prompt["num_prompts"])
        g.u32(p + "hidden", prompt["hidden"])
        g.str(p + "field", prompt["field"])
        g.str(p + "activation", "relu")  # fixed in NeMo's PromptStreamingMixin
        g.strs(p + "dictionary.locales", prompt["locales"])
        g.i32s(p + "dictionary.indices", prompt["indices"])
        if prompt["auto"] is not None:
            g.u32(p + "auto_id", prompt["auto"])
    frontend(g, pre)

    use_bias = conformer_encoder(e, enc, "enc.blocks")
    g.bool(f"{ns}.use_bias", use_bias)
    e.move("decoder.prediction.embed.weight", "pred.embed.weight")
    for i in range(n_lstm):
        p = f"decoder.prediction.dec_rnn.lstm."
        e.move(f"{p}weight_ih_l{i}", f"pred.lstm.{i}.Wx", matrix=True)
        e.move(f"{p}weight_hh_l{i}", f"pred.lstm.{i}.Wh", matrix=True)
        e.put(f"pred.lstm.{i}.bias", e.arr(f"{p}bias_ih_l{i}") + e.arr(f"{p}bias_hh_l{i}"))
    for a, b in (("enc", "enc"), ("pred", "pred"), ("joint_net.2", "out")):
        e.move(f"joint.{a}.weight", f"joint.{b}.weight", matrix=True)
        e.move(f"joint.{a}.bias", f"joint.{b}.bias")
    if prompt:
        for j in ("0", "2"):
            e.move(f"prompt_kernel.{j}.weight", f"prompt.mlp.{j}.weight", matrix=True)
            e.move(f"prompt_kernel.{j}.bias", f"prompt.mlp.{j}.bias")
    for layer in spk_layers:
        for kind in ("spk_kernels", "bg_spk_kernels"):
            if kind == "bg_spk_kernels" and not cfg.get("add_bg_spk_kernel"):
                continue
            dst = "enc.spk_kernel" if kind == "spk_kernels" else "enc.bg_spk_kernel"
            for j in ("0", "3"):
                e.move(f"{kind}.{layer}.{j}.weight", f"{dst}.{layer}.{j}.weight", matrix=True)
                e.move(f"{kind}.{layer}.{j}.bias", f"{dst}.{layer}.{j}.bias")
    e.check_all_used(("preprocessor.", "ctc_decoder."))
    return g


# --------------------------------------------------------------- Sortformer


TRANSFORMER_BLOCK = [
    ("layer_norm_1", "norm_1", False),
    ("first_sub_layer.query_net", "attn.q", True),
    ("first_sub_layer.key_net", "attn.k", True),
    ("first_sub_layer.value_net", "attn.v", True),
    ("first_sub_layer.out_projection", "attn.out", True),
    ("layer_norm_2", "norm_2", False),
    ("second_sub_layer.dense_in", "ff.in", True),
    ("second_sub_layer.dense_out", "ff.out", True),
]

SORTFORMER_HEAD = [
    ("encoder_proj", "encoder_proj"),
    ("first_hidden_to_hidden", "fc1"),
    ("hidden_to_spks", "spk_head"),
    ("single_hidden_to_spks", "single_spk_head"),
]


def convert_sortformer(nemo: Nemo, qtype: str, name: str) -> GGUF:
    cfg = nemo.cfg
    enc, tf, pre, sm = cfg["encoder"], cfg["transformer_encoder"], cfg["preprocessor"], cfg["sortformer_modules"]
    g = GGUF("sortformer")
    identity(g, nemo, name, "sortformer", qtype)
    g.str("stt.variant", nemo.path.stem)
    frontend(g, pre)
    sr = int(pre["sample_rate"])
    g.bool("stt.capability.streaming", True)
    g.bool("stt.capability.speaker_diarization", True)
    g.u32("stt.sortformer.max_speakers", int(cfg.get("max_num_of_spks", 4)))
    g.u32("stt.sortformer.frame_hop", int(round(float(pre["window_stride"]) * sr)) * int(enc["subsampling_factor"]))
    # The Go Sortformer encoder is full-context: regular attention and
    # centered convolutions.
    if str(enc.get("att_context_style", "regular")) != "regular":
        raise SystemExit(f"Sortformer attention style {enc.get('att_context_style')} is not supported")
    k = int(enc["conv_kernel_size"])
    if conv_context(enc.get("conv_context_size"), k) != conv_context(None, k):
        raise SystemExit(f"Sortformer conv_context_size {enc.get('conv_context_size')} is not supported")
    ns = "stt.sortformer.encoder"
    encoder_settings(g, enc, ns)
    g.u32(f"{ns}.feat_in", int(enc["feat_in"]))
    g.bool(f"{ns}.xscaling", bool(enc.get("xscaling", True)))
    (left, right), _ = att_context(enc.get("att_context_size"))
    g.i32(f"{ns}.att_context_left", left)
    g.i32(f"{ns}.att_context_right", right)
    g.bool(f"{ns}.causal_downsampling", bool(enc.get("causal_downsampling", False)))
    g.u32("stt.sortformer.transformer.n_layers", int(tf["num_layers"]))
    g.u32("stt.sortformer.transformer.d_model", int(tf["hidden_size"]))
    g.u32("stt.sortformer.transformer.n_heads", int(tf["num_attention_heads"]))
    g.u32("stt.sortformer.transformer.d_ff", int(tf["inner_size"]))
    g.str("stt.sortformer.transformer.activation", str(tf.get("hidden_act", "relu")))
    g.bool("stt.sortformer.transformer.pre_ln", bool(tf.get("pre_ln", False)))
    for k in ("chunk_len", "spkcache_len", "fifo_len", "spkcache_update_period"):
        g.u32(f"stt.sortformer.stream.{k}", int(sm[k]))
    for k in ("chunk_left_context", "chunk_right_context", "spkcache_sil_frames_per_spk"):
        if k in sm:
            g.u32(f"stt.sortformer.stream.{k}", int(sm[k]))
    for k in ("pred_score_threshold", "scores_boost_latest", "sil_threshold", "strong_boost_rate",
              "weak_boost_rate", "min_pos_scores_rate"):
        if k in sm:
            g.f32(f"stt.sortformer.stream.{k}", float(sm[k]))

    e = Emitter(g, nemo.sd, qtype)
    conformer_encoder(e, enc, "enc.blocks")
    for i in range(int(tf["num_layers"])):
        for a, b, mat in TRANSFORMER_BLOCK:
            e.move(f"transformer_encoder.layers.{i}.{a}.weight", f"tf.blocks.{i}.{b}.weight", matrix=mat)
            e.move(f"transformer_encoder.layers.{i}.{a}.bias", f"tf.blocks.{i}.{b}.bias")
    for a, b in SORTFORMER_HEAD:
        e.move(f"sortformer_modules.{a}.weight", f"diar.{b}.weight")
        e.move(f"sortformer_modules.{a}.bias", f"diar.{b}.bias")
    e.check_all_used()
    return g


# ---------------------------------------------------------------- MarbleNet


def convert_marblenet(nemo: Nemo, qtype: str, name: str) -> GGUF:
    """MarbleNet frame VAD: a stack of 1-D separable convolution blocks
    (NeMo JasperBlock) and one linear layer giving speech and non-speech
    scores every 20 ms. Batch norms are folded into the convolutions here,
    so the file holds plain convolution weights and biases."""
    cfg = nemo.cfg
    enc, dec, pre = cfg["encoder"], cfg["decoder"], cfg["preprocessor"]
    if "ConvASREncoder" not in str(enc.get("_target_", "")):
        raise SystemExit("not a MarbleNet (ConvASREncoder) model")
    g = GGUF("marblenet")
    identity(g, nemo, name, "marblenet", "f32")
    g.str("stt.variant", nemo.path.stem)
    frontend(g, pre)
    act = str(enc.get("activation", "relu"))
    if act != "relu" or str(dec.get("activation", "relu")) != "relu":
        raise SystemExit("only ReLU MarbleNets are supported")
    labels = [str(x) for x in cfg.get("labels", ["0", "1"])]
    g.strs("stt.marblenet.labels", labels)
    e = Emitter(g, nemo.sd, "f32")

    def fold(conv_w: np.ndarray, bn: str) -> tuple[np.ndarray, np.ndarray]:
        w, b = e.arr(bn + ".weight"), e.arr(bn + ".bias")
        mean, var = e.arr(bn + ".running_mean"), e.arr(bn + ".running_var")
        s = w / np.sqrt(var + 1e-3)  # JasperBlock's BatchNorm1d uses eps=1e-3
        return (conv_w * s.reshape(-1, *([1] * (conv_w.ndim - 1)))).astype(np.float32), (b - mean * s).astype(np.float32)

    blocks = enc["jasper"]
    g.u32("stt.marblenet.n_blocks", len(blocks))
    feat = int(enc["feat_in"])
    for i, blk in enumerate(blocks):
        reps = int(blk.get("repeat", 1))
        k, st, dil = int(blk["kernel"][0]), int(blk["stride"][0]), int(blk["dilation"][0])
        if float(blk.get("kernel_size_factor", 1.0)) != 1.0:
            raise SystemExit("kernel_size_factor other than 1 is not supported")
        sep = bool(blk.get("separable", False))
        p = f"stt.marblenet.block.{i}."
        g.u32(p + "filters", int(blk["filters"]))
        g.u32(p + "repeat", reps)
        g.u32(p + "kernel", k)
        g.u32(p + "stride", st)
        g.u32(p + "dilation", dil)
        g.bool(p + "residual", bool(blk.get("residual", False)))
        g.bool(p + "separable", sep)
        src = f"encoder.encoder.{i}."
        j = 0
        for r in range(reps):
            dst = f"vad.blocks.{i}.{r}."
            if sep:
                e.move(f"{src}mconv.{j}.conv.weight", dst + "dw.weight")  # [in, 1, k]
                pw = e.arr(f"{src}mconv.{j + 1}.conv.weight")  # [out, in, 1]
                w, b = fold(pw[:, :, 0], f"{src}mconv.{j + 2}")
                j += 3
            else:
                cw = e.arr(f"{src}mconv.{j}.conv.weight")  # [out, in, k]
                w, b = fold(cw, f"{src}mconv.{j + 1}")
                j += 2
            e.put(dst + "pw.weight", w)
            e.put(dst + "pw.bias", b)
            j += 2  # activation and dropout between repeats
        if blk.get("residual"):
            rw = e.arr(f"{src}res.0.0.conv.weight")
            w, b = fold(rw[:, :, 0], f"{src}res.0.1")
            e.put(f"vad.blocks.{i}.res.weight", w)
            e.put(f"vad.blocks.{i}.res.bias", b)
        feat = int(blk["filters"])
    n = int(dec.get("num_layers", 1))
    for i in range(n):
        e.move(f"decoder.layer{i}.weight", f"vad.dec.{i}.weight")
        e.move(f"decoder.layer{i}.bias", f"vad.dec.{i}.bias")
    g.u32("stt.marblenet.decoder_layers", n)
    e.check_all_used()
    return g


# --------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nemo", type=Path, help="NVIDIA .nemo file")
    ap.add_argument("--type", choices=["f32", "f16", "q8_0"], default="q8_0")
    ap.add_argument("--out", type=Path, help="output file (default: beside the .nemo)")
    ap.add_argument("--name", help="display name stored in the file")
    a = ap.parse_args(argv)
    nemo = Nemo(a.nemo)
    target = str(nemo.cfg.get("target", ""))
    name = a.name or a.nemo.stem
    kind = a.type
    if "Sortformer" in target:
        g = convert_sortformer(nemo, a.type, name)
    elif "FrameClassification" in target or "ConvASREncoder" in str(nemo.cfg.get("encoder", {}).get("_target_", "")):
        g = convert_marblenet(nemo, a.type, name)
        kind = "f32"  # MarbleNet is tiny and stays f32
    elif "RNNT" in target or "Multitalker" in target or "Transducer" in target:
        g = convert_parakeet(nemo, a.type, name)
    else:
        raise SystemExit(f"model type {target!r} is not supported")
    out = a.out or a.nemo.with_name(f"{a.nemo.stem}-{kind.upper()}.gguf")
    digest = g.write(out)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    print(f"source {a.nemo.name} sha256 {nemo.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
