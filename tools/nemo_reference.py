#!/usr/bin/env python3
"""Make NVIDIA NeMo reference outputs for the conformer/ parity tests.

This is a developer tool, not part of the engine: it needs NVIDIA NeMo
(pip install "nemo_toolkit[asr]") and the .nemo originals. It runs each
model in float32 on the CPU and saves what the Go tests compare against,
into the folder GORELAXOR_CONFORMER_REF names:

    python nemo_reference.py asr nemotron-speech-streaming-en-0.6b.nemo jfk.wav REF/nemo-nemotron --stream 13 6 1 0
    python nemo_reference.py asr parakeet-tdt-0.6b-v2.nemo jfk.wav REF/nemo-parakeet
    python nemo_reference.py vad frame_vad_multilingual_marblenet_v2.0.nemo jfk.wav REF/nemo-vad
    python nemo_reference.py sortformer diar_streaming_sortformer_4spk-v2.1.nemo REF/nemo-sortformer a.wav b.wav c.wav
    python nemo_reference.py multitalker multitalker-parakeet-streaming-0.6b-v1.nemo \\
        diar_streaming_sortformer_4spk-v2.1.nemo multitalker-2spk-mix.wav REF/nemo-multitalker

Nemotron 3.5 takes a language: each clip once with its locale and once
with auto (the default), streaming at R = 13 and 3:

    for c in jfk:en-US fleurs-de:de-DE fleurs-es:es-US fleurs-fr:fr-FR \\
             fleurs-ja:ja-JP fleurs-zh:zh-CN ko-short:ko-KR; do
      python nemo_reference.py asr nemotron-3.5-asr-streaming-0.6b.nemo ${c%:*}.wav \\
          REF/nemo-nemotron35-${c%:*} --lang ${c#*:} --stream 13 3
      python nemo_reference.py asr nemotron-3.5-asr-streaming-0.6b.nemo ${c%:*}.wav \\
          REF/nemo-nemotron35-${c%:*}-auto --stream 13 3
    done

The multitalker step needs multitalker_transcript_config.py from the
model's Hugging Face page beside this script.

Each tensor is saved as NAME.f32 (raw little-endian float32, row-major)
with NAME.json giving its shape. The model files the tests load go in REF
too: nemotron.gguf, nemotron35.gguf and parakeet.gguf (convert_nemo.py
--type f32), sortformer.gguf, vad.gguf, and the Q8_0 files
nemotron-q8.gguf, nemotron35-q8.gguf, parakeet-q8.gguf and
multitalker-q8.gguf, with the .wav clips above.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def save(out: Path, name: str, arr) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    arr.tofile(out / f"{name}.f32")
    (out / f"{name}.json").write_text(json.dumps({"name": name, "shape": list(arr.shape), "source": {"kind": "nemo"}}))


def load_wav(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: want 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def restore(cls, path):
    """Loads a .nemo file, or a folder it was extracted to (NeMo unpacks
    the archive to a temporary folder otherwise, which needs the disk
    space of the weights again)."""
    if Path(path).is_dir():
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

        class Mapped(SaveRestoreConnector):
            # the weights stay in the file until copied into the model,
            # so they are in memory once, not twice
            @staticmethod
            def _load_state_dict_from_disk(model_weights, map_location="cpu"):
                return torch.load(model_weights, map_location=map_location, mmap=True)

        c = Mapped()
        c.model_extracted_dir = str(Path(path).resolve())
        return cls.restore_from(str(path), map_location="cpu", save_restore_connector=c).eval()
    return cls.restore_from(path, map_location="cpu").eval()


def asr(a) -> None:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pcm = load_wav(a.wav)
    m = restore(nemo_asr.models.ASRModel, a.nemo)
    m.preprocessor.featurizer.dither = 0.0
    sig, ln = torch.from_numpy(pcm)[None], torch.tensor([len(pcm)])
    mel, ml = m.preprocessor(input_signal=sig, length=ln)
    save(out, "mel", mel[0].numpy())
    enc, el = m.encoder(audio_signal=mel, length=ml)
    save(out, "enc", enc[0, :, : int(el[0])].T.numpy())
    # Prompt models (Nemotron 3.5) feed the decoder the encoder output
    # after the language prompt network: the model's own forward.
    lang = None
    if getattr(m, "concat", False):
        lang = a.lang or "auto"
        pid = int(m.cfg.model_defaults.prompt_dictionary[lang])
        enc, el = m.forward(processed_signal=mel, processed_signal_length=ml, prompt_indices=torch.tensor([pid]))
        save(out, "prompted", enc[0, :, : int(el[0])].T.numpy())
        m.set_inference_prompt(lang)  # for conformer_stream_step below
    elif a.lang:
        raise SystemExit("this model has no language prompt")
    hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=el, return_hypotheses=True)
    h = hyp[0][0] if isinstance(hyp, tuple) else hyp[0]
    h = h[0] if isinstance(h, list) else h
    ts = h.timestamp.tolist() if hasattr(h.timestamp, "tolist") else list(h.timestamp or [])
    toks = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
    ref = {"text": h.text, "tokens": toks, "timestamps": [int(t) for t in ts]}
    if lang:
        ref["lang"] = lang
    (out / "text.json").write_text(json.dumps(ref))
    print(h.text)
    left = {R: L for L, R in m.encoder.att_context_size_all}
    for R in a.stream:
        m.encoder.set_default_att_context_size([left[R], R])
        m.encoder.setup_streaming_params()
        cfg = m.encoder.streaming_cfg
        encs = []
        buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio(pcm)
        c0, c1, c2 = m.encoder.get_initial_cache_state(batch_size=1)
        for i, (chunk, clen) in enumerate(iter(buf)):
            drop = 0 if i == 0 else cfg.drop_extra_pre_encoded
            e, elen, c0, c1, c2 = m.encoder.cache_aware_stream_step(
                processed_signal=chunk, processed_signal_length=clen, cache_last_channel=c0, cache_last_time=c1,
                cache_last_channel_len=c2, keep_all_outputs=buf.is_buffer_empty(), drop_extra_pre_encoded=drop)
            encs.append(e[0, :, : int(elen[0])].T.numpy())
        save(out, f"stream{R}.enc", np.concatenate(encs, 0))
        buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio(pcm)
        c0, c1, c2 = m.encoder.get_initial_cache_state(batch_size=1)
        prev = pred = texts = None
        for i, (chunk, clen) in enumerate(iter(buf)):
            drop = 0 if i == 0 else cfg.drop_extra_pre_encoded
            pred, texts, c0, c1, c2, prev = m.conformer_stream_step(
                processed_signal=chunk, processed_signal_length=clen, cache_last_channel=c0, cache_last_time=c1,
                cache_last_channel_len=c2, keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev,
                previous_pred_out=pred, drop_extra_pre_encoded=drop, return_transcription=True)[:6]
        t = texts[0].text if hasattr(texts[0], "text") else texts[0]
        (out / f"stream{R}.json").write_text(json.dumps({"text": t}))
        print(R, t)


def vad(a) -> None:
    from nemo.collections.asr.models import EncDecFrameClassificationModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # quiet noise either side of the speech, so there is silence to find
    rng = np.random.default_rng(7)
    pcm = np.concatenate([rng.normal(0, 0.002, 24000), load_wav(a.wav), rng.normal(0, 0.002, 16000)]).astype(np.float32)
    save(out, "pcm", pcm)
    m = EncDecFrameClassificationModel.restore_from(a.nemo, map_location="cpu", strict=False).eval()
    m.preprocessor.featurizer.dither = 0.0
    sig, ln = torch.from_numpy(pcm)[None], torch.tensor([len(pcm)])
    mel, _ = m.preprocessor(input_signal=sig, length=ln)
    save(out, "mel", mel[0].numpy())
    logits = m.forward(input_signal=sig, input_signal_length=ln)
    logits = logits[0] if isinstance(logits, tuple) else logits
    save(out, "logits", logits[0].numpy())
    save(out, "prob", torch.softmax(logits[0], -1)[:, 1].numpy())


# "high" is NVIDIA's very-high-latency preset (conformer.DiarVeryHighLatency);
# the name stays so existing reference folders keep working.
PRESETS = {"high": (340, 40, 40, 300, 188), "low": (6, 7, 188, 144, 188)}


def sortformer(a) -> None:
    from nemo.collections.asr.models import SortformerEncLabelModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pcm = np.concatenate([load_wav(w) for w in a.wavs])
    save(out, "pcm", pcm)
    m = SortformerEncLabelModel.restore_from(a.nemo, map_location="cpu", strict=False).eval()
    m.preprocessor.featurizer.dither = 0.0
    sig, ln = torch.from_numpy(pcm)[None], torch.tensor([len(pcm)])
    for name, (ch, rc, fifo, upd, cache) in PRESETS.items():
        sm = m.sortformer_modules
        sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len = ch, rc, fifo, upd, cache
        sm._check_streaming_parameters()
        save(out, f"preds.{name}", m.forward(audio_signal=sig, audio_signal_length=ln)[0].numpy())


def multitalker(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from multitalker_transcript_config import MultitalkerTranscriptionConfig
    from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
    from nemo.collections.asr.parts.utils import multispk_transcribe_utils as mu
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from omegaconf import OmegaConf

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    diar = SortformerEncLabelModel.restore_from(a.diar, map_location="cpu", strict=False).eval()
    asr_m = ASRModel.restore_from(a.nemo, map_location="cpu").eval()
    asr_m.preprocessor.featurizer.dither = 0.0
    diar.preprocessor.featurizer.dither = 0.0
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.audio_file, cfg.output_path, cfg.log, cfg.device = a.wav, str(out / "seglst.json"), False, "cpu"
    diar = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar)
    buf = CacheAwareStreamingAudioBuffer(model=asr_m, online_normalization=cfg.online_normalization,
                                         pad_and_drop_preencoded=cfg.pad_and_drop_preencoded)
    buf.append_audio_file(audio_filepath=a.wav, stream_id=-1)
    streamer = mu.SpeakerTaggedASR(cfg, asr_m, diar)
    # NVIDIA's example runs under autocast; float32 keeps the reference exact
    for step, (chunk, clen) in enumerate(iter(buf)):
        drop = 0 if step == 0 else asr_m.encoder.streaming_cfg.drop_extra_pre_encoded
        streamer.perform_parallel_streaming_stt_spk(step_num=step, chunk_audio=chunk, chunk_lengths=clen,
                                                    is_buffer_empty=buf.is_buffer_empty(), drop_extra_pre_encoded=drop)
    st = streamer.instance_manager.batch_asr_states[0]
    texts = {str(s): (None if st.previous_hypothesis[s] is None else st.previous_hypothesis[s].text) for s in st.get_speakers()}
    save(out, "diar", streamer.instance_manager.diar_states.diar_pred_out_stream[0].numpy())
    (out / "result.json").write_text(json.dumps({"texts": texts}))
    print(texts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("asr")
    p.add_argument("nemo", help=".nemo file or the folder it was extracted to")
    p.add_argument("wav"), p.add_argument("out")
    p.add_argument("--stream", type=int, nargs="*", default=[], help="right contexts to stream with")
    p.add_argument("--lang", help="prompt models: a prompt_dictionary locale such as de-DE (default auto)")
    p = sub.add_parser("vad")
    p.add_argument("nemo"), p.add_argument("wav"), p.add_argument("out")
    p = sub.add_parser("sortformer")
    p.add_argument("nemo"), p.add_argument("out"), p.add_argument("wavs", nargs="+")
    p = sub.add_parser("multitalker")
    p.add_argument("nemo"), p.add_argument("diar"), p.add_argument("wav"), p.add_argument("out")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"asr": asr, "vad": vad, "sortformer": sortformer, "multitalker": multitalker}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
