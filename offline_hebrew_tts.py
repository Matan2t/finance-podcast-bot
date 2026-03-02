import json
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OfflineHebrewTTSConfig:
    # Hebrew -> IPA phonemes ONNX model (snapshot download)
    g2p_model_id: str = "thewh1teagle/gemma3-270b-heb-g2p"
    g2p_local_dirname: str = "gemma3_onnx"

    # Piper ONNX voice model/config (file download)
    voice_repo_id: str = "thewh1teagle/phonikud-tts-checkpoints"
    voice_model_filename: str = "shaul.onnx"
    voice_config_filename: str = "model.config.json"

    # Phonikud (public) Hebrew text -> IPA phonemes (offline)
    phonikud_repo_id: str = "thewh1teagle/phonikud-onnx"
    phonikud_model_filenames: tuple[str, ...] = ("phonikud-1.0.int8.onnx", "phonikud-onnx-1.0.int8.onnx")

    # Cache root (HuggingFace + downloaded files)
    cache_dir: str = ".cache/offline_hebrew_tts"

    # Output file format (MoviePy can read WAV reliably)
    output_suffix: str = ".wav"

    # Speaking rate (Piper): higher => slower, lower => faster
    default_length_scale: float = 1.25


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_cache_root(cfg: OfflineHebrewTTSConfig) -> pathlib.Path:
    return pathlib.Path(cfg.cache_dir).expanduser().resolve()


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return None
    token = token.strip()
    if not token or token.upper().startswith("DUMMY"):
        return None
    # Most HF user access tokens look like "hf_...". If it's not, treat it as unset.
    if not token.startswith("hf_"):
        return None
    return token


def _debug_enabled() -> bool:
    return _is_truthy_env("OFFLINE_HEBREW_TTS_DEBUG")


def _hf() -> Any:
    try:
        import huggingface_hub  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: huggingface_hub") from e
    return huggingface_hub


def _download_file(repo_id: str, filename: str, dest_dir: pathlib.Path) -> pathlib.Path:
    hf = _hf()
    downloaded = hf.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        token=_hf_token(),
    )
    return pathlib.Path(downloaded)


def _download_voice_files(cfg: OfflineHebrewTTSConfig, cache_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    voice_model_path = cache_root / cfg.voice_model_filename
    voice_config_path = cache_root / cfg.voice_config_filename

    if not voice_model_path.exists():
        voice_model_path = _download_file(cfg.voice_repo_id, cfg.voice_model_filename, cache_root)
    if not voice_config_path.exists():
        voice_config_path = _download_file(cfg.voice_repo_id, cfg.voice_config_filename, cache_root)

    return voice_model_path, voice_config_path


def _download_models(cfg: OfflineHebrewTTSConfig) -> tuple[pathlib.Path | None, pathlib.Path, pathlib.Path]:
    """
    Returns:
      g2p_dir (or None if unavailable), voice_model_path, voice_config_path
    """
    cache_root = _get_cache_root(cfg)
    _ensure_dir(cache_root)
    token = _hf_token()

    g2p_unavailable_marker = cache_root / ".g2p_unavailable"
    g2p_dir: pathlib.Path | None = cache_root / cfg.g2p_local_dirname
    if g2p_dir.exists():
        # Use existing local snapshot if present (even if the HF repo is gated).
        pass
    else:
        # If we previously detected the repo is gated/unavailable and we still don't have a token,
        # avoid retrying on every run.
        if token is None and g2p_unavailable_marker.exists():
            g2p_dir = None
        else:
            try:
                _hf().snapshot_download(
                    repo_id=cfg.g2p_model_id,
                    local_dir=str(g2p_dir),
                    token=token,
                )
            except Exception as e:
                if _debug_enabled():
                    print(f"OfflineHebrewTTS: G2P model unavailable ({e}); falling back to Phonikud/Piper.")
                # Repo can be gated/private; allow the rest of the pipeline to proceed
                # (voice files are public) and fall back to Piper's built-in phonemizer.
                g2p_dir = None
                if token is None:
                    try:
                        g2p_unavailable_marker.write_text("unavailable\n", encoding="utf-8")
                    except OSError:
                        pass

    voice_model_path, voice_config_path = _download_voice_files(cfg, cache_root)

    return g2p_dir, voice_model_path, voice_config_path


def _hebrew_to_phonemes(text: str, g2p_dir: pathlib.Path) -> str:
    """
    Hebrew -> IPA phonemes, using the community ONNX model (optimum-onnx + transformers).
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: transformers and optimum.onnxruntime") from e

    tokenizer = AutoTokenizer.from_pretrained(str(g2p_dir))
    model = ORTModelForCausalLM.from_pretrained(str(g2p_dir))

    system_message = (
        "Given the following Hebrew sentence, convert it to IPA phonemes.\n"
        "Input Format: A Hebrew sentence.\n"
        "Output Format: A string of IPA phonemes.\n"
    )
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]

    formatted_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95,
        top_k=64,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids([" ", " "]),
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if " model" in response:
        response = response.split(" model")[-1].strip()
    for end_token in [" ", " ", " "]:
        response = response.replace(end_token, "")

    return response.strip()


def _phonemes_to_wav(phonemes: str, voice_model_path: pathlib.Path, voice_config_path: pathlib.Path, out_wav_path: pathlib.Path) -> None:
    """
    IPA phonemes -> audio WAV using piper-onnx.
    """
    try:
        from piper_onnx import Piper  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: piper-onnx and soundfile") from e

    tts = Piper(str(voice_model_path), str(voice_config_path))
    samples, sample_rate = tts.create(phonemes, is_phonemes=True)
    sf.write(str(out_wav_path), samples, sample_rate)


def _env_length_scale() -> float | None:
    """
    Optional runtime control:
      OFFLINE_HEBREW_TTS_LENGTH_SCALE=1.25  (higher => slower)
    """
    raw = os.environ.get("OFFLINE_HEBREW_TTS_LENGTH_SCALE", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    if val <= 0:
        return None
    return val


def _voice_config_with_length_scale(voice_config_path: pathlib.Path, length_scale: float, cache_root: pathlib.Path) -> pathlib.Path:
    """
    Create a temporary config JSON with overridden inference.length_scale.
    """
    data: dict[str, Any] = json.loads(voice_config_path.read_text(encoding="utf-8"))
    inference = data.get("inference") or {}
    inference["length_scale"] = float(length_scale)
    data["inference"] = inference

    tmp_path = cache_root / f"model.config.length_scale_{length_scale:.3f}.json"
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tmp_path


def _resolve_voice_config_path(
    voice_config_path: pathlib.Path,
    cache_root: pathlib.Path,
    length_scale: float | None,
    cfg: OfflineHebrewTTSConfig,
) -> pathlib.Path:
    if length_scale is None:
        length_scale = _env_length_scale()
    if length_scale is None:
        # Default to slightly slower speech without requiring env vars.
        length_scale = cfg.default_length_scale
    if length_scale <= 0:
        return voice_config_path
    if abs(length_scale - 1.0) < 1e-9:
        return voice_config_path
    return _voice_config_with_length_scale(voice_config_path, length_scale, cache_root)


def _ensure_phonikud_model(cfg: OfflineHebrewTTSConfig, cache_root: pathlib.Path) -> pathlib.Path:
    """
    Download Phonikud ONNX model (public) used to add diacritics.
    """
    for filename in cfg.phonikud_model_filenames:
        dest = cache_root / filename
        if dest.exists():
            return dest
        try:
            return _download_file(cfg.phonikud_repo_id, filename, cache_root)
        except Exception:
            continue
    raise RuntimeError(f"Unable to download Phonikud model from {cfg.phonikud_repo_id}")


def _text_to_phonemes_via_phonikud(text: str, cfg: OfflineHebrewTTSConfig, cache_root: pathlib.Path) -> str:
    """
    Hebrew text -> IPA phonemes using Phonikud:
      text -> (add diacritics via ONNX) -> phonemize(diacritics) -> IPA-ish phonemes string
    """
    try:
        from phonikud_onnx import Phonikud  # type: ignore
        from phonikud import phonemize  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: phonikud and phonikud-onnx") from e

    model_path = _ensure_phonikud_model(cfg, cache_root)
    phonikud = Phonikud(str(model_path))
    vocalized = phonikud.add_diacritics(text)
    phonemes = phonemize(vocalized)
    return phonemes.strip()


def _text_to_wav_via_piper(text: str, voice_model_path: pathlib.Path, voice_config_path: pathlib.Path, out_wav_path: pathlib.Path) -> None:
    """
    Direct text -> audio using Piper's built-in phonemizer (espeak-ng).

    This is a fallback when the Hebrew G2P model is not accessible (e.g., gated HF repo).
    """
    try:
        from piper_onnx import Piper  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: piper-onnx and soundfile") from e

    tts = Piper(str(voice_model_path), str(voice_config_path))
    samples, sample_rate = tts.create(text, is_phonemes=False)
    sf.write(str(out_wav_path), samples, sample_rate)


def synthesize_hebrew_audio(text: str, cfg: OfflineHebrewTTSConfig | None = None, *, length_scale: float | None = None) -> str:
    """
    Offline Hebrew text-to-speech using a community pipeline.

    - Downloads/caches models on first run
    - Returns path to a generated audio file (WAV by default)
    """
    cfg = cfg or OfflineHebrewTTSConfig()
    g2p_dir, voice_model_path, voice_config_path = _download_models(cfg)
    cache_root = _get_cache_root(cfg)
    voice_config_to_use = _resolve_voice_config_path(voice_config_path, cache_root, length_scale, cfg)

    fd, out_path = tempfile.mkstemp(suffix=cfg.output_suffix)
    os.close(fd)
    out = pathlib.Path(out_path)
    if g2p_dir is not None:
        try:
            phonemes = _hebrew_to_phonemes(text, g2p_dir)
            _phonemes_to_wav(phonemes, voice_model_path, voice_config_to_use, out)
            return str(out)
        except Exception:
            pass

    # If the (best-quality) ONNX LLM G2P model is gated/unavailable, fall back to Phonikud
    # to generate the phoneme alphabet this Piper voice expects.
    try:
        phonemes = _text_to_phonemes_via_phonikud(text, cfg, cache_root)
        if _debug_enabled():
            print(f"OfflineHebrewTTS: using Phonikud phonemes: {phonemes[:120]}{'...' if len(phonemes) > 120 else ''}")
        _phonemes_to_wav(phonemes, voice_model_path, voice_config_to_use, out)
        return str(out)
    except Exception:
        # Last-resort fallback: Piper's built-in phonemizer (may be less accurate for this voice).
        _text_to_wav_via_piper(text, voice_model_path, voice_config_to_use, out)
    return str(out)

