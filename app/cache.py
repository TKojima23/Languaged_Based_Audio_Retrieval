import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import (
    ART_DIR,
    CKPT_PATH,
    ENABLE_FUSION,
    HOP_SEC,
    VAD_FRAME_SEC,
    VAD_HOP_SEC,
    VAD_MARGIN_SEC,
    VAD_MAX_SEG_SEC,
    VAD_MIN_DUR_SEC,
    VAD_THRESHOLD_DB,
    WINDOW_SEC,
)
from .models import SliceSpec


def collect_wav_files(audio_dir: str) -> list[str]:
    p = Path(audio_dir)
    if not p.exists():
        raise FileNotFoundError(f"Audio dir not found: {p.resolve()}")
    wavs = sorted(str(x) for x in p.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No wav files found under: {p.resolve()}")
    return wavs


def signature_for_files(paths: list[str]) -> str:
    h = hashlib.sha1()
    for s in paths:
        p = Path(s)
        st = p.stat()
        h.update(s.replace("\\", "/").encode())
        h.update(b"|")
        h.update(str(st.st_size).encode())
        h.update(b"|")
        h.update(str(st.st_mtime_ns).encode())
        h.update(b"\n")
    return h.hexdigest()


def cache_stem(mode: str = "window", vad_params: dict | None = None) -> str:
    ckpt = Path(CKPT_PATH).stem
    if mode == "window":
        w = f"{WINDOW_SEC:g}".replace(".", "p")
        h = f"{HOP_SEC:g}".replace(".", "p")
        return f"clotho_slices__{ckpt}__w{w}s_h{h}s__fusion{int(ENABLE_FUSION)}"

    p = vad_params or {}
    fr = f"{p.get('frame_sec', VAD_FRAME_SEC):g}".replace(".", "p")
    thr = f"{p.get('threshold_db', VAD_THRESHOLD_DB):g}".replace(".", "p").replace("-", "n")
    mg = f"{p.get('margin_sec', VAD_MARGIN_SEC):g}".replace(".", "p")
    mn = f"{p.get('min_dur_sec', VAD_MIN_DUR_SEC):g}".replace(".", "p")
    mx = f"{p.get('max_seg_sec', VAD_MAX_SEG_SEC):g}".replace(".", "p")
    hp = f"{p.get('hop_sec', VAD_HOP_SEC):g}".replace(".", "p")
    return (
        f"clotho_slices__{ckpt}__vad"
        f"_fr{fr}_thr{thr}_mg{mg}_mn{mn}_mx{mx}_hp{hp}__fusion{int(ENABLE_FUSION)}"
    )


def cache_paths(mode: str = "window", vad_params: dict | None = None):
    stem = cache_stem(mode, vad_params)
    emb_path = ART_DIR / f"{stem}__emb.npy"
    meta_path = ART_DIR / f"{stem}__meta.json"
    man_path = ART_DIR / f"{stem}__manifest.json"
    return emb_path, meta_path, man_path


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def _slices_to_meta(slices: list[SliceSpec]) -> list[dict]:
    return [
        {"path": s.path, "start": s.start_sec, "end": s.end_sec, "mode": s.mode}
        for s in slices
    ]


def _meta_to_slices(meta: list[dict]) -> list[SliceSpec]:
    return [
        SliceSpec(
            path=m["path"],
            start_sec=m["start"],
            end_sec=m["end"],
            mode=m.get("mode", "window"),
        )
        for m in meta
    ]


def load_cache_if_valid(
    wavs: list[str],
    emb_path: Path,
    meta_path: Path,
    man_path: Path,
    mode: str = "window",
    vad_params: dict | None = None,
    mmap_mode: str | None = None,
):
    if not (emb_path.exists() and meta_path.exists() and man_path.exists()):
        return None

    man = json.loads(man_path.read_text(encoding="utf-8"))
    if (
        man.get("ckpt_path") != CKPT_PATH
        or man.get("enable_fusion") != ENABLE_FUSION
        or man.get("slice_mode") != mode
    ):
        return "MISMATCH_PARAMS"

    if mode == "window":
        if man.get("window_sec") != WINDOW_SEC or man.get("hop_sec") != HOP_SEC:
            return "MISMATCH_PARAMS"
    else:
        p = vad_params or {}
        for key, default in [
            ("frame_sec", VAD_FRAME_SEC),
            ("threshold_db", VAD_THRESHOLD_DB),
            ("margin_sec", VAD_MARGIN_SEC),
            ("min_dur_sec", VAD_MIN_DUR_SEC),
            ("max_seg_sec", VAD_MAX_SEG_SEC),
            ("hop_sec", VAD_HOP_SEC),
        ]:
            if man.get(f"vad_{key}") != p.get(key, default):
                return "MISMATCH_PARAMS"

    if man.get("files_signature") != signature_for_files(wavs):
        return "MISMATCH_FILES"

    emb = np.load(emb_path, mmap_mode=mmap_mode)
    slices = _meta_to_slices(json.loads(meta_path.read_text(encoding="utf-8")))
    return emb, slices


def save_cache(
    wavs: list[str],
    emb: np.ndarray,
    slices: list[SliceSpec],
    emb_path: Path,
    meta_path: Path,
    man_path: Path,
    mode: str = "window",
    vad_params: dict | None = None,
):
    np.save(emb_path, emb.astype(np.float32))
    meta_path.write_text(
        json.dumps(_slices_to_meta(slices), ensure_ascii=False), encoding="utf-8"
    )

    p = vad_params or {}
    man: dict = {
        "ckpt_path": CKPT_PATH,
        "enable_fusion": ENABLE_FUSION,
        "slice_mode": mode,
        "files_signature": signature_for_files(wavs),
        "num_slices": len(slices),
        "embed_shape": list(emb.shape),
    }
    if mode == "window":
        man["window_sec"] = WINDOW_SEC
        man["hop_sec"] = HOP_SEC
    else:
        for key, default in [
            ("frame_sec", VAD_FRAME_SEC),
            ("threshold_db", VAD_THRESHOLD_DB),
            ("margin_sec", VAD_MARGIN_SEC),
            ("min_dur_sec", VAD_MIN_DUR_SEC),
            ("max_seg_sec", VAD_MAX_SEG_SEC),
            ("hop_sec", VAD_HOP_SEC),
        ]:
            man[f"vad_{key}"] = p.get(key, default)

    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


