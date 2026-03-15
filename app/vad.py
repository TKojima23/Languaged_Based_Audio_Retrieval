import numpy as np

from .audio import read_audio
from .config import (
    VAD_FRAME_SEC,
    VAD_HOP_SEC,
    VAD_MARGIN_SEC,
    VAD_MAX_SEG_SEC,
    VAD_MIN_DUR_SEC,
    VAD_THRESHOLD_DB,
)
from .models import SliceSpec


def compute_vad_segments(
    y: np.ndarray,
    sr: int,
    frame_sec: float = VAD_FRAME_SEC,
    threshold_db: float = VAD_THRESHOLD_DB,
    min_dur_sec: float = VAD_MIN_DUR_SEC,
    margin_sec: float = VAD_MARGIN_SEC,
) -> list[tuple[float, float]]:
    frame_len = max(1, int(sr * frame_sec))
    n_frames = int(np.ceil(len(y) / frame_len))

    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = y[i * frame_len : (i + 1) * frame_len]
        if len(chunk) > 0:
            rms[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    is_voiced = db >= threshold_db

    def frame_to_sec(idx: int) -> float:
        return idx * frame_sec

    segments: list[tuple[float, float]] = []
    in_voiced = False
    seg_start = 0
    for i, voiced in enumerate(is_voiced):
        if voiced and not in_voiced:
            seg_start = i
            in_voiced = True
        elif not voiced and in_voiced:
            segments.append((frame_to_sec(seg_start), frame_to_sec(i)))
            in_voiced = False
    if in_voiced:
        segments.append((frame_to_sec(seg_start), frame_to_sec(n_frames)))

    total_dur = len(y) / sr
    expanded: list[tuple[float, float]] = []
    for s, e in segments:
        expanded.append((max(0.0, s - margin_sec), min(total_dur, e + margin_sec)))

    if not expanded:
        return []
    expanded.sort(key=lambda x: x[0])

    merged: list[tuple[float, float]] = [expanded[0]]
    for s, e in expanded[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if (e - s) >= min_dur_sec]


def make_vad_slices(
    path: str,
    max_seg_sec: float = VAD_MAX_SEG_SEC,
    hop_sec: float = VAD_HOP_SEC,
    frame_sec: float = VAD_FRAME_SEC,
    threshold_db: float = VAD_THRESHOLD_DB,
    min_dur_sec: float = VAD_MIN_DUR_SEC,
    margin_sec: float = VAD_MARGIN_SEC,
) -> list[SliceSpec]:
    y, sr = read_audio(path)
    vad_segs = compute_vad_segments(y, sr, frame_sec, threshold_db, min_dur_sec, margin_sec)

    slices: list[SliceSpec] = []
    for seg_start, seg_end in vad_segs:
        if seg_end - seg_start <= max_seg_sec:
            slices.append(
                SliceSpec(path=path, start_sec=seg_start, end_sec=seg_end, mode="vad")
            )
            continue

        t = seg_start
        while t < seg_end:
            end = min(t + max_seg_sec, seg_end)
            slices.append(SliceSpec(path=path, start_sec=t, end_sec=end, mode="vad"))
            t += hop_sec
            if t >= seg_end:
                break

    return slices


def load_vad_segments_for_display(
    path: str,
    vad_params: dict | None = None,
) -> list[tuple[float, float]]:
    p = vad_params or {}
    y, sr = read_audio(path)
    return compute_vad_segments(
        y,
        sr,
        frame_sec=p.get("frame_sec", VAD_FRAME_SEC),
        threshold_db=p.get("threshold_db", VAD_THRESHOLD_DB),
        min_dur_sec=p.get("min_dur_sec", VAD_MIN_DUR_SEC),
        margin_sec=p.get("margin_sec", VAD_MARGIN_SEC),
    )


