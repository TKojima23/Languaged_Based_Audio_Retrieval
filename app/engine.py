import os
import tempfile
from pathlib import Path

import numpy as np

from .audio import extract_slice_wav
from .cache import (
    cache_paths,
    collect_wav_files,
    l2norm,
    load_cache_if_valid,
    save_cache,
)
from .config import (
    ART_DIR,
    BATCH_SIZE,
    CKPT_PATH,
    CLOTHO_AUDIO_DIR,
    ENABLE_FUSION,
    FORCE_RECOMPUTE,
    VAD_FRAME_SEC,
    VAD_HOP_SEC,
    VAD_MARGIN_SEC,
    VAD_MAX_SEG_SEC,
    VAD_MIN_DUR_SEC,
    VAD_THRESHOLD_DB,
    WINDOW_SEC,
    DEVICE,
    HOP_SEC,
)
from .slices import make_window_slices
from .models import SliceResult, SliceSpec
from .vad import make_vad_slices

from CLAP.src import laion_clap


class ClapModel:
    def __init__(self):
        self._model = None

    def get(self):
        if self._model is None:
            model = laion_clap.CLAP_Module(enable_fusion=ENABLE_FUSION, device=DEVICE)
            model.load_ckpt(CKPT_PATH)
            self._model = model
        return self._model


def compute_slice_embeddings(
    slices: list[SliceSpec],
    model,
    progress_cb=None,
) -> np.ndarray:
    embs = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="clap_slices_"))
    try:
        for i in range(0, len(slices), BATCH_SIZE):
            batch = slices[i : i + BATCH_SIZE]
            tmp_wavs = []
            for j, spec in enumerate(batch):
                tmp_wav = str(tmp_dir / f"slice_{i+j:06d}.wav")
                seg_dur = min(spec.end_sec - spec.start_sec, VAD_MAX_SEG_SEC) if spec.mode == "vad" else WINDOW_SEC
                extract_slice_wav(spec, seg_dur, tmp_wav)
                tmp_wavs.append(tmp_wav)

            a = model.get_audio_embedding_from_filelist(x=tmp_wavs, use_tensor=True)
            a = l2norm(a)
            embs.append(a.detach().cpu().numpy().astype(np.float32))

            for w in tmp_wavs:
                try:
                    os.remove(w)
                except OSError:
                    pass

            if progress_cb:
                progress_cb(i + len(batch), len(slices))
        return np.concatenate(embs, axis=0)
    finally:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


class SliceSearchEngine:
    def __init__(self, use_memmap_index: bool = True):
        self.slices: list[SliceSpec] = []
        self.audio_emb: np.ndarray | None = None
        self._model = ClapModel()
        self._use_memmap_index = use_memmap_index

    def ensure_ready(
        self,
        mode: str = "window",
        vad_params: dict | None = None,
        progress_cb=None,
    ):
        ART_DIR.mkdir(parents=True, exist_ok=True)
        params = vad_params or {}
        wavs = collect_wav_files(CLOTHO_AUDIO_DIR)
        emb_path, meta_path, man_path = cache_paths(mode, params)

        cached = load_cache_if_valid(
            wavs,
            emb_path,
            meta_path,
            man_path,
            mode,
            params,
            mmap_mode="r" if self._use_memmap_index else None,
        )
        if isinstance(cached, tuple):
            emb, slices = cached
            self.audio_emb = emb
            self.slices = slices
            print(f"[OK] Cache loaded: {emb_path} shape={self.audio_emb.shape}")
        else:
            if isinstance(cached, str) and not FORCE_RECOMPUTE:
                print(f"[INFO] Cache mismatch ({cached}). Recomputing...")

            all_slices: list[SliceSpec] = []
            if mode == "window":
                for wav in wavs:
                    all_slices.extend(make_window_slices(wav, WINDOW_SEC, HOP_SEC))
            else:
                for wav in wavs:
                    all_slices.extend(
                        make_vad_slices(
                            wav,
                            max_seg_sec=params.get("max_seg_sec", VAD_MAX_SEG_SEC),
                            hop_sec=params.get("hop_sec", VAD_HOP_SEC),
                            frame_sec=params.get("frame_sec", VAD_FRAME_SEC),
                            threshold_db=params.get("threshold_db", VAD_THRESHOLD_DB),
                            min_dur_sec=params.get("min_dur_sec", VAD_MIN_DUR_SEC),
                            margin_sec=params.get("margin_sec", VAD_MARGIN_SEC),
                        )
                    )

            if not all_slices:
                raise RuntimeError(
                    "No slices generated. VAD threshold may be too strict."
                )

            model = self._model.get()
            emb = compute_slice_embeddings(all_slices, model, progress_cb=progress_cb)
            save_cache(wavs, emb, all_slices, emb_path, meta_path, man_path, mode, params)
            self.slices = all_slices
            self.audio_emb = (
                np.load(emb_path, mmap_mode="r") if self._use_memmap_index else emb
            )
            print(f"[OK] Saved: {emb_path} shape={emb.shape}")

        self._model.get()

    def search(self, text: str, topk: int) -> list[SliceResult]:
        if self.audio_emb is None:
            raise RuntimeError("Engine not ready")
        model = self._model.get()

        t = model.get_text_embedding([text], use_tensor=True)
        t = l2norm(t).detach().cpu().numpy().astype(np.float32)[0]
        scores = self.audio_emb @ t

        n = len(scores)
        if n == 0:
            return []
        k = min(max(1, int(topk)), n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results = []
        for rank, i in enumerate(idx, start=1):
            s = self.slices[int(i)]
            results.append(
                SliceResult(
                    rank=rank,
                    score=float(scores[i]),
                    path=s.path,
                    start_sec=s.start_sec,
                    end_sec=s.end_sec,
                    mode=s.mode,
                )
            )
        return results


