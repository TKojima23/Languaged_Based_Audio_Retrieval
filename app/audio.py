import numpy as np

from .config import TARGET_SR, WAVE_MAX_POINTS
from .models import SliceSpec


def _read_audio_sf(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def _read_audio_wave(path: str) -> tuple[np.ndarray, int]:
    """
    soundfileが失敗した時のみ利用されるので無視してOK
    :param path:
    :return:
    """
    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sw}")

    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, int(sr)


def read_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        return _read_audio_sf(path)
    except Exception:
        return _read_audio_wave(path)


def resample_if_needed(y: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """
    リサンプリング方法で面倒見すぎ．単一の方法で良い
    :param y:
    :param sr:
    :param target_sr:
    :return:
    """
    if sr == target_sr:
        return y, sr
    try:
        import resampy

        return resampy.resample(y, sr, target_sr), target_sr
    except ImportError:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        return resample_poly(y, target_sr // g, sr // g), target_sr
    except ImportError:
        pass

    duration = len(y) / sr
    n_out = int(duration * target_sr)
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, y).astype(np.float32), target_sr


def extract_slice_wav(spec: SliceSpec, duration_sec: float, out_path: str):
    import wave

    y, sr = read_audio(spec.path)
    y, sr = resample_if_needed(y, sr, TARGET_SR)

    n_window = int(sr * duration_sec)
    start_sample = int(sr * spec.start_sec)
    end_sample = start_sample + n_window

    segment = y[start_sample:end_sample]
    if len(segment) < n_window:
        segment = np.concatenate(
            [segment, np.zeros(n_window - len(segment), dtype=np.float32)]
        )

    yi16 = np.clip(segment, -1.0, 1.0)
    yi16 = (yi16 * 32767.0).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(yi16.tobytes())


def load_audio_for_display(path: str, max_points: int = WAVE_MAX_POINTS):
    y, sr = read_audio(path)
    if len(y) > max_points:
        step = int(np.ceil(len(y) / max_points))
        y_ds = y[::step]
    else:
        y_ds = y
    t = np.arange(len(y_ds), dtype=np.float32) * (len(y) / sr) / len(y_ds)
    return t, y_ds, sr, len(y) / sr


