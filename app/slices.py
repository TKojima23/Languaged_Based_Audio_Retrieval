from .audio import read_audio
from .models import SliceSpec


def make_window_slices(path: str, window_sec: float, hop_sec: float) -> list[SliceSpec]:
    y, sr = read_audio(path)
    duration = len(y) / sr

    slices: list[SliceSpec] = []
    start = 0.0
    while start < duration:
        slices.append(
            SliceSpec(path=path, start_sec=start, end_sec=start + window_sec, mode="window")
        )
        start += hop_sec
    return slices


