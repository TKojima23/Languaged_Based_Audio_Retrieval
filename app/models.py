from dataclasses import dataclass


@dataclass
class SliceSpec:
    path: str
    start_sec: float
    end_sec: float
    mode: str = "window"


@dataclass
class SliceResult:
    rank: int
    score: float
    path: str
    start_sec: float
    end_sec: float
    mode: str = "window"


