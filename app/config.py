from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAP_DIR = PROJECT_ROOT / "CLAP"

CKPT_PATH = "checkpoints/630k-audioset-best.pt"
ENABLE_FUSION = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLOTHO_AUDIO_DIR = r"path_to/clotho_audio_evaluation/evaluation"

WINDOW_SEC = 10.0
HOP_SEC = 5.0

VAD_FRAME_SEC = 0.01
VAD_THRESHOLD_DB = -40.0
VAD_MIN_DUR_SEC = 0.3
VAD_MARGIN_SEC = 0.1
VAD_MAX_SEG_SEC = 10.0
VAD_HOP_SEC = 5.0

FORCE_RECOMPUTE = False
BATCH_SIZE = 32

ART_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TOP_K = 20
WAVE_MAX_POINTS = 8000
PREVIEW_DIR = ART_DIR / "preview_wav"

TARGET_SR = 48000


