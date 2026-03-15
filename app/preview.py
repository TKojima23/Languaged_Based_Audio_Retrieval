import hashlib
from pathlib import Path

from .audio import extract_slice_wav
from .config import PREVIEW_DIR
from .models import SliceSpec


class PreviewManager:
    def __init__(self, preview_dir: Path = PREVIEW_DIR):
        self.preview_dir = preview_dir
        self._created_paths: set[Path] = set()

    def get_preview(self, path: str, start_sec: float, end_sec: float) -> str:
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        st = src.stat()
        key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{start_sec}|{end_sec}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        out_path = (
            self.preview_dir
            / f"{src.stem}__{digest}__{start_sec:.1f}-{end_sec:.1f}.wav"
        )
        if not out_path.exists():
            spec = SliceSpec(path=path, start_sec=start_sec, end_sec=end_sec)
            extract_slice_wav(spec, end_sec - start_sec, str(out_path))
            self._created_paths.add(out_path)
        return str(out_path)

    def cleanup(self):
        for p in list(self._created_paths):
            try:
                p.unlink()
            except OSError:
                pass
        self._created_paths.clear()
        try:
            self.preview_dir.rmdir()
        except OSError:
            pass


