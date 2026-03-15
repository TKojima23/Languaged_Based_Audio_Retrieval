import os
import queue
import threading
from pathlib import Path

import numpy as np
import tkinter as tk
import winsound
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox, ttk

from .audio import load_audio_for_display
from .config import (
    DEFAULT_TOP_K,
    HOP_SEC,
    VAD_FRAME_SEC,
    VAD_MARGIN_SEC,
    VAD_MAX_SEG_SEC,
    VAD_MIN_DUR_SEC,
    VAD_THRESHOLD_DB,
    VAD_HOP_SEC,
    WINDOW_SEC,
)
from .engine import SliceSearchEngine
from .preview import PreviewManager
from .models import SliceResult
from .vad import load_vad_segments_for_display


class App(tk.Tk):
    def __init__(self, engine: SliceSearchEngine):
        super().__init__()
        self.title("CLAP Segment Search [VAD]")
        self.geometry("1380x800")

        self.engine = engine
        self.preview_manager = PreviewManager()
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self._selected: SliceResult | None = None
        self._vad_segs_cache: dict[str, list[tuple[float, float]]] = {}

        self._build_ui()
        self._set_status("initializing...")
        self._run_async(self._init_engine)
        self.after(50, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.LabelFrame(root, text="Search Settings", padding=6)
        top_frame.pack(fill=tk.X, pady=(0, 6))

        row1 = ttk.Frame(top_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Text query:").pack(side=tk.LEFT)
        self.query_var = tk.StringVar(value="dog barking")
        ttk.Entry(row1, textvariable=self.query_var, width=60).pack(side=tk.LEFT, padx=8)

        ttk.Label(row1, text="TopK:").pack(side=tk.LEFT, padx=(12, 0))
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        ttk.Spinbox(row1, from_=1, to=500, textvariable=self.topk_var, width=6).pack(
            side=tk.LEFT, padx=8
        )

        self.search_btn = ttk.Button(row1, text="Search", command=self._on_search)
        self.search_btn.pack(side=tk.LEFT, padx=(12, 0))
        self.reindex_btn = ttk.Button(row1, text="Re-index", command=self._on_reindex)
        self.reindex_btn.pack(side=tk.LEFT, padx=6)

        row2 = ttk.Frame(top_frame)
        row2.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(row2, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="window")
        ttk.Radiobutton(
            row2,
            text="Fixed Window",
            variable=self.mode_var,
            value="window",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(
            row2,
            text="VAD",
            variable=self.mode_var,
            value="vad",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.win_frame = ttk.Frame(row2)
        self.win_frame.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(
            self.win_frame,
            text=f"window={WINDOW_SEC}s  hop={HOP_SEC}s",
            foreground="gray",
        ).pack(side=tk.LEFT)

        self.vad_frame = ttk.Frame(row2)
        ttk.Label(self.vad_frame, text="Frame(s):").pack(side=tk.LEFT, padx=(16, 2))
        self.vad_fr_var = tk.DoubleVar(value=VAD_FRAME_SEC)
        ttk.Spinbox(
            self.vad_frame,
            from_=0.005,
            to=0.2,
            increment=0.005,
            textvariable=self.vad_fr_var,
            width=6,
            format="%.3f",
        ).pack(side=tk.LEFT)

        ttk.Label(self.vad_frame, text="Threshold(dB):").pack(side=tk.LEFT, padx=(10, 2))
        self.vad_thr_var = tk.DoubleVar(value=VAD_THRESHOLD_DB)
        ttk.Spinbox(
            self.vad_frame,
            from_=-80.0,
            to=0.0,
            increment=1.0,
            textvariable=self.vad_thr_var,
            width=7,
            format="%.1f",
        ).pack(side=tk.LEFT)

        ttk.Label(self.vad_frame, text="Margin(s):").pack(side=tk.LEFT, padx=(10, 2))
        self.vad_mg_var = tk.DoubleVar(value=VAD_MARGIN_SEC)
        ttk.Spinbox(
            self.vad_frame,
            from_=0.0,
            to=2.0,
            increment=0.05,
            textvariable=self.vad_mg_var,
            width=6,
            format="%.2f",
        ).pack(side=tk.LEFT)

        ttk.Label(self.vad_frame, text="MinDur(s):").pack(side=tk.LEFT, padx=(10, 2))
        self.vad_mn_var = tk.DoubleVar(value=VAD_MIN_DUR_SEC)
        ttk.Spinbox(
            self.vad_frame,
            from_=0.05,
            to=5.0,
            increment=0.05,
            textvariable=self.vad_mn_var,
            width=6,
            format="%.2f",
        ).pack(side=tk.LEFT)

        ttk.Label(self.vad_frame, text="MaxSeg(s):").pack(side=tk.LEFT, padx=(10, 2))
        self.vad_mx_var = tk.DoubleVar(value=VAD_MAX_SEG_SEC)
        ttk.Spinbox(
            self.vad_frame,
            from_=1.0,
            to=10.0,
            increment=0.5,
            textvariable=self.vad_mx_var,
            width=6,
            format="%.1f",
        ).pack(side=tk.LEFT)

        self.status_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0.0)
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(12, 0))

        mid = ttk.Frame(root)
        mid.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cols = ("rank", "score", "start", "end", "dur", "mode", "filename", "path")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=26)
        for c, t, w, anc in [
            ("rank", "Rank", 50, tk.CENTER),
            ("score", "Score", 100, tk.E),
            ("start", "Start [s]", 70, tk.E),
            ("end", "End [s]", 70, tk.E),
            ("dur", "Dur [s]", 60, tk.E),
            ("mode", "Mode", 60, tk.CENTER),
            ("filename", "File", 200, tk.W),
            ("path", "Path", 380, tk.W),
        ]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=anc)

        ysb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

        right = ttk.Frame(mid, padding=(12, 0, 0, 0))
        right.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right, text="Selected segment:").pack(anchor=tk.W)
        self.seg_label_var = tk.StringVar(value="(none)")
        ttk.Label(right, textvariable=self.seg_label_var, wraplength=380).pack(
            anchor=tk.W, pady=(0, 6)
        )

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btns, text="Copy Path", command=self._copy_path).pack(side=tk.LEFT)
        ttk.Button(btns, text="Open Folder", command=self._open_folder).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Play Segment", command=self._play_segment).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Button(btns, text="Stop", command=self._stop_audio).pack(side=tk.LEFT, padx=6)

        ttk.Label(right, text="Waveform (orange=selected / green=VAD voiced):").pack(
            anchor=tk.W
        )

        self.fig = Figure(figsize=(4.8, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        self._clear_waveform()

    def _on_mode_change(self):
        if self.mode_var.get() == "vad":
            self.win_frame.pack_forget()
            self.vad_frame.pack(side=tk.LEFT, padx=(16, 0))
        else:
            self.vad_frame.pack_forget()
            self.win_frame.pack(side=tk.LEFT, padx=(16, 0))

    def _get_vad_params(self) -> dict:
        return {
            "frame_sec": float(self.vad_fr_var.get()),
            "threshold_db": float(self.vad_thr_var.get()),
            "margin_sec": float(self.vad_mg_var.get()),
            "min_dur_sec": float(self.vad_mn_var.get()),
            "max_seg_sec": float(self.vad_mx_var.get()),
            "hop_sec": VAD_HOP_SEC,
        }

    def _set_status(self, s: str):
        self.status_var.set(s)

    def _run_async(self, fn):
        if self.worker and self.worker.is_alive():
            return
        self.search_btn.config(state=tk.DISABLED)
        self.reindex_btn.config(state=tk.DISABLED)
        self.worker = threading.Thread(target=fn, daemon=True)
        self.worker.start()

    def _progress_cb(self, done: int, total: int):
        self.q.put(("progress", 100.0 * done / max(total, 1)))

    def _init_engine(self):
        try:
            mode = self.mode_var.get()
            params = self._get_vad_params() if mode == "vad" else None
            self.engine.ensure_ready(mode=mode, vad_params=params, progress_cb=self._progress_cb)
            self.q.put(("status", "ready"))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _reindex_task(self):
        try:
            mode = self.mode_var.get()
            params = self._get_vad_params() if mode == "vad" else None
            self.q.put(("progress", 0.0))
            self.engine.ensure_ready(mode=mode, vad_params=params, progress_cb=self._progress_cb)
            self.q.put(("status", "ready (re-indexed)"))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _search_task(self, text: str, topk: int):
        try:
            self.q.put(("results", self.engine.search(text, topk)))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _wave_task(self, result: SliceResult):
        try:
            t, y, sr, total_dur = load_audio_for_display(result.path)
            vad_segs = None
            if self.mode_var.get() == "vad":
                key = result.path
                if key not in self._vad_segs_cache:
                    self._vad_segs_cache[key] = load_vad_segments_for_display(
                        result.path, self._get_vad_params()
                    )
                vad_segs = self._vad_segs_cache[key]
            self.q.put(("wave", (result, t, y, sr, total_dur, vad_segs)))
        except Exception as e:
            self.q.put(("wave_error", (result.path, str(e))))

    def _on_search(self):
        text = self.query_var.get().strip()
        if not text:
            return
        self._set_status("searching...")
        self._run_async(lambda: self._search_task(text, int(self.topk_var.get())))

    def _on_reindex(self):
        self._set_status("re-indexing...")
        self._vad_segs_cache.clear()
        self._run_async(self._reindex_task)

    def _on_close(self):
        self._stop_audio()
        self.preview_manager.cleanup()
        self.destroy()

    def _on_select_row(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        self._stop_audio()
        values = self.tree.item(sel[0], "values")
        if not values or len(values) < 8:
            return

        rank, score_str, start_str, end_str, dur_str, mode_str, _, path = values
        self._selected = SliceResult(
            rank=int(rank),
            score=float(score_str),
            path=path,
            start_sec=float(start_str),
            end_sec=float(end_str),
            mode=mode_str,
        )
        self.seg_label_var.set(
            f"{Path(path).name}\n"
            f"{start_str}s - {end_str}s ({dur_str}s) [{mode_str}] score={score_str}"
        )
        self._set_status("loading waveform...")
        self._run_async(lambda: self._wave_task(self._selected))

    def _copy_path(self):
        if not self._selected:
            return
        self.clipboard_clear()
        self.clipboard_append(self._selected.path)
        self._set_status("copied")

    def _open_folder(self):
        if not self._selected:
            return
        try:
            os.startfile(str(Path(self._selected.path).parent))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _play_segment(self):
        if not self._selected:
            return
        self._stop_audio()
        try:
            preview = self.preview_manager.get_preview(
                self._selected.path, self._selected.start_sec, self._selected.end_sec
            )
            winsound.PlaySound(preview, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._set_status(
                f"playing ({self._selected.end_sec - self._selected.start_sec:.1f}s segment)..."
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _stop_audio(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                    self.search_btn.config(state=tk.NORMAL)
                    self.reindex_btn.config(state=tk.NORMAL)
                    self.progress_var.set(0.0)
                elif kind == "progress":
                    self.progress_var.set(payload)
                elif kind == "results":
                    self._populate(payload)
                    self._set_status(f"done: {len(payload)} results")
                    self.search_btn.config(state=tk.NORMAL)
                    self.reindex_btn.config(state=tk.NORMAL)
                elif kind == "wave":
                    result, t, y, sr, total_dur, vad_segs = payload
                    if (
                        self._selected
                        and result.path == self._selected.path
                        and result.start_sec == self._selected.start_sec
                    ):
                        self._draw_waveform(
                            t, y, sr, total_dur, result.start_sec, result.end_sec, vad_segs
                        )
                        self._set_status("ready")
                    self.search_btn.config(state=tk.NORMAL)
                    self.reindex_btn.config(state=tk.NORMAL)
                elif kind == "wave_error":
                    path, msg = payload
                    if self._selected and path == self._selected.path:
                        self._clear_waveform()
                        self._set_status("waveform error")
                        messagebox.showerror("Waveform Error", msg)
                    self.search_btn.config(state=tk.NORMAL)
                    self.reindex_btn.config(state=tk.NORMAL)
                elif kind == "error":
                    self.search_btn.config(state=tk.NORMAL)
                    self.reindex_btn.config(state=tk.NORMAL)
                    messagebox.showerror("Error", payload)
                    self._set_status("error")
                    self.progress_var.set(0.0)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _populate(self, results: list[SliceResult]):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in results:
            dur = r.end_sec - r.start_sec
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r.rank,
                    f"{r.score:.4f}",
                    f"{r.start_sec:.2f}",
                    f"{r.end_sec:.2f}",
                    f"{dur:.2f}",
                    r.mode,
                    Path(r.path).name,
                    r.path,
                ),
            )
        self._selected = None
        self.seg_label_var.set("(none)")
        self._clear_waveform()

    def _clear_waveform(self):
        self.ax.clear()
        self.ax.set_title("No selection")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.canvas.draw_idle()

    def _draw_waveform(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sr: int,
        total_dur: float,
        start_sec: float,
        end_sec: float,
        vad_segs: list[tuple[float, float]] | None = None,
    ):
        self.ax.clear()
        if vad_segs:
            for vs, ve in vad_segs:
                self.ax.axvspan(vs, ve, alpha=0.15, color="green", linewidth=0)
        self.ax.plot(t, y, linewidth=0.6, color="#4a90d9")
        self.ax.axvspan(start_sec, min(end_sec, total_dur), alpha=0.35, color="orange")

        import matplotlib.patches as mpatches

        handles = [mpatches.Patch(facecolor="orange", alpha=0.35, label="selected")]
        if vad_segs:
            handles.append(mpatches.Patch(facecolor="green", alpha=0.4, label="VAD voiced"))
        self.ax.legend(handles=handles, loc="upper right", fontsize=8)

        self.ax.set_title(f"sr={sr} duration={total_dur:.1f}s")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(0, total_dur)
        self.ax.grid(True)
        self.canvas.draw_idle()


