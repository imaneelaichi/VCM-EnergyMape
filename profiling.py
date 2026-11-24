"""
profiling.py -- per-stage wall-clock profiling (time.perf_counter), independent of the codec.

Concepts
--------
* phase     : "init" (model construction / checkpoint loading), "warmup" (frames run
              but discarded from statistics), "measured" (statistics).
* category  : "compute" (neural nets, PMF evaluation, range coder, bit manipulation),
              "io" (image read, image write, bitstream file read/write).
              The *compute time* of a frame is the sum of its "compute" LEAF stages;
              the *total time* additionally contains I/O and any un-instrumented gap.
* leaf stages only: a stage must not contain another stage, otherwise time would be
  counted twice.  `frame()` measures the whole frame independently (wall time) so the
  coverage  sum(leaf stages) / wall  can be checked (reported in the summary).
* On CPU, PyTorch ops are synchronous: no synchronisation is needed.  If a CUDA device
  is used, `sync()` calls torch.cuda.synchronize() before every timestamp.

Usage
-----
    prof = Profiler(device="cpu")
    with prof.phase("measured"):
        with prof.frame(idx, "P", direction="enc"):
            with prof.stage("motion_estimation"): ...
            with prof.stage("image_read", category="io"): ...
            prof.add("M_y_pmf", seconds)        # for sub-timers accumulated inside loops
    prof.export("results/profile")              # -> .csv (raw) + _summary.csv + .json
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

IO_STAGES = {"image_read", "image_write", "bitstream_write", "bitstream_read", "image_decode_png",
             "preprocess_to_tensor"}


def _stats(v: List[float]) -> Dict[str, float]:
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return {"n": n, "total_s": sum(v), "mean_s": m, "min_s": min(v), "max_s": max(v), "std_s": sd}


class Profiler:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.records: List[dict] = []          # one row per (frame, stage)
        self.frames: List[dict] = []           # one row per frame (wall time)
        self.init_times: Dict[str, float] = {}
        self._phase = "measured"
        self._cur: Optional[dict] = None

    # -------------------------------------------------------------- low level
    def sync(self):
        if self.device != "cpu":
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str):
        old, self._phase = self._phase, name
        try:
            yield
        finally:
            self._phase = old

    @contextmanager
    def init_timer(self, name: str):
        t0 = time.perf_counter()
        yield
        self.init_times[name] = self.init_times.get(name, 0.0) + (time.perf_counter() - t0)

    @contextmanager
    def frame(self, idx: int, ftype: str, direction: str):
        """direction: 'enc' or 'dec'. Measures the wall time of the whole frame."""
        self._cur = {"frame": idx, "type": ftype, "dir": direction, "phase": self._phase}
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            wall = time.perf_counter() - t0
            row = dict(self._cur, wall_s=wall)
            self.frames.append(row)
            self._cur = None

    @contextmanager
    def stage(self, name: str, category: Optional[str] = None):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.add(name, time.perf_counter() - t0, category)

    def add(self, name: str, seconds: float, category: Optional[str] = None):
        if self._cur is None:
            return
        cat = category or ("io" if name in IO_STAGES else "compute")
        self.records.append(dict(self._cur, stage=name, category=cat, seconds=seconds))

    # -------------------------------------------------------------- reporting
    def summary(self, phase: str = "measured") -> List[dict]:
        """Per (dir, type, stage): n / total / mean / min / max / std over frames."""
        groups = defaultdict(list)
        for r in self.records:
            if r["phase"] == phase:
                groups[(r["dir"], r["type"], r["stage"], r["category"])].append(r["seconds"])
        out = []
        for (d, t, s, c), v in sorted(groups.items()):
            out.append(dict(dir=d, type=t, stage=s, category=c, **_stats(v)))
        return out

    def frame_summary(self, phase: str = "measured") -> List[dict]:
        """Per (dir, type): wall time, sum of compute leaf stages, sum of io leaf stages, coverage."""
        by = defaultdict(lambda: {"wall": [], "compute": [], "io": []})
        per_frame = defaultdict(lambda: {"compute": 0.0, "io": 0.0})
        for r in self.records:
            if r["phase"] == phase:
                per_frame[(r["dir"], r["type"], r["frame"])][r["category"]] += r["seconds"]
        for f in self.frames:
            if f["phase"] != phase:
                continue
            k = (f["dir"], f["type"])
            pf = per_frame[(f["dir"], f["type"], f["frame"])]
            by[k]["wall"].append(f["wall_s"])
            by[k]["compute"].append(pf["compute"])
            by[k]["io"].append(pf["io"])
        out = []
        for (d, t), v in sorted(by.items()):
            w, c, i = v["wall"], v["compute"], v["io"]
            row = dict(dir=d, type=t, n_frames=len(w),
                       wall_mean_s=sum(w) / len(w), wall_min_s=min(w), wall_max_s=max(w),
                       wall_std_s=_stats(w)["std_s"], wall_total_s=sum(w),
                       compute_mean_s=sum(c) / len(c), io_mean_s=sum(i) / len(i),
                       coverage=(sum(c) + sum(i)) / sum(w) if sum(w) > 0 else float("nan"))
            out.append(row)
        return out

    def export(self, prefix: str, meta: Optional[dict] = None):
        import os
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        if self.records:
            with open(prefix + "_raw.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
                w.writeheader()
                w.writerows(self.records)
        summ = self.summary()
        if summ:
            with open(prefix + "_summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
                w.writeheader()
                w.writerows(summ)
        with open(prefix + ".json", "w") as f:
            json.dump({"meta": meta or {}, "init_times_s": self.init_times,
                       "frame_summary": self.frame_summary(), "stage_summary": summ,
                       "warmup_frames": sum(1 for x in self.frames if x["phase"] == "warmup"),
                       "measured_frames": sum(1 for x in self.frames if x["phase"] == "measured")},
                      f, indent=2)
