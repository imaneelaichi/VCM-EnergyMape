"""
energy_profiling.py -- energy instrumentation on top of CodeCarbon (tested API: 3.3.2).

WHAT IS (AND IS NOT) MEASURED
-----------------------------
CodeCarbon does not read a wattmeter.  Depending on the machine it either
  * reads Intel RAPL counters (Linux, readable powercap)    -> measured CPU package energy, or
  * estimates CPU power from CPU load x TDP ("cpu_load")    -> MODELLED energy (typical on Windows), and
  * estimates RAM power from the installed RAM size         -> MODELLED.
`method_info()` reports which one was used; every exported record carries
`cpu_power_method` and `energy_kind` ("measured_rapl" or "modelled_tdp_x_load").
A modelled value is a PROXY, never to be reported as a measured energy.  It sees
utilisation, not what the silicon actually draws.  For a defensible measurement use an
external wattmeter (wall socket / USB meter) and keep this module for the time/energy
bookkeeping.

SCOPE
-----
* Energy is measured for a *block of work repeated for >= min_seconds* (CodeCarbon samples
  every `measure_power_secs`, so a single 1-2 s frame is far below its resolution).
* Energy is NEVER split between modules/stages by this module: only whole blocks
  (e.g. "encode the whole sequence", "decode the whole sequence", "encode I-frame only")
  are measured.  Per-stage energy = proxy only (time share x block energy) and must be
  labelled as such if shown.
* One tracker at a time (guarded here).  Initialisation (model build / checkpoint load)
  and warm-up are done OUTSIDE the measured task.
* Idle baseline: the machine is measured doing nothing for `idle_seconds`; the reported
  `net_*` figures subtract idle_power x duration.  Gross figures are always reported too.
* Scope = whole machine (tracking_mode="machine"), which includes the OS and other
  processes -> close other applications, plug the laptop to AC, fix the power plan.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

KWH_TO_J = 3.6e6
_ACTIVE = {"tracker": None}


class EnergyMeter:
    def __init__(self, country_iso_code: str = "MAR", measure_power_secs: float = 1.0,
                 tracking_mode: str = "machine", project_name: str = "VCM-EnergyMap"):
        if _ACTIVE["tracker"] is not None:
            raise RuntimeError("An EnergyMeter is already active: only one CodeCarbon tracker at a time.")
        from codecarbon import OfflineEmissionsTracker   # offline: no network lookup, energy only
        self.measure_power_secs = measure_power_secs
        self.t = OfflineEmissionsTracker(country_iso_code=country_iso_code, project_name=project_name,
                                         measure_power_secs=measure_power_secs, save_to_file=False,
                                         log_level="error", tracking_mode=tracking_mode)
        self.t.start()
        _ACTIVE["tracker"] = self
        self.tasks: List[dict] = []

    # --------------------------------------------------------------- disclosure
    def method_info(self) -> dict:
        info = {"codecarbon_measure_power_secs": self.measure_power_secs}
        for h in getattr(self.t, "_hardware", []):
            name = type(h).__name__
            info[f"{name}_mode"] = getattr(h, "_mode", None)
            if name == "CPU":
                info["cpu_tdp_w"] = getattr(h, "_tdp", None)
        mode = str(info.get("CPU_mode"))
        info["cpu_power_method"] = mode
        info["energy_kind"] = "measured_rapl" if "rapl" in mode.lower() else "modelled_tdp_x_load"
        try:
            info["hardware"] = self.t.get_detected_hardware()
        except Exception:
            pass
        return info

    # --------------------------------------------------------------- tasks
    @contextmanager
    def task(self, name: str):
        self.t.start_task(name)
        t0 = time.perf_counter()
        holder: dict = {}
        try:
            yield holder
        finally:
            wall = time.perf_counter() - t0
            d = self.t.stop_task()
            rec = {"task": name, "wall_s": wall, "cc_duration_s": d.duration,
                   "energy_total_j": d.energy_consumed * KWH_TO_J,
                   "energy_cpu_j": d.cpu_energy * KWH_TO_J, "energy_ram_j": d.ram_energy * KWH_TO_J,
                   "energy_gpu_j": d.gpu_energy * KWH_TO_J,
                   "mean_power_total_w": d.energy_consumed * KWH_TO_J / max(wall, 1e-9),
                   "cpu_power_w_last": d.cpu_power, "ram_power_w_last": d.ram_power}
            holder.update(rec)
            self.tasks.append(rec)

    def close(self):
        try:
            self.t.stop()
        finally:
            _ACTIVE["tracker"] = None

    # --------------------------------------------------------------- protocols
    def measure_idle(self, seconds: float) -> dict:
        with self.task("idle") as r:
            time.sleep(seconds)
        return r

    def measure_repeated(self, label: str, fn: Callable[[], int], idle: Optional[dict],
                         min_seconds: float = 20.0, warmup_iters: int = 1, repeats: int = 3) -> dict:
        """fn() performs ONE repetition of the block and returns the number of frames it processed.
        Each of the `repeats` independent measurements runs fn() in a loop for >= min_seconds.
        Returns per-repeat and aggregated (mean/std) energy per iteration and per frame."""
        if min_seconds < 10 * self.measure_power_secs:
            print(f"[energy] WARNING: min_seconds={min_seconds} < 10 x measure_power_secs "
                  f"({self.measure_power_secs}): energy resolution is poor")
        for _ in range(warmup_iters):          # warm-up OUTSIDE any measured task
            fn()
        runs = []
        idle_w = idle["mean_power_total_w"] if idle else 0.0
        for k in range(repeats):
            iters = frames = 0
            with self.task(f"{label}#{k}") as r:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < min_seconds or iters == 0:
                    frames += fn()
                    iters += 1
            net = r["energy_total_j"] - idle_w * r["wall_s"]
            runs.append({"repeat": k, "iterations": iters, "frames": frames, "wall_s": r["wall_s"],
                         "time_per_frame_s": r["wall_s"] / frames,
                         "gross_j_per_frame": r["energy_total_j"] / frames,
                         "net_j_per_frame": net / frames,
                         "gross_cpu_j_per_frame": r["energy_cpu_j"] / frames,
                         "gross_ram_j_per_frame": r["energy_ram_j"] / frames,
                         "mean_power_w": r["mean_power_total_w"]})

        def ms(key):
            v = [x[key] for x in runs]
            m = sum(v) / len(v)
            sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
            return m, sd
        agg = {}
        for key in ("time_per_frame_s", "gross_j_per_frame", "net_j_per_frame", "mean_power_w"):
            agg[key + "_mean"], agg[key + "_std"] = ms(key)
        return {"label": label, "runs": runs, "aggregate": agg, "idle_power_w": idle_w,
                "min_seconds": min_seconds, "repeats": repeats}


def dump(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
