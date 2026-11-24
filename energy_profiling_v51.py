"""
energy_profiling_v51.py -- energy instrumentation on top of CodeCarbon (tested API: 3.3.2).  [revision V5.1]
(contenu identique à energy_profiling_v41.py = energy_profiling.py V4 + energy_breakdown + idle_power_stats ;
 seul le nom change, pour que V5.1 soit autonome : les fichiers V4, V4.1 et V5 sont conservés non modifiés)

WHAT IS (AND IS NOT) MEASURED
-----------------------------
CodeCarbon does not read a wattmeter.  Depending on the machine it either
  * reads Intel RAPL counters (Linux powercap, or Windows EMI)  -> CPU PACKAGE energy from hardware counters
  * estimates CPU power from CPU load x TDP ("cpu_load")        -> MODELLED energy, and
  * estimates RAM power from the installed RAM size             -> MODELLED (always, in CodeCarbon 3.3.2).

`method_info()` reports what was really used.  Since V4 the classification is NOT deduced from the
mode string: for `windows_emi` the WindowsEMI instance really used by the tracker is inspected
(`_intel_interface._devices`) and the CPU energy is labelled `measured_rapl_via_windows_emi` ONLY IF
  (1) the selected channel is exactly one PACKAGE channel (e.g. RAPL_Package0_PKG), no DRAM / PP0 / PP1, and
  (2) the counter really delivered energy (> 0 J) on every task measured so far.
Otherwise the label is explicit about what is missing (never "measured").

RAPL scope = CPU PACKAGE (cores + uncore + iGPU).  It is NOT the whole machine (no display, SSD, fans,
DRAM...).  RAM energy is a model (constant power x time) and is never a RAPL measurement here.
The total (CPU + RAM [+ GPU]) is therefore a HYBRID figure: measured CPU + modelled RAM.

SCOPE
-----
* Energy is measured for a block of work (a task).  CodeCarbon samples every `measure_power_secs`.
* Energy is NEVER split between modules by this module: only whole tasks are measured.
* One tracker at a time (guarded here).  Initialisation and warm-up are done OUTSIDE the measured task.
* Idle baseline: the machine is measured doing nothing; `net_*` = gross - idle_power x DURATION OF THE
  ENERGY WINDOW (cc_duration_s, the same clock CodeCarbon used to integrate the energy; wall_s as a
  documented fallback only).  Gross figures are always reported too.  Negative nets are kept as is.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

KWH_TO_J = 3.6e6
_ACTIVE = {"tracker": None}


def _is_package_channel_name(name: str) -> bool:
    """True for a RAPL *package* channel (e.g. RAPL_Package0_PKG); False for DRAM / PP0 / PP1 / core / psys."""
    n = str(name).lower()
    return "pkg" in n and not any(t in n for t in ("dram", "pp0", "pp1", "core", "psys"))


def energy_window(rec: dict) -> float:
    """Duration (s) over which CodeCarbon integrated the energy of a task record: cc_duration_s if valid,
    else wall_s.  Single place where the choice is made (used for power AND for the idle subtraction)."""
    d = rec.get("cc_duration_s")
    if isinstance(d, (int, float)) and math.isfinite(d) and d > 0:
        return float(d)
    return float(rec.get("wall_s", float("nan")))


class EnergyMeter:
    def __init__(self, country_iso_code: str = "MAR", measure_power_secs: float = 1.0,
                 tracking_mode: str = "machine", project_name: str = "VCM-EnergyMap"):
        if _ACTIVE["tracker"] is not None:
            raise RuntimeError("An EnergyMeter is already active: only one CodeCarbon tracker at a time.")
        from codecarbon import OfflineEmissionsTracker   # offline: no network lookup, energy only
        self.measure_power_secs = measure_power_secs
        self.country_iso_code = country_iso_code
        self.tracking_mode = tracking_mode
        # rapl_include_dram=False is CodeCarbon's default; passed explicitly so it is recorded and cannot drift
        self.t = OfflineEmissionsTracker(country_iso_code=country_iso_code, project_name=project_name,
                                         measure_power_secs=measure_power_secs, save_to_file=False,
                                         log_level="error", tracking_mode=tracking_mode,
                                         rapl_include_dram=False)
        self.t.start()
        _ACTIVE["tracker"] = self
        self.tasks: List[dict] = []
        self.cpu_nonpositive_tasks: List[str] = []   # tasks whose CPU energy was <= 0 J (counter not delivering)

    # --------------------------------------------------------------- disclosure
    def _cpu_hw(self):
        for h in getattr(self.t, "_hardware", []):
            if type(h).__name__ == "CPU":
                return h
        return None

    def probe_emi(self) -> dict:
        """Inspect the WindowsEMI instance REALLY used by the tracker (not a fresh one)."""
        out: dict = {"probe_ok": False}
        h = self._cpu_hw()
        iface = getattr(h, "_intel_interface", None) if h is not None else None
        if iface is None:
            out["probe_error"] = "CPU._intel_interface absent (mode is not windows_emi / intel_rapl)"
            return out
        out["interface_class"] = type(iface).__name__
        out["include_dram"] = getattr(iface, "emi_include_dram", None)
        devices = getattr(iface, "_devices", None)
        if devices is None:
            out["probe_error"] = "attribute _devices not found on the interface (CodeCarbon API changed?)"
            return out
        sel, monitored = [], []
        for path, names, idxs in devices:
            monitored.extend(names)
            for i in idxs:
                sel.append({"device_path": path, "channel_index": i, "channel": names[i]})
        out["selected_channels"] = sel
        out["selected_channel_names"] = [s["channel"] for s in sel]
        out["channels_on_monitored_devices"] = monitored
        out["mirrored_candidates_pending"] = [
            {"channel": iface._channel_name(k), "mirrors": iface._channel_name(v)}
            for k, v in getattr(iface, "_mirrored_candidates", {}).items()] if hasattr(iface, "_channel_name") else []
        try:   # full inventory of the machine (all devices), if the Windows helpers are importable
            from codecarbon.core.windows_emi import _read_device_channels, list_emi_device_paths
            inv = []
            for p in list_emi_device_paths():
                try:
                    inv.append({"device_path": p, "channels": _read_device_channels(p)})
                except Exception as e:                       # noqa: BLE001
                    inv.append({"device_path": p, "error": repr(e)})
            out["emi_inventory"] = inv
        except Exception as e:                               # noqa: BLE001
            out["emi_inventory_error"] = repr(e)
        names = out["selected_channel_names"]
        out["only_package_channels_selected"] = bool(names) and all(_is_package_channel_name(n) for n in names)
        out["single_channel_selected"] = len(names) == 1
        out["package_channel_confirmed"] = out["only_package_channels_selected"] and out["single_channel_selected"]
        out["probe_ok"] = True
        return out

    def method_info(self) -> dict:
        """Disclosure of what is REALLY used.  Call again at the end of the run: the selection can change
        (CodeCarbon may drop a channel it detects as a mirror) and the counter check needs measured tasks."""
        try:                                                  # hardware is created lazily in some versions
            self.t._ensure_hardware_ready()
        except Exception:                                     # noqa: BLE001
            pass
        info = {"codecarbon_measure_power_secs": self.measure_power_secs, "tracking_mode": self.tracking_mode,
                "country_iso_code": self.country_iso_code}
        try:
            import importlib.metadata as imd
            info["codecarbon_version"] = imd.version("codecarbon")
        except Exception:                                     # noqa: BLE001
            info["codecarbon_version"] = None
        for h in getattr(self.t, "_hardware", []):
            name = type(h).__name__
            info[f"{name}_mode"] = getattr(h, "_mode", None)     # RAM has no _mode -> None (not informative)
            if name == "CPU":
                info["cpu_tdp_w"] = getattr(h, "_tdp", None)
        info["hardware_classes"] = [type(h).__name__ for h in getattr(self.t, "_hardware", [])]
        mode = str(info.get("CPU_mode")).lower()
        info["cpu_power_method"] = mode

        emi = self.probe_emi() if mode in ("windows_emi", "intel_rapl") else {"probe_ok": False}
        info["emi_probe"] = emi
        n_tasks = len(self.tasks)
        info["n_tasks_measured"] = n_tasks
        info["n_tasks_cpu_energy_nonpositive"] = len(self.cpu_nonpositive_tasks)
        # "compteur fonctionnel" n'a de sens que pour un compteur materiel (EMI / RAPL) ; None pour un mode modelise
        functional = (None if (n_tasks == 0 or mode not in ("windows_emi", "intel_rapl"))
                      else (len(self.cpu_nonpositive_tasks) == 0))
        info["cpu_counter_functional"] = functional

        if mode == "windows_emi":
            if not emi.get("package_channel_confirmed"):
                cpu_src = "windows_emi_package_channel_NOT_confirmed"
            elif functional is None:
                cpu_src = "windows_emi_package_channel_confirmed_energy_not_yet_checked"
            elif functional:
                cpu_src = "measured_rapl_via_windows_emi"
            else:
                cpu_src = "windows_emi_package_channel_confirmed_but_counter_returned_no_energy"
        elif mode == "intel_rapl":
            cpu_src = "intel_rapl_powercap_domain_not_verified_by_this_module"
        elif mode == "cpu_load":
            cpu_src = "modelled_tdp_x_load"
        elif mode == "constant":
            cpu_src = "modelled_constant_tdp"
        else:
            cpu_src = f"unclassified_mode_{mode}"
        info["cpu_energy_source"] = cpu_src
        info["cpu_is_confirmed_rapl_measurement"] = (cpu_src == "measured_rapl_via_windows_emi")
        # RAM: CodeCarbon 3.3.2 has no counter-based RAM path -> always an estimate (power from installed size x time)
        info["ram_energy_source"] = "modelled"
        info["ram_energy_detail"] = "CodeCarbon RAM power estimate (installed RAM size) x duration; not RAPL/DRAM"
        info["total_energy_kind"] = ("hybrid_cpu_measured_rapl_plus_ram_modelled"
                                     if info["cpu_is_confirmed_rapl_measurement"]
                                     else "not_a_confirmed_hardware_measurement")
        # kept for backward compatibility with V3 consumers
        info["energy_kind"] = cpu_src
        try:
            info["ram_total_size_gb"] = self.t._conf.get("ram_total_size")
            info["cpu_model"] = self.t._conf.get("cpu_model")
        except Exception:                                     # noqa: BLE001
            pass
        try:
            info["hardware"] = self.t.get_detected_hardware()
        except Exception:                                     # noqa: BLE001
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
            if d is None:
                raise RuntimeError(f"CodeCarbon stop_task() returned None for task '{name}'")
            e_tot, e_cpu = d.energy_consumed * KWH_TO_J, d.cpu_energy * KWH_TO_J
            e_ram, e_gpu = d.ram_energy * KWH_TO_J, d.gpu_energy * KWH_TO_J
            rec = {"task": name, "wall_s": wall, "cc_duration_s": d.duration,
                   "energy_total_j": e_tot, "energy_cpu_j": e_cpu, "energy_ram_j": e_ram, "energy_gpu_j": e_gpu,
                   "cpu_power_w_last": d.cpu_power, "ram_power_w_last": d.ram_power}
            win = energy_window(rec)
            rec["energy_window_s"] = win
            rec["energy_window_source"] = "cc_duration_s" if win == rec["cc_duration_s"] else "wall_s"
            # mean powers = energy / ENERGY WINDOW (not wall_s): same clock as the integration
            rec["mean_power_total_w"] = e_tot / max(win, 1e-9)
            rec["mean_power_cpu_w"] = e_cpu / max(win, 1e-9)
            rec["mean_power_ram_w"] = e_ram / max(win, 1e-9)
            rec["mean_power_gpu_w"] = e_gpu / max(win, 1e-9)
            if not (e_cpu > 0):
                self.cpu_nonpositive_tasks.append(name)
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
        Returns per-repeat and aggregated (mean/std) energy per iteration and per frame.
        Net = gross - idle_power x energy window (same clock as the energy), per component."""
        if min_seconds < 10 * self.measure_power_secs:
            print(f"[energy] WARNING: min_seconds={min_seconds} < 10 x measure_power_secs "
                  f"({self.measure_power_secs}): energy resolution is poor")
        for _ in range(warmup_iters):          # warm-up OUTSIDE any measured task
            fn()
        runs = []
        idle_w = idle["mean_power_total_w"] if idle else 0.0
        idle_cpu_w = idle.get("mean_power_cpu_w", 0.0) if idle else 0.0
        idle_ram_w = idle.get("mean_power_ram_w", 0.0) if idle else 0.0
        for k in range(repeats):
            iters = frames = 0
            with self.task(f"{label}#{k}") as r:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < min_seconds or iters == 0:
                    frames += fn()
                    iters += 1
            win = r["energy_window_s"]
            net = r["energy_total_j"] - idle_w * win
            runs.append({"repeat": k, "iterations": iters, "frames": frames, "wall_s": r["wall_s"],
                         "energy_window_s": win, "time_per_frame_s": r["wall_s"] / frames,
                         "gross_j_per_frame": r["energy_total_j"] / frames,
                         "net_j_per_frame": net / frames,
                         "gross_cpu_j_per_frame": r["energy_cpu_j"] / frames,
                         "gross_ram_j_per_frame": r["energy_ram_j"] / frames,
                         "net_cpu_j_per_frame": (r["energy_cpu_j"] - idle_cpu_w * win) / frames,
                         "net_ram_j_per_frame": (r["energy_ram_j"] - idle_ram_w * win) / frames,
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


# ======================================================================================
# V4.1 : calcul brut / net par composant et controle de coherence (fonctions pures, testables sans CodeCarbon)
# ======================================================================================
NET_ABS_TOL_J = 1e-9      # tolerance absolue (J) pour |net - net_direct|
NET_REL_TOL = 1e-9        # tolerance relative a max(1 J, |brut total|, |E repos total|)


def _f(x, default=0.0):
    return default if x is None else float(x)


def energy_breakdown(rec: dict, idle: dict) -> dict:
    """Energies BRUTES et NETTES par composant pour UNE tache, avec controle de coherence.

    rec  : energy_total_j, energy_cpu_j, energy_ram_j, [energy_gpu_j], energy_window_s (sinon cc_duration_s / wall_s)
    idle : power_total_w, power_cpu_w, power_ram_w, [power_gpu_w]   (baseline a vide, W)

    net composant  = brut composant - P_repos(composant) x fenetre          (fenetre = energy_window, meme horloge que l'energie)
    net total      = somme des nets composants   (CPU + RAM + GPU)
    net direct     = brut total - P_repos(total) x fenetre                  (controle independant)
    Les valeurs NEGATIVES sont conservees telles quelles : jamais de clamp a 0, jamais de valeur absolue.
    Les deux definitions du net ne different que si le total CodeCarbon != CPU+RAM+GPU (tache) ou si la puissance
    totale a vide != somme des puissances a vide ; `consistent` le teste avec une tolerance numerique stricte."""
    win = rec.get("energy_window_s")
    if not (isinstance(win, (int, float)) and math.isfinite(win) and win > 0):
        win = energy_window(rec)
    e_t, e_c, e_r = float(rec["energy_total_j"]), float(rec["energy_cpu_j"]), float(rec["energy_ram_j"])
    e_g = _f(rec.get("energy_gpu_j"))
    p_t, p_c, p_r = float(idle["power_total_w"]), float(idle["power_cpu_w"]), float(idle["power_ram_w"])
    p_g = _f(idle.get("power_gpu_w"))
    idle_c, idle_r, idle_g, idle_t = p_c * win, p_r * win, p_g * win, p_t * win
    net_c, net_r, net_g = e_c - idle_c, e_r - idle_r, e_g - idle_g
    net = net_c + net_r + net_g
    net_direct = e_t - idle_t
    diff = net - net_direct
    scale = max(1.0, abs(e_t), abs(idle_t))
    consistent = bool(math.isfinite(diff) and abs(diff) <= NET_ABS_TOL_J + NET_REL_TOL * scale)
    return {"window_s": win,
            "gross_total_j": e_t, "gross_cpu_j": e_c, "gross_ram_j": e_r, "gross_gpu_j": e_g,
            "idle_energy_total_j": idle_t, "idle_energy_cpu_j": idle_c, "idle_energy_ram_j": idle_r, "idle_energy_gpu_j": idle_g,
            "net_total_j": net, "net_cpu_j": net_c, "net_ram_j": net_r, "net_gpu_j": net_g,
            "net_direct_j": net_direct, "net_difference_j": diff, "consistent": consistent,
            "task_component_residual_j": e_t - (e_c + e_r + e_g),
            "idle_component_residual_w": p_t - (p_c + p_r + p_g)}


def idle_power_stats(idle_rows: list) -> dict:
    """Statistiques des baselines a vide (toutes les mesures) : puissance CPU / RAM / GPU / total, en W."""
    out = {"n_baselines": len(idle_rows)}
    for comp in ("cpu", "ram", "gpu", "total"):
        v = [float(r[f"power_{comp}_w"]) for r in idle_rows
             if isinstance(r.get(f"power_{comp}_w"), (int, float)) and math.isfinite(r[f"power_{comp}_w"])]
        if v:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
            out[comp] = {"n": len(v), "mean_w": m, "std_w": sd, "min_w": min(v), "max_w": max(v),
                         "cv": sd / abs(m) if (m and math.isfinite(sd)) else float("nan")}
        else:
            out[comp] = {"n": 0}
    return out
