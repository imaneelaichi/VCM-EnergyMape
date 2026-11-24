#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
experiment3_frame_components_v3.py -- VCM-EnergyMap, experience 3 (VERSION 3)
ENERGIE ET TEMPS PAR IMAGE (I / P, encodage / decodage) ET PAR COMPOSANT DU CODEC,
avec repetitions completes independantes, validation automatique et metadonnees de mesure.

A placer dans svc_code\ (a cote de test_base_profiled.py). NE MODIFIE AUCUN script existant et
NE MODIFIE PAS le codec : il reutilise BaseLayerCodec, Profiler, EnergyMeter, BitstreamWriter /
BitstreamReader / SequenceHeader et _frame_path / _save_png / _env_meta de test_base_profiled.py.
Les signatures reelles de ces API sont verifiees (inspect) au demarrage, y compris en --dry_run.

PROTOCOLE
  0. pre-controles (fichiers, API, processus CodeCarbon concurrents, fichiers de verrou) ;
  1. HORS mesure : construction + chargement du checkpoint, warm-up (--warmup_frames) ;
  2. creation du tracker CodeCarbon (UN seul, journaux captures), pause, baseline a vide ;
  3. pour chaque repetition r = 0..R-1 (independantes) :
       baseline a vide courte (sauf r=0), reset encodeur ET decodeur, NOUVEAU flux .vcmb,
       ENCODAGE (1 tache CodeCarbon / image), DECODAGE depuis le .vcmb (1 tache / image),
       sauvegarde des reconstructions (reconstructed/rep<r>/), VALIDATION automatique,
       sauvegarde progressive (CSV, progress.json) ;
  4. statistiques / figures a partir des mesures deja enregistrees (le codec n'est pas relance).

CE QUI EST MESURE / ESTIME (ne jamais confondre)
  MESURE  : temps (time.perf_counter) de la tache, du profiler, des composants.
  RAPPORTE PAR CODECARBON : energie par image (brute = CPU + RAM). Sa NATURE (compteur materiel ou
            modele TDP x charge) depend du mode CPU ; CPU_mode=windows_emi n'est PAS une preuve de
            compteurs presents (doc CodeCarbon : sous Windows 10, EMI seulement avec materiel de mesure
            dedie). L'energie RAM est TOUJOURS modelisee par CodeCarbon. -> a presenter comme
            « energie CodeCarbon », jamais comme mesure electrique confirmee.
  ESTIME  : energie par COMPOSANT = part de temps du composant x energie de l'image (proxy).
  SECONDAIRE : energie NETTE = brute - P_repos x duree ; peut etre negative ; jamais le resultat principal.
  NON MESURE : detection YOLO.

PERIMETRE (voir metadata.json -> outside_measured_tasks)
  Dans les taches mesurees : lecture PNG, conversion tenseur, encode_frame, ajout au flux EN MEMOIRE
  (BitstreamWriter.add_frame) ; lecture d'un enregistrement du flux, decode_frame, ecriture du PNG.
  HORS taches mesurees (chronometre mais energie non attribuee) : ecriture reelle du fichier .vcmb
  (BitstreamWriter.close), ouverture/fermeture du lecteur de flux, validation, hachages, ecriture
  des CSV, baselines a vide, warm-up, construction du codec.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as imd
import importlib.util
import inspect
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import warnings as pywarnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NAN = float("nan")
VERSION = "v3"
REQUIRED = ["test_base_profiled.py", "energy_profiling.py", "profiling.py", "base_codec.py", "bitstream.py",
            "entropy_codec.py"]
MIN_FINAL_REPEATS = 8

# --------------------------------------------------------------------------------------
# Groupes de composants (noms d'etapes verifies dans base_codec.py et entropy_codec.py)
#   base_codec.py   : align, crop_clamp, motion_estimation, motion_prediction, motion_compensation,
#                     {I,M,R}_{analysis,hyper_analysis,cond_net,synthesis}
#   entropy_codec.py: {tag}_quantize, {tag}_pmf, {tag}_rangecode, {tag}_rangedecode,
#                     {tag}_dequantize, {tag}_context_net      (tag = I_z, I_y, M_z, M_y, R_z, R_y)
#   profiling.IO_STAGES : image_read, image_write, bitstream_write, bitstream_read, ...
# Toutes ces etapes sont des FEUILLES (aucun `with stage` n'en contient un autre) ; les temps
# internes de la boucle autoregressive (context_net, pmf, rangecode) sont cumules par prof.add().
# --------------------------------------------------------------------------------------
GROUPS = [
    ("motion_estimation", "Estimation de mouvement (PWC-Net)"),
    ("motion_prediction", "Prédiction de mouvement (MWNet)"),
    ("motion_compensation", "Compensation de mouvement (warp + MCNet)"),
    ("analysis", "Réseaux d'analyse (analysis + hyper-analysis)"),
    ("synthesis", "Réseau de synthèse"),
    ("cond_net", "Réseau conditionnel"),
    ("align_crop", "Alignement + recadrage/clamp"),
    ("quantization", "Quantification"),
    ("pmf", "PMF (évaluation des probabilités)"),
    ("context_net", "Réseau de contexte"),
    ("entropy_coding", "Codage entropique (range coder)"),
    ("io", "Entrées / sorties (images, flux)"),
    ("uninstrumented", "Non instrumenté (reste)"),
]
GROUP_LABEL = dict(GROUPS)
GROUP_KEYS = [g for g, _ in GROUPS]
_EXACT = {"motion_estimation": "motion_estimation", "motion_prediction": "motion_prediction",
          "motion_compensation": "motion_compensation", "align": "align_crop", "crop_clamp": "align_crop"}
_SUFFIX = [("_context_net", "context_net"), ("_hyper_analysis", "analysis"), ("_analysis", "analysis"),
           ("_cond_net", "cond_net"), ("_synthesis", "synthesis"), ("_quantize", "quantization"),
           ("_dequantize", "quantization"), ("_pmf", "pmf"), ("_rangecode", "entropy_coding"),
           ("_rangedecode", "entropy_coding")]

FRAME_KEYS = ["time_task_s", "time_profiler_s", "cc_duration_s", "tracker_overhead_s", "energy_gross_j",
              "energy_cpu_gross_j", "energy_ram_gross_j", "energy_net_secondary_j", "mean_power_gross_w",
              "profiler_coverage"]
TYPE_KEYS = ["time_task_s", "time_profiler_s", "cc_duration_s", "energy_gross_j", "energy_cpu_gross_j",
             "energy_ram_gross_j", "mean_power_gross_w", "energy_net_secondary_j"]


def stage_group(name, io_stages):
    """(groupe, reconnu). Une etape inconnue est comptee 'non instrumente' ET signalee."""
    if name in io_stages:
        return "io", True
    if name in _EXACT:
        return _EXACT[name], True
    for suf, g in _SUFFIX:
        if name.endswith(suf):
            return g, True
    return "uninstrumented", False


# --------------------------------------------------------------------------------------
# utilitaires
# --------------------------------------------------------------------------------------
def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def agg(v):
    """n, moyenne, ecart-type (n-1), min, max, CV = std/|moyenne| (NaN si non defini)."""
    v = [x for x in v if isnum(x)]
    if not v:
        return {"n": 0, "mean": NAN, "std": NAN, "min": NAN, "max": NAN, "cv": NAN}
    m = sum(v) / len(v)
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) if len(v) > 1 else NAN
    cv = sd / abs(m) if isnum(sd) and m != 0 else NAN
    return {"n": len(v), "mean": m, "std": sd, "min": min(v), "max": max(v), "cv": cv}


def mean_std(v):
    s = agg(v)
    return s["mean"], s["std"]


def flat(prefix, s):
    return {f"{prefix}_{k}": s[k] for k in ("mean", "std", "min", "max", "cv")}


def jsonable(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def dump_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def hms(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt(v, d=3):
    return "n/a" if not isnum(v) else f"{v:.{d}f}"


def pm(m, sd, d=3):
    return f"{fmt(m, d)} ± {fmt(sd, d)}" if isnum(sd) else f"{fmt(m, d)}"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class CsvSink:
    """Ajout progressif (une ligne par image) puis reecriture atomique apres validation."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = None

    def append(self, rows):
        if not rows:
            return
        if self.fields is None:
            self.fields = list(rows[0].keys())
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)

    def rewrite(self, rows):
        if not rows:
            return True
        tmp = str(self.path) + ".tmp"
        try:
            write_csv(tmp, rows, self.fields)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False


class Log:
    def __init__(self):
        self.path = None

    def __call__(self, msg=""):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


log = Log()


def pkg_version(name):
    try:
        return imd.version(name)
    except Exception:
        return None


def energy_label(info):
    """Label NEUTRE : mode CPU brut de CodeCarbon, aucune affirmation de mesure electrique."""
    return (f"énergie CodeCarbon (CPU_mode={info.get('CPU_mode')}, RAM_mode={info.get('RAM_mode')}) ; "
            f"compteurs EMI/RAPL non vérifiés sur ce matériel ; RAM toujours modélisée")


# --------------------------------------------------------------------------------------
# verifications d'API (signatures reelles) et de l'environnement
# --------------------------------------------------------------------------------------
def load_modules():
    """Imports lourds. Retourne le dictionnaire de modules/classes reutilises."""
    import torch
    import test_base_profiled as tbp
    from base_codec import BaseLayerCodec
    from bitstream import BitstreamReader, BitstreamWriter, SequenceHeader
    from energy_profiling import EnergyMeter
    from profiling import IO_STAGES, Profiler
    return dict(torch=torch, tbp=tbp, BaseLayerCodec=BaseLayerCodec, BitstreamReader=BitstreamReader,
                BitstreamWriter=BitstreamWriter, SequenceHeader=SequenceHeader, EnergyMeter=EnergyMeter,
                IO_STAGES=IO_STAGES, Profiler=Profiler)


def check_api(M):
    """Verifie par introspection que les API utilisees existent avec les parametres attendus."""
    problems = []

    def params(fn):
        ps = list(inspect.signature(fn).parameters)
        return ps[1:] if ps and ps[0] == "self" else ps

    def need(cls, name, expected):
        fn = getattr(cls, name, None)
        if fn is None:
            problems.append(f"{cls.__name__}.{name} introuvable")
            return
        got = params(fn)
        if got[:len(expected)] != expected:
            problems.append(f"{cls.__name__}.{name}{tuple(got)} != attendu {tuple(expected)}")

    need(M["BaseLayerCodec"], "encode_frame", ["x", "idx", "gop", "prof"])
    need(M["BaseLayerCodec"], "decode_frame", ["payload", "ranges", "idx", "gop", "shapes", "hw", "prof"])
    need(M["BaseLayerCodec"], "reset_sequence", [])
    need(M["BaseLayerCodec"], "load_checkpoint", ["path"])
    need(M["BitstreamWriter"], "add_frame", ["payload", "ranges"])
    need(M["BitstreamWriter"], "close", [])
    need(M["BitstreamReader"], "next_frame", [])
    need(M["BitstreamReader"], "close", [])
    need(M["Profiler"], "frame", ["idx", "ftype", "direction"])
    need(M["Profiler"], "stage", ["name"])
    need(M["Profiler"], "phase", ["name"])
    need(M["Profiler"], "export", ["prefix"])
    need(M["EnergyMeter"], "task", ["name"])
    need(M["EnergyMeter"], "method_info", [])
    need(M["EnergyMeter"], "close", [])
    need(M["EnergyMeter"], "__init__", ["country_iso_code", "measure_power_secs"])
    fields = getattr(M["SequenceHeader"], "__dataclass_fields__", {})
    for f in ("width", "height", "n_frames", "gop", "ckpt_id", "closed_gop", "shapes"):
        if f not in fields:
            problems.append(f"SequenceHeader.{f} introuvable")
    for fn in ("_frame_path", "_save_png", "_env_meta"):
        if not callable(getattr(M["tbp"], fn, None)):
            problems.append(f"test_base_profiled.{fn} introuvable")
    return problems


def find_concurrent_codecarbon():
    """Cherche (lecture seule) d'autres processus Python lies a CodeCarbon / aux experiences et des fichiers de verrou."""
    found = {"processes": [], "lock_files": [], "psutil": importlib.util.find_spec("psutil") is not None}
    if found["psutil"]:
        import psutil
        me = os.getpid()
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.info["pid"] == me:
                    continue
                cmd = " ".join(p.info.get("cmdline") or [])
                if "python" in (p.info.get("name") or "").lower() and any(
                        s in cmd.lower() for s in ("codecarbon", "experiment3", "run_vcm_energymap", "energy_")):
                    found["processes"].append({"pid": p.info["pid"], "cmd": cmd[:200]})
            except Exception:
                continue
    try:
        for f in Path(tempfile.gettempdir()).iterdir():
            if "codecarbon" in f.name.lower():
                found["lock_files"].append(str(f))
    except Exception:
        pass
    return found


def collect_machine_metadata(a):
    md = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
          "processor": platform.processor(), "os_cpu_count_logical": os.cpu_count(),
          "versions": {p: pkg_version(p) for p in ("torch", "torchvision", "numpy", "codecarbon", "constriction",
                                                   "matplotlib", "Pillow", "psutil")},
          "argv": sys.argv[1:], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if hasattr(sys, "getwindowsversion"):
        v = sys.getwindowsversion()
        md["windows_build"] = v.build
        md["windows_major_label"] = "Windows 11 (build >= 22000)" if v.build >= 22000 else "Windows 10 ou antérieur"
    if importlib.util.find_spec("psutil") is not None:
        import psutil
        try:
            md["cpu_cores_physical"] = psutil.cpu_count(logical=False)
            md["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
            fr = psutil.cpu_freq()
            md["cpu_freq_mhz"] = {"current": fr.current, "max": fr.max} if fr else None
            bat = psutil.sensors_battery()
            md["battery"] = {"percent": bat.percent, "power_plugged": bat.power_plugged} if bat else None
        except Exception as e:
            md["psutil_error"] = str(e)
    if os.name == "nt":
        try:
            r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
            md["power_scheme"] = r.stdout.strip()
        except Exception as e:
            md["power_scheme"] = f"indisponible ({e})"
    return md


# --------------------------------------------------------------------------------------
# l'experience
# --------------------------------------------------------------------------------------
class Experiment:
    def __init__(self, a, M, out_dir: Path):
        self.a, self.M, self.out = a, M, out_dir
        self.k = a.checkpoint
        self.frames_rows, self.comp_rows, self.task_rows, self.rep_rows = [], [], [], []
        self.warnings, self.errors = [], []
        self.validation = {}
        self.meter = None
        self.active = None
        self.t0_run = time.perf_counter()
        self.sink_frames = CsvSink(out_dir / "frame_measurements.csv")
        self.sink_comp = CsvSink(out_dir / "component_time_energy.csv")
        self.sink_tasks = CsvSink(out_dir / "energy_tasks.csv")
        self.idle_w = self.idle_cpu_w = NAN
        self.low_res = defaultdict(set)
        self.partial = {}          # infos partielles d'une repetition (utiles si elle echoue en cours de route)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)
            log(f"[AVERTISSEMENT] {msg}")

    # ---------- tache CodeCarbon : jamais imbriquee ni chevauchante
    @contextmanager
    def task(self, name):
        if self.active is not None:
            raise RuntimeError(f"chevauchement de tâches CodeCarbon : '{self.active}' encore active, "
                               f"'{name}' demandée")
        self.active = name
        t_enter = time.perf_counter()
        holder = None
        try:
            with self.meter.task(name) as holder:
                yield holder
        finally:
            t_exit = time.perf_counter()
            self.active = None
            if holder is not None:
                holder["outer_wall_s"] = t_exit - t_enter
                holder["tracker_overhead_s"] = (t_exit - t_enter) - holder.get("wall_s", NAN)
                self.task_rows.append({
                    "task": name, "t_enter_s": t_enter - self.t0_run, "t_exit_s": t_exit - self.t0_run,
                    "outer_wall_s": t_exit - t_enter, "time_task_s": holder.get("wall_s", NAN),
                    "cc_duration_s": holder.get("cc_duration_s", NAN),
                    "tracker_overhead_s": holder["tracker_overhead_s"],
                    "energy_total_j": holder.get("energy_total_j", NAN), "energy_cpu_j": holder.get("energy_cpu_j", NAN),
                    "energy_ram_j": holder.get("energy_ram_j", NAN), "energy_gpu_j": holder.get("energy_gpu_j", NAN),
                    "cpu_power_w_last": holder.get("cpu_power_w_last", NAN),
                    "ram_power_w_last": holder.get("ram_power_w_last", NAN)})

    def measure_idle(self, name, seconds):
        with self.task(name) as h:
            time.sleep(seconds)
        w = h["energy_total_j"] / h["wall_s"] if h["wall_s"] > 0 else NAN
        wc = h["energy_cpu_j"] / h["wall_s"] if h["wall_s"] > 0 else NAN
        wr = h["energy_ram_j"] / h["wall_s"] if h["wall_s"] > 0 else NAN
        if seconds < 10 * self.a.measure_power_secs:
            self.warn(f"baseline {name} : {seconds:g} s < 10 x measure_power_secs : résolution faible")
        return {"name": name, "seconds": h["wall_s"], "power_total_w": w, "power_cpu_w": wc, "power_ram_w": wr}

    # ---------- init + warm-up (HORS mesure)
    def init_codecs(self):
        from PIL import Image
        from torchvision.transforms import ToTensor
        M, a, tbp = self.M, self.a, self.M["tbp"]
        ckpt = Path(a.ckpt_dir) / f"checkpoint_base_{self.k}.pth.tar"
        n, gop = a.no_frames, a.gop
        log(f"construction + chargement du checkpoint {ckpt.name} (hors mesure)")
        self.enc = M["BaseLayerCodec"]("enc", root=a.root, closed_gop=a.closed_gop)
        self.ck_info = self.enc.load_checkpoint(str(ckpt))
        self.dec = M["BaseLayerCodec"]("dec", root=a.root, closed_gop=a.closed_gop)
        self.dec.load_checkpoint(str(ckpt))
        self.W, self.H = Image.open(tbp._frame_path(a.inp_path, a.prefix, a.start_index)).size
        if a.warmup_frames > 0:
            nw = min(a.warmup_frames, n)
            log(f"warm-up encodage/décodage sur {nw} image(s) (hors mesure, écarté)")
            pw = M["Profiler"]()
            recs = []
            with pw.phase("warmup"):
                for i in range(nw):
                    x = ToTensor()(Image.open(tbp._frame_path(a.inp_path, a.prefix, a.start_index + i)).convert("RGB")).unsqueeze(0)
                    with pw.frame(i, "I" if i % gop == 0 else "P", "enc"):
                        r = self.enc.encode_frame(x, i, gop, pw)
                    recs.append((r["payload"], r["ranges"], {kk: tuple(v[1:]) for kk, v in r["shapes"].items()}))
                shapes = {}
                for _, _, sh in recs:
                    shapes.update(sh)
                for i, (pl, rg, _) in enumerate(recs):
                    with pw.frame(i, "I" if i % gop == 0 else "P", "dec"):
                        self.dec.decode_frame(pl, rg, i, gop, shapes, (self.H, self.W), pw)
        self.enc.reset_sequence()
        self.dec.reset_sequence()

    # ---------- enregistrement d'une mesure d'image
    def record(self, rep, direction, i, ftype, holder, prof, extra, idle):
        a = self.a
        IO_STAGES = self.M["IO_STAGES"]
        pw = next(f for f in prof.frames if f["dir"] == direction and f["frame"] == i and f["phase"] == "measured")
        wall_task = holder["wall_s"]
        cc_dur = holder.get("cc_duration_s", NAN)
        e_g, e_c, e_r = holder["energy_total_j"], holder["energy_cpu_j"], holder["energy_ram_j"]
        net = e_g - idle["power_total_w"] * wall_task
        gt = defaultdict(float)
        for r in prof.records:
            if r["phase"] == "measured" and r["dir"] == direction and r["frame"] == i:
                g, known = stage_group(r["stage"], IO_STAGES)
                if not known:
                    self.warn(f"étape non reconnue '{r['stage']}' comptée dans 'non instrumenté'")
                gt[g] += r["seconds"]
        tot_st = sum(gt.values())
        pwall = pw["wall_s"]
        coverage = tot_st / pwall if pwall > 0 else NAN
        reliable = isnum(coverage) and a.cov_min <= coverage <= a.cov_max
        if not reliable:
            self.warn(f"{direction} image {i} rép. {rep} : profiler_coverage={fmt(coverage, 4)} hors de "
                      f"[{a.cov_min}, {a.cov_max}] -> décomposition par composant NON FIABLE pour cette image")
        gt["uninstrumented"] += max(0.0, pwall - tot_st)
        denom = max(pwall, tot_st)
        if not (wall_task >= 10 * a.measure_power_secs):
            self.low_res[direction].add(i)
        row = {
            "level": self.k, "repeat": rep, "repeat_valid": "pending", "frame": i,
            "frame_file_index": a.start_index + i, "type": ftype, "direction": direction,
            "task": holder.get("name", ""),
            "time_task_s": wall_task, "time_profiler_s": pwall, "cc_duration_s": cc_dur,
            "outer_wall_s": holder.get("outer_wall_s", NAN), "tracker_overhead_s": holder.get("tracker_overhead_s", NAN),
            "gap_task_minus_profiler_s": wall_task - pwall,
            "gap_task_minus_cc_s": wall_task - cc_dur if isnum(cc_dur) else NAN,
            "sampling_period_over_wall_pct": 100 * a.measure_power_secs / wall_task if wall_task > 0 else NAN,
            "profiler_coverage": coverage, "decomposition_reliable": reliable,
            "energy_gross_j": e_g, "energy_cpu_gross_j": e_c, "energy_ram_gross_j": e_r,
            "cpu_share_of_energy": e_c / e_g if e_g else NAN,
            "mean_power_gross_w": e_g / wall_task if wall_task > 0 else NAN,
            "cpu_power_w_last": holder.get("cpu_power_w_last", NAN), "ram_power_w_last": holder.get("ram_power_w_last", NAN),
            "idle_power_w_used": idle["power_total_w"], "idle_source": idle["name"],
            "energy_net_secondary_j": net, "energy_net_negative": net < 0,
            "energy_resolution_ok_ge_10x_sampling": wall_task >= 10 * a.measure_power_secs, **extra}
        comp = []
        for g in GROUP_KEYS:
            t = gt.get(g, 0.0)
            share = t / denom if denom > 0 else NAN
            comp.append({"level": self.k, "repeat": rep, "repeat_valid": "pending", "frame": i, "type": ftype,
                         "direction": direction, "group": g, "group_label": GROUP_LABEL[g], "time_s": t,
                         "time_share": share, "decomposition_reliable": reliable,
                         "energy_gross_estimated_j": e_g * share,
                         "energy_net_secondary_estimated_j": net * share,
                         "energy_nature": "ESTIMEE (part de temps x energie de l'image)"})
        self.frames_rows.append(row)
        self.comp_rows.extend(comp)
        self.sink_frames.append([row])
        self.sink_comp.append(comp)
        return row

    # ---------- une repetition complete et independante
    def run_repetition(self, rep, idle):
        from PIL import Image
        from torchvision.transforms import ToTensor
        M, a, tbp = self.M, self.a, self.M["tbp"]
        n, gop = a.no_frames, a.gop
        Profiler = M["Profiler"]
        info = {"repeat": rep, "outside": {}}
        self.enc.reset_sequence()
        self.dec.reset_sequence()
        p_enc, p_dec = Profiler(), Profiler()
        bs_path = str(self.out / "bitstreams" / f"base_{self.k}_rep{rep:02d}.vcmb")
        rec_dir = self.out / "reconstructed" / f"rep{rep:02d}"
        rec_dir.mkdir(parents=True, exist_ok=False)
        info["bitstream_path"], info["rec_dir"] = bs_path, str(rec_dir)
        self.partial[rep] = info
        hdr = M["SequenceHeader"](self.W, self.H, n, gop, ckpt_id=self.k, closed_gop=a.closed_gop)
        writer = M["BitstreamWriter"](bs_path, hdr)
        pix = self.H * self.W
        t_rep = time.perf_counter()

        log(f"rép. {rep + 1}/{a.repeats} — ENCODAGE de {n} images (1 tâche CodeCarbon par image)")
        for i in range(n):
            ftype = "I" if i % gop == 0 else "P"
            path = tbp._frame_path(a.inp_path, a.prefix, a.start_index + i)
            name = f"L{self.k}_r{rep:02d}_enc_frame{i:03d}"
            with self.task(name) as holder:
                with p_enc.frame(i, ftype, "enc"):
                    with p_enc.stage("image_read"):
                        img = Image.open(path).convert("RGB")
                    with p_enc.stage("preprocess_to_tensor"):
                        x = ToTensor()(img).unsqueeze(0)
                    r = self.enc.encode_frame(x, i, gop, p_enc)
                    with p_enc.stage("bitstream_write"):          # EN MEMOIRE (add_frame) : pas d'E/S disque
                        rec_bytes = writer.add_frame(r["payload"], r["ranges"])
            holder["name"] = name
            hdr.shapes.update({kk: tuple(v[1:]) for kk, v in r["shapes"].items()})
            fr = self.record(rep, "enc", i, ftype, holder, p_enc,
                             {"payload_bytes": len(r["payload"]), "bpp_frame": 8 * rec_bytes / pix}, idle)
            log(f"  r{rep + 1} enc {i + 1}/{n} [{ftype}] {fr['time_task_s']:.2f} s | {fr['energy_gross_j']:.1f} J brut | "
                f"cov {fmt(fr['profiler_coverage'], 4)} | écoulé {hms(time.perf_counter() - self.t0_run)}")
        t0 = time.perf_counter()
        sizes = writer.close()                                     # ecriture reelle du fichier : HORS tache
        info["outside"]["bitstream_file_write_close_s"] = time.perf_counter() - t0
        info["sizes"] = sizes

        log(f"rép. {rep + 1}/{a.repeats} — DÉCODAGE de {n} images depuis {Path(bs_path).name}")
        t0 = time.perf_counter()
        rd = M["BitstreamReader"](bs_path)                         # ouverture : HORS tache
        info["outside"]["bitstream_open_s"] = time.perf_counter() - t0
        h = rd.header
        self.dec.reset_sequence()
        self.dec.closed_gop = h.closed_gop
        decoded = 0
        try:
            for i in range(h.n_frames):
                ftype = "I" if i % h.gop == 0 else "P"
                name = f"L{self.k}_r{rep:02d}_dec_frame{i:03d}"
                with self.task(name) as holder:
                    with p_dec.frame(i, ftype, "dec"):
                        with p_dec.stage("bitstream_read"):
                            payload, ranges, _ = rd.next_frame()
                        rec = self.dec.decode_frame(payload, ranges, i, h.gop, h.shapes, (h.height, h.width), p_dec)
                        with p_dec.stage("image_write"):
                            tbp._save_png(rec, tbp._frame_path(str(rec_dir), a.prefix, a.start_index + i))
                holder["name"] = name
                decoded += 1
                fr = self.record(rep, "dec", i, ftype, holder, p_dec, {"payload_bytes": len(payload), "bpp_frame": NAN}, idle)
                log(f"  r{rep + 1} dec {i + 1}/{h.n_frames} [{ftype}] {fr['time_task_s']:.2f} s | "
                    f"{fr['energy_gross_j']:.1f} J brut | cov {fmt(fr['profiler_coverage'], 4)} | "
                    f"écoulé {hms(time.perf_counter() - self.t0_run)}")
        finally:
            t0 = time.perf_counter()
            rd.close()
            info["outside"]["bitstream_close_read_s"] = time.perf_counter() - t0
        info["decoded"] = decoded
        info["rep_wall_s"] = time.perf_counter() - t_rep
        p_enc.export(str(self.out / "metrics" / f"base_{self.k}_rep{rep:02d}_enc_profile"),
                     meta={**self.M["tbp"]._env_meta(a), **sizes, "repeat": rep})
        p_dec.export(str(self.out / "metrics" / f"base_{self.k}_rep{rep:02d}_dec_profile"),
                     meta={**self.M["tbp"]._env_meta(a), "repeat": rep})
        info["bitstream_path"] = bs_path
        info["rec_dir"] = str(rec_dir)
        return info

    # ---------- validation automatique d'une repetition
    def validate(self, rep, info, error=None):
        from PIL import Image
        import numpy as np
        a, M, tbp = self.a, self.M, self.M["tbp"]
        n = a.no_frames
        errs, checks = [], {}
        if error:
            errs.append(f"exception pendant la répétition : {error}")
        rows = [r for r in self.frames_rows if r["repeat"] == rep]
        enc = {r["frame"]: r for r in rows if r["direction"] == "enc"}
        dec = {r["frame"]: r for r in rows if r["direction"] == "dec"}
        checks["n_encoded"], checks["n_decoded"] = len(enc), len(dec)
        if len(enc) != n:
            errs.append(f"images encodées : {len(enc)} != {n}")
        if len(dec) != n:
            errs.append(f"images décodées : {len(dec)} != {n}")
        if set(enc) != set(dec):
            errs.append(f"indices encodés != indices décodés (manquants au décodage : {sorted(set(enc) - set(dec))})")
        bs = info.get("bitstream_path") if info else None
        checks["bitstream_exists"] = bool(bs and os.path.isfile(bs) and os.path.getsize(bs) > 0)
        if not checks["bitstream_exists"]:
            errs.append("bitstream absent ou vide")
        else:
            try:
                rd = M["BitstreamReader"](bs)
                h = rd.header
                if (h.n_frames, h.gop, h.width, h.height) != (n, a.gop, self.W, self.H):
                    errs.append(f"en-tête du flux incohérent : {(h.n_frames, h.gop, h.width, h.height)} != "
                                f"{(n, a.gop, self.W, self.H)}")
                cnt = 0
                try:
                    while True:
                        pl, _, _ = rd.next_frame()
                        if cnt in enc and len(pl) != enc[cnt]["payload_bytes"]:
                            errs.append(f"image {cnt} : taille de payload du flux != taille à l'encodage")
                        cnt += 1
                        if cnt > n:
                            break
                except EOFError:
                    pass
                rd.close()
                checks["frames_in_bitstream"] = cnt
                if cnt != n:
                    errs.append(f"enregistrements dans le flux : {cnt} != {n}")
            except Exception as e:
                errs.append(f"flux illisible : {e!r}")
        # correspondance encodage / decodage
        for i in sorted(set(enc) & set(dec)):
            if enc[i]["type"] != dec[i]["type"]:
                errs.append(f"image {i} : type I/P encodé != décodé")
            if enc[i]["payload_bytes"] != dec[i]["payload_bytes"]:
                errs.append(f"image {i} : payload encodé ({enc[i]['payload_bytes']}) != décodé ({dec[i]['payload_bytes']})")
        # reconstructions
        rec_dir = Path(info["rec_dir"]) if info else None
        psnrs, rec_hash = [], hashlib.sha256()
        missing, invalid, wrong_size = [], [], []
        for i in range(n):
            p = tbp._frame_path(str(rec_dir), a.prefix, a.start_index + i) if rec_dir else None
            if not p or not os.path.isfile(p):
                missing.append(i)
                continue
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.size != (self.W, self.H):
                        wrong_size.append((i, im.size))
                    rec = np.asarray(im.convert("RGB"), dtype=np.float64)
                rec_hash.update(Path(p).read_bytes())
                with Image.open(tbp._frame_path(a.inp_path, a.prefix, a.start_index + i)) as io:
                    orig = np.asarray(io.convert("RGB"), dtype=np.float64)
                if orig.shape == rec.shape:
                    mse = float(np.mean((orig - rec) ** 2))
                    psnrs.append(10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"))
            except Exception as e:
                invalid.append((i, repr(e)))
        checks.update({"reconstructions_missing": missing, "reconstructions_invalid": invalid,
                       "reconstructions_wrong_size": wrong_size})
        if missing:
            errs.append(f"reconstructions manquantes : images {missing}")
        if invalid:
            errs.append(f"reconstructions illisibles : {invalid}")
        if wrong_size:
            errs.append(f"dimensions des reconstructions != {(self.W, self.H)} : {wrong_size}")
        if rec_dir and rec_dir.is_dir():
            extra = sorted(set(os.listdir(rec_dir)) - {Path(tbp._frame_path("", a.prefix, a.start_index + i)).name
                                                       for i in range(n)})
            if extra:
                checks["reconstructions_unexpected_files"] = extra
        fin = [p for p in psnrs if math.isfinite(p)]
        checks["psnr_dec_png8_db_mean"] = sum(fin) / len(fin) if fin else NAN
        checks["psnr_dec_png8_db_min"] = min(fin) if fin else NAN
        #if fin and min(fin) < a.min_psnr_db:
            #errs.append(f"PSNR minimal {min(fin):.2f} dB < {a.min_psnr_db} dB : reconstruction aberrante probable")
        checks["bitstream_sha256"] = sha256_file(bs) if checks["bitstream_exists"] else None
        checks["reconstruction_sha256"] = rec_hash.hexdigest()
        ok = not errs
        v = {"repeat": rep, "valid": ok, "errors": errs, "checks": checks}
        self.validation[rep] = v
        if not ok:
            for e in errs:
                self.errors.append(f"rép. {rep} : {e}")
                log(f"[ERREUR VALIDATION] rép. {rep} : {e}")
        else:
            log(f"validation rép. {rep + 1} : OK (PSNR PNG8 moyen {fmt(checks['psnr_dec_png8_db_mean'], 2)} dB)")
        # independance : comparaison avec la repetition valide de reference
        ref = next((self.validation[r] for r in sorted(self.validation) if r != rep and self.validation[r]["valid"]), None)
        if ok and ref:
            if ref["checks"]["bitstream_sha256"] != checks["bitstream_sha256"]:
                self.warn(f"rép. {rep} : bitstream différent de celui de la rép. {ref['repeat']} (codec non déterministe "
                          f"d'une passe à l'autre ?) — à examiner")
            if ref["checks"]["reconstruction_sha256"] != checks["reconstruction_sha256"]:
                self.warn(f"rép. {rep} : reconstructions différentes de celles de la rép. {ref['repeat']}")
        return ok

    def finalize_repetition(self, rep, ok):
        for r in self.frames_rows:
            if r["repeat"] == rep:
                r["repeat_valid"] = bool(ok)
        for r in self.comp_rows:
            if r["repeat"] == rep:
                r["repeat_valid"] = bool(ok)
        if not (self.sink_frames.rewrite(self.frames_rows) and self.sink_comp.rewrite(self.comp_rows)):
            self.warn("réécriture des CSV impossible (fichier ouvert ailleurs ?) : repeat_valid reste 'pending' dans les CSV ; "
                      "voir repeat_summary.csv / validation.json")
        write_csv(self.out / "energy_tasks.csv", self.task_rows)


# --------------------------------------------------------------------------------------
# statistiques (uniquement sur les repetitions VALIDES)
# --------------------------------------------------------------------------------------
def frame_stats(frames_rows):
    g = defaultdict(list)
    for r in frames_rows:
        g[(r["direction"], r["frame"], r["type"])].append(r)
    out = []
    for (d, i, t), rs in sorted(g.items()):
        row = {"direction": d, "frame": i, "type": t, "n_repeats": len(rs)}
        for key in FRAME_KEYS:
            row.update(flat(key, agg([r[key] for r in rs])))
        row["decomposition_reliable_all_repeats"] = all(r["decomposition_reliable"] for r in rs)
        out.append(row)
    return out


def type_stats(frames_rows):
    """Par (direction, type). mean / std_between_repeats / min / max / cv_between_repeats : sur les moyennes PAR
    REPETITION (moyenne des images du type dans la passe) ; std_between_frames : ecart-type entre les moyennes
    par image (moyennees sur les repetitions), NaN s'il n'y a qu'une image de ce type."""
    out = {}
    for d in ("enc", "dec"):
        for t in ("I", "P"):
            rows = [r for r in frames_rows if r["direction"] == d and r["type"] == t]
            if not rows:
                continue
            reps = sorted({r["repeat"] for r in rows})
            frs = sorted({r["frame"] for r in rows})
            s = {"n_frames": len(frs), "n_repeats": len(reps)}
            for key in TYPE_KEYS:
                per_rep = [agg([r[key] for r in rows if r["repeat"] == rp])["mean"] for rp in reps]
                per_frame = [agg([r[key] for r in rows if r["frame"] == fr])["mean"] for fr in frs]
                ar = agg(per_rep)
                s[key] = {"mean": ar["mean"], "std_between_repeats": ar["std"], "min": ar["min"], "max": ar["max"],
                          "cv_between_repeats": ar["cv"], "std_between_frames": agg(per_frame)["std"]}
            s["n_net_negative"] = sum(1 for r in rows if r["energy_net_negative"])
            s["n_measurements"] = len(rows)
            out[f"{d}_{t}"] = s
    for d in ("enc", "dec"):
        rows = [r for r in frames_rows if r["direction"] == d]
        if not rows:
            continue
        reps = sorted({r["repeat"] for r in rows})
        s = {"n_repeats": len(reps)}
        for key in ("time_task_s", "energy_gross_j"):
            s[key + "_sum_over_images_per_pass"] = flat(
                "v", agg([sum(r[key] for r in rows if r["repeat"] == rp) for rp in reps]))
        out[f"{d}_phase_total"] = s
    return out


def component_stats_by_frame(comp_rows):
    g = defaultdict(list)
    for r in comp_rows:
        g[(r["direction"], r["frame"], r["type"], r["group"])].append(r)
    out = []
    for (d, i, t, grp), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], GROUP_KEYS.index(kv[0][3]))):
        row = {"direction": d, "frame": i, "type": t, "group": grp, "group_label": GROUP_LABEL[grp], "n_repeats": len(rs)}
        row.update(flat("time_s", agg([r["time_s"] for r in rs])))
        row.update(flat("time_share", agg([r["time_share"] for r in rs])))
        row.update(flat("energy_gross_estimated_j", agg([r["energy_gross_estimated_j"] for r in rs])))
        row["decomposition_reliable_all_repeats"] = all(r["decomposition_reliable"] for r in rs)
        row["energy_nature"] = "ESTIMEE (proxy temps x energie de l'image)"
        out.append(row)
    return out


def component_stats_by_type(comp_rows):
    g = defaultdict(list)
    for r in comp_rows:
        g[(r["direction"], r["type"], r["group"])].append(r)
    out = []
    for (d, t, grp), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], GROUP_KEYS.index(kv[0][2]))):
        reps = sorted({r["repeat"] for r in rs})
        frs = sorted({r["frame"] for r in rs})
        row = {"direction": d, "type": t, "group": grp, "group_label": GROUP_LABEL[grp],
               "n_frames": len(frs), "n_repeats": len(reps)}
        for key in ("time_s", "time_share", "energy_gross_estimated_j"):
            per_rep = [agg([r[key] for r in rs if r["repeat"] == rp])["mean"] for rp in reps]
            per_frame = [agg([r[key] for r in rs if r["frame"] == fr])["mean"] for fr in frs]
            row.update(flat(key, agg(per_rep)))                  # ecart-type ENTRE REPETITIONS
            row[key + "_std_between_frames"] = agg(per_frame)["std"]
        row["decomposition_reliable_all"] = all(r["decomposition_reliable"] for r in rs)
        row["energy_nature"] = "ESTIMEE (proxy temps x energie de l'image)"
        out.append(row)
    return out


def repeat_summary(exp, rep_infos):
    out = []
    for rep in sorted(exp.validation):
        v = exp.validation[rep]
        rows = [r for r in exp.frames_rows if r["repeat"] == rep]
        info = rep_infos.get(rep, {})
        row = {"repeat": rep, "valid": v["valid"], "n_errors": len(v["errors"]),
               "errors": " | ".join(v["errors"]),
               "idle_power_w_used": rows[0]["idle_power_w_used"] if rows else NAN,
               "idle_source": rows[0]["idle_source"] if rows else "",
               "bitstream_bytes": (info.get("sizes") or {}).get("total_bytes", NAN)}
        for d in ("enc", "dec"):
            rd = [r for r in rows if r["direction"] == d]
            row[f"{d}_time_sum_s"] = sum(r["time_task_s"] for r in rd) if rd else NAN
            row[f"{d}_energy_gross_sum_j"] = sum(r["energy_gross_j"] for r in rd) if rd else NAN
            row[f"{d}_tracker_overhead_sum_s"] = sum(r["tracker_overhead_s"] for r in rd) if rd else NAN
            for t in ("I", "P"):
                rt = [r for r in rd if r["type"] == t]
                row[f"{d}_{t}_time_mean_s"] = agg([r["time_task_s"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_gross_mean_j"] = agg([r["energy_gross_j"] for r in rt])["mean"]
        cov = [r["profiler_coverage"] for r in rows if isnum(r["profiler_coverage"])]
        row["coverage_min"], row["coverage_max"] = (min(cov), max(cov)) if cov else (NAN, NAN)
        row["n_frames_decomposition_unreliable"] = sum(1 for r in rows if not r["decomposition_reliable"])
        row["n_net_negative"] = sum(1 for r in rows if r["energy_net_negative"])
        row["psnr_dec_png8_db_mean"] = v["checks"].get("psnr_dec_png8_db_mean", NAN)
        for k_, lab in (("bitstream_file_write_close_s", "outside_bitstream_file_write_close_s"),
                        ("bitstream_open_s", "outside_bitstream_open_s"),
                        ("bitstream_close_read_s", "outside_bitstream_close_read_s")):
            row[lab] = (info.get("outside") or {}).get(k_, NAN)
        row["rep_wall_s"] = info.get("rep_wall_s", NAN)
        row["bitstream_sha256"] = v["checks"].get("bitstream_sha256")
        row["reconstruction_sha256"] = v["checks"].get("reconstruction_sha256")
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
# graphiques
# --------------------------------------------------------------------------------------
def make_figures(exp, fstats, cs_frame, cs_type, tstats, valid_rows, label, unreliable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    a, k = exp.a, exp.k
    figdir = exp.out / "figures"
    figdir.mkdir(exist_ok=True)
    COL = {"I": "#8c564b", "P": "#2ca02c"}
    made = []
    reps_used = sorted({r["repeat"] for r in valid_rows})
    foot = (f"Niveau {k} · GOP={a.gop} · {a.no_frames} images · {len(reps_used)} répétition(s) valide(s) · "
            f"barres d'erreur = écart-type entre répétitions · MESURÉ = temps ; énergie = CodeCarbon (nature non vérifiée, "
            f"RAM modélisée) ; ESTIMÉ = énergie par composant")

    def done(fig, name):
        fig.text(0.01, 0.004, foot, fontsize=5.8, color="#555", va="bottom")
        if unreliable and name.startswith(("fig5", "fig6", "fig8")):
            fig.text(0.5, 0.97, "⚠ DÉCOMPOSITION NON FIABLE (profiler_coverage hors limites sur ≥ 1 image)",
                     color="red", ha="center", fontsize=9, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(figdir / f"{name}.png", dpi=300)
        fig.savefig(figdir / f"{name}.pdf")
        plt.close(fig)
        made.append(name)

    def err(sd):
        return sd if isnum(sd) else 0.0

    # fig1 / fig2 : par image, I et P distingues
    for name, key, yl, ttl in (
            ("fig1_energy_per_frame", "energy_gross_j", "Énergie brute par image (J, CPU+RAM, CodeCarbon)",
             "Énergie brute par image [CodeCarbon, nature non vérifiée]"),
            ("fig2_time_per_frame", "time_task_s", "Temps de traitement par image (s)", "Temps par image [MESURÉ]")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
        for ax, (d, dn) in zip(axes, (("enc", "Encodage"), ("dec", "Décodage"))):
            rs = sorted([r for r in fstats if r["direction"] == d], key=lambda r: r["frame"])
            ax.plot([r["frame"] for r in rs], [r[key + "_mean"] for r in rs], "-", color="#999", lw=.8, zorder=1)
            for t, mk, sz in (("I", "D", 90), ("P", "o", 40)):
                sub = [r for r in rs if r["type"] == t]
                if sub:
                    ax.errorbar([r["frame"] for r in sub], [r[key + "_mean"] for r in sub],
                                yerr=[err(r[key + "_std"]) for r in sub], fmt=mk, color=COL[t], capsize=3,
                                ms=sz ** .5, label=f"Image {t}", zorder=3)
            raw = [(r["frame"], r[key]) for r in valid_rows if r["direction"] == d]
            ax.scatter([x for x, _ in raw], [y for _, y in raw], s=5, color="k", alpha=.25, zorder=2,
                       label="répétitions individuelles")
            ax.set_title(dn)
            ax.set_xlabel("Numéro d'image (0 = image I)")
            ax.set_ylabel(yl, fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.suptitle(ttl, fontweight="bold")
        done(fig, name)

    # fig3 : encodage vs decodage
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (key, yl) in zip(axes, (("time_task_s", "Temps moyen par image (s) [MESURÉ]"),
                                    ("energy_gross_j", "Énergie brute moyenne par image (J) [CodeCarbon]"))):
        w = .36
        for j, (d, dn, c) in enumerate((("enc", "Encodage", "#1f77b4"), ("dec", "Décodage", "#d62728"))):
            for xi, t in enumerate(("I", "P")):
                s = tstats.get(f"{d}_{t}")
                if s:
                    ax.bar(xi + (j - .5) * w, s[key]["mean"], w, yerr=err(s[key]["std_between_repeats"]), capsize=4,
                           color=c, alpha=.85, edgecolor="k", lw=.5, label=dn if xi == 0 else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Image I", "Image P"])
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(axis="y", alpha=.3)
        ax.legend()
    fig.suptitle("Encodage vs décodage (barre = moyenne ± écart-type entre répétitions)", fontweight="bold")
    done(fig, "fig3_encode_vs_decode")

    # fig4 : I vs P
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (d, key, yl) in zip(axes.ravel(), (("enc", "time_task_s", "Temps (s/image) [MESURÉ]"),
                                               ("dec", "time_task_s", "Temps (s/image) [MESURÉ]"),
                                               ("enc", "energy_gross_j", "Énergie brute (J/image) [CodeCarbon]"),
                                               ("dec", "energy_gross_j", "Énergie brute (J/image) [CodeCarbon]"))):
        labels = []
        for t in ("I", "P"):
            s = tstats.get(f"{d}_{t}")
            if not s:
                continue
            ax.bar(len(labels), s[key]["mean"], yerr=err(s[key]["std_between_repeats"]), capsize=5, color=COL[t],
                   alpha=.8, edgecolor="k", lw=.6)
            pts = [r[key + "_mean"] for r in fstats if r["direction"] == d and r["type"] == t]
            ax.scatter([len(labels)] * len(pts), pts, color="k", s=14, zorder=3)
            labels.append(f"{t}\n({s['n_frames']} img × {s['n_repeats']} rép.)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(("Encodage" if d == "enc" else "Décodage"))
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Image I vs image P (barre ± écart-type entre répétitions ; points = moyennes par image)",
                 fontweight="bold")
    done(fig, "fig4_I_vs_P")

    # fig5 / fig6 : composants (temps mesure / energie estimee)
    for name, key, xl, ttl, hatch in (
            ("fig5_component_time", "time_s", "Temps (s) [MESURÉ]", "Temps par composant [MESURÉ, perf_counter]", None),
            ("fig6_component_energy_estimated", "energy_gross_estimated_j", "Énergie brute ESTIMÉE (J)",
             "Énergie par composant [ESTIMÉE = part de temps × énergie de l'image]", "//")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        for ax, (d, t) in zip(axes.ravel(), (("enc", "I"), ("enc", "P"), ("dec", "I"), ("dec", "P"))):
            rs = [r for r in cs_type if r["direction"] == d and r["type"] == t and r[key + "_mean"] > 0]
            rs.sort(key=lambda r: r[key + "_mean"])
            ax.barh([r["group_label"] for r in rs], [r[key + "_mean"] for r in rs],
                    xerr=[err(r[key + "_std"]) for r in rs], color=COL[t], alpha=.8, edgecolor="k", lw=.4,
                    hatch=hatch, capsize=2)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_title(f"{'Encodage' if d == 'enc' else 'Décodage'} — image {t}", fontsize=10)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="x", alpha=.3)
        fig.suptitle(ttl, fontweight="bold")
        done(fig, name)

    # fig7 : variabilite entre repetitions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (d, key, yl) in zip(axes.ravel(), (("enc", "time_task_s", "Temps moyen (s) [MESURÉ]"),
                                               ("dec", "time_task_s", "Temps moyen (s) [MESURÉ]"),
                                               ("enc", "energy_gross_j", "Énergie brute moyenne (J) [CodeCarbon]"),
                                               ("dec", "energy_gross_j", "Énergie brute moyenne (J) [CodeCarbon]"))):
        for t in ("I", "P"):
            ys, es = [], []
            for rp in reps_used:
                v = [r[key] for r in valid_rows if r["direction"] == d and r["type"] == t and r["repeat"] == rp]
                m, sd = mean_std(v)
                ys.append(m)
                es.append(err(sd))
            s = tstats.get(f"{d}_{t}")
            if not s:
                continue
            ax.errorbar(reps_used, ys, yerr=es if t == "P" else None, fmt="-o", color=COL[t], capsize=3, ms=4,
                        label=f"{t} : CV entre rép. = {fmt(100 * s[key]['cv_between_repeats'], 1)} %")
            ax.axhline(s[key]["mean"], color=COL[t], ls="--", lw=.8)
            if isnum(s[key]["std_between_repeats"]):
                ax.axhspan(s[key]["mean"] - s[key]["std_between_repeats"], s[key]["mean"] + s[key]["std_between_repeats"],
                           color=COL[t], alpha=.12)
        ax.set_xlabel("Numéro de répétition")
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title("Encodage" if d == "enc" else "Décodage")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("Variabilité entre répétitions (points = moyenne de la passe ; barre d'erreur P = écart-type entre images ; "
                 "bande = ±1 écart-type entre répétitions)", fontweight="bold", fontsize=9)
    done(fig, "fig7_variability_across_repeats")

    # fig8 : part de temps par composant
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cmap = plt.get_cmap("tab20")
    for ax, (d, dn) in zip(axes, (("enc", "Encodage"), ("dec", "Décodage"))):
        types = [t for t in ("I", "P") if any(r["type"] == t and r["direction"] == d for r in cs_type)]
        left = [0.0] * len(types)
        for gi, g in enumerate(GROUP_KEYS):
            sh = []
            for t in types:
                v = [r["time_share_mean"] for r in cs_type if r["group"] == g and r["type"] == t and r["direction"] == d]
                sh.append(100 * v[0] if v and isnum(v[0]) else 0.0)
            if not any(s > 0 for s in sh):
                continue
            ax.barh(types, sh, left=left, color=cmap(gi), edgecolor="w", label=GROUP_LABEL[g])
            left = [l_ + s for l_, s in zip(left, sh)]
        ax.set_xlabel("Part du temps de l'image (%) [MESURÉ] = clé de l'estimation d'énergie")
        ax.set_title(dn)
        ax.set_xlim(0, 100)
    axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, .5))
    fig.suptitle("Part temporelle de chaque composant", fontweight="bold")
    done(fig, "fig8_component_time_share")
    return made


# --------------------------------------------------------------------------------------
def print_summary(exp, tstats, cs_type, label):
    a = exp.a
    print("\n" + "=" * 104)
    print(f"RÉCAPITULATIF — niveau {exp.k} ({a.repeats} répétition(s) demandée(s) ; ± = écart-type ENTRE RÉPÉTITIONS)")
    print(f"[{label}]")
    print("=" * 104)
    print(f"| {'Phase / type':<14} | {'img':>3} | {'Temps moy (s)':>18} | {'E brute (J)':>18} | {'P moy (W)':>13} | {'E nette* neg.':>13} |")
    print("|" + "-" * 102 + "|")
    for key, s in tstats.items():
        if key.endswith("phase_total"):
            continue
        d, t = key.split("_")
        print(f"| {('Encodage ' if d == 'enc' else 'Décodage ') + t:<14} | {s['n_frames']:>3} | "
              f"{pm(s['time_task_s']['mean'], s['time_task_s']['std_between_repeats']):>18} | "
              f"{pm(s['energy_gross_j']['mean'], s['energy_gross_j']['std_between_repeats']):>18} | "
              f"{pm(s['mean_power_gross_w']['mean'], s['mean_power_gross_w']['std_between_repeats'], 2):>13} | "
              f"{s['n_net_negative']:>5}/{s['n_measurements']:<7} |")
    print("(* énergie nette = indicateur secondaire, non utilisée comme résultat)")
    print("\nCOMPOSANTS (temps = mesuré ; énergie = ESTIMÉE par part de temps)")
    for d in ("enc", "dec"):
        for t in ("I", "P"):
            rows = [r for r in cs_type if r["direction"] == d and r["type"] == t and r["time_s_mean"] > 0]
            if not rows:
                continue
            print(f"\n  {'Encodage' if d == 'enc' else 'Décodage'} — image {t}")
            for r in sorted(rows, key=lambda r: -r["time_s_mean"]):
                print(f"    {r['group_label']:<48} {pm(r['time_s_mean'], r['time_s_std']):>18} s "
                      f"{100 * r['time_share_mean']:>6.1f} % {fmt(r['energy_gross_estimated_j_mean']):>9} J (est.)")


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="VCM-EnergyMap exp.3 v3 : énergie/temps par image (I/P) et par composant.")
    ap.add_argument("--no_frames", type=int, default=10, help="nombre d'images")
    ap.add_argument("--repeats", type=int, default=8,
                    help="passes COMPLÈTES indépendantes (>= 8 pour le résultat final ; 2 pour le pilote)")
    ap.add_argument("--gop", type=int, default=32)
    ap.add_argument("--checkpoint", type=int, default=1, choices=[1, 2, 3, 4], help="niveau de qualité (checkpoint_base_k)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--run_name", default="experiment3_v3")
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--inp_path", default=str(ROOT / "videos_5" / "input_png"))
    ap.add_argument("--prefix", default="frame_")
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "base"))
    ap.add_argument("--root", default=str(ROOT), help="racine du dépôt (./models, ./config)")
    ap.add_argument("--results_root", default=str(ROOT / "results" / "VCM_EnergyMap"))
    ap.add_argument("--warmup_frames", type=int, default=2)
    ap.add_argument("--idle_seconds", type=float, default=30.0, help="baseline à vide initiale")
    ap.add_argument("--idle_between_seconds", type=float, default=15.0,
                    help="baseline courte avant chaque répétition suivante (0 = désactivée : réutilise la baseline initiale)")
    ap.add_argument("--measure_power_secs", type=float, default=1.0)
    ap.add_argument("--country", default="MAR")
    ap.add_argument("--cooldown", type=float, default=30.0, help="pause avant la baseline initiale")
    ap.add_argument("--closed_gop", action="store_true")
    ap.add_argument("--cov_min", type=float, default=0.90, help="couverture minimale du profiler (fiabilité)")
    ap.add_argument("--cov_max", type=float, default=1.001, help="couverture maximale (au-dessus : double comptage)")
    #ap.add_argument("--min_psnr_db", type=float, default=20.0, help="PSNR minimal (PNG 8 bits) pour valider une reconstruction")
    ap.add_argument("--dry_run", action="store_true", help="contrôles + plan, sans construire le codec ni mesurer")
    ap.add_argument("--est_enc_I_s", type=float, default=66.1, help="estimation s, encodage I (pilote V2)")
    ap.add_argument("--est_enc_P_s", type=float, default=43.9)
    ap.add_argument("--est_dec_I_s", type=float, default=74.7)
    ap.add_argument("--est_dec_P_s", type=float, default=33.1)
    ap.add_argument("--max_hours", type=float, default=2.0)
    return ap


def estimate_seconds(a):
    n, gop = a.no_frames, a.gop
    n_i = len([i for i in range(n) if i % gop == 0])
    n_p = n - n_i
    per_pass = n_i * (a.est_enc_I_s + a.est_dec_I_s) + n_p * (a.est_enc_P_s + a.est_dec_P_s)
    nw = min(a.warmup_frames, n)
    warm = sum((a.est_enc_I_s + a.est_dec_I_s) if i % gop == 0 else (a.est_enc_P_s + a.est_dec_P_s) for i in range(nw))
    idle = a.idle_seconds + max(0, a.repeats - 1) * a.idle_between_seconds
    return a.repeats * per_pass * 1.03 + warm + idle + a.cooldown + 60, per_pass


def main():
    a = build_parser().parse_args()
    n, gop, R = a.no_frames, a.gop, a.repeats
    if R < 1 or n < 1:
        sys.exit("[ERREUR] --repeats et --no_frames doivent être >= 1")
    miss = [f for f in REQUIRED if not (ROOT / f).exists()]
    if miss:
        sys.exit(f"[ERREUR] fichiers manquants dans {ROOT} : {miss}")
    out_dir = Path(a.results_root) / a.run_name / f"level_{a.checkpoint}"
    n_i = len([i for i in range(n) if i % gop == 0])
    frames_missing = [a.start_index + i for i in range(n)
                      if not os.path.exists(os.path.join(a.inp_path, f"{a.prefix}{a.start_index + i:03d}.png"))]
    ckpt = Path(a.ckpt_dir) / f"checkpoint_base_{a.checkpoint}.pth.tar"
    total, per_pass = estimate_seconds(a)

    print("=" * 80)
    print(f"PLAN — exp.3 {VERSION} : énergie/temps par image (I/P) et par composant")
    print("=" * 80)
    print(f"Images          : {n} (fichiers {a.prefix}{a.start_index:03d}..{a.prefix}{a.start_index + n - 1:03d}) ; "
          f"GOP={gop} -> {n_i} image(s) I + {n - n_i} P")
    print(f"Répétitions     : {R} passe(s) complète(s) indépendantes -> {R * n} mesures d'encodage + {R * n} de décodage")
    print(f"Checkpoint      : {ckpt}")
    print(f"Résultats       : {out_dir}")
    print(f"Durée estimée   : {hms(total)} (≈ {hms(per_pass)} par passe ; limite {a.max_hours:g} h)")
    errors, notes = [], []
    if frames_missing:
        errors.append(f"images introuvables : {frames_missing[:10]} dans {a.inp_path}")
    if not ckpt.exists():
        errors.append(f"checkpoint introuvable : {ckpt}")
    if out_dir.exists() and any(out_dir.iterdir()):
        errors.append(f"{out_dir} existe déjà : rien n'est écrasé ; choisissez un autre --run_name")
    if R < MIN_FINAL_REPEATS:
        notes.append(f"{R} répétition(s) < {MIN_FINAL_REPEATS} : exécution PILOTE, pas un résultat final")
    if R < 2:
        notes.append("1 seule répétition : écart-type entre répétitions non défini")
    if total > a.max_hours * 3600:
        errors.append(f"durée estimée {hms(total)} > {a.max_hours:g} h : réduisez --repeats/--no_frames ou passez --max_hours")
    if n_i == 1:
        notes.append("une seule image I par passe : sa variabilité vient uniquement des répétitions")
    cc = find_concurrent_codecarbon()
    if cc["processes"]:
        notes.append(f"autres processus Python liés à CodeCarbon/expériences : {cc['processes']} — fermez-les avant la mesure")
    if cc["lock_files"]:
        notes.append(f"fichiers CodeCarbon dans le dossier temporaire (lecture seule, non supprimés) : {cc['lock_files']}")
    if not cc["psutil"]:
        notes.append("psutil absent : détection des processus concurrents et métadonnées machine limitées")
    if importlib.util.find_spec("codecarbon") is None:
        errors.append("codecarbon n'est pas installé dans cet environnement")
    # verification d'API (imports lourds : torch, codec ; AUCUN modèle construit, AUCUN tracker lancé)
    M, api_problems = None, []
    try:
        M = load_modules()
        api_problems = check_api(M)
    except Exception as e:
        errors.append(f"import des modules du projet impossible : {e!r}")
    for p in api_problems:
        errors.append(f"API : {p}")
    if M is not None and not api_problems:
        print("API             : signatures vérifiées (BaseLayerCodec, Bitstream*, Profiler, EnergyMeter, test_base_profiled)")
    for nt in notes:
        print(f"[INFO] {nt}")
    for e in errors:
        print(f"[ERREUR] {e}")
    if a.dry_run:
        print("\n--dry_run : rien n'a été lancé." + (" Corrigez les erreurs ci-dessus." if errors else " Contrôles OK."))
        sys.exit(2 if errors else 0)
    if errors:
        sys.exit(2)

    M["torch"].set_num_threads(a.threads)
    out_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("bitstreams", "reconstructed", "metrics", "figures"):
        (out_dir / sub).mkdir()
    log.path = out_dir / "experiment_log.txt"
    exp = Experiment(a, M, out_dir)
    for nt in notes:
        exp.warn(nt)
    log(f"Début exp.3 {VERSION} : {n} images, {R} répétition(s), GOP={gop}, checkpoint {a.checkpoint}, threads={a.threads}")
    log("Fermez les autres applications, secteur branché, plan d'alimentation fixe, ne touchez pas à la machine.")
    status, rep_infos = "SUCCESS", {}
    meta = {"version": VERSION, "machine": collect_machine_metadata(a), "params": vars(a),
            "concurrent_codecarbon_check": cc, "api_check": "OK"}
    handler = None
    cc_msgs, py_warns = [], []
    try:
        exp.init_codecs()                       # HORS mesure (avant le tracker)
        # ---- tracker CodeCarbon : un seul, journaux et avertissements captures
        class _H(logging.Handler):
            def emit(self, rec):
                cc_msgs.append(f"{rec.levelname}: {rec.getMessage()}")
        handler = _H()
        handler.setLevel(logging.DEBUG)
        logging.getLogger("codecarbon").addHandler(handler)
        with pywarnings.catch_warnings(record=True) as wl:
            pywarnings.simplefilter("always")
            exp.meter = M["EnergyMeter"](country_iso_code=a.country, measure_power_secs=a.measure_power_secs)
        py_warns = [str(w.message) for w in wl]
        info = exp.meter.method_info()
        label = energy_label(info)
        log(f"Méthode d'énergie : {label}")
        if cc_msgs or py_warns:
            log(f"Messages CodeCarbon à l'initialisation : {cc_msgs + py_warns}")
        if any("Multiple instances" in m for m in cc_msgs + py_warns):
            exp.warn("CodeCarbon signale « Multiple instances ... allowed » : en 3.x allow_multiple_runs=True par défaut "
                     "(avertissement seulement). Vérifiez qu'aucun autre tracker/script ne tourne (voir metadata.json -> "
                     "concurrent_codecarbon_check) ; les tâches de CE script sont strictement séquentielles (contrôle ci-dessous).")
        mach = meta["machine"]
        if str(info.get("CPU_mode")).lower() == "windows_emi" and mach.get("windows_build", 0) < 22000:
            exp.warn("CPU_mode=windows_emi sous Windows 10 : la doc CodeCarbon indique qu'EMI ne rapporte la puissance CPU que "
                     "sur du matériel de mesure dédié ; présence de compteurs NON vérifiée ici -> énergie CodeCarbon, "
                     "pas une mesure électrique confirmée.")
        bat = mach.get("battery")
        if bat and not bat.get("power_plugged"):
            exp.warn("le portable fonctionne sur batterie : branchez le secteur")
        meta.update({"energy_method": info, "energy_label": label, "codecarbon_log_messages": cc_msgs + py_warns})
        dump_json(out_dir / "metadata.json", meta)

        log(f"Pause {a.cooldown:g} s puis baseline à vide {a.idle_seconds:g} s")
        time.sleep(a.cooldown)
        idle0 = exp.measure_idle("idle_initial", a.idle_seconds)
        log(f"Baseline initiale : {idle0['power_total_w']:.3f} W total (CPU {idle0['power_cpu_w']:.3f} W, RAM {idle0['power_ram_w']:.3f} W)")
        exp.idle_w, exp.idle_cpu_w = idle0["power_total_w"], idle0["power_cpu_w"]

        idles = [idle0]
        for rep in range(R):
            idle = idle0
            if rep > 0 and a.idle_between_seconds > 0:
                idle = exp.measure_idle(f"idle_before_rep{rep:02d}", a.idle_between_seconds)
                idles.append(idle)
                log(f"Baseline avant rép. {rep + 1} : {idle['power_total_w']:.3f} W total")
            info_r, err_msg = None, None
            try:
                info_r = exp.run_repetition(rep, idle)
            except KeyboardInterrupt:
                status = "INTERRUPTED"
                exp.errors.append(f"rép. {rep} interrompue par l'utilisateur")
                log("[INTERRUPTION] arrêt demandé : sauvegarde des mesures déjà validées")
                exp.validate(rep, info_r or exp.partial.get(rep, {}), error="interruption utilisateur")
                exp.finalize_repetition(rep, False)
                break
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                log(f"[ERREUR] rép. {rep} : {err_msg}\n{traceback.format_exc()}")
            info_v = info_r or exp.partial.get(rep, {})
            rep_infos[rep] = info_v
            ok = exp.validate(rep, info_v, error=err_msg)
            exp.finalize_repetition(rep, ok)
            dump_json(out_dir / "progress.json", {"repeats_requested": R, "repeats_done": rep + 1,
                                                  "valid": [r for r, v in exp.validation.items() if v["valid"]],
                                                  "errors": exp.errors, "elapsed_s": time.perf_counter() - exp.t0_run})
            if err_msg and exp.active is not None:
                exp.active = None
        meta["idle_baselines"] = idles
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        log("[INTERRUPTION] pendant l'initialisation")
    except Exception as e:
        status = "FAILED"
        exp.errors.append(f"{type(e).__name__}: {e}")
        log(f"[ERREUR FATALE] {e}\n{traceback.format_exc()}")
    finally:
        if exp.meter is not None:
            try:
                exp.meter.close()
            except Exception as e:
                log(f"[AVERTISSEMENT] fermeture du tracker : {e!r}")
        if handler is not None:
            logging.getLogger("codecarbon").removeHandler(handler)

    # ------------------------------------------------------------------ analyse
    valid_reps = sorted(r for r, v in exp.validation.items() if v["valid"])
    n_ok = len(valid_reps)
    if status == "SUCCESS" and n_ok != R:
        status = "PARTIAL" if n_ok > 0 else "FAILED"
    valid_rows = [r for r in exp.frames_rows if r["repeat"] in valid_reps]
    valid_comp = [r for r in exp.comp_rows if r["repeat"] in valid_reps]
    write_csv(out_dir / "energy_tasks.csv", exp.task_rows)
    write_csv(out_dir / "repeat_summary.csv", repeat_summary(exp, rep_infos))
    dump_json(out_dir / "validation.json", exp.validation)

    # controle de non-chevauchement des taches CodeCarbon (fenetres perf_counter)
    ts = sorted(exp.task_rows, key=lambda r: r["t_enter_s"])
    overlaps = [(x["task"], y["task"]) for x, y in zip(ts, ts[1:]) if x["t_exit_s"] > y["t_enter_s"] + 1e-9]
    if overlaps:
        exp.warn(f"chevauchement de tâches CodeCarbon détecté : {overlaps[:5]}")
    label = meta.get("energy_label", "énergie CodeCarbon (nature non déterminée)")
    summary = {"experiment": "exp3_frame_components_v3", "status": status, "level": exp.k,
               "repeats_requested": R, "repeats_completed": len(exp.validation), "repeats_valid": n_ok,
               "is_final_protocol": bool(n_ok >= MIN_FINAL_REPEATS and status == "SUCCESS"),
               "errors": exp.errors, "warnings": exp.warnings, "energy_label": label,
               "task_overlap_check": {"n_tasks": len(ts), "overlaps": overlaps},
               "codecarbon_tasks_expected": 2 * R * n + R}
    tstats, cs_frame, cs_type, fstats = {}, [], [], []
    if valid_rows:
        fstats = frame_stats(valid_rows)
        cs_frame = component_stats_by_frame(valid_comp)
        cs_type = component_stats_by_type(valid_comp)
        tstats = type_stats(valid_rows)
        write_csv(out_dir / "frame_stats_across_repeats.csv", fstats)
        write_csv(out_dir / "component_stats_by_frame.csv", cs_frame)
        write_csv(out_dir / "component_stats_by_type.csv", cs_type)
        cov = [r["profiler_coverage"] for r in valid_rows if isnum(r["profiler_coverage"])]
        unreliable = [r for r in valid_rows if not r["decomposition_reliable"]]
        mp = [r["mean_power_gross_w"] for r in valid_rows if isnum(r["mean_power_gross_w"])]
        cpu_p = [r["cpu_power_w_last"] for r in valid_rows if isnum(r["cpu_power_w_last"])]
        ram_p = [r["ram_power_w_last"] for r in valid_rows if isnum(r["ram_power_w_last"])]
        neg = [r for r in valid_rows if r["energy_net_negative"]]
        idle_ref = exp.idle_w
        if neg:
            exp.warn(f"énergie nette négative pour {len(neg)}/{len(valid_rows)} mesures : l'énergie nette est un indicateur "
                     f"SECONDAIRE non fiable ici ; utiliser l'énergie BRUTE")
        if mp and isnum(idle_ref) and idle_ref > 0.9 * (sum(mp) / len(mp)):
            exp.warn(f"puissance à vide ({idle_ref:.2f} W) ≥ 90 % de la puissance moyenne en traitement "
                     f"({sum(mp) / len(mp):.2f} W) : contraste charge/repos faible, énergie nette non interprétable")
        if len(set(round(x, 6) for x in ram_p)) <= 1 and ram_p:
            exp.warn("puissance RAM identique pour toutes les tâches : énergie RAM = puissance constante x durée (modèle), "
                     "pas une mesure")
        if unreliable:
            exp.warn(f"{len(unreliable)} mesure(s) avec profiler_coverage hors [{a.cov_min}, {a.cov_max}] : "
                     f"décomposition par composant NON FIABLE ; ne pas la présenter comme telle")
        for d_, idxs in exp.low_res.items():
            exp.warn(f"{d_} : image(s) {sorted(idxs)} de durée < 10 x {a.measure_power_secs} s : résolution énergie insuffisante")
        summary.update({
            "profiler_coverage": {"min": min(cov) if cov else NAN, "max": max(cov) if cov else NAN,
                                  "bounds_used": [a.cov_min, a.cov_max],
                                  "n_measurements_unreliable": len(unreliable),
                                  "decomposition_reliable_all": not unreliable},
            "energy_diagnostics": {
                "idle_baseline_initial_w": exp.idle_w, "idle_baseline_initial_cpu_w": exp.idle_cpu_w,
                "mean_task_power_w": agg(mp)["mean"],
                "cpu_power_w_last": {"min": min(cpu_p) if cpu_p else NAN, "max": max(cpu_p) if cpu_p else NAN,
                                     "n_distinct": len(set(round(x, 6) for x in cpu_p))},
                "ram_power_w_last": {"min": min(ram_p) if ram_p else NAN, "max": max(ram_p) if ram_p else NAN,
                                     "n_distinct": len(set(round(x, 6) for x in ram_p))},
                "cpu_share_of_energy_mean": agg([r["cpu_share_of_energy"] for r in valid_rows])["mean"],
                "n_net_negative": len(neg), "n_measurements": len(valid_rows)},
            "tracker_overhead_s": {"mean": agg([r["tracker_overhead_s"] for r in valid_rows])["mean"],
                                   "max": agg([r["tracker_overhead_s"] for r in valid_rows])["max"]},
            "stats_by_direction_and_type": tstats})
        try:
            figs = make_figures(exp, fstats, cs_frame, cs_type, tstats, valid_rows, label, bool(unreliable))
        except Exception as e:
            figs = []
            exp.warn(f"échec de la génération des figures : {e!r}")
        summary["figures"] = figs
        print_summary(exp, tstats, cs_type, label)
        print(f"\nprofiler_coverage : min={fmt(min(cov), 4)} max={fmt(max(cov), 4)} (borne de fiabilité {a.cov_min}..{a.cov_max})")
    summary["warnings"] = exp.warnings
    summary["definitions"] = {
        "energy_gross_j": "énergie CodeCarbon de la tâche (CPU+RAM) englobant l'image entière ; RÉSULTAT PRINCIPAL",
        "energy_net_secondary_j": "brute - P_repos x durée ; indicateur SECONDAIRE, peut être négatif",
        "component_energy": "ESTIMÉE = énergie de l'image x temps du composant / temps de l'image",
        "time_task_s": "EnergyMeter : perf_counter entre le retour de start_task et l'appel de stop_task",
        "time_profiler_s": "Profiler.frame() : durée de l'image à l'intérieur de la tâche",
        "cc_duration_s": "durée de la tâche rapportée par CodeCarbon",
        "outer_wall_s": "perf_counter autour de TOUT le bloc de tâche, start_task/stop_task inclus",
        "tracker_overhead_s": "outer_wall_s - time_task_s : coût de start_task + stop_task (pas un overhead du profiler)",
        "gap_task_minus_profiler_s": "temps passé dans la tâche hors Profiler.frame() (entrée/sortie du gestionnaire de contexte)",
        "gap_task_minus_cc_s": "écart de chronométrage entre EnergyMeter et CodeCarbon",
        "sampling_period_over_wall_pct": "période d'échantillonnage CodeCarbon / durée de la tâche : borne d'incertitude "
                                         "de bord de l'énergie de la tâche",
        "std_between_repeats": "écart-type des moyennes par passe (mêmes images, passes différentes)",
        "std_between_frames": "écart-type entre les moyennes par image (images différentes)",
        "cv": "écart-type / |moyenne|", "group_mapping": {g: GROUP_LABEL[g] for g in GROUP_KEYS},
        "yolo": "non mesuré dans cette expérience"}
    meta["outside_measured_tasks"] = {
        "inside_tasks_encode": ["lecture PNG + conversion RGB", "ToTensor", "encode_frame", "add_frame (en mémoire)"],
        "inside_tasks_decode": ["lecture d'un enregistrement du flux (fichier ouvert)", "decode_frame", "écriture PNG"],
        "outside_tasks_timed_not_energy": ["BitstreamWriter.close (écriture réelle du .vcmb)", "BitstreamReader() ouverture",
                                           "BitstreamReader.close"],
        "outside_tasks_not_measured": ["construction du codec + checkpoint", "warm-up", "validation + PSNR + hachages",
                                       "écriture des CSV/JSON", "baselines à vide", "reset_sequence"],
        "uninstrumented_inside_image": "encodeur I : estimation de bits (_likelihood par position) et boucle autorégressive "
                                       "hors context_net/pmf/rangecode ; FrameEncoder()/finish() ; _push_recon ; etc."}
    meta["limitations"] = [
        "L'énergie CodeCarbon est une énergie machine entière ; RAM toujours modélisée ; CPU : nature selon CPU_mode (non vérifiée).",
        "Énergie par composant = estimation proportionnelle au temps ; deux composants de même durée mais de puissances "
        "différentes sont indiscernables.",
        "Une tâche ne peut pas être plus fine que la période d'échantillonnage CodeCarbon (voir sampling_period_over_wall_pct).",
        "Résultats valables pour cette machine, ce checkpoint, ces images et ce GOP uniquement."]
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump_json(out_dir / "summary.json", summary)
    dump_json(out_dir / "metadata.json", meta)
    dump_json(out_dir / "progress.json", {"status": status, "repeats_valid": n_ok, "repeats_requested": R})

    print("\n" + "=" * 80)
    print(f"STATUT : {status} — {n_ok}/{R} répétition(s) valide(s)")
    print("=" * 80)
    if exp.errors:
        print("ERREURS")
        for e in exp.errors:
            print(" -", e)
    print("AVERTISSEMENTS")
    for w in exp.warnings:
        print(" -", w)
    print(f"\nDossier : {out_dir}")
    for f_ in ("frame_measurements.csv", "component_time_energy.csv", "frame_stats_across_repeats.csv",
               "component_stats_by_frame.csv", "component_stats_by_type.csv", "repeat_summary.csv",
               "energy_tasks.csv", "summary.json", "metadata.json", "validation.json"):
        print(f"  {f_}")
    log("Terminé : " + status)
    sys.exit(0 if status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
