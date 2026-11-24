#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
experiment3_frame_components_v51.py -- VCM-EnergyMap, expérience 3 (V5.1) = V5 (configuration YAML) + V4.1 (profilage énergétique).

Fusion : la couche V5 (YAML, validations, --probe/--run/--batch, traçabilité) est appliquée PAR-DESSUS V4.1 (énergie brute/nette
CPU-RAM-total avec cohérence net/net direct, RAPL via Windows EMI vérifié, temps séparés prétraitement / encodage / décodage /
flux / post-traitement, bpp et bit/s, statistiques des puissances au repos, négatifs conservés). Les originaux V4, V4.1 et V5
ne sont pas modifiés. Le codec, ses poids et son architecture ne sont jamais modifiés ; la résolution est appliquée aux IMAGES
D'ENTRÉE (cache dédié) avant le codec. UNE seule source de paramètres par expérience : experiment_config_v51.yaml.

ACTIONS (sans action explicite : validation statique seulement, rien n'est lancé)
  --list                          expériences du YAML
  --id C1 | --ids C1,C2           validation statique (fichiers, GOP, structure I/P attendue, images, résolution, détecteur, énergie, fps)
  --deep                          + inspection des checkpoints (formes de tenseurs, champs de configuration)
  --probe                         + essai RÉEL du codec sur quelques images, sans énergie (GOP accepté, I/P observé, taille décodée)
  --run --id C1                   mode individuel
  --batch --ids C1,C2,C3,C4       mode lot : validation de TOUT le lot, puis exécution successive (un processus par configuration)
  --check_energy                  test de la collecte d'énergie seule

Chaque configuration écrit dans results\VCM_EnergyMap\exp3_v51\<id>_gop<g>_<L>x<H>_<checkpoint>_<vidéo>_n<n>_r<r>\ ;
rien n'est jamais écrasé, les résultats de configurations différentes ne se mélangent pas.
Dépendances locales : exp_config_v51.py, energy_profiling_v51.py (+ le projet VCM : test_base_profiled, base_codec, ...).
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
VERSION = "v51"
RUN_NAME_DEFAULT = "exp3_v51_RAPL_10rep"
import exp_config_v51 as xc
IDLE_MIN_RATIO = 10      # une baseline doit durer >= 10 x measure_power_secs
EXPECTED_CHANNEL = "RAPL_Package0_PKG"
REQUIRED = ["test_base_profiled.py", "energy_profiling_v51.py", "profiling.py", "base_codec.py", "bitstream.py",
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
    ("preprocess", "Prétraitement (lecture PNG + conversion tenseur)"),
    ("bitstream_io", "Flux binaire (en mémoire)"),
    ("postprocess", "Post-traitement (écriture PNG reconstruit)"),
    ("io", "Autres entrées / sorties"),
    ("uninstrumented", "Non instrumenté (reste)"),
]
GROUP_LABEL = dict(GROUPS)
GROUP_KEYS = [g for g, _ in GROUPS]
GROUP_SHORT = {"motion_estimation": "Estim. mouvement", "motion_prediction": "Préd. mouvement",
               "motion_compensation": "Comp. mouvement", "analysis": "Analyse", "synthesis": "Synthèse",
               "cond_net": "Réseau cond.", "align_crop": "Align./recadrage", "quantization": "Quantification",
               "pmf": "PMF", "context_net": "Contexte", "entropy_coding": "Codage entropique",
               "preprocess": "Prétraitement", "bitstream_io": "Flux binaire", "postprocess": "Post-traitement",
               "io": "Autres E/S", "uninstrumented": "Non instrumenté"}
STAGE_IO_GROUP = {"image_read": "preprocess", "preprocess_to_tensor": "preprocess", "bitstream_write": "bitstream_io",
                  "bitstream_read": "bitstream_io", "image_write": "postprocess"}
_EXACT = {"motion_estimation": "motion_estimation", "motion_prediction": "motion_prediction",
          "motion_compensation": "motion_compensation", "align": "align_crop", "crop_clamp": "align_crop"}
_SUFFIX = [("_context_net", "context_net"), ("_hyper_analysis", "analysis"), ("_analysis", "analysis"),
           ("_cond_net", "cond_net"), ("_synthesis", "synthesis"), ("_quantize", "quantization"),
           ("_dequantize", "quantization"), ("_pmf", "pmf"), ("_rangecode", "entropy_coding"),
           ("_rangedecode", "entropy_coding")]

FRAME_KEYS = ["time_task_s", "time_profiler_s", "cc_duration_s", "tracker_overhead_s", "energy_window_s",
              "energy_gross_j", "energy_cpu_gross_j", "energy_ram_gross_j",
              "energy_net_j", "energy_cpu_net_j", "energy_ram_net_j", "mean_power_gross_w", "profiler_coverage"]
TYPE_KEYS = ["time_task_s", "time_profiler_s", "cc_duration_s", "energy_gross_j", "energy_cpu_gross_j",
             "energy_ram_gross_j", "energy_net_j", "energy_cpu_net_j", "energy_ram_net_j", "mean_power_gross_w"]
COMP_ENERGY_KEYS = ["energy_gross_estimated_j", "energy_cpu_gross_estimated_j", "energy_ram_gross_estimated_j",
                    "energy_net_estimated_j", "energy_cpu_net_estimated_j", "energy_ram_net_estimated_j"]
# V4.1 : temps MESURES par phase (NaN = sans objet pour la direction) et debit
TIME_SPLIT_KEYS = ["time_preprocess_s", "time_encode_s", "time_decode_s", "time_bitstream_s", "time_postprocess_s",
                   "time_other_residual_s"]
BITRATE_KEYS = ["bits_payload", "bpp_payload", "bitrate_payload_bps", "bits_record", "bpp_frame", "bitrate_record_bps"]
FRAME_KEYS = FRAME_KEYS + TIME_SPLIT_KEYS + BITRATE_KEYS
TYPE_KEYS = TYPE_KEYS + TIME_SPLIT_KEYS + BITRATE_KEYS


def bitrate_metrics(payload_bytes, record_bytes, pix, fps, fps_assumed=False):
    """Débit d'UNE image. bits_payload = 8 x octets du payload codé ; bits_record = 8 x octets de l'enregistrement du flux
    (payload + cadrage ; connu à l'encodage seulement, NaN au décodage). bpp = bits / pixels ; bit/s = bits x fps
    (débit ÉQUIVALENT si des images de ce type se succédaient à la cadence fps ; pas un débit mesuré en temps réel)."""
    ok = isnum(fps) and fps > 0 and pix > 0

    def bits(b):
        return 8.0 * b if isnum(b) else NAN
    bp, br = bits(payload_bytes), bits(record_bytes)
    return {"fps": fps, "fps_assumed": bool(fps_assumed), "bits_payload": bp,
            "bpp_payload": bp / pix if ok and isnum(bp) else NAN,
            "bitrate_payload_bps": bp * fps if ok and isnum(bp) else NAN,
            "bits_record": br, "bpp_frame": br / pix if ok and isnum(br) else NAN,
            "bitrate_record_bps": br * fps if ok and isnum(br) else NAN}


def sequence_bitrate(file_bytes, payload_bytes_sum, record_bytes_sum, n_frames, pix, fps):
    """Débit de la SÉQUENCE : bits totaux / durée vidéo (n_frames / fps). file = fichier .vcmb complet (en-tête inclus)."""
    if not (isnum(fps) and fps > 0 and n_frames > 0 and pix > 0):
        return {k: NAN for k in ("fps", "sequence_duration_s", "bitrate_file_bps", "bitrate_payload_bps",
                                 "bitrate_record_bps", "bpp_file_mean", "bpp_payload_mean")}
    dur = n_frames / fps

    def r(b, div):
        return 8.0 * b / div if isnum(b) else NAN
    return {"fps": fps, "sequence_duration_s": dur, "bitrate_file_bps": r(file_bytes, dur),
            "bitrate_payload_bps": r(payload_bytes_sum, dur), "bitrate_record_bps": r(record_bytes_sum, dur),
            "bpp_file_mean": r(file_bytes, n_frames * pix), "bpp_payload_mean": r(payload_bytes_sum, n_frames * pix)}


def stage_group(name, io_stages):
    """(groupe, reconnu). Une etape inconnue est comptee 'non instrumente' ET signalee."""
    if name in STAGE_IO_GROUP:
        return STAGE_IO_GROUP[name], True
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
    """Label fonde sur la classification VERIFIEE (energy_profiling.method_info), jamais sur le seul nom du mode."""
    probe = info.get("emi_probe") or {}
    chans = ", ".join(probe.get("selected_channel_names") or []) or "aucun"
    if info.get("cpu_is_confirmed_rapl_measurement"):
        return (f"CPU : RAPL Package mesuré via Windows EMI (canal {chans}) ; RAM : estimée par CodeCarbon ; "
                f"total = CPU mesuré + RAM estimée (énergie hybride)")
    if str(info.get("cpu_energy_source", "")).endswith("energy_not_yet_checked"):
        return (f"CPU : canal Package {chans} confirmé, compteur pas encore vérifié (attente de la 1re mesure) ; "
                f"RAM : estimée par CodeCarbon")
    return (f"CPU : NON confirmé comme mesure RAPL (source={info.get('cpu_energy_source')}, CPU_mode={info.get('CPU_mode')}, "
            f"canaux={chans}) ; RAM : estimée par CodeCarbon ; énergie CodeCarbon, pas une mesure matérielle confirmée")


def energy_source_block(info):
    """Bloc 'source energetique' ecrit dans metadata.json / summary.json."""
    probe = info.get("emi_probe") or {}
    return {"cpu_backend": info.get("CPU_mode"),
            "codecarbon_version": info.get("codecarbon_version"),
            "rapl_channels_selected": probe.get("selected_channel_names"),
            "rapl_channels_on_monitored_devices": probe.get("channels_on_monitored_devices"),
            "rapl_dram_included": probe.get("include_dram"),
            "rapl_package_channel_confirmed": probe.get("package_channel_confirmed"),
            "rapl_expected_channel": EXPECTED_CHANNEL,
            "rapl_expected_channel_selected": EXPECTED_CHANNEL.lower() in [str(n).lower() for n in
                                                                          probe.get("selected_channel_names") or []],
            "cpu_counter_functional": info.get("cpu_counter_functional"),
            "n_tasks_cpu_energy_nonpositive": info.get("n_tasks_cpu_energy_nonpositive"),
            "cpu_energy_source": info.get("cpu_energy_source"),
            "ram_energy_source": info.get("ram_energy_source"),
            "ram_energy_detail": info.get("ram_energy_detail"),
            "total_energy_kind": info.get("total_energy_kind"),
            "cpu_scope": "package CPU (coeurs + uncore + iGPU) ; DRAM, écran, disque, ventilateurs exclus",
            "tracking_mode": info.get("tracking_mode"),
            "measure_power_secs": info.get("codecarbon_measure_power_secs"),
            "ram_total_size_gb": info.get("ram_total_size_gb"),
            "cpu_model": info.get("cpu_model")}


def rapl_problem(info, stage, expected_channel=EXPECTED_CHANNEL):
    """None si le CPU est confirme RAPL via EMI ; sinon la raison. stage='init' : structure ; 'after_idle' : + compteur actif."""
    probe = info.get("emi_probe") or {}
    if str(info.get("CPU_mode")).lower() != "windows_emi":
        return f"CPU_mode={info.get('CPU_mode')} (attendu : windows_emi)"
    if not probe.get("probe_ok"):
        return f"sonde EMI impossible : {probe.get('probe_error')}"
    names = probe.get("selected_channel_names") or []
    if not probe.get("package_channel_confirmed"):
        return f"canal Package non confirmé : canaux sélectionnés = {names}"
    if expected_channel and expected_channel.lower() not in [n.lower() for n in names]:
        return f"canal attendu {expected_channel} absent de la sélection {names}"
    if stage == "after_idle" and not info.get("cpu_is_confirmed_rapl_measurement"):
        return f"le compteur n'a pas délivré d'énergie CPU (source = {info.get('cpu_energy_source')})"
    return None


# --------------------------------------------------------------------------------------
# verifications d'API (signatures reelles) et de l'environnement
# --------------------------------------------------------------------------------------
def load_modules():
    """Imports lourds. Retourne le dictionnaire de modules/classes reutilises."""
    import torch
    import test_base_profiled as tbp
    from base_codec import BaseLayerCodec
    from bitstream import BitstreamReader, BitstreamWriter, SequenceHeader
    from energy_profiling_v51 import EnergyMeter, energy_window, energy_breakdown, idle_power_stats   # revision V4.1 requise
    from profiling import IO_STAGES, Profiler
    return dict(torch=torch, tbp=tbp, BaseLayerCodec=BaseLayerCodec, BitstreamReader=BitstreamReader,
                BitstreamWriter=BitstreamWriter, SequenceHeader=SequenceHeader, EnergyMeter=EnergyMeter, energy_window=energy_window,
                energy_breakdown=energy_breakdown, idle_power_stats=idle_power_stats,
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
    need(M["EnergyMeter"], "probe_emi", [])
    need(M["EnergyMeter"], "measure_idle", ["seconds"])
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
                if p.info["pid"] == me or str(p.info["pid"]) == os.environ.get("VCM_BATCH_PARENT"):
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


def collect_repro_metadata(a):
    """Tout ce qui permet de refaire l'experience : empreintes des scripts, du checkpoint et des images d'entree,
    commit git, paquets Python, ligne de commande, variables d'environnement de threads."""
    def sh(p):
        try:
            return sha256_file(p)
        except Exception:
            return None
    ck = Path(a.checkpoint_file)
    d = {"command_line": " ".join([sys.executable] + sys.argv),
         "scripts_sha256": {f: sh(ROOT / f) for f in sorted(set(REQUIRED + [Path(__file__).name]))},
         "checkpoint": str(ck), "checkpoint_sha256": sh(ck),
         "input_frames_sha256": {f"{a.prefix}{a.start_index + i:03d}.png":
                                 sh(os.path.join(a.inp_path, f"{a.prefix}{a.start_index + i:03d}.png"))
                                 for i in range(a.no_frames)},
         "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
         "protocol": {"layer": "base", "checkpoint": a.checkpoint_name, "checkpoint_id": a.checkpoint, "resolution": f"{a.width}x{a.height}", "no_frames": a.no_frames, "gop": a.gop,
                      "repeats": a.repeats, "closed_gop": a.closed_gop, "threads": a.threads,
                      "warmup_frames": a.warmup_frames, "idle_seconds": a.idle_seconds,
                      "idle_between_seconds": a.idle_between_seconds, "cooldown": a.cooldown,
                      "measure_power_secs": a.measure_power_secs}}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        d["git_commit"] = r.stdout.strip() or None
        r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        d["git_dirty"] = bool(r.stdout.strip())
    except Exception:
        d["git_commit"] = None
    try:
        d["python_packages"] = {(x.metadata["Name"] or "?"): x.version for x in imd.distributions()}
    except Exception:
        d["python_packages"] = None
    return d


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
        self.idle_rows = []          # toutes les mesures a vide (idle_baselines.csv)
        self.type_check = defaultdict(int)   # (direction, attendu, observé) -> nombre d'images
        self.energy_info = {}        # derniere classification verifiee (method_info)
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
                    "energy_window_s": holder.get("energy_window_s", NAN),
                    "energy_window_source": holder.get("energy_window_source", ""),
                    "tracker_overhead_s": holder["tracker_overhead_s"],
                    "energy_total_j": holder.get("energy_total_j", NAN), "energy_cpu_j": holder.get("energy_cpu_j", NAN),
                    "energy_ram_j": holder.get("energy_ram_j", NAN), "energy_gpu_j": holder.get("energy_gpu_j", NAN),
                    "mean_power_total_w": holder.get("mean_power_total_w", NAN),
                    "mean_power_cpu_w": holder.get("mean_power_cpu_w", NAN),
                    "mean_power_ram_w": holder.get("mean_power_ram_w", NAN),
                    "cpu_power_w_last": holder.get("cpu_power_w_last", NAN),
                    "ram_power_w_last": holder.get("ram_power_w_last", NAN)})

    def measure_idle(self, name, seconds):
        """Baseline a vide. Puissances = energie / FENETRE D'ENERGIE (cc_duration_s), la meme horloge que celle utilisee
        ensuite pour soustraire P_repos x duree. Chaque mesure est conservee (idle_baselines.csv)."""
        with self.task(name) as h:
            time.sleep(seconds)
        win = h["energy_window_s"]
        okw = isnum(win) and win > 0
        row = {"name": name, "requested_s": seconds, "wall_s": h["wall_s"], "cc_duration_s": h.get("cc_duration_s", NAN),
               "energy_window_s": win, "energy_window_source": h.get("energy_window_source", ""),
               "energy_total_j": h["energy_total_j"], "energy_cpu_j": h["energy_cpu_j"],
               "energy_ram_j": h["energy_ram_j"], "energy_gpu_j": h.get("energy_gpu_j", 0.0),
               "power_total_w": h["energy_total_j"] / win if okw else NAN,
               "power_cpu_w": h["energy_cpu_j"] / win if okw else NAN,
               "power_ram_w": h["energy_ram_j"] / win if okw else NAN,
               "power_gpu_w": h.get("energy_gpu_j", 0.0) / win if okw else NAN,
               "sampling_period_s": self.a.measure_power_secs,
               "window_over_sampling": win / self.a.measure_power_secs if okw else NAN,
               "t_start_s": self.task_rows[-1]["t_enter_s"] if self.task_rows else NAN}
        row["seconds"] = row["wall_s"]
        if win < IDLE_MIN_RATIO * self.a.measure_power_secs:
            self.warn(f"baseline {name} : {win:g} s < {IDLE_MIN_RATIO} x measure_power_secs : résolution faible")
        self.idle_rows.append(row)
        write_csv(self.out / "idle_baselines.csv", self.idle_rows)
        return row

    # ---------- init + warm-up (HORS mesure)
    def type_check_summary(self):
        tot = sum(self.type_check.values())
        bad = {f"{d}:{e}->{o}": n for (d, e, o), n in self.type_check.items() if o in ("I", "P") and o != e}
        unk = sum(n for (d, e, o), n in self.type_check.items() if o == "?")
        return {"n_frames_checked": tot, "n_mismatch": sum(bad.values()), "mismatches": bad, "n_type_unknown": unk,
                "counts": {f"{d}:{e}->{o}": n for (d, e, o), n in sorted(self.type_check.items())},
                "method": "type observé = I si les étapes exécutées sont identiques à celles de l'image 0 (intra par construction), sinon P",
                "all_consistent": tot > 0 and not bad}

    def init_codecs(self):
        from PIL import Image
        from torchvision.transforms import ToTensor
        M, a, tbp = self.M, self.a, self.M["tbp"]
        ckpt = Path(a.checkpoint_file)
        n, gop = a.no_frames, a.gop
        log(f"construction + chargement du checkpoint {ckpt.name} (hors mesure)")
        self.enc = M["BaseLayerCodec"]("enc", root=a.root, closed_gop=a.closed_gop)
        self.ck_info = self.enc.load_checkpoint(str(ckpt))
        self.dec = M["BaseLayerCodec"]("dec", root=a.root, closed_gop=a.closed_gop)
        self.dec.load_checkpoint(str(ckpt))
        self.W, self.H = Image.open(tbp._frame_path(a.inp_path, a.prefix, a.start_index)).size
        if (self.W, self.H) != (a.width, a.height):
            raise RuntimeError(f"résolution réelle des images d'entrée {self.W}x{self.H} != résolution demandée "
                               f"{a.width}x{a.height}")
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
        """Energie brute / nette par composant (CPU, RAM, total).
        - TEMPS (resultats de temps)   : wall_task = holder["wall_s"] (perf_counter autour du travail).
        - FENETRE D'ENERGIE            : win = cc_duration_s (duree sur laquelle CodeCarbon a integre l'energie) ;
          puissances et soustraction de la baseline utilisent TOUTES win. wall_s n'est un repli que si cc_duration_s
          est invalide (energy_window_source le dit). L'ecart wall_s / cc_duration_s est enregistre.
        - NET par composant = brut - P_repos(composant) x win ; NET TOTAL = somme des nets des composants
          (coherent par construction) ; la valeur directe brut_total - P_repos_total x win est enregistree pour controle.
        - Les nets negatifs sont CONSERVES (ni 0, ni valeur absolue)."""
        a = self.a
        IO_STAGES = self.M["IO_STAGES"]
        pw = next(f for f in prof.frames if f["dir"] == direction and f["frame"] == i and f["phase"] == "measured")
        obs = observed_frame_type(prof, direction, i)
        self.type_check[(direction, ftype, obs)] += 1
        if obs in ("I", "P") and obs != ftype:
            raise FrameTypeMismatch(f"structure I/P incohérente avec GOP={a.gop} : {direction} image {i} attendue {ftype}, "
                                    f"observée {obs} d'après les étapes exécutées par le codec")
        wall_task = holder["wall_s"]
        cc_dur = holder.get("cc_duration_s", NAN)
        win = holder.get("energy_window_s", NAN)
        win_src = holder.get("energy_window_source", "")
        e_g, e_c, e_r = holder["energy_total_j"], holder["energy_cpu_j"], holder["energy_ram_j"]
        e_gpu = holder.get("energy_gpu_j", 0.0)
        ip_c, ip_r = idle["power_cpu_w"], idle["power_ram_w"]
        ip_g, ip_t = idle.get("power_gpu_w", 0.0), idle["power_total_w"]
        eb = self.M["energy_breakdown"]({**holder, "energy_window_s": win}, idle)   # energy_profiling_v51 (fonction pure testee)
        net_c, net_r, net_gpu, net, net_direct = eb["net_cpu_j"], eb["net_ram_j"], eb["net_gpu_j"], eb["net_total_j"], eb["net_direct_j"]
        residual = eb["task_component_residual_j"]         # doit etre ~0 : total CodeCarbon = CPU + RAM + GPU
        if not eb["consistent"]:
            self.warn("energy_net_j != energy_net_direct_j au-delà de la tolérance sur ≥ 1 tâche (total CodeCarbon != CPU+RAM+GPU "
                      "ou puissance totale à vide != somme des composantes) : voir energy_net_consistent")
        if isnum(residual) and abs(residual) > 1e-6 * max(1.0, abs(e_g)):
            self.warn("énergie totale CodeCarbon != CPU + RAM + GPU sur ≥ 1 tâche (voir energy_component_residual_j)")
        if isnum(e_gpu) and e_gpu > 0:
            self.warn("énergie GPU > 0 sur ≥ 1 tâche : incluse dans le total, absente des colonnes CPU/RAM")
        if not (isnum(e_c) and e_c > 0):
            self.warn(f"énergie CPU <= 0 J sur ≥ 1 tâche (ex. {direction} image {i} rép. {rep}) : compteur RAPL non fiable")
        gap_pct = 100 * abs(wall_task - win) / win if isnum(win) and win > 0 else NAN
        if isnum(gap_pct) and gap_pct > 2.0:
            self.warn("écart > 2 % entre wall_s et cc_duration_s sur ≥ 1 tâche (voir window_gap_pct)")
        gt = defaultdict(float)
        for r in prof.records:
            if r["phase"] == "measured" and r["dir"] == direction and r["frame"] == i:
                g, known = stage_group(r["stage"], IO_STAGES)
                if not known:
                    self.warn(f"étape non reconnue '{r['stage']}' comptée dans 'non instrumenté'")
                gt[g] += r["seconds"]
        tot_st = sum(gt.values())
        # temps MESURES par phase (durees des etapes du profiler, perf_counter) ; le residu est CALCULE
        t_pre, t_bs, t_post = gt.get("preprocess", 0.0), gt.get("bitstream_io", 0.0), gt.get("postprocess", 0.0)
        extra = dict(extra)
        t_codec = extra.pop("time_codec_call_s", NAN)       # duree de l'appel encode_frame / decode_frame (perf_counter)
        rec_bytes = extra.pop("record_bytes", NAN)
        is_enc = direction == "enc"
        t_other = wall_task - (t_pre + t_bs + t_post + t_codec) if isnum(t_codec) else NAN
        bm = bitrate_metrics(extra.get("payload_bytes"), rec_bytes, self.W * self.H, a.fps, a.fps_assumed)
        pwall = pw["wall_s"]
        coverage = tot_st / pwall if pwall > 0 else NAN
        reliable = isnum(coverage) and a.cov_min <= coverage <= a.cov_max
        if not reliable:
            self.warn(f"{direction} image {i} rép. {rep} : profiler_coverage={fmt(coverage, 4)} hors de "
                      f"[{a.cov_min}, {a.cov_max}] -> décomposition par composant NON FIABLE pour cette image")
        gt["uninstrumented"] += max(0.0, pwall - tot_st)
        denom = max(pwall, tot_st)
        low = not (isnum(win) and win >= 10 * a.measure_power_secs)
        if low:
            self.low_res[direction].add(i)
        row = {
            "config_id": a.config_id, "checkpoint": a.checkpoint_name, "gop": a.gop, "resolution": f"{self.W}x{self.H}",
            "repeat": rep, "repeat_valid": "pending", "frame": i,
            "frame_file_index": a.start_index + i, "type": ftype, "direction": direction,
            "task": holder.get("name", ""),
            "time_task_s": wall_task, "time_profiler_s": pwall, "cc_duration_s": cc_dur,
            "time_preprocess_s": t_pre if is_enc else NAN, "time_encode_s": t_codec if is_enc else NAN,
            "time_decode_s": NAN if is_enc else t_codec, "time_bitstream_s": t_bs,
            "time_postprocess_s": NAN if is_enc else t_post, "time_other_residual_s": t_other,
            "energy_window_s": win, "energy_window_source": win_src, "window_gap_pct": gap_pct, "type_observed": obs,
            "outer_wall_s": holder.get("outer_wall_s", NAN), "tracker_overhead_s": holder.get("tracker_overhead_s", NAN),
            "gap_task_minus_profiler_s": wall_task - pwall,
            "gap_task_minus_cc_s": wall_task - cc_dur if isnum(cc_dur) else NAN,
            "sampling_period_over_window_pct": 100 * a.measure_power_secs / win if isnum(win) and win > 0 else NAN,
            "profiler_coverage": coverage, "decomposition_reliable": reliable,
            # ---- energie BRUTE (J) : CPU = RAPL via EMI si confirme ; RAM = modele ; total = hybride
            "energy_gross_j": e_g, "energy_cpu_gross_j": e_c, "energy_ram_gross_j": e_r, "energy_gpu_gross_j": e_gpu,
            "energy_component_residual_j": residual,
            "cpu_share_of_energy": e_c / e_g if e_g else NAN,
            "mean_power_gross_w": e_g / win if isnum(win) and win > 0 else NAN,
            "mean_power_cpu_w": e_c / win if isnum(win) and win > 0 else NAN,
            "mean_power_ram_w": e_r / win if isnum(win) and win > 0 else NAN,
            "cpu_power_w_last": holder.get("cpu_power_w_last", NAN), "ram_power_w_last": holder.get("ram_power_w_last", NAN),
            # ---- baseline a vide utilisee (W) et energie NETTE (J) = brut - P_repos x energy_window_s
            "idle_source": idle["name"], "idle_power_w_used": ip_t, "idle_power_cpu_w_used": ip_c,
            "idle_power_ram_w_used": ip_r, "idle_power_gpu_w_used": ip_g,
            "energy_net_j": net, "energy_cpu_net_j": net_c, "energy_ram_net_j": net_r, "energy_gpu_net_j": net_gpu,
            "energy_net_direct_j": net_direct, "energy_net_sum_check_j": eb["net_difference_j"],
            "energy_net_consistent": eb["consistent"], "idle_component_residual_w": eb["idle_component_residual_w"],
            "energy_net_negative": net < 0, "energy_cpu_net_negative": net_c < 0, "energy_ram_net_negative": net_r < 0,
            "energy_resolution_ok_ge_10x_sampling": not low, **bm, **extra}
        comp = []
        for g in GROUP_KEYS:
            t = gt.get(g, 0.0)
            share = t / denom if denom > 0 else NAN
            comp.append({"config_id": a.config_id, "checkpoint": a.checkpoint_name, "repeat": rep, "repeat_valid": "pending", "frame": i, "type": ftype,
                         "direction": direction, "group": g, "group_label": GROUP_LABEL[g], "time_s": t,
                         "time_share": share, "decomposition_reliable": reliable,
                         "energy_gross_estimated_j": e_g * share,
                         "energy_cpu_gross_estimated_j": e_c * share,
                         "energy_ram_gross_estimated_j": e_r * share,
                         "energy_net_estimated_j": net * share,
                         "energy_cpu_net_estimated_j": net_c * share,
                         "energy_ram_net_estimated_j": net_r * share,
                         "energy_nature": "ESTIMEE (part de temps x energie de l'image ; CPU: RAPL si confirme, RAM: modele)"})
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
        bs_path = str(self.out / "bitstreams" / f"{self.a.checkpoint_name}_rep{rep:02d}.vcmb")
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
            name = f"{self.a.config_id}_r{rep:02d}_enc_frame{i:03d}"
            with self.task(name) as holder:
                with p_enc.frame(i, ftype, "enc"):
                    with p_enc.stage("image_read"):
                        img = Image.open(path).convert("RGB")
                    with p_enc.stage("preprocess_to_tensor"):
                        x = ToTensor()(img).unsqueeze(0)
                    t_c0 = time.perf_counter()
                    r = self.enc.encode_frame(x, i, gop, p_enc)
                    t_codec = time.perf_counter() - t_c0          # temps d'ENCODAGE du codec seul (mesure directe)
                    with p_enc.stage("bitstream_write"):          # EN MEMOIRE (add_frame) : pas d'E/S disque
                        rec_bytes = writer.add_frame(r["payload"], r["ranges"])
            holder["name"] = name
            hdr.shapes.update({kk: tuple(v[1:]) for kk, v in r["shapes"].items()})
            fr = self.record(rep, "enc", i, ftype, holder, p_enc,
                             {"payload_bytes": len(r["payload"]), "record_bytes": rec_bytes, "time_codec_call_s": t_codec}, idle)
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
                name = f"{self.a.config_id}_r{rep:02d}_dec_frame{i:03d}"
                with self.task(name) as holder:
                    with p_dec.frame(i, ftype, "dec"):
                        with p_dec.stage("bitstream_read"):
                            payload, ranges, _ = rd.next_frame()
                        t_c0 = time.perf_counter()
                        rec = self.dec.decode_frame(payload, ranges, i, h.gop, h.shapes, (h.height, h.width), p_dec)
                        t_codec = time.perf_counter() - t_c0      # temps de DÉCODAGE du codec seul (mesure directe)
                        with p_dec.stage("image_write"):
                            tbp._save_png(rec, tbp._frame_path(str(rec_dir), a.prefix, a.start_index + i))
                holder["name"] = name
                decoded += 1
                fr = self.record(rep, "dec", i, ftype, holder, p_dec, {"payload_bytes": len(payload), "record_bytes": NAN, "time_codec_call_s": t_codec}, idle)
                log(f"  r{rep + 1} dec {i + 1}/{h.n_frames} [{ftype}] {fr['time_task_s']:.2f} s | "
                    f"{fr['energy_gross_j']:.1f} J brut | cov {fmt(fr['profiler_coverage'], 4)} | "
                    f"écoulé {hms(time.perf_counter() - self.t0_run)}")
        finally:
            t0 = time.perf_counter()
            rd.close()
            info["outside"]["bitstream_close_read_s"] = time.perf_counter() - t0
        info["decoded"] = decoded
        info["rep_wall_s"] = time.perf_counter() - t_rep
        p_enc.export(str(self.out / "metrics" / f"{self.a.checkpoint_name}_rep{rep:02d}_enc_profile"),
                     meta={**self.M["tbp"]._env_meta(a), **sizes, "repeat": rep})
        p_dec.export(str(self.out / "metrics" / f"{self.a.checkpoint_name}_rep{rep:02d}_dec_profile"),
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
        inc = [r for r in rows if not r.get("energy_net_consistent", True)]
        checks["net_consistency"] = {
            "n_rows": len(rows), "n_inconsistent": len(inc), "tolerance": "1e-9 J absolu + 1e-9 x max(1 J, |brut|, |E repos|)",
            "max_abs_difference_j": max((abs(r["energy_net_sum_check_j"]) for r in rows if isnum(r["energy_net_sum_check_j"])),
                                        default=NAN),
            "n_net_negative_kept": sum(1 for r in rows if r["energy_net_negative"])}
        if inc:
            self.warn(f"rép. {rep} : {len(inc)} mesure(s) avec energy_net_j != energy_net_direct_j (voir validation.json) ; "
                      f"données conservées, non supprimées")
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
        for key in ("time_task_s", "energy_gross_j", "energy_cpu_gross_j", "energy_ram_gross_j", "energy_net_j",
                    "energy_cpu_net_j", "energy_ram_net_j"):
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
        for ek in COMP_ENERGY_KEYS:
            row.update(flat(ek, agg([r[ek] for r in rs])))
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
        for key in ("time_s", "time_share", *COMP_ENERGY_KEYS):
            per_rep = [agg([r[key] for r in rs if r["repeat"] == rp])["mean"] for rp in reps]
            per_frame = [agg([r[key] for r in rs if r["frame"] == fr])["mean"] for fr in frs]
            row.update(flat(key, agg(per_rep)))                  # ecart-type ENTRE REPETITIONS
            row[key + "_std_between_frames"] = agg(per_frame)["std"]
        row["decomposition_reliable_all"] = all(r["decomposition_reliable"] for r in rs)
        row["energy_nature"] = "ESTIMEE (proxy temps x energie de l'image)"
        out.append(row)
    return out


def bitrate_block(exp, rep_infos, valid_reps, tstats):
    """Bloc 'bitrate' de summary.json : séquence (moyenne entre répétitions) et par (direction, type)."""
    a = exp.a
    if a.fps_assumed:
        exp.warn(f"--fps non fourni : cadence supposée {a.fps:g} images/s pour tous les débits en bit/s (les bpp, eux, "
                 f"n'en dépendent pas) ; renseignez --fps avec la cadence réelle de la vidéo")
    rs = [r for r in repeat_summary(exp, rep_infos) if r["repeat"] in valid_reps]
    seq = {k[4:]: agg([r[k] for r in rs]) for k in (rs[0] if rs else {}) if k.startswith("seq_")}
    by = {key: {k: s[k]["mean"] for k in BITRATE_KEYS} for key, s in tstats.items() if not key.endswith("phase_total")}
    return {"fps": a.fps, "fps_assumed": a.fps_assumed,
            "definition": "bit/s = bits par image x fps (équivalent) ; séquence = bits totaux / (n_images / fps) ; "
                          "payload = octets codés de l'image ; record = payload + cadrage du flux (encodage) ; "
                          "file = fichier .vcmb complet (en-tête inclus)",
            "sequence_across_repeats": seq, "per_frame_by_direction_type": by}


def repeat_summary(exp, rep_infos):
    out = []
    for rep in sorted(exp.validation):
        v = exp.validation[rep]
        rows = [r for r in exp.frames_rows if r["repeat"] == rep]
        info = rep_infos.get(rep, {})
        row = {"repeat": rep, "valid": v["valid"], "n_errors": len(v["errors"]),
               "errors": " | ".join(v["errors"]),
               "idle_power_w_used": rows[0]["idle_power_w_used"] if rows else NAN,
               "idle_power_cpu_w_used": rows[0]["idle_power_cpu_w_used"] if rows else NAN,
               "idle_power_ram_w_used": rows[0]["idle_power_ram_w_used"] if rows else NAN,
               "idle_source": rows[0]["idle_source"] if rows else "",
               "bitstream_bytes": (info.get("sizes") or {}).get("total_bytes", NAN)}
        enc_rows = [r for r in rows if r["direction"] == "enc"]
        sb = sequence_bitrate(row["bitstream_bytes"], sum(r["payload_bytes"] for r in enc_rows),
                              sum(r["bits_record"] / 8.0 for r in enc_rows if isnum(r["bits_record"])), len(enc_rows),
                              exp.W * exp.H, exp.a.fps)
        row.update({f"seq_{k}": v_ for k, v_ in sb.items()})
        row["fps_assumed"] = exp.a.fps_assumed
        for d in ("enc", "dec"):
            rd = [r for r in rows if r["direction"] == d]
            row[f"{d}_time_sum_s"] = sum(r["time_task_s"] for r in rd) if rd else NAN
            for ek in ("energy_gross_j", "energy_cpu_gross_j", "energy_ram_gross_j", "energy_net_j",
                       "energy_cpu_net_j", "energy_ram_net_j"):
                row[f"{d}_{ek[:-2]}_sum_j"] = sum(r[ek] for r in rd) if rd else NAN
            row[f"{d}_tracker_overhead_sum_s"] = sum(r["tracker_overhead_s"] for r in rd) if rd else NAN
            for tk in TIME_SPLIT_KEYS:
                vv = [r[tk] for r in rd if isnum(r[tk])]
                row[f"{d}_{tk[:-2]}_sum_s"] = sum(vv) if vv else NAN
            for t in ("I", "P"):
                rt = [r for r in rd if r["type"] == t]
                row[f"{d}_{t}_time_mean_s"] = agg([r["time_task_s"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_gross_mean_j"] = agg([r["energy_gross_j"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_cpu_gross_mean_j"] = agg([r["energy_cpu_gross_j"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_ram_gross_mean_j"] = agg([r["energy_ram_gross_j"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_net_mean_j"] = agg([r["energy_net_j"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_cpu_net_mean_j"] = agg([r["energy_cpu_net_j"] for r in rt])["mean"]
                row[f"{d}_{t}_energy_ram_net_mean_j"] = agg([r["energy_ram_net_j"] for r in rt])["mean"]
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
    """Figures simples : UN titre general par figure, legendes courtes (Énergie brute / Énergie nette,
    Encodage / Décodage, Image I / Image P). Les explications (source des energies, barres d'erreur, nombre de
    repetitions) sont dans summary.json / metadata.json, pas dans les figures. Les valeurs negatives sont tracees telles quelles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    figdir = exp.out / "figures"
    figdir.mkdir(exist_ok=True)
    COL = {"I": "#8c564b", "P": "#2ca02c"}
    C_GROSS, C_NET = "#1f77b4", "#d62728"
    C_DIR = {"enc": "#1f77b4", "dec": "#d62728"}
    DIRN = {"enc": "Encodage", "dec": "Décodage"}
    BASE = "VCM Base Layer"
    cpu_rapl = bool((exp.energy_info or {}).get("cpu_is_confirmed_rapl_measurement"))
    CPU_T = "CPU (RAPL)" if cpu_rapl else "CPU"
    made = []
    reps_used = sorted({r["repeat"] for r in valid_rows})

    def done(fig, name, title):
        fig.suptitle(title, fontweight="bold")
        warn_txt = unreliable and name.startswith(("fig5", "fig6", "fig8"))
        if warn_txt:
            fig.text(0.5, 0.01, "Décomposition non fiable", color="red", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04 if warn_txt else 0, 1, 0.95))
        fig.savefig(figdir / f"{name}.png", dpi=300)
        fig.savefig(figdir / f"{name}.pdf")
        plt.close(fig)
        made.append(name)

    def err(sd):
        return sd if isnum(sd) else 0.0

    def tag(ax, text):
        ax.text(0.0, 1.01, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=9, color="#444")

    def ticks(rs):
        return ["I" if r["type"] == "I" else f"P{r['frame']}" for r in rs]

    # ---- figures par image : energie brute / nette (total, CPU, RAM)
    for name, gk, nk, ttl in (
            ("fig1_energy_per_frame", "energy_gross_j", "energy_net_j", f"Énergie par image — {BASE}"),
            ("fig1b_cpu_energy_per_frame", "energy_cpu_gross_j", "energy_cpu_net_j", f"Énergie {CPU_T} par image — {BASE}"),
            ("fig1c_ram_energy_per_frame", "energy_ram_gross_j", "energy_ram_net_j", f"Énergie RAM (estimée) par image — {BASE}")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, d in zip(axes, ("enc", "dec")):
            rs = sorted([r for r in fstats if r["direction"] == d], key=lambda r: r["frame"])
            x = list(range(len(rs)))
            for key, col, lab in ((gk, C_GROSS, "Énergie brute"), (nk, C_NET, "Énergie nette")):
                ax.errorbar(x, [r[key + "_mean"] for r in rs], yerr=[err(r[key + "_std"]) for r in rs],
                            fmt="-o", color=col, capsize=3, ms=4, lw=1.2, label=lab)
            ax.axhline(0, color="#888", lw=.6)
            ax.set_xticks(x)
            ax.set_xticklabels(ticks(rs))
            ax.set_xlabel("Image")
            ax.set_ylabel("Énergie (J)")
            ax.grid(alpha=.3)
            tag(ax, DIRN[d])
        axes[0].legend(fontsize=8, loc="center right")
        done(fig, name, ttl)

    # ---- temps par image
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, d in zip(axes, ("enc", "dec")):
        rs = sorted([r for r in fstats if r["direction"] == d], key=lambda r: r["frame"])
        x = list(range(len(rs)))
        ax.errorbar(x, [r["time_task_s_mean"] for r in rs], yerr=[err(r["time_task_s_std"]) for r in rs],
                    fmt="-o", color=C_DIR[d], capsize=3, ms=4, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks(rs))
        ax.set_xlabel("Image")
        ax.set_ylabel("Temps (s)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.3)
        tag(ax, DIRN[d])
    done(fig, "fig2_time_per_frame", f"Temps par image — {BASE}")

    # ---- encodage vs decodage / I vs P : barres groupees (temps, energie brute, energie nette)
    METRICS = (("time_task_s", "Temps (s)"), ("energy_gross_j", "Énergie brute (J)"), ("energy_net_j", "Énergie nette (J)"))

    def grouped(name, title, xgroups, series, entry, colors):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        w = .36
        for ax, (key, yl) in zip(axes, METRICS):
            for j, (sk, sl) in enumerate(series):
                for xi, (xk, xl) in enumerate(xgroups):
                    s = entry(xk, sk)
                    if s:
                        ax.bar(xi + (j - .5) * w, s[key]["mean"], w, yerr=err(s[key]["std_between_repeats"]), capsize=4,
                               color=colors[sk], alpha=.85, edgecolor="k", lw=.5, label=sl if xi == 0 else None)
            ax.axhline(0, color="#888", lw=.6)
            ax.set_xticks(range(len(xgroups)))
            ax.set_xticklabels([xl for _, xl in xgroups])
            ax.set_ylabel(yl)
            ax.grid(axis="y", alpha=.3)
        axes[0].legend(fontsize=8)
        done(fig, name, title)

    grouped("fig3_encode_vs_decode", f"Encodage vs décodage — {BASE}",
            [("I", "Image I"), ("P", "Image P")], [("enc", "Encodage"), ("dec", "Décodage")],
            lambda t, d: tstats.get(f"{d}_{t}"), C_DIR)
    grouped("fig4_I_vs_P", f"Image I vs image P — {BASE}",
            [("enc", "Encodage"), ("dec", "Décodage")], [("I", "Image I"), ("P", "Image P")],
            lambda d, t: tstats.get(f"{d}_{t}"), COL)

    # ---- composants : temps (mesure) et energie brute / nette (estimees)
    panels = (("enc", "I"), ("enc", "P"), ("dec", "I"), ("dec", "P"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (d, t) in zip(axes.ravel(), panels):
        rs = [r for r in cs_type if r["direction"] == d and r["type"] == t and r["time_s_mean"] > 0]
        rs.sort(key=lambda r: r["time_s_mean"])
        ax.barh([GROUP_SHORT[r["group"]] for r in rs], [r["time_s_mean"] for r in rs],
                xerr=[err(r["time_s_std"]) for r in rs], color=COL[t], alpha=.8, edgecolor="k", lw=.4, capsize=2)
        ax.set_xlabel("Temps (s)")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.3)
        tag(ax, f"{DIRN[d]} · image {t}")
    done(fig, "fig5_component_time", f"Temps par composant — {BASE}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (d, t) in zip(axes.ravel(), panels):
        rs = [r for r in cs_type if r["direction"] == d and r["type"] == t and r["time_s_mean"] > 0]
        rs.sort(key=lambda r: r["energy_gross_estimated_j_mean"])
        ypos = list(range(len(rs)))
        h = .38
        ax.barh([y + h / 2 for y in ypos], [r["energy_gross_estimated_j_mean"] for r in rs],
                h, xerr=[err(r["energy_gross_estimated_j_std"]) for r in rs], color=C_GROSS, alpha=.85,
                edgecolor="k", lw=.4, capsize=2, label="Énergie brute")
        ax.barh([y - h / 2 for y in ypos], [r["energy_net_estimated_j_mean"] for r in rs],
                h, xerr=[err(r["energy_net_estimated_j_std"]) for r in rs], color=C_NET, alpha=.85,
                edgecolor="k", lw=.4, capsize=2, label="Énergie nette")
        ax.axvline(0, color="#888", lw=.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([GROUP_SHORT[r["group"]] for r in rs], fontsize=8)
        ax.set_xlabel("Énergie (J)")
        ax.grid(axis="x", alpha=.3)
        tag(ax, f"{DIRN[d]} · image {t}")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    done(fig, "fig6_component_energy_estimated", f"Énergie par composant — {BASE}")

    # ---- variabilite entre repetitions (temps, energie brute, energie nette)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row_i, (key, yl) in enumerate(METRICS):
        for col_i, d in enumerate(("enc", "dec")):
            ax = axes[row_i, col_i]
            for t in ("I", "P"):
                ys = []
                for rp in reps_used:
                    v = [r[key] for r in valid_rows if r["direction"] == d and r["type"] == t and r["repeat"] == rp]
                    ys.append(mean_std(v)[0])
                if any(isnum(y) for y in ys):
                    ax.plot(reps_used, ys, "-o", color=COL[t], ms=4, lw=1.2, label=f"Image {t}")
            ax.axhline(0, color="#888", lw=.6)
            ax.set_xlabel("Répétition")
            ax.set_ylabel(yl)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.3)
            tag(ax, DIRN[d])
    axes[0, 0].legend(fontsize=8)
    done(fig, "fig7_variability_across_repeats", f"Variabilité entre répétitions — {BASE}")

    # ---- part de temps par composant
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    cmap = plt.get_cmap("tab20")
    for ax, d in zip(axes, ("enc", "dec")):
        types = [t for t in ("I", "P") if any(r["type"] == t and r["direction"] == d for r in cs_type)]
        left = [0.0] * len(types)
        for gi, g in enumerate(GROUP_KEYS):
            sh = []
            for t in types:
                v = [r["time_share_mean"] for r in cs_type if r["group"] == g and r["type"] == t and r["direction"] == d]
                sh.append(100 * v[0] if v and isnum(v[0]) else 0.0)
            if not any(s > 0 for s in sh):
                continue
            ax.barh(types, sh, left=left, color=cmap(gi), edgecolor="w", label=GROUP_SHORT[g])
            left = [l_ + s for l_, s in zip(left, sh)]
        ax.set_xlabel("Part du temps (%)")
        ax.set_xlim(0, 100)
        tag(ax, DIRN[d])
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5))
    done(fig, "fig8_component_time_share", f"Part du temps par composant — {BASE}")
    return made


# --------------------------------------------------------------------------------------
def print_summary(exp, tstats, cs_type, label):
    a = exp.a
    print("\n" + "=" * 110)
    print(f"RÉCAPITULATIF — config {a.config_id} (GOP {a.gop}, {exp.W}x{exp.H}, {a.checkpoint_name} ; {a.repeats} répétition(s) demandée(s) ; ± = écart-type ENTRE RÉPÉTITIONS)")
    print(f"[{label}]")
    print("=" * 110)
    print(f"| {'Phase / type':<14} | {'img':>3} | {'Temps (s)':>16} | {'E brute (J)':>16} | {'E nette (J)':>16} | "
          f"{'P moy (W)':>12} | {'nets<0':>9} |")
    print("|" + "-" * 108 + "|")
    for key, s in tstats.items():
        if key.endswith("phase_total"):
            continue
        d, t = key.split("_")
        print(f"| {('Encodage ' if d == 'enc' else 'Décodage ') + t:<14} | {s['n_frames']:>3} | "
              f"{pm(s['time_task_s']['mean'], s['time_task_s']['std_between_repeats']):>16} | "
              f"{pm(s['energy_gross_j']['mean'], s['energy_gross_j']['std_between_repeats']):>16} | "
              f"{pm(s['energy_net_j']['mean'], s['energy_net_j']['std_between_repeats']):>16} | "
              f"{pm(s['mean_power_gross_w']['mean'], s['mean_power_gross_w']['std_between_repeats'], 2):>12} | "
              f"{s['n_net_negative']:>3}/{s['n_measurements']:<5} |")
    print("\nCPU (RAPL si confirmé) / RAM (estimée) — énergie brute et nette par image (J)")
    print(f"| {'Phase / type':<14} | {'CPU brute':>14} | {'CPU nette':>14} | {'RAM brute':>14} | {'RAM nette':>14} |")
    print("|" + "-" * 82 + "|")
    for key, s in tstats.items():
        if key.endswith("phase_total"):
            continue
        d, t = key.split("_")

        def c(k):
            return pm(s[k]["mean"], s[k]["std_between_repeats"], 2)
        print(f"| {('Encodage ' if d == 'enc' else 'Décodage ') + t:<14} | {c('energy_cpu_gross_j'):>14} | "
              f"{c('energy_cpu_net_j'):>14} | {c('energy_ram_gross_j'):>14} | {c('energy_ram_net_j'):>14} |")
    print("(net = brut - P_repos x cc_duration_s ; valeurs négatives conservées)")
    print("\nTEMPS MESURÉS (perf_counter), s par image — l'énergie par composant, elle, est ESTIMÉE")
    print(f"| {'Phase / type':<14} | {'Prétrait.':>10} | {'Codec':>10} | {'Flux':>10} | {'Post-trait.':>11} | {'Résidu calc.':>12} |")
    print("|" + "-" * 86 + "|")
    for key, s in tstats.items():
        if key.endswith("phase_total"):
            continue
        d, t = key.split("_")

        def m(k):
            return fmt(s[k]["mean"], 3)
        codec = "time_encode_s" if d == "enc" else "time_decode_s"
        print(f"| {('Encodage ' if d == 'enc' else 'Décodage ') + t:<14} | {m('time_preprocess_s'):>10} | {m(codec):>10} | "
              f"{m('time_bitstream_s'):>10} | {m('time_postprocess_s'):>11} | {m('time_other_residual_s'):>12} |")
    print("(résidu = temps de la tâche − somme des temps mesurés : valeur CALCULÉE ; « n/a » = sans objet pour la direction)")
    print(f"\nDÉBIT (fps = {a.fps:g}{' SUPPOSÉ' if a.fps_assumed else ''}) — bit/s = bits par image x fps")
    print(f"| {'Phase / type':<14} | {'bits payload':>13} | {'bpp payload':>12} | {'bit/s payload':>14} | {'bpp flux':>9} | {'bit/s flux':>12} |")
    print("|" + "-" * 90 + "|")
    for key, s in tstats.items():
        if key.endswith("phase_total"):
            continue
        d, t = key.split("_")
        print(f"| {('Encodage ' if d == 'enc' else 'Décodage ') + t:<14} | {fmt(s['bits_payload']['mean'], 0):>13} | "
              f"{fmt(s['bpp_payload']['mean'], 4):>12} | {fmt(s['bitrate_payload_bps']['mean'], 0):>14} | "
              f"{fmt(s['bpp_frame']['mean'], 4):>9} | {fmt(s['bitrate_record_bps']['mean'], 0):>12} |")
    print("\nCOMPOSANTS (temps = mesuré ; énergie = ESTIMÉE par part de temps)")
    for d in ("enc", "dec"):
        for t in ("I", "P"):
            rows = [r for r in cs_type if r["direction"] == d and r["type"] == t and r["time_s_mean"] > 0]
            if not rows:
                continue
            print(f"\n  {'Encodage' if d == 'enc' else 'Décodage'} — image {t}")
            for r in sorted(rows, key=lambda r: -r["time_s_mean"]):
                print(f"    {r['group_label']:<48} {pm(r['time_s_mean'], r['time_s_std']):>18} s "
                      f"{100 * r['time_share_mean']:>6.1f} % {fmt(r['energy_gross_estimated_j_mean']):>9} J brut (est.) "
                      f"{fmt(r['energy_net_estimated_j_mean']):>9} J net (est.)")


# --------------------------------------------------------------------------------------
def estimate_seconds(a):
    n, gop = a.no_frames, a.gop
    n_i = len([i for i in range(n) if i % gop == 0])
    n_p = n - n_i
    per_pass = n_i * (a.est_enc_I_s + a.est_dec_I_s) + n_p * (a.est_enc_P_s + a.est_dec_P_s)
    nw = min(a.warmup_frames, n)
    warm = sum((a.est_enc_I_s + a.est_dec_I_s) if i % gop == 0 else (a.est_enc_P_s + a.est_dec_P_s) for i in range(nw))
    idle = a.idle_seconds + max(0, a.repeats - 1) * a.idle_between_seconds
    return a.repeats * per_pass * 1.03 + warm + idle + a.cooldown + 60, per_pass


def check_energy(a):
    """Test de la collecte d'energie SEULE (aucun codec, aucun checkpoint) : cree le tracker, verifie le backend et le
    canal selectionne, mesure une baseline puis une charge CPU, controle que le compteur reagit, que la RAM est constante
    (modele), que total = CPU + RAM, et que wall_s ~ cc_duration_s. N'ecrit PAS dans le dossier de resultats V4
    (fichier results_root/_energy_check_v51.json). Code de sortie 0 = tout OK, 2 = au moins un controle en echec."""
    import energy_profiling_v51 as ep
    secs = a.check_seconds
    checks, notes = {}, []
    meter = ep.EnergyMeter(country_iso_code=a.country, measure_power_secs=a.measure_power_secs)
    try:
        info0 = meter.method_info()
        print(f"Backend CPU      : {info0.get('CPU_mode')} (CodeCarbon {info0.get('codecarbon_version')})")
        pr = info0.get("emi_probe") or {}
        print(f"Canaux EMI choisis : {pr.get('selected_channel_names')}")
        print(f"Canaux des appareils surveillés : {pr.get('channels_on_monitored_devices')}")
        print(f"DRAM incluse     : {pr.get('include_dram')}")
        idle = meter.measure_idle(secs)
        busy_rec = {}
        with meter.task("busy") as b:
            t0, x = time.perf_counter(), 0.0
            while time.perf_counter() - t0 < secs:
                x = (x + math.sqrt(x + 1.0)) % 1e6
        busy_rec.update(b)
        info = meter.method_info()
    finally:
        meter.close()

    def pw(r, k):
        return r[k] / r["energy_window_s"]
    p_idle_c, p_busy_c = pw(idle, "energy_cpu_j"), pw(busy_rec, "energy_cpu_j")
    p_idle_r, p_busy_r = pw(idle, "energy_ram_j"), pw(busy_rec, "energy_ram_j")
    problem = rapl_problem(info, "after_idle", a.expected_channel)
    checks["backend_is_windows_emi"] = str(info.get("CPU_mode")).lower() == "windows_emi"
    checks["package_channel_confirmed_and_expected"] = problem is None
    checks["cpu_energy_positive_idle_and_busy"] = idle["energy_cpu_j"] > 0 and busy_rec["energy_cpu_j"] > 0
    checks["cpu_power_rises_under_load_x1.2"] = p_busy_c >= 1.2 * p_idle_c > 0
    checks["ram_power_constant_model"] = abs(p_busy_r - p_idle_r) <= 0.02 * max(p_busy_r, p_idle_r, 1e-12)
    checks["total_equals_cpu_plus_ram_plus_gpu"] = all(
        abs(r["energy_total_j"] - (r["energy_cpu_j"] + r["energy_ram_j"] + r["energy_gpu_j"]))
        <= 1e-6 * max(1.0, r["energy_total_j"]) for r in (idle, busy_rec))
    checks["wall_vs_cc_duration_within_2pct"] = all(
        abs(r["wall_s"] - r["energy_window_s"]) <= 0.02 * r["energy_window_s"] for r in (idle, busy_rec))
    checks["duration_ge_10x_sampling"] = secs >= IDLE_MIN_RATIO * a.measure_power_secs
    checks["cpu_source_is_measured_rapl_via_windows_emi"] = info.get("cpu_energy_source") == "measured_rapl_via_windows_emi"
    if problem:
        notes.append(f"RAPL non confirmé : {problem}")
    ok = all(checks.values())
    print("\n" + "=" * 78)
    print(f"Idle  {secs:g} s : CPU {p_idle_c:.3f} W | RAM {p_idle_r:.3f} W | total {pw(idle, 'energy_total_j'):.3f} W")
    print(f"Busy  {secs:g} s : CPU {p_busy_c:.3f} W | RAM {p_busy_r:.3f} W | total {pw(busy_rec, 'energy_total_j'):.3f} W")
    print(f"Source CPU : {info.get('cpu_energy_source')} ; RAM : {info.get('ram_energy_source')} ; total : {info.get('total_energy_kind')}")
    print("-" * 78)
    for k, v in checks.items():
        print(f"  [{'OK ' if v else 'ÉCHEC'}] {k}")
    for n in notes:
        print(f"  [NOTE] {n}")
    print("=" * 78)
    print("RÉSULTAT :", "COLLECTE D'ÉNERGIE VALIDÉE (RAPL Package via Windows EMI)" if ok else "COLLECTE NON VALIDÉE : ne pas lancer l'expérience complète")
    out = Path(a.results_root)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "_energy_check_v51.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": secs, "checks": checks, "notes": notes,
        "idle": idle, "busy": busy_rec, "power_w": {"idle_cpu": p_idle_c, "busy_cpu": p_busy_c, "idle_ram": p_idle_r,
                                                    "busy_ram": p_busy_r},
        "energy_method": info, "energy_source": energy_source_block(info), "machine": collect_machine_metadata(a)})
    print(f"Détail : {out / '_energy_check_v51.json'}")
    return 0 if ok else 2


def run_experiment(a, trace):
    """Exécute UNE configuration (paramètres déjà résolus depuis le YAML dans `a`, traçabilité dans `trace`)."""
    n, gop, R = a.no_frames, a.gop, a.repeats
    if R < 1 or n < 1:
        sys.exit("[ERREUR] --repeats et --no_frames doivent être >= 1")
    miss = [f for f in REQUIRED if not (ROOT / f).exists()]
    if miss:
        sys.exit(f"[ERREUR] fichiers manquants dans {ROOT} : {miss}")
    out_dir = Path(a.out_dir)
    n_i = len([i for i in range(n) if i % gop == 0])
    frames_missing = [a.start_index + i for i in range(n)
                      if not os.path.exists(os.path.join(a.inp_path, f"{a.prefix}{a.start_index + i:03d}.png"))]
    ckpt = Path(a.checkpoint_file)
    total, per_pass = estimate_seconds(a)

    print("=" * 80)
    print(f"PLAN — exp.3 {VERSION} : énergie/temps par image (I/P) et par composant")
    print("=" * 80)
    print(f"Images          : {n} (fichiers {a.prefix}{a.start_index:03d}..{a.prefix}{a.start_index + n - 1:03d}) ; "
          f"GOP={gop} -> {n_i} image(s) I + {n - n_i} P")
    print(f"Répétitions     : {R} passe(s) complète(s) indépendantes -> {R * n} mesures d'encodage + {R * n} de décodage")
    print(f"Checkpoint      : {ckpt}")
    print(f"Résultats       : {out_dir}")
    print(f"Durée estimée   : {hms(total)} (environ {hms(per_pass)} par passe ; limite {a.max_hours:g} h)")
    errors, notes = [], []
    if frames_missing:
        errors.append(f"images introuvables : {frames_missing[:10]} dans {a.inp_path}")
    if not ckpt.exists():
        errors.append(f"checkpoint introuvable : {ckpt}")
    if out_dir.exists() and any(out_dir.iterdir()):
        errors.append(f"{out_dir} existe déjà : rien n'est écrasé ; changez l'identifiant ou un paramètre dans le YAML")
    if R < MIN_FINAL_REPEATS:
        notes.append(f"{R} répétition(s) < {MIN_FINAL_REPEATS} : exécution PILOTE, pas un résultat final")
    if R < 2:
        notes.append("1 seule répétition : écart-type entre répétitions non défini")
    if total > a.max_hours * 3600:
        errors.append(f"durée estimée {hms(total)} > {a.max_hours:g} h : réduisez --repeats/--no_frames ou passez --max_hours")
    if n_i == 1:
        notes.append("une seule image I par passe : sa variabilité vient uniquement des répétitions")
    if a.idle_seconds < IDLE_MIN_RATIO * a.measure_power_secs:
        errors.append(f"--idle_seconds={a.idle_seconds:g} < {IDLE_MIN_RATIO} x measure_power_secs ({a.measure_power_secs:g}) : "
                      f"baseline trop courte pour la résolution de CodeCarbon")
    if 0 < a.idle_between_seconds < IDLE_MIN_RATIO * a.measure_power_secs:
        errors.append(f"--idle_between_seconds={a.idle_between_seconds:g} < {IDLE_MIN_RATIO} x measure_power_secs : "
                      f"baseline trop courte")
    if not a.fps > 0:
        errors.append(f"--fps={a.fps} doit être > 0")
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
    dump_json(out_dir / "detector_params.json", {"detector": trace["detector"], "hash": trace["detector_hash"]})
    dump_json(out_dir / "config_resolved.json", trace["resolved_parameters"])
    try:
        (out_dir / "experiment_config_snapshot.yaml").write_bytes(Path(trace["config_file"]).read_bytes())
    except Exception:
        pass
    exp = Experiment(a, M, out_dir)
    for nt in notes:
        exp.warn(nt)
    log(f"Début exp.3 {VERSION} : {n} images, {R} répétition(s), GOP={gop}, checkpoint {a.checkpoint_name}, {a.width}x{a.height}, threads={a.threads}")
    log("Fermez les autres applications, secteur branché, plan d'alimentation fixe, ne touchez pas à la machine.")
    status, rep_infos = "SUCCESS", {}
    meta = {"version": VERSION, "machine": collect_machine_metadata(a), "params": vars(a),
            "concurrent_codecarbon_check": cc, "api_check": "OK",
            "reproducibility": collect_repro_metadata(a), "config_trace": trace}
    handler = None
    cc_msgs, py_warns = [], []
    try:
        exp.init_codecs()                       # HORS mesure (avant le tracker)
        meta["codec_runtime"] = {"gop_used": gop, "closed_gop": a.closed_gop, "resolution_actual": f"{exp.W}x{exp.H}",
                                 "checkpoint_name": a.checkpoint_name, "checkpoint_id_in_stream_header": exp.k,
                                 "checkpoint_file": a.checkpoint_file, "checkpoint_loaded_return": jsonable(exp.ck_info),
                                 "n_frames": n, "repeats": R, "threads": a.threads}
        # ---- tracker CodeCarbon : un seul, journaux et avertissements captures
        class _H(logging.Handler):
            def emit(self, rec):
                cc_msgs.append(f"{rec.levelname}: {rec.getMessage()}")
        handler = _H()
        handler.setLevel(logging.DEBUG)
        logging.getLogger("codecarbon").addHandler(handler)
        with pywarnings.catch_warnings(record=True) as wl:
            pywarnings.simplefilter("always")
            exp.meter = M["EnergyMeter"](country_iso_code=a.country, measure_power_secs=a.measure_power_secs,
                                         tracking_mode=a.tracking_mode)
        py_warns = [str(w.message) for w in wl]
        info = exp.meter.method_info()
        exp.energy_info = info
        label = energy_label(info)
        log(f"Méthode d'énergie : {label}")
        probe = info.get("emi_probe") or {}
        log(f"Backend CPU = {info.get('CPU_mode')} ; canaux EMI sélectionnés = {probe.get('selected_channel_names')} ; "
            f"canaux des appareils surveillés = {probe.get('channels_on_monitored_devices')} ; DRAM incluse = {probe.get('include_dram')}")
        if cc_msgs or py_warns:
            log(f"Messages CodeCarbon à l'initialisation : {cc_msgs + py_warns}")
        if any("Multiple instances" in m for m in cc_msgs + py_warns):
            exp.warn("CodeCarbon signale « Multiple instances ... allowed » : en 3.x allow_multiple_runs=True par défaut "
                     "(avertissement seulement). Vérifiez qu'aucun autre tracker/script ne tourne (voir metadata.json -> "
                     "concurrent_codecarbon_check) ; les tâches de CE script sont strictement séquentielles (contrôle ci-dessous).")
        mach = meta["machine"]
        problem = rapl_problem(info, "init", a.expected_channel)
        if problem:
            exp.warn(f"RAPL via Windows EMI NON confirmé : {problem}")
            if not a.allow_unconfirmed_rapl:
                raise RuntimeError(f"RAPL via Windows EMI non confirmé ({problem}) : arrêt AVANT toute mesure. "
                                   f"Lancez d'abord --check_energy ; --allow_unconfirmed_rapl pour passer outre.")
        bat = mach.get("battery")
        if bat and not bat.get("power_plugged"):
            exp.warn("le portable fonctionne sur batterie : branchez le secteur")
        meta.update({"energy_method": info, "energy_source": energy_source_block(info), "energy_label": label,
                     "codecarbon_log_messages": cc_msgs + py_warns})
        dump_json(out_dir / "metadata.json", meta)

        log(f"Pause {a.cooldown:g} s puis baseline à vide {a.idle_seconds:g} s")
        time.sleep(a.cooldown)
        idle0 = exp.measure_idle("idle_initial", a.idle_seconds)
        log(f"Baseline initiale : {idle0['power_total_w']:.3f} W total (CPU {idle0['power_cpu_w']:.3f} W, RAM {idle0['power_ram_w']:.3f} W)")
        exp.idle_w, exp.idle_cpu_w = idle0["power_total_w"], idle0["power_cpu_w"]
        # re-classification APRES une première tâche mesurée : le compteur CPU doit avoir délivré de l'énergie (> 0 J)
        info = exp.meter.method_info()
        exp.energy_info = info
        label = energy_label(info)
        meta.update({"energy_method": info, "energy_source": energy_source_block(info), "energy_label": label})
        dump_json(out_dir / "metadata.json", meta)
        log(f"Source après baseline : CPU = {info.get('cpu_energy_source')} ; RAM = {info.get('ram_energy_source')} ; "
            f"total = {info.get('total_energy_kind')}")
        problem = rapl_problem(info, "after_idle", a.expected_channel)
        if problem:
            exp.warn(f"RAPL via Windows EMI NON confirmé après la baseline : {problem}")
            if not a.allow_unconfirmed_rapl:
                raise RuntimeError(f"RAPL via Windows EMI non confirmé après la baseline ({problem}) : arrêt avant les répétitions.")

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
            if err_msg and err_msg.startswith("FrameTypeMismatch"):
                status = "FAILED"
                log("[ARRÊT] structure I/P incohérente avec le GOP demandé : les répétitions suivantes seraient invalides")
                break
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
            try:                                   # re-sonde de l'instance réellement utilisée, avant fermeture
                exp.energy_info = exp.meter.method_info()
                meta["energy_method_final"] = exp.energy_info
            except Exception as e:
                log(f"[AVERTISSEMENT] sonde finale de la source d'énergie : {e!r}")
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
    meta["frame_type_check"] = exp.type_check_summary()
    if not meta["frame_type_check"]["all_consistent"]:
        exp.warn(f"structure I/P : contrôle non concluant ou incohérent {meta['frame_type_check']}")
    final_info = meta.get("energy_method_final") or meta.get("energy_method") or {}
    label = energy_label(final_info) if final_info else "énergie CodeCarbon (nature non déterminée)"
    meta["energy_source_final"] = energy_source_block(final_info) if final_info else None
    meta["energy_label_final"] = label
    sel0 = ((meta.get("energy_method") or {}).get("emi_probe") or {}).get("selected_channel_names")
    sel1 = (final_info.get("emi_probe") or {}).get("selected_channel_names")
    meta["rapl_channel_selection_unchanged_during_run"] = (sel0 == sel1)
    if sel0 != sel1:
        exp.warn(f"la sélection de canaux EMI a changé pendant l'expérience : {sel0} -> {sel1}")
    if final_info and not final_info.get("cpu_is_confirmed_rapl_measurement"):
        exp.warn(f"en fin d'expérience, le CPU n'est PAS confirmé comme mesure RAPL ({final_info.get('cpu_energy_source')}) : "
                 f"ne pas présenter l'énergie CPU comme une mesure RAPL")
    try:
        with open(out_dir / "codecarbon_messages.log", "w", encoding="utf-8") as f_:
            f_.write("\n".join(cc_msgs + py_warns) + "\n")
    except Exception:
        pass
    summary = {"experiment": "exp3_frame_components_v51", "status": status, "config_id": a.config_id, "checkpoint": a.checkpoint_name,
               "gop": gop, "resolution": f"{exp.W}x{exp.H}",
               "repeats_requested": R, "repeats_completed": len(exp.validation), "repeats_valid": n_ok,
               "is_final_protocol": bool(n_ok >= MIN_FINAL_REPEATS and status == "SUCCESS"),
               "errors": exp.errors, "warnings": exp.warnings, "energy_label": label,
               "energy_source": meta["energy_source_final"],
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
            exp.warn(f"énergie nette négative pour {len(neg)}/{len(valid_rows)} mesures (conservées telles quelles) : "
                     f"énergie de la tâche < puissance de repos x durée ; la baseline à vide était supérieure à la "
                     f"puissance moyenne de la tâche (fluctuation de la baseline, activité de fond au repos, résolution "
                     f"de l'échantillonnage) — à interpréter avec la dispersion de la baseline (idle_baselines.csv)")
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
                "n_net_negative": len(neg), "n_measurements": len(valid_rows),
                "n_cpu_net_negative": sum(1 for r in valid_rows if r["energy_cpu_net_negative"]),
                "n_ram_net_negative": sum(1 for r in valid_rows if r["energy_ram_net_negative"]),
                "max_abs_component_residual_j": max((abs(r["energy_component_residual_j"]) for r in valid_rows
                                                     if isnum(r["energy_component_residual_j"])), default=NAN),
                "max_abs_net_sum_check_j": max((abs(r["energy_net_sum_check_j"]) for r in valid_rows
                                                if isnum(r["energy_net_sum_check_j"])), default=NAN),
                "n_tasks_cpu_energy_nonpositive": (meta.get("energy_method_final") or {}).get("n_tasks_cpu_energy_nonpositive"),
                "n_net_inconsistent": sum(1 for r in valid_rows if not r["energy_net_consistent"]),
                "net_consistency_all_ok": all(r["energy_net_consistent"] for r in valid_rows),
                "idle_power_stats": exp.M["idle_power_stats"](exp.idle_rows)},
            "bitrate": bitrate_block(exp, rep_infos, valid_reps, tstats),
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
        "energy_gross_j": "énergie enregistrée pendant la tâche (image entière) = CPU + RAM (+ GPU) ; total HYBRIDE",
        "energy_cpu_gross_j": "énergie CPU brute : RAPL Package via Windows EMI si energy_source.cpu_energy_source = "
                              "measured_rapl_via_windows_emi, sinon non confirmée",
        "energy_ram_gross_j": "énergie RAM brute : ESTIMÉE par CodeCarbon (puissance modélisée x durée), jamais RAPL",
        "energy_window_s": "durée sur laquelle CodeCarbon a intégré l'énergie (cc_duration_s) ; utilisée pour les puissances "
                           "et pour la soustraction de la baseline ; wall_s seulement en repli (energy_window_source)",
        "energy_net_j": "énergie nette = somme des nets des composants ; net composant = brut - P_repos(composant) x "
                        "energy_window_s ; peut être NÉGATIF (conservé : tâche moins énergivore que la baseline x durée)",
        "energy_cpu_net_j / energy_ram_net_j": "nets CPU et RAM (même formule) ; RAM nette ≈ 0 attendu si la puissance RAM "
                                               "modélisée est constante",
        "time_preprocess_s": "MESURÉ (étapes image_read + preprocess_to_tensor du profiler, perf_counter) ; encodage seulement",
        "time_encode_s / time_decode_s": "MESURÉ : durée de l'appel encode_frame / decode_frame (perf_counter), une seule des deux "
                                         "par ligne",
        "time_bitstream_s": "MESURÉ : bitstream_write (mémoire) à l'encodage, bitstream_read au décodage",
        "time_postprocess_s": "MESURÉ : écriture du PNG reconstruit ; décodage seulement",
        "time_other_residual_s": "CALCULÉ (pas mesuré) : time_task_s − (prétraitement + codec + flux + post-traitement)",
        "energy_net_consistent": "True si |energy_net_j − energy_net_direct_j| <= 1e-9 J + 1e-9 x max(1, |brut|, |E repos|) ; "
                                 "les valeurs négatives ne sont jamais corrigées",
        "bits_payload / bpp_payload / bitrate_payload_bps": "8 x octets du payload ; /pixels ; x fps (débit équivalent)",
        "bits_record / bpp_frame / bitrate_record_bps": "idem avec l'enregistrement du flux (payload + cadrage), encodage seulement",
        "seq_bitrate_*_bps": "bits totaux de la séquence / (n_images / fps) ; file = fichier .vcmb complet",
        "fps": "cadence utilisée pour les bit/s ; fps_assumed=True si non fournie (--fps) : valeur SUPPOSÉE",
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
        "Énergie CPU = compteur RAPL du PACKAGE (coeurs + uncore + iGPU) : ni la machine entière, ni la DRAM, ni l'écran/disque ; "
        "RAM = estimation CodeCarbon (puissance modélisée x durée) ; total = CPU mesuré + RAM estimée (hybride).",
        "Énergie par composant = estimation proportionnelle au temps ; deux composants de même durée mais de puissances "
        "différentes sont indiscernables.",
        "Une tâche ne peut pas être plus fine que la période d'échantillonnage CodeCarbon (voir sampling_period_over_wall_pct).",
        "Les temps par phase sont des mesures directes (perf_counter) ; le résidu est calculé ; l'énergie de chaque phase ou "
        "composant (prétraitement, codec, flux, post-traitement) est une ESTIMATION par part de temps, pas une mesure.",
        "Les bit/s dépendent de la cadence --fps ; s'ils sont calculés avec la valeur par défaut, fps_assumed vaut True.",
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
               "energy_tasks.csv", "idle_baselines.csv", "summary.json", "metadata.json", "validation.json",
               "experiment_log.txt", "codecarbon_messages.log"):
        print(f"  {f_}")
    log("Terminé : " + status)
    sys.exit(0 if status == "SUCCESS" else 1)


class FrameTypeMismatch(RuntimeError):
    """Le type d'image (I/P) réellement observé dans le codec diffère du type attendu d'après le GOP."""


def stage_signature(prof, direction, i, phase="measured"):
    return frozenset(r["stage"] for r in prof.records if r["dir"] == direction and r["frame"] == i and r["phase"] == phase)


def observed_frame_type(prof, direction, i, phase="measured"):
    """Type d'image OBSERVÉ, sans aucune hypothèse sur les noms d'étapes du codec : l'image 0 d'une séquence est
    forcément intra (aucune référence) ; son ensemble d'étapes exécutées sert de référence. Une image qui exécute
    exactement les mêmes étapes est observée 'I', une image qui en diffère (étapes de mouvement/résiduel) est 'P'.
    '?' si les étapes sont indisponibles."""
    ref, cur = stage_signature(prof, direction, 0, phase), stage_signature(prof, direction, i, phase)
    if not ref or not cur:
        return "?"
    return "I" if cur == ref else "P"


def jsonable(o):
    return json.loads(json.dumps(o, default=str))


def library_versions():
    out = {}
    for pkg in ("torch", "torchvision", "numpy", "Pillow", "codecarbon", "PyYAML", "ultralytics", "constriction",
                "psutil", "matplotlib"):
        try:
            out[pkg] = imd.version(pkg)
        except Exception:
            out[pkg] = None
    out["python"] = sys.version.split()[0]
    return out


def other_experiment_processes():
    """Autres processus Python d'expériences/CodeCarbon (hors moi et hors le lot parent) : leur CPU fausserait une mesure."""
    res = []
    try:
        import psutil
    except ImportError:
        return None
    skip = {os.getpid(), os.getppid()}
    try:
        skip.add(int(os.environ.get("VCM_BATCH_PARENT", "0")))
    except ValueError:
        pass
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if p.info["pid"] not in skip and "python" in (p.info.get("name") or "").lower() and any(
                    s in cmd.lower() for s in ("experiment3", "codecarbon", "run_vcm_energymap")):
                res.append({"pid": p.info["pid"], "cmd": cmd[:160]})
        except Exception:
            continue
    return res


def heavy_guard(action, force):
    procs = other_experiment_processes()
    if procs:
        msg = (f"une autre expérience tourne ({procs}) : '{action}' consomme du CPU et FAUSSERAIT sa mesure d'énergie "
               f"(compteur RAPL du package entier). Attendez sa fin" + (" (--force passé : poursuite à vos risques)" if force else
                                                                         " ou utilisez --force sous votre responsabilité."))
        if not force:
            sys.exit(f"[REFUSÉ] {msg}")
        print(f"[ATTENTION] {msg}")
    elif procs is None:
        print("[ATTENTION] psutil absent : impossible de vérifier qu'aucune autre expérience ne tourne.")


def verify_input_files(R, inputs_dir):
    """Vérifie sur les FICHIERS que la résolution demandée est bien celle des images qui alimenteront le codec."""
    from PIL import Image
    bad, v = [], R["video"]
    for i in range(R["frames_count"]):
        f = xc.frame_path(inputs_dir, v["prefix"], v["start_index"] + i)
        try:
            with Image.open(f) as im:
                if im.size != (R["W"], R["H"]):
                    bad.append(f"{os.path.basename(f)}: {im.size[0]}x{im.size[1]}")
        except Exception as e:
            bad.append(f"{os.path.basename(f)}: {e!r}")
    return bad


def probe_config(R, a, M):
    """Essai RÉEL du codec (aucune énergie mesurée) sur les `probe_frames` premières images : charge le checkpoint,
    encode puis décode, observe I/P via les étapes du profiler, contrôle la taille décodée, relève bpp et PSNR
    (information sur le checkpoint, SANS hypothèse sur un niveau de qualité)."""
    import tempfile
    import numpy as np
    from PIL import Image
    from torchvision.transforms import ToTensor
    tbp, gop, W, H = M["tbp"], a.gop, R["W"], R["H"]
    checks, rep = [], {"frames": []}

    def add(check, status, detail):
        checks.append({"check": check, "status": status, "detail": detail})
    k = max(1, min(int(R["validation"]["probe_frames"]), a.no_frames))
    try:
        enc = M["BaseLayerCodec"]("enc", root=a.root, closed_gop=a.closed_gop)
        rep["checkpoint_load_return"] = jsonable(enc.load_checkpoint(a.checkpoint_file))
        dec = M["BaseLayerCodec"]("dec", root=a.root, closed_gop=a.closed_gop)
        dec.load_checkpoint(a.checkpoint_file)
        add("checkpoint chargé par le codec (compatible avec le modèle)", xc.OK, f"{Path(a.checkpoint_file).name}")
    except Exception as e:
        add("checkpoint chargé par le codec (compatible avec le modèle)", xc.FAIL, f"{type(e).__name__}: {e}")
        return checks, rep
    pe, pd = M["Profiler"](), M["Profiler"]()
    recs, exp_types, obs_enc, obs_dec = [], xc.expected_types(k, gop), [], []
    try:
        with pe.phase("measured"):
            for i in range(k):
                path = tbp._frame_path(a.inp_path, a.prefix, a.start_index + i)
                x = ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
                with pe.frame(i, exp_types[i], "enc"):
                    r = enc.encode_frame(x, i, gop, pe)
                obs_enc.append(observed_frame_type(pe, "enc", i))
                recs.append((r["payload"], r["ranges"], {kk: tuple(v[1:]) for kk, v in r["shapes"].items()}))
                rep["frames"].append({"frame": i, "expected": exp_types[i], "observed_enc": obs_enc[-1],
                                      "bytes": len(r["payload"]), "bpp": 8 * len(r["payload"]) / (W * H),
                                      "bitrate_payload_bps_at_fps": 8 * len(r["payload"]) * a.fps})
        add("GOP accepté par le codec (encodage)", xc.OK, f"GOP={gop}, {k} image(s) encodée(s)")
    except Exception as e:
        add("GOP accepté par le codec (encodage)", xc.FAIL, f"{type(e).__name__}: {e}")
        return checks, rep
    try:
        shapes = {}
        for _, _, sh in recs:
            shapes.update(sh)
        with pd.phase("measured"):
            for i, (pl, rg, _) in enumerate(recs):
                with pd.frame(i, exp_types[i], "dec"):
                    out = dec.decode_frame(pl, rg, i, gop, shapes, (H, W), pd)
                obs_dec.append(observed_frame_type(pd, "dec", i))
                with tempfile.TemporaryDirectory() as td:
                    tp = os.path.join(td, "r.png")
                    tbp._save_png(out, tp)
                    with Image.open(tp) as im:
                        size = im.size
                        rec_arr = np.asarray(im.convert("RGB"), dtype=np.float64)
                with Image.open(tbp._frame_path(a.inp_path, a.prefix, a.start_index + i)) as im0:
                    src = np.asarray(im0.convert("RGB"), dtype=np.float64)
                mse = float(((rec_arr - src) ** 2).mean()) if rec_arr.shape == src.shape else float("nan")
                rep["frames"][i].update({"observed_dec": obs_dec[-1], "decoded_size": f"{size[0]}x{size[1]}",
                                         "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse > 0 else float("inf")
                                         if mse == mse else float("nan")})
        add("GOP accepté par le codec (décodage)", xc.OK, "décodage séquentiel réussi")
    except Exception as e:
        add("GOP accepté par le codec (décodage)", xc.FAIL, f"{type(e).__name__}: {e}")
        return checks, rep
    bad = [f for f in rep["frames"] if f["decoded_size"] != f"{W}x{H}"]
    add("résolution appliquée (taille décodée)", xc.FAIL if bad else xc.OK,
        f"{bad[0]['decoded_size']} != {W}x{H}" if bad else f"toutes les images décodées font {W}x{H}")
    mism = [(i, exp_types[i], obs_enc[i], obs_dec[i]) for i in range(k)
            if any(o in ("I", "P") and o != exp_types[i] for o in (obs_enc[i], obs_dec[i]))]
    unk = [i for i in range(k) if "?" in (obs_enc[i], obs_dec[i])]
    add("structure I/P réellement produite = GOP", xc.FAIL if mism else (xc.WARN if unk else xc.OK),
        f"attendu {''.join(exp_types)} ; observé enc {''.join(obs_enc)} / dec {''.join(obs_dec)}" +
        (f" ; INCOHÉRENCES (img, attendu, enc, dec) : {mism}" if mism else "") +
        (" ; type inconnu (aucune étape) sur " + str(unk) if unk else ""))
    return checks, rep


def build_trace(R, plan, checks, ckpt_report, probe_report):
    return jsonable({
        "config_id": R["id"], "config_file": R["config_file"], "config_sha256": R["config_sha256"],
        "results_dirname": R["paths"]["dirname"], "resolved_parameters": {k: R[k] for k in (
            "gop", "resolution", "checkpoint", "ckpt_id", "frames_count", "repeats", "video", "codec", "resize",
            "detector", "energy", "measurement", "validation", "estimate", "limits")},
        "gop_requested": R["gop"], "expected_frame_types": xc.expected_types(R["frames_count"], R["gop"]),
        "resolution_requested": R["resolution"], "fps": R["fps"], "fps_assumed": R["fps_assumed"],
        "video": {**R["video"], "n_frames": R["frames_count"], "source_dir": R["paths"]["frames_dir"],
                  "source_resolution": plan.get("source_resolution"), "input_action": plan.get("action"),
                  "inputs_dir_used": plan.get("inputs_dir"), "resize_method": R["resize"]["method"] if plan.get("action") == "resize" else None,
                  "inputs_sha256": plan.get("inputs_sha256"), "cache_files_created": plan.get("created"),
                  "cache_files_reused": plan.get("reused")},
        "checkpoint": {"name": R["checkpoint"], "id": R["ckpt_id"], "file": R["paths"]["checkpoint_file"],
                       "inspection": ckpt_report,
                       "note": "aucune hypothèse n'est faite sur une signification de qualité de base_k ; voir "
                               "config_like_fields et probe_report (bpp/PSNR observés)"},
        "detector": R["detector"], "detector_hash": R["detector_hash"],
        "energy_config": R["energy"], "libraries": library_versions(),
        "validation_checks": checks, "probe_report": probe_report})


def validate_ids(cfg, ids, deep, force_inputs=False):
    """Validation statique (+ profonde) de une ou plusieurs expériences + contrôles croisés. Écrit le rapport dans
    <results_group>/_validation/ (jamais dans un dossier de résultats d'expérience). Retourne (Rs, rapports, echec)."""
    root = Path(__file__).resolve().parent
    Rs, reports, cache, fail = [], {}, {}, False
    for i in ids:
        R = xc.resolve(cfg, i, root)
        ch = xc.static_checks(R, deep=deep, ckpt_cache=cache)
        reports[i] = ch
        Rs.append(R)
        xc.print_checks(f"{i} -> {R['paths']['dirname']}", ch)
        fail = fail or xc.has_fail(ch)
    cross = xc.cross_checks(Rs, deep=deep, ckpt_cache=cache) if len(Rs) > 1 or deep else []
    if cross:
        xc.print_checks("contrôles croisés", cross)
        fail = fail or xc.has_fail(cross)
    reports["_cross"] = cross
    vdir = Path(Rs[0]["paths"]["results_root"]) / Rs[0]["paths"]["results_group"] / "_validation"
    vdir.mkdir(parents=True, exist_ok=True)
    rp = vdir / f"validation_{time.strftime('%Y%m%d_%H%M%S')}_{'-'.join(ids)}.json"
    dump_json(rp, {"ids": ids, "deep": deep, "failed": fail, "checks": reports,
                   "checkpoint_reports": cache, "config_sha256": cfg["_sha256"]})
    print(f"\nRapport de validation : {rp}")
    return Rs, reports, cache, fail


def prepare_and_probe(R, probe, cache):
    """Prépare les entrées (cache redimensionné), vérifie la résolution sur fichiers, option : essai réel du codec."""
    plan = xc.prepare_inputs(R, execute=True)
    if plan["inputs_dir"] is None:
        sys.exit("[ERREUR] aucune image source exploitable")
    bad = verify_input_files(R, plan["inputs_dir"])
    checks = [{"check": "résolution appliquée sur les fichiers d'entrée", "status": xc.FAIL if bad else xc.OK,
               "detail": f"{bad[:5]}" if bad else f"{R['frames_count']} images {R['resolution']} dans {plan['inputs_dir']}"
               + (f" ({len(plan['created'])} créées, {len(plan['reused'])} réutilisées)" if plan["action"] == "resize" else "")}]
    probe_report = None
    if probe and not bad:
        M = load_modules()
        problems = check_api(M)
        if problems:
            sys.exit(f"[ERREUR] API : {problems}")
        a = xc.to_namespace(R, plan["inputs_dir"])
        M["torch"].set_num_threads(a.threads)
        pc, probe_report = probe_config(R, a, M)
        checks += pc
    xc.print_checks(f"{R['id']} : préparation des entrées{' + essai réel du codec' if probe else ''}", checks)
    return plan, checks, probe_report


def batch_run(cfg, ids, args):
    root = Path(__file__).resolve().parent
    Rs, _, cache, fail = validate_ids(cfg, ids, deep=True)
    if fail:
        sys.exit("\n[ARRÊT] la validation a échoué : AUCUNE configuration n'a été lancée.")
    tot = sum(R["_estimated_seconds"] for R in Rs) + (len(Rs) - 1) * float(cfg.get("batch", {}).get("cooldown_between_configs_s", 60))
    print("\nPLAN DU LOT (ordre d'exécution, aucune autre combinaison) :")
    for R in Rs:
        print(f"  {R['id']:<4} GOP {R['gop']:<3} {R['resolution']:<8} {R['checkpoint']:<8} -> {R['paths']['dirname']}")
    print(f"Durée estimée totale (grossière) : {hms(tot)}")
    bdir = Path(Rs[0]["paths"]["results_root"]) / Rs[0]["paths"]["results_group"] / f"_batch_{time.strftime('%Y%m%d_%H%M%S')}"
    bdir.mkdir(parents=True, exist_ok=False)
    cooldown = float(cfg.get("batch", {}).get("cooldown_between_configs_s", 60))
    stop = bool(cfg.get("batch", {}).get("stop_on_failure", True))
    rows = []
    for n, R in enumerate(Rs):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", cfg["_file"], "--id", R["id"], "--run"]
        if args.probe:
            cmd.append("--probe")
        if args.force:
            cmd.append("--force")
        env = dict(os.environ, VCM_BATCH_PARENT=str(os.getpid()))
        print(f"\n{'=' * 80}\n[LOT {n + 1}/{len(Rs)}] {R['id']} : {' '.join(cmd[2:])}\n{'=' * 80}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, env=env, cwd=str(root))
        status = None
        try:
            status = json.load(open(Path(R["paths"]["out_dir"]) / "summary.json", encoding="utf-8")).get("status")
        except Exception:
            pass
        rows.append({"id": R["id"], "dirname": R["paths"]["dirname"], "out_dir": R["paths"]["out_dir"], "return_code": rc,
                     "status": status, "elapsed_s": time.time() - t0})
        dump_json(bdir / "batch_summary.json", {"ids": ids, "config_sha256": cfg["_sha256"], "runs": rows})
        write_csv(bdir / "batch_summary.csv", rows)
        if rc != 0 and stop:
            print(f"[LOT] {R['id']} en échec (code {rc}) : arrêt du lot (stop_on_failure).")
            break
        if n < len(Rs) - 1 and cooldown > 0:
            print(f"[LOT] pause thermique {cooldown:g} s", flush=True)
            time.sleep(cooldown)
    print(f"\nRésumé du lot : {bdir / 'batch_summary.json'}")
    sys.exit(0 if all(r["return_code"] == 0 for r in rows) and len(rows) == len(Rs) else 1)


def main():
    ap = argparse.ArgumentParser(
        description="VCM-EnergyMap exp.3 v5.1 : expériences pilotées par experiment_config_v51.yaml. "
                    "SANS action (--run / --batch / --probe / --check_energy) : validation statique uniquement.")
    ap.add_argument("--config", default=str(ROOT / "experiment_config_v51.yaml"))
    ap.add_argument("--list", action="store_true", help="liste les expériences du YAML")
    ap.add_argument("--id", help="expérience unique (ex. C1)")
    ap.add_argument("--ids", help="liste explicite pour --batch ou --validate (ex. C1,C2,C3)")
    ap.add_argument("--deep", action="store_true", help="validation profonde : inspection torch des checkpoints (plus lourd)")
    ap.add_argument("--probe", action="store_true",
                    help="essai RÉEL du codec sur quelques images, sans énergie (GOP, I/P, checkpoint, résolution décodée)")
    ap.add_argument("--run", action="store_true", help="mode individuel : lance l'expérience --id")
    ap.add_argument("--batch", action="store_true", help="mode lot : lance successivement les expériences de --ids")
    ap.add_argument("--dry_run", action="store_true", help="avec --run : plan + vérification d'API du moteur, sans mesure")
    ap.add_argument("--check_energy", action="store_true", help="teste uniquement la collecte d'énergie (sans codec)")
    ap.add_argument("--check_seconds", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="outrepasse le refus si une autre expérience tourne (déconseillé)")
    args = ap.parse_args()
    try:
        cfg = xc.load_config(args.config)
    except xc.ConfigError as e:
        sys.exit(f"[ERREUR] {e}")
    root = Path(__file__).resolve().parent
    if args.list:
        print(f"{'id':<5}{'actif':<7}{'GOP':<5}{'résolution':<12}{'checkpoint':<12}dossier")
        for e in cfg["experiments"]:
            R = xc.resolve(cfg, e["id"], root)
            print(f"{e['id']:<5}{str(e.get('enabled', True)):<7}{R['gop']:<5}{R['resolution']:<12}{R['checkpoint']:<12}{R['paths']['dirname']}")
        return
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else ([args.id] if args.id else [])
    try:
        if args.check_energy:
            R = xc.resolve(cfg, ids[0] if ids else cfg["experiments"][0]["id"], root)
            heavy_guard("--check_energy", args.force)
            a = xc.to_namespace(R, R["paths"]["frames_dir"], {"check_seconds": args.check_seconds})
            sys.exit(check_energy(a))
        if args.batch:
            if not ids:
                sys.exit("[ERREUR] --batch exige --ids C1,C2,... (aucune combinaison n'est lancée automatiquement)")
            disabled = [i for i in ids if not next(e for e in cfg["experiments"] if e["id"] == i).get("enabled", True)]
            if disabled:
                sys.exit(f"[ERREUR] expériences désactivées dans le YAML : {disabled}")
            heavy_guard("--batch", args.force)
            batch_run(cfg, ids, args)
            return
        if not ids:
            sys.exit("[ERREUR] précisez --id Cx (ou --ids pour valider plusieurs expériences) ; --list pour les voir")
        deep = args.deep or args.probe or args.run
        if deep or args.run:
            heavy_guard("validation profonde / essai / exécution", args.force)
        Rs, reports, cache, fail = validate_ids(cfg, ids, deep=deep)
        if fail:
            sys.exit("\n[ÉCHEC] validation : corrigez le YAML ou l'environnement (rien n'a été lancé).")
        if not (args.probe or args.run):
            print("\nValidation statique OK. Rien n'a été lancé. Étapes suivantes : --deep, --probe, puis --run.")
            return
        if len(Rs) != 1:
            sys.exit("[ERREUR] --probe / --run : une seule expérience (--id)")
        R = Rs[0]
        plan, pchecks, probe_report = prepare_and_probe(R, args.probe, cache)
        if xc.has_fail(pchecks):
            sys.exit("\n[ÉCHEC] préparation/essai : l'expérience n'est pas lancée.")
        if not args.run:
            print("\nEssai OK (aucune énergie mesurée, aucun résultat d'expérience créé).")
            return
        a = xc.to_namespace(R, plan["inputs_dir"], {"dry_run": args.dry_run})
        mism = xc.namespace_consistency(R, a)       # une seule configuration : moteur == YAML résolu
        if mism:
            sys.exit(f"[ERREUR] incohérence entre le YAML résolu et les paramètres du moteur : {mism}")
        trace = build_trace(R, plan, reports[R["id"]] + pchecks, cache.get(R["checkpoint"]), probe_report)
        trace["config_consistency"] = {"engine_namespace_equals_resolved_yaml": True, "mismatches": mism}
        run_experiment(a, trace)
    except xc.ConfigError as e:
        sys.exit(f"[ERREUR] {e}")


if __name__ == "__main__":
    main()
