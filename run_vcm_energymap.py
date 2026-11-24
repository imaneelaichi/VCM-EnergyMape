#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_vcm_energymap.py -- orchestrateur VCM-EnergyMap (Base Layer, 4 checkpoints, sequentiel).

NE MODIFIE AUCUN script existant. Il se contente de les APPELER :
  * test_base_profiled.py  encode / decode / evaluate / energy   (sous-processus, un par etape)
  * compare_detection_145.py  (importe dans un sous-processus ; ses constantes de chemins sont
                               redefinies a l'execution, le fichier n'est jamais modifie)
et d'agreger leurs sorties (JSON/CSV reels) dans results/<run>/comparison/.

A placer dans svc_code\ (a cote de test_base_profiled.py). Python 3.10, PowerShell, environnement svc_env_hp.

PILOTE 10 IMAGES (4 checkpoints) :
  python run_vcm_energymap.py --no_frames 10 --gop 32 --start_index 1 --repeats 3 --min_seconds 20
VERIFIER SANS RIEN LANCER :
  python run_vcm_energymap.py --no_frames 10 --dry_run
REPRENDRE APRES INTERRUPTION :
  python run_vcm_energymap.py --no_frames 10 --resume
145 IMAGES (memes parametres) :
  python run_vcm_energymap.py --no_frames 145 --gop 32 --start_index 1 --repeats 3 --min_seconds 20 --energy_frames 33 --no_dump_tensors

Sorties (results\<run_name>\, run_name = pilot_<no_frames> par defaut) :
  base_k\bitstream\base_k.vcmb          vrai bitstream
  base_k\metrics\base_k_*               JSON/CSV produits par test_base_profiled.py
  base_k\reconstructed\                 images decodees
  base_k\detection\                     sorties de compare_detection_145.py (original vs reconstruit)
  base_k\logs\<etape>.log, status.json  journaux et etat (reprise)
  comparison\all_checkpoints.csv/.json, stage_breakdown.csv, experiment_log.txt, figures\*.png|pdf

REGLES DE SECURITE : aucun resultat n'est supprime ni ecrase. Un dossier base_k existant sans --resume est
laisse intact (le checkpoint est saute). Avec --resume, les fichiers PARTIELS d'une etape non terminee sont
DEPLACES dans base_k\_partial_<horodatage>\ avant de la relancer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAN = float("nan")
STEPS = ["encode", "decode", "evaluate", "detection", "energy"]
REQUIRED_SCRIPTS = ["test_base_profiled.py", "energy_profiling.py", "profiling.py", "base_codec.py",
                    "compare_detection_145.py"]


# ======================================================================================
# utilitaires
# ======================================================================================
def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else NAN
    except (TypeError, ValueError):
        return NAN


def mean_std(v):
    v = [x for x in v if isnum(x)]
    if not v:
        return NAN, NAN
    m = sum(v) / len(v)
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) if len(v) > 1 else NAN
    return m, sd


def jsonable(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(p):
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


class Log:
    """Ecrit a la console ET dans experiment_log.txt (ajout, jamais d'ecrasement)."""

    def __init__(self):
        self.path = None

    def set_file(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, msg="", console=True):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if console:
            print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


log = Log()


# ======================================================================================
# execution d'un sous-processus (journal + progression, jamais en parallele)
# ======================================================================================
def run_cmd(cmd, log_path: Path, label: str, heartbeat_s=60):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("COMMANDE [" + label + "] : " + subprocess.list2cmdline(cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("$ " + subprocess.list2cmdline(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)

        def pump():
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                sys.stdout.write("    | " + line if line.endswith("\n") else "    | " + line + "\n")
                sys.stdout.flush()

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        last = t0
        while p.poll() is None:
            time.sleep(1.0)
            if time.perf_counter() - last >= heartbeat_s:
                last = time.perf_counter()
                el = int(last - t0)
                print(f"    ... {label} en cours ({el // 60:d} min {el % 60:02d} s)", flush=True)
        th.join(timeout=10)
        rc = p.returncode
        lf.write(f"\n[return code {rc}]\n")
    dt = time.perf_counter() - t0
    log(f"FIN [{label}] code={rc} duree={dt:.1f}s (journal: {log_path})", console=True)
    return rc, dt


# ======================================================================================
# worker de detection : reutilise compare_detection_145.py SANS le modifier
# ======================================================================================
def detect_worker(cfg_path):
    cfg = read_json(cfg_path)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    if cfg.get("threads"):
        import torch
        torch.set_num_threads(int(cfg["threads"]))
    import compare_detection_145 as cd
    need = ["ORIGINAL_DIR", "RECONSTRUCTED_DIR", "LABELS_DIR", "OUTPUT_DIR", "ORIGINAL_OUTPUT_DIR",
            "RECONSTRUCTED_OUTPUT_DIR", "main"]
    missing = [n for n in need if not hasattr(cd, n)]
    if missing:
        raise RuntimeError(f"compare_detection_145.py ne fournit pas : {missing} (version differente ?)")
    out = Path(cfg["output_dir"])
    cd.ORIGINAL_DIR = Path(cfg["original_dir"])
    cd.RECONSTRUCTED_DIR = Path(cfg["reconstructed_dir"])
    cd.LABELS_DIR = Path(cfg["labels_dir"])
    cd.OUTPUT_DIR = out
    cd.ORIGINAL_OUTPUT_DIR = out / "original"
    cd.RECONSTRUCTED_OUTPUT_DIR = out / "reconstructed"
    cd.main()


# ======================================================================================
# etat / reprise
# ======================================================================================
class Status:
    def __init__(self, path: Path):
        self.path = path
        self.d = read_json(path) if path.exists() else {"steps": {}, "params": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jsonable(self.d), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def step(self, name):
        return self.d["steps"].get(name, {})

    def set(self, name, **kw):
        self.d["steps"].setdefault(name, {}).update(kw)
        self.save()


def quarantine(paths, ckpt_dir: Path):
    """Deplace (jamais ne supprime) les fichiers partiels existants avant de relancer une etape."""
    moved = []
    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        return moved
    q = ckpt_dir / f"_partial_{time.strftime('%Y%m%d_%H%M%S')}"
    q.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p = Path(p)
        rel = p.relative_to(ckpt_dir)
        dst = q / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        moved.append(str(rel))
    log(f"  fichiers partiels deplaces vers {q.name}: {moved}")
    return moved


# ======================================================================================
# collecte des metriques (lecture des JSON/CSV REELS produits par les scripts existants)
# ======================================================================================
def cpu_kind(mode):
    m = str(mode).lower()
    if m in ("windows_emi", "intel_rapl"):
        return f"CPU mesuree (compteurs RAPL via {m})"
    if m == "intel_power_gadget":
        return "CPU mesuree (Intel Power Gadget)"
    if m in ("cpu_load", "constant", "tdp", "none", "none_type"):
        return f"CPU MODELISEE ({m} : TDP x charge)"
    return f"CPU : methode inconnue ({mode}) - a verifier"


def block_stats(blocks, label, idle):
    """Retourne un dict de metriques d'un bloc d'energie (labels : encode_sequence...)."""
    b = next((x for x in blocks if x.get("label") == label), None)
    if not b:
        return None
    runs = b.get("runs", [])
    g = b.get("aggregate", {})
    idle_cpu_w = NAN
    if idle and fnum(idle.get("wall_s")) > 0:
        idle_cpu_w = fnum(idle.get("energy_cpu_j")) / fnum(idle.get("wall_s"))
    net_cpu = [fnum(r["gross_cpu_j_per_frame"]) - idle_cpu_w * fnum(r["time_per_frame_s"]) for r in runs]
    cpu_m, cpu_sd = mean_std([fnum(r["gross_cpu_j_per_frame"]) for r in runs])
    ram_m, ram_sd = mean_std([fnum(r["gross_ram_j_per_frame"]) for r in runs])
    ncpu_m, ncpu_sd = mean_std(net_cpu)
    return {
        "n_repeats": len(runs), "iterations_per_repeat": [r.get("iterations") for r in runs],
        "time_s_per_frame_mean": fnum(g.get("time_per_frame_s_mean")), "time_s_per_frame_std": fnum(g.get("time_per_frame_s_std")),
        "gross_j_per_frame_mean": fnum(g.get("gross_j_per_frame_mean")), "gross_j_per_frame_std": fnum(g.get("gross_j_per_frame_std")),
        "net_j_per_frame_mean": fnum(g.get("net_j_per_frame_mean")), "net_j_per_frame_std": fnum(g.get("net_j_per_frame_std")),
        "gross_cpu_j_per_frame_mean": cpu_m, "gross_cpu_j_per_frame_std": cpu_sd,
        "gross_ram_j_per_frame_mean": ram_m, "gross_ram_j_per_frame_std": ram_sd,
        "net_cpu_j_per_frame_mean": ncpu_m, "net_cpu_j_per_frame_std": ncpu_sd,
        "mean_power_w_mean": fnum(g.get("mean_power_w_mean")), "mean_power_w_std": fnum(g.get("mean_power_w_std")),
        "idle_power_total_w": fnum(b.get("idle_power_w")), "idle_power_cpu_w": idle_cpu_w,
    }


def collect_checkpoint(k, ckpt_dir: Path, ckpt_file: Path, status: Status, args):
    notes, row = [], {}
    stem = f"base_{k}"
    mdir = ckpt_dir / "metrics"
    row["checkpoint"] = k
    row["checkpoint_file"] = ckpt_file.name
    row["checkpoint_size_bytes"] = ckpt_file.stat().st_size if ckpt_file.exists() else NAN
    row["checkpoint_sha256"] = status.d.get("checkpoint_sha256", "")
    row["steps_state"] = ";".join(f"{s}={status.step(s).get('state', 'absent')}" for s in STEPS)
    row["n_frames"], row["gop"], row["start_index"], row["threads"] = args.no_frames, args.gop, args.start_index, args.threads
    stage_rows, extra = [], {}

    # ---- profil encodeur / decodeur (temps par image et par etape, E/S comprises)
    prof = {}
    for d in ("enc", "dec"):
        p = mdir / f"{stem}_{d}_profile.json"
        if p.exists():
            prof[d] = read_json(p)
        else:
            notes.append(f"{d}: profil absent ({p.name}) -> temps par image/etape = NaN")
    for d, pj in prof.items():
        fs = pj.get("frame_summary", [])
        tot = sum(fnum(r["wall_mean_s"]) * r["n_frames"] for r in fs)
        nfr = sum(r["n_frames"] for r in fs)
        row[f"{d}_time_s_per_frame_profile"] = tot / nfr if nfr else NAN
        for t in ("I", "P"):
            r = next((x for x in fs if x["type"] == t), None)
            row[f"{d}_time_s_per_frame_profile_{t}"] = fnum(r["wall_mean_s"]) if r else NAN
            row[f"{d}_n_frames_{t}"] = r["n_frames"] if r else 0
        row[f"{d}_init_model_build_s"] = fnum(pj.get("init_times_s", {}).get("model_build"))
        row[f"{d}_init_checkpoint_load_s"] = fnum(pj.get("init_times_s", {}).get("checkpoint_load"))
        row[f"{d}_warmup_frames"] = pj.get("warmup_frames")
        row[f"{d}_profile_coverage_mean"] = (sum(fnum(r["coverage"]) * r["n_frames"] for r in fs) / nfr) if nfr else NAN
        for s in pj.get("stage_summary", []):
            stage_rows.append({"checkpoint": k, "dir": d, "type": s["type"], "stage": s["stage"], "category": s["category"],
                               "n": s["n"], "total_s": s["total_s"], "mean_s": s["mean_s"], "std_s": s["std_s"]})
        extra[f"{d}_wall_s_per_frame_all"] = row[f"{d}_time_s_per_frame_profile"]
        extra[f"{d}_measured_frames"] = pj.get("measured_frames")
    if "enc" in prof:
        m = prof["enc"].get("meta", {})
        row["bitstream_total_bytes"] = fnum(m.get("total_bytes"))
        row["bitstream_header_bytes"] = fnum(m.get("header_bytes"))
        row["bitstream_frames_bytes"] = fnum(m.get("frames_bytes"))
        row["bpp_file"] = fnum(m.get("bpp_file"))
        extra["ckpt_info"] = m.get("ckpt_info")
        extra["torch"] = m.get("torch")
    else:
        for c in ("bitstream_total_bytes", "bitstream_header_bytes", "bitstream_frames_bytes", "bpp_file"):
            row[c] = NAN

    # ---- evaluation (debit estime/reel, PSNR, MSE, MS-SSIM)
    pe = mdir / f"{stem}_eval_summary.json"
    if pe.exists():
        ev = read_json(pe)
        for key, col in (("bpp_est_mean", "bpp_est_mean"), ("bpp_payload_mean", "bpp_payload_mean"),
                         ("bpp_frame_mean", "bpp_frame_mean"), ("bpp_real_over_est_ratio", "bpp_real_over_est_ratio"),
                         ("bpp_payload_mean_I", "bpp_payload_mean_I"), ("bpp_payload_mean_P", "bpp_payload_mean_P"),
                         ("bpp_est_mean_I", "bpp_est_mean_I"), ("bpp_est_mean_P", "bpp_est_mean_P"),
                         ("psnr_dec_float_from_mean_mse", "psnr_db_float"), ("msssim_dec_float_mean", "msssim_float_mean"),
                         ("psnr_dec_png8_mean", "psnr_db_png8_mean_of_frames"), ("msssim_dec_png8_mean", "msssim_png8_mean"),
                         ("max_abs_diff_enc_vs_dec", "max_abs_diff_enc_vs_dec")):
            row[col] = fnum(ev.get(key))
        row["all_frames_bit_exact"] = ev.get("all_frames_bit_exact", "")
        if not isnum(row["psnr_db_float"]):
            notes.append("PSNR/MS-SSIM float : absents (evaluate lance sans --dump_tensors) -> seules les valeurs PNG 8 bits existent")
        else:
            row["mse_mean_float"] = 10 ** (-row["psnr_db_float"] / 10)   # DERIVE : PSNR calcule depuis la MSE moyenne
        pf = mdir / f"{stem}_eval_frames.csv"
        if pf.exists():
            fr = read_csv(pf)
            for t in ("I", "P"):
                v = [fnum(r.get("psnr_dec_float")) for r in fr if r["type"] == t]
                row[f"psnr_db_float_mean_{t}"] = mean_std(v)[0]
    else:
        notes.append(f"evaluate: {pe.name} absent -> debit estime/reel, PSNR, MSE, MS-SSIM = NaN")
        for c in ("bpp_est_mean", "bpp_payload_mean", "bpp_frame_mean", "bpp_real_over_est_ratio", "psnr_db_float",
                  "mse_mean_float", "msssim_float_mean", "psnr_db_png8_mean_of_frames", "msssim_png8_mean"):
            row[c] = NAN

    # ---- detection
    pd_ = ckpt_dir / "detection" / "results.json"
    if pd_.exists():
        dj = read_json(pd_)
        o, r = dj["original"], dj["reconstructed"]
        for m_ in ("precision", "recall", "mAP50", "mAP50_95"):
            row[f"det_{m_}"] = fnum(r.get(m_))
            row[f"det_{m_}_original"] = fnum(o.get(m_))
            row[f"det_{m_}_delta"] = row[f"det_{m_}"] - row[f"det_{m_}_original"]
        row["det_images"] = r.get("images")
        row["det_gt_objects"] = r.get("ground_truth_objects")
        row["det_predictions_reconstructed"] = r.get("predictions")
        row["yolo_mean_inference_time_s_reconstructed"] = fnum(r.get("mean_inference_time_s"))
        extra["detection_per_class"] = dj.get("per_class")
    else:
        notes.append("detection: results.json absent -> precision/recall/mAP = NaN")
        for m_ in ("precision", "recall", "mAP50", "mAP50_95"):
            for suf in ("", "_original", "_delta"):
                row[f"det_{m_}{suf}"] = NAN
    row["yolo_energy_j_per_frame"] = NAN
    notes.append("energie YOLO : non mesuree par ce script (volontairement non ajoutee a l'energie du codec)")

    # ---- energie (blocs entiers uniquement)
    pen = mdir / f"{stem}_energy.json"
    row["energy_frames"] = NAN
    if pen.exists():
        ej = read_json(pen)
        meta = ej.get("meta", {})
        mode = meta.get("cpu_power_method")
        row["cpu_power_method"] = mode
        row["energy_kind_reported_by_script"] = meta.get("energy_kind")
        row["energy_cpu_kind"] = cpu_kind(mode)
        row["energy_ram_kind"] = "RAM MODELISEE (puissance fixe selon la memoire installee, CodeCarbon)"
        row["energy_kind_corrected"] = ("MIXTE : CPU mesuree (RAPL via windows_emi) + RAM modelisee"
                                        if str(mode).lower() in ("windows_emi", "intel_rapl") else "MODELISEE (proxy, pas une mesure)")
        row["energy_frames"] = meta.get("frames", NAN)
        idle = ej.get("idle")
        row["idle_power_total_w"] = fnum((idle or {}).get("mean_power_total_w"))
        for blabel, pref in (("encode_sequence", "enc"), ("decode_sequence", "dec"), ("encode_iframe_only", "encI")):
            bs = block_stats(ej.get("blocks", []), blabel, idle)
            if bs is None:
                if pref != "encI":
                    notes.append(f"energie: bloc {blabel} absent")
                bs = {}
            for key in ("n_repeats", "time_s_per_frame_mean", "time_s_per_frame_std", "gross_j_per_frame_mean",
                        "gross_j_per_frame_std", "net_j_per_frame_mean", "net_j_per_frame_std",
                        "gross_cpu_j_per_frame_mean", "gross_cpu_j_per_frame_std", "gross_ram_j_per_frame_mean",
                        "net_cpu_j_per_frame_mean", "net_cpu_j_per_frame_std", "mean_power_w_mean", "mean_power_w_std",
                        "idle_power_cpu_w"):
                row[f"{pref}_{key}"] = bs.get(key, NAN)
            for key in ("gross_j_per_frame_mean", "net_j_per_frame_mean"):
                v = bs.get(key, NAN)
                row[f"{pref}_{key.replace('_j_', '_mwh_')}"] = v / 3.6 if isnum(v) else NAN     # 1 mWh = 3.6 J
        if isnum(row.get("enc_gross_j_per_frame_mean")) and isnum(row.get("dec_gross_j_per_frame_mean")):
            row["codec_gross_j_per_frame_enc_plus_dec"] = row["enc_gross_j_per_frame_mean"] + row["dec_gross_j_per_frame_mean"]
            row["codec_net_j_per_frame_enc_plus_dec"] = row.get("enc_net_j_per_frame_mean", NAN) + row.get("dec_net_j_per_frame_mean", NAN)
        extra["energy_idle"] = idle
        extra["energy_meta"] = meta
        notes.append("l'energie d'encodage inclut le calcul des bits estimes (est_bits : une evaluation de vraisemblance "
                     "supplementaire absente d'un vrai encodeur) -> surestimation de l'encodage, a quantifier")
    else:
        notes.append(f"energie: {pen.name} absent -> toutes les energies = NaN")
    row["notes"] = " | ".join(notes)
    return row, stage_rows, extra


COLUMN_UNITS = {
    "bitstream_*_bytes": "octets", "bpp_*": "bits par pixel (H*W = taille originale non paddee)",
    "psnr_*": "dB", "mse_mean_float": "MSE moyenne (echelle [0,1]^2, derivee du PSNR)", "msssim_*": "sans unite [0,1]",
    "det_*": "sans unite [0,1] (YOLOv5s, classes person+car, conf 0.25)",
    "*_time_s_*": "secondes (par image)", "*_j_*": "joules (par image) ; 1 mWh = 3.6 J", "*_mwh_*": "mWh (par image)",
    "*_power_*_w": "watts (puissance moyenne)", "gross": "energie brute (CPU+RAM, bloc entier)",
    "net": "energie brute - puissance au repos x duree (repos mesure avant les blocs)",
}


# ======================================================================================
# figures (matplotlib) : uniquement des donnees reellement mesurees
# ======================================================================================
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
CK_STYLE = {1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 3: ("#1baf7a", "^"), 4: ("#eda100", "D")}   # palette validee (ordre fixe)
STAGE_GROUPS = [  # 7 groupes <= 8 teintes
    ("Mouvement (estim.+pred.+compens.)", ("motion_estimation", "motion_prediction", "motion_compensation")),
    ("Reseaux analyse/synthese/cond.", ("_analysis", "_hyper_analysis", "_cond_net", "_synthesis", "align", "crop_clamp")),
    ("Quantification", ("_quantize", "_dequantize")),
    ("PMF + reseau de contexte", ("_pmf", "_context_net")),
    ("Codage entropique", ("_rangecode", "_rangedecode")),
    ("E/S (lecture/ecriture, bitstream)", ("image_read", "image_write", "bitstream_write", "bitstream_read",
                                           "preprocess_to_tensor", "image_decode_png")),
    ("Non instrumente (reste)", ()),
]
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def stage_group(name):
    for gname, pats in STAGE_GROUPS[:-1]:
        for p in pats:
            if name == p or (p.startswith("_") and name.endswith(p)):
                return gname
    return STAGE_GROUPS[-1][0]


def make_figures(rows, stage_rows, outdir: Path, args, footnote):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"FIGURES NON GENEREES : matplotlib indisponible ({e}). pip install matplotlib")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False})
    made = []
    rows = sorted(rows, key=lambda r: r["checkpoint"])

    def save(fig, name):
        fig.text(0.01, 0.005, footnote, fontsize=7, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        made.append(name)
        log(f"figure generee : {name}.png/.pdf")

    def avail(cols, label):
        rr = [r for r in rows if all(isnum(r.get(c)) for c in cols)]
        if not rr:
            log(f"figure NON generee ({label}) : colonnes sans valeur mesuree {cols}")
        elif len(rr) < len(rows):
            log(f"figure {label} : {len(rr)}/{len(rows)} checkpoints disponibles")
        return rr

    def scatter(name, title, xcol, xerr, xlabel, ycol, ylabel, ref=None):
        rr = avail([xcol, ycol], name)
        if not rr:
            return
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        for r in rr:
            c, mk = CK_STYLE[r["checkpoint"]]
            xe = r.get(xerr) if xerr and isnum(r.get(xerr)) else None
            ax.errorbar(r[xcol], r[ycol], xerr=xe, fmt=mk, color=c, markersize=9, markeredgecolor=INK, markeredgewidth=0.8,
                        ecolor=INK2, elinewidth=1, capsize=3, label=f"checkpoint {r['checkpoint']}")
            ax.annotate(f"base_{r['checkpoint']}", (r[xcol], r[ycol]), textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK)
        if ref is not None:
            v = [r[ref] for r in rr if isnum(r.get(ref))]
            if v:
                ax.axhline(sum(v) / len(v), ls="--", lw=1, color=INK2)
                ax.annotate("images originales (sans compression)", (ax.get_xlim()[0], sum(v) / len(v)), textcoords="offset points",
                            xytext=(4, 4), fontsize=8, color=INK2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        save(fig, name)

    scatter("fig1_map50_vs_bitrate", "Précision de détection vs débit", "bpp_file", None, "Débit (bpp, fichier .vcmb complet)",
            "det_mAP50", "mAP@50 (YOLOv5s)", ref="det_mAP50_original")
    scatter("fig2_map50_vs_energie_encodage", "Précision de détection vs énergie d'encodage par image", "enc_gross_j_per_frame_mean",
            "enc_gross_j_per_frame_std", "Énergie d'encodage brute (J / image, CPU+RAM ; barre = écart-type des répétitions)",
            "det_mAP50", "mAP@50 (YOLOv5s)", ref="det_mAP50_original")
    scatter("fig3_map50_95_vs_bitrate", "mAP@50:95 vs débit", "bpp_file", None, "Débit (bpp, fichier .vcmb complet)",
            "det_mAP50_95", "mAP@50:95 (YOLOv5s)", ref="det_mAP50_95_original")

    def grouped(name, title, specs, ylabel, panels):
        """specs: [(label, colprefix_mean, colprefix_std, color)] ; panels: [(subtitle, mean_suffix, std_suffix)]"""
        ok = False
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
        for ax, (sub, msuf, ssuf) in zip(axes[0], panels):
            xs = list(range(len(rows)))
            w = 0.8 / len(specs)
            for j, (lab, pre, color) in enumerate(specs):
                vals = [r.get(f"{pre}_{msuf}", NAN) for r in rows]
                errs = [r.get(f"{pre}_{ssuf}", NAN) for r in rows] if ssuf else [NAN] * len(rows)
                xp = [x - 0.4 + w * (j + 0.5) for x in xs]
                ys = [v if isnum(v) else 0 for v in vals]
                ax.bar(xp, ys, width=w * 0.92, color=color, edgecolor=SURF, linewidth=1.5, label=lab)
                for x_, v, e in zip(xp, vals, errs):
                    if isnum(v):
                        ok = True
                        if isnum(e):
                            ax.errorbar(x_, v, yerr=e, color=INK, capsize=3, lw=1)
                    else:
                        ax.annotate("NaN", (x_, 0), ha="center", va="bottom", fontsize=8, color=INK2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"base_{r['checkpoint']}" for r in rows])
            ax.set_ylabel(ylabel)
            ax.set_title(sub, fontsize=10)
            ax.legend(loc="best")
        if not ok:
            plt.close(fig)
            log(f"figure NON generee ({name}) : aucune valeur mesuree")
            return
        fig.suptitle(title, fontweight="bold")
        save(fig, name)

    ENC, DEC = "#2a78d6", "#eb6834"
    grouped("fig4_energie_encodage_decodage", "Énergie d'encodage et de décodage par image",
            [("encodage", "enc", ENC), ("décodage", "dec", DEC)], "Énergie (J / image)",
            [("Brute (CPU+RAM)", "gross_j_per_frame_mean", "gross_j_per_frame_std"),
             ("Nette (brute − repos)", "net_j_per_frame_mean", "net_j_per_frame_std")])
    have_blocks = all(isnum(r.get("enc_time_s_per_frame_mean")) and isnum(r.get("dec_time_s_per_frame_mean")) for r in rows)
    if have_blocks:
        grouped("fig5_temps_encodage_decodage", "Temps d'encodage et de décodage par image (blocs répétés)",
                [("encodage", "enc", ENC), ("décodage", "dec", DEC)], "Temps (s / image)",
                [("moyenne ± écart-type des répétitions", "time_s_per_frame_mean", "time_s_per_frame_std")])
    else:
        grouped("fig5_temps_encodage_decodage", "Temps d'encodage et de décodage par image (profil, 1 passe, E/S incluses)",
                [("encodage", "enc", ENC), ("décodage", "dec", DEC)], "Temps (s / image)",
                [("moyenne sur les images mesurées", "time_s_per_frame_profile", None)])

    # fig6 : PSNR / MS-SSIM (valeurs float du decodeur = comparables a test_base.py)
    rr = avail(["psnr_db_float", "msssim_float_mean"], "fig6")
    if rr:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
        for ax, col, yl, sub in ((axes[0], "psnr_db_float", "PSNR (dB)", "PSNR (depuis la MSE moyenne)"),
                                 (axes[1], "msssim_float_mean", "MS-SSIM", "MS-SSIM moyen")):
            xs = list(range(len(rr)))
            ax.bar(xs, [r[col] for r in rr], width=0.6, color=[CK_STYLE[r["checkpoint"]][0] for r in rr], edgecolor=SURF, linewidth=1.5)
            for x_, r in zip(xs, rr):
                ax.annotate(f"{r[col]:.3f}", (x_, r[col]), ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"base_{r['checkpoint']}" for r in rr])
            ax.set_ylabel(yl)
            ax.set_title(sub, fontsize=10)
            if col == "msssim_float_mean":
                lo = min(r[col] for r in rr)
                ax.set_ylim(max(0, lo - 0.05), 1.0)
                ax.annotate("axe tronqué", (0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=7, color=INK2)
        fig.suptitle("Qualité de reconstruction par checkpoint (sortie flottante du décodeur)", fontweight="bold")
        save(fig, "fig6_psnr_msssim")

    # fig7 : decomposition temporelle par groupe d'etapes (temps, PAS energie)
    if stage_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        any_ = False
        for ax, d, sub in ((axes[0], "enc", "Encodage"), (axes[1], "dec", "Décodage")):
            xs, bottoms = [], {}
            cks = [r for r in rows if isnum(r.get(f"{d}_time_s_per_frame_profile"))]
            if not cks:
                continue
            for xi, r in enumerate(cks):
                nfr = sum(r.get(f"{d}_n_frames_{t}", 0) or 0 for t in ("I", "P"))
                per = {g[0]: 0.0 for g in STAGE_GROUPS}
                for s in stage_rows:
                    if s["checkpoint"] == r["checkpoint"] and s["dir"] == d and nfr:
                        per[stage_group(s["stage"])] += s["total_s"] / nfr
                inst = sum(v for kx, v in per.items() if kx != STAGE_GROUPS[-1][0])
                per[STAGE_GROUPS[-1][0]] = max(r[f"{d}_time_s_per_frame_profile"] - inst, 0.0)
                bot = 0.0
                for gi, g in enumerate(STAGE_GROUPS):
                    v = per[g[0]]
                    ax.bar(xi, v, bottom=bot, width=0.6, color=GROUP_COLORS[gi], edgecolor=SURF, linewidth=1.5,
                           label=g[0] if xi == 0 else None)
                    bot += v
                any_ = True
            ax.set_xticks(range(len(cks)))
            ax.set_xticklabels([f"base_{r['checkpoint']}" for r in cks])
            ax.set_ylabel("Temps moyen (s / image)")
            ax.set_title(sub, fontsize=10)
        if any_:
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.045))
            fig.suptitle("Décomposition du TEMPS par groupe d'étapes (1 passe de profil ; ce n'est pas de l'énergie)", fontweight="bold")
            fig.tight_layout(rect=(0, 0.17, 1, 0.95))
            fig.text(0.01, 0.005, footnote, fontsize=7, color=INK2)
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"fig7_decomposition_temps.{ext}", dpi=200)
            made.append("fig7_decomposition_temps")
            log("figure generee : fig7_decomposition_temps.png/.pdf")
        plt.close(fig)
    else:
        log("figure NON generee (fig7) : aucun profil par etape disponible")

    log("figure NON generee (fig8, courbe de puissance) : les JSON d'energie ne contiennent que des puissances MOYENNES par "
        "tache, pas de serie temporelle -> aucune courbe ne peut etre construite sans fabriquer des donnees")
    return made


# ======================================================================================
# programme principal
# ======================================================================================
def build_parser():
    ap = argparse.ArgumentParser(description="Orchestrateur VCM-EnergyMap (voir l'en-tete du fichier)")
    ap.add_argument("--no_frames", type=int, default=10)
    ap.add_argument("--gop", type=int, default=32)
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="repetitions des blocs d'energie")
    ap.add_argument("--min_seconds", type=float, default=20.0, help="duree minimale d'un bloc d'energie mesure")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument("--resume", action="store_true", help="reprendre : saute les etapes terminees, deplace les partiels")
    ap.add_argument("--run_name", default=None, help="defaut : pilot_<no_frames>")
    ap.add_argument("--results_root", default=str(ROOT / "results"))
    ap.add_argument("--inp_path", default=str(ROOT / "videos_5" / "input_png"))
    ap.add_argument("--labels_dir", default=str(ROOT / "videos_5" / "labels"))
    ap.add_argument("--prefix", default="frame_")
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "base"))
    ap.add_argument("--threads", type=int, default=2, help="threads PyTorch (identiques pour tous les checkpoints)")
    ap.add_argument("--warmup_frames", type=int, default=2, help="images de warm-up (profil enc/dec), ecartees")
    ap.add_argument("--warmup_iters", type=int, default=1, help="iterations de warm-up avant chaque bloc d'energie")
    ap.add_argument("--idle_seconds", type=float, default=30.0)
    ap.add_argument("--measure_power_secs", type=float, default=1.0)
    ap.add_argument("--country", default="MAR")
    ap.add_argument("--energy_frames", type=int, default=None, help="images par bloc d'energie (defaut = no_frames)")
    ap.add_argument("--no_iframe_block", action="store_true", help="ne pas mesurer le bloc 'I-frame seule'")
    ap.add_argument("--closed_gop", action="store_true", help="deviation documentee : remet a zero le tampon de flots a chaque I")
    ap.add_argument("--no_dump_tensors", action="store_true", help="economise disque/RAM (145 images) : pas de test bit-exact ni de PSNR float")
    ap.add_argument("--skip_energy", action="store_true")
    ap.add_argument("--skip_detection", action="store_true")
    ap.add_argument("--no_hash", action="store_true", help="ne pas calculer le SHA256 des checkpoints")
    ap.add_argument("--cooldown", type=float, default=30.0, help="pause (s) avant l'etape d'energie et entre checkpoints")
    ap.add_argument("--machine_label", default="Intel i5-6300U, CPU seul, Windows")
    ap.add_argument("--collect_only", action="store_true", help="n'execute rien : agrege et trace a partir des resultats existants")
    ap.add_argument("--dry_run", action="store_true", help="verifie les entrees et affiche les commandes sans les executer")
    return ap


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--_detect_worker" in sys.argv:
        detect_worker(sys.argv[sys.argv.index("--_detect_worker") + 1])
        return 0
    args = build_parser().parse_args()
    py = sys.executable
    script = str(ROOT / "test_base_profiled.py")
    run_name = args.run_name or f"pilot_{args.no_frames}"
    run_dir = Path(args.results_root) / run_name
    comp = run_dir / "comparison"
    log.set_file(comp / "experiment_log.txt")
    args.energy_frames = args.energy_frames or args.no_frames
    n, k0 = args.no_frames, args.start_index

    log("=" * 78)
    log(f"VCM-EnergyMap orchestrateur | run={run_name} | frames={n} (indices {k0}..{k0 + n - 1}) | gop={args.gop} | "
        f"checkpoints={args.checkpoints} | threads={args.threads} | resume={args.resume}")
    log(f"python {sys.version.split()[0]} | {platform.platform()} | torch {pkg_version('torch')} | codecarbon "
        f"{pkg_version('codecarbon')} | constriction {pkg_version('constriction')} | matplotlib {pkg_version('matplotlib')}")
    log("commande : " + subprocess.list2cmdline([py] + sys.argv))

    # ------------------------------------------------------------ verifications
    problems = []
    for s in REQUIRED_SCRIPTS:
        if not (ROOT / s).is_file():
            problems.append(f"script introuvable : {ROOT / s}")
    for mod in ("torch", "codecarbon", "constriction"):
        if importlib.util.find_spec(mod) is None and not args.collect_only:
            problems.append(f"module Python manquant dans CET interpreteur : {mod} (activez svc_env_hp)")
    inp = Path(args.inp_path)
    frames = [inp / f"{args.prefix}{i:03d}.png" for i in range(k0, k0 + n)]
    for f in frames:
        if not f.is_file():
            problems.append(f"image manquante : {f}")
    if not args.skip_detection:
        for i in range(k0, k0 + n):
            lab = Path(args.labels_dir) / f"frame_{i:04d}.txt"       # convention de compare_detection_145.py
            if not lab.is_file():
                problems.append(f"annotation manquante : {lab}")
        if not (ROOT / "yolov5s.pt").is_file():
            problems.append("yolov5s.pt introuvable")
    ckpts = {k: Path(args.ckpt_dir) / f"checkpoint_base_{k}.pth.tar" for k in args.checkpoints}
    for k, p in ckpts.items():
        if not p.is_file():
            problems.append(f"checkpoint introuvable : {p}")
    if args.energy_frames > n:
        problems.append("--energy_frames ne peut pas depasser --no_frames")
    if n <= args.gop:
        log(f"AVERTISSEMENT : {n} images <= GOP {args.gop} -> 1 seule I-frame et {n - 1} P-frames ; aucun 2e GOP, "
            f"les statistiques I/P ne couvrent pas la remise a zero d'un nouveau GOP (valider ensuite 65 images).")
    if args.warmup_frames > n:
        args.warmup_frames = n
    if problems and not args.collect_only:
        for p in problems:
            log("ERREUR : " + p)
        return 2

    # ------------------------------------------------------------ plan
    def paths_for(k):
        cd = run_dir / f"base_{k}"
        return {"dir": cd, "bs": cd / "bitstream" / f"base_{k}.vcmb", "metrics": cd / "metrics", "recon": cd / "reconstructed",
                "det": cd / "detection", "logs": cd / "logs", "stem": f"base_{k}"}

    dump = not args.no_dump_tensors

    def cmds_for(k):
        P = paths_for(k)
        common = ["--root", str(ROOT), "--checkpoint", str(ckpts[k]), "--checkpoint_number", str(k), "--bitstream", str(P["bs"]),
                  "--prefix", args.prefix, "--start_index", str(k0), "--results", str(P["metrics"]),
                  "--threads", str(args.threads), "--warmup_frames", str(args.warmup_frames)]
        cg = ["--closed_gop"] if args.closed_gop else []
        dmp = ["--dump_tensors"] if dump else []
        c = {
            "encode": [py, script, "encode", *common, *dmp, "--inp_path", str(inp), "--no_frames", str(n), "--gop", str(args.gop), *cg],
            "decode": [py, script, "decode", *common, *dmp, "--out_path", str(P["recon"])],
            "evaluate": [py, script, "evaluate", *common, *dmp, "--inp_path", str(inp), "--out_path", str(P["recon"])],
            "energy": [py, script, "energy", *common, "--inp_path", str(inp), "--no_frames", str(args.energy_frames),
                       "--gop", str(args.gop), "--country", args.country, "--measure_power_secs", str(args.measure_power_secs),
                       "--idle_seconds", str(args.idle_seconds), "--min_seconds", str(args.min_seconds),
                       "--repeats", str(args.repeats), "--warmup_iters", str(args.warmup_iters),
                       *([] if args.no_iframe_block else ["--also_iframe"]), *cg],
        }
        return c

    def outputs_for(k, step):
        P = paths_for(k)
        m, st = P["metrics"], P["stem"]
        return {
            "encode": [P["bs"], m / f"{st}_enc_profile.json", m / f"{st}_enc_frames.csv"] + ([m / f"{st}_enc_rec.pt"] if dump else []),
            "decode": [m / f"{st}_dec_profile.json"] + ([m / f"{st}_dec_rec.pt"] if dump else []),
            "evaluate": [m / f"{st}_eval_summary.json", m / f"{st}_eval_frames.csv"],
            "detection": [P["det"] / "results.json", P["det"] / "metrics_summary.csv"],
            "energy": [m / f"{st}_energy.json"],
        }[step]

    if args.dry_run:
        log("DRY RUN : aucune commande executee. Entrees verifiees.")
        for k in args.checkpoints:
            for step, c in cmds_for(k).items():
                log(f"  [base_{k}/{step}] " + subprocess.list2cmdline(c))
            log(f"  [base_{k}/detection] import de compare_detection_145 avec chemins : original={run_dir / '_yolo_inputs' / 'original'} "
                f"reconstruit={paths_for(k)['recon']} labels={run_dir / '_yolo_inputs' / 'labels'} -> {paths_for(k)['det']}")
        return 0

    # ------------------------------------------------------------ execution sequentielle
    summary = {}
    if not args.collect_only:
        run_dir.mkdir(parents=True, exist_ok=True)
        yolo_orig, yolo_lab = run_dir / "_yolo_inputs" / "original", run_dir / "_yolo_inputs" / "labels"
        if not args.skip_detection:
            yolo_orig.mkdir(parents=True, exist_ok=True)
            yolo_lab.mkdir(parents=True, exist_ok=True)
            for i, f in zip(range(k0, k0 + n), frames):          # copies (jamais de modification des originaux)
                d = yolo_orig / f.name
                if not d.exists():
                    shutil.copy2(f, d)
                lab = Path(args.labels_dir) / f"frame_{i:04d}.txt"
                dl = yolo_lab / lab.name
                if not dl.exists():
                    shutil.copy2(lab, dl)
        fp_keys = {"no_frames": n, "gop": args.gop, "start_index": k0, "prefix": args.prefix, "inp_path": str(inp),
                   "threads": args.threads, "warmup_frames": args.warmup_frames, "repeats": args.repeats,
                   "min_seconds": args.min_seconds, "idle_seconds": args.idle_seconds, "warmup_iters": args.warmup_iters,
                   "measure_power_secs": args.measure_power_secs, "energy_frames": args.energy_frames,
                   "closed_gop": args.closed_gop, "iframe_block": not args.no_iframe_block, "dump_tensors": dump,
                   "labels_dir": args.labels_dir}
        for idx_k, k in enumerate(args.checkpoints, 1):
            P = paths_for(k)
            tag = f"[checkpoint {k}  ({idx_k}/{len(args.checkpoints)})]"
            log("-" * 78)
            log(f"{tag} debut -> {P['dir']}")
            if P["dir"].exists() and not args.resume:
                log(f"{tag} le dossier existe deja : LAISSE INTACT, checkpoint saute. Utilisez --resume pour continuer ou "
                    f"--run_name <autre> pour une nouvelle serie.")
                summary[k] = {"all": "saute (dossier existant)"}
                continue
            P["dir"].mkdir(parents=True, exist_ok=True)
            st = Status(P["dir"] / "status.json")
            if st.d["params"] and st.d["params"] != fp_keys:
                diff = [x for x in fp_keys if st.d["params"].get(x) != fp_keys[x]]
                log(f"{tag} ERREUR : les parametres different de ceux de la serie existante ({diff}). Ne pas melanger : "
                    f"utilisez --run_name <autre>.")
                summary[k] = {"all": "parametres incoherents"}
                continue
            st.d["params"] = fp_keys
            ck = ckpts[k]
            st.d["checkpoint_file"] = str(ck)
            st.d["checkpoint_size_bytes"] = ck.stat().st_size
            if not args.no_hash and not st.d.get("checkpoint_sha256"):
                log(f"{tag} calcul du SHA256 du checkpoint ({ck.stat().st_size / 1e6:.0f} Mo)...")
                st.d["checkpoint_sha256"] = sha256_file(ck)
            st.save()
            cm = cmds_for(k)
            res = {}
            for step in STEPS:
                if step == "detection" and args.skip_detection:
                    res[step] = "ignore (--skip_detection)"
                    continue
                if step == "energy" and args.skip_energy:
                    res[step] = "ignore (--skip_energy)"
                    continue
                outs = outputs_for(k, step)
                sstat = st.step(step)
                if args.resume and sstat.get("state") == "done" and all(Path(o).exists() for o in outs):
                    log(f"{tag} etape {step} : deja terminee, saut (--resume)")
                    res[step] = "deja fait"
                    continue
                deps = {"decode": ["encode"], "evaluate": ["encode", "decode"], "detection": ["decode"]}.get(step, [])
                if any(res.get(d_, "") not in ("ok", "deja fait") for d_ in deps):
                    log(f"{tag} etape {step} : IGNOREE (dependance en echec : {deps})")
                    res[step] = "ignore (dependance)"
                    st.set(step, state="skipped_dependency")
                    continue
                move_targets = list(outs)
                if step == "decode":
                    move_targets.append(P["recon"])
                if step == "detection":
                    move_targets.append(P["det"])
                quarantine(move_targets, P["dir"])
                if step == "energy" and args.cooldown > 0:
                    log(f"{tag} pause de {args.cooldown:.0f} s avant la mesure d'energie (refroidissement). Fermez les autres applications.")
                    time.sleep(args.cooldown)
                t0 = time.perf_counter()
                st.set(step, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    if step == "detection":
                        cfg = {"original_dir": str(yolo_orig), "reconstructed_dir": str(P["recon"]), "labels_dir": str(yolo_lab),
                               "output_dir": str(P["det"]), "threads": args.threads}
                        # controle : le decodeur a bien ecrit exactement n images
                        nrec = len(list(P["recon"].glob("*.png"))) if P["recon"].exists() else 0
                        if nrec != n:
                            raise RuntimeError(f"{nrec} images reconstruites trouvees, {n} attendues")
                        cfgp = P["dir"] / "detection_worker_config.json"
                        cfgp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                        cmd = [py, str(Path(__file__).resolve()), "--_detect_worker", str(cfgp)]
                        rc, dt = run_cmd(cmd, P["logs"] / "detection.log", f"base_{k}/detection")
                    else:
                        rc, dt = run_cmd(cm[step], P["logs"] / f"{step}.log", f"base_{k}/{step}")
                    missing = [str(o) for o in outs if not Path(o).exists()]
                    if rc != 0:
                        raise RuntimeError(f"code de retour {rc}")
                    if missing:
                        raise RuntimeError(f"sorties attendues absentes : {missing}")
                    st.set(step, state="done", seconds=dt, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                           cmd=subprocess.list2cmdline(cm[step]) if step in cm else "detect_worker",
                           outputs=[str(o) for o in outs])
                    res[step] = "ok"
                except Exception as e:                                    # on continue avec l'etape/checkpoint suivant
                    log(f"{tag} ERREUR etape {step} : {e}")
                    log(traceback.format_exc(), console=False)
                    st.set(step, state="failed", error=str(e), seconds=time.perf_counter() - t0)
                    res[step] = f"ECHEC ({e})"
            summary[k] = res
            log(f"{tag} termine : {res}")
            if idx_k < len(args.checkpoints) and args.cooldown > 0:
                time.sleep(args.cooldown)

    # ------------------------------------------------------------ agregation
    log("-" * 78)
    log("agregation des resultats reels dans " + str(comp))
    rows, stage_rows, extras = [], [], {}
    for k in args.checkpoints:
        P = paths_for(k)
        if not P["dir"].exists():
            log(f"checkpoint {k} : aucun resultat ({P['dir']})")
            continue
        try:
            row, srows, extra = collect_checkpoint(k, P["dir"], ckpts[k], Status(P["dir"] / "status.json"), args)
            rows.append(row)
            stage_rows += srows
            extras[k] = extra
        except Exception as e:
            log(f"checkpoint {k} : agregation impossible ({e})")
            log(traceback.format_exc(), console=False)
    if not rows:
        log("rien a agreger.")
        return 1
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    comp.mkdir(parents=True, exist_ok=True)
    with open(comp / "all_checkpoints.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("NaN" if isinstance(r.get(c), float) and not math.isfinite(r[c]) else r.get(c, "NaN")) for c in cols})
    if stage_rows:
        with open(comp / "stage_breakdown.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
            w.writeheader()
            w.writerows(stage_rows)
    shas = {r["checkpoint"]: r.get("checkpoint_sha256") for r in rows}
    distinct = len({v for v in shas.values() if v})
    meta = {"run_name": run_name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:],
            "machine_label": args.machine_label, "platform": platform.platform(), "python": sys.version.split()[0],
            "versions": {p: pkg_version(p) for p in ("torch", "torchvision", "numpy", "codecarbon", "constriction", "matplotlib")},
            "params": {k_: v for k_, v in vars(args).items()},
            "checkpoint_sha256": shas,
            "checkpoints_are_distinct_files": (distinct == len([v for v in shas.values() if v])) if distinct else "non verifie",
            "warning_checkpoints": "Les 4 checkpoints ne doivent pas etre presentes comme 4 niveaux de qualite distincts tant que leurs "
                                   "configurations/parametres d'entrainement ne sont pas verifies (README original = indication seulement).",
            "warning_energy": "Energie = CPU (mesuree si windows_emi/intel_rapl) + RAM (toujours modelisee), machine entiere. "
                              "Le champ energy_kind du script energy_profiling.py classe a tort 'windows_emi' comme modelise ; "
                              "energy_kind_corrected le remplace ici.",
            "column_units": COLUMN_UNITS}
    (comp / "all_checkpoints.json").write_text(json.dumps(jsonable({"meta": meta, "rows": rows, "extra": extras}), indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    log(f"ecrit : all_checkpoints.csv / .json ({len(rows)} checkpoints)")
    kind = next((r.get("energy_kind_corrected") for r in rows if r.get("energy_kind_corrected")), "energie non mesuree")
    foot = (f"{args.machine_label} | N={n} images, GOP={args.gop} | énergie: {kind} (CodeCarbon {pkg_version('codecarbon')}) | "
            f"{len(rows)}/4 checkpoints | brute = CPU+RAM bloc entier, nette = brute - repos")
    make_figures(rows, stage_rows, comp / "figures", args, foot)

    log("RESUME DES ETAPES : " + json.dumps(jsonable(summary), ensure_ascii=False))
    failed = [k for k, v in summary.items() for s, x in v.items() if str(x).startswith(("ECHEC", "parametres"))]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())