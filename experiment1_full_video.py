#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment1_full_video.py  --  VCM-EnergyMap

Exploite des resultats DEJA existants (3 repetitions sur les memes 145 images)
et genere des figures scientifiques. AUCUNE mesure n'est relancee, AUCUN
fichier de resultats n'est modifie : le script LIT uniquement .csv / .json et
n'ecrit que dans le dossier de sortie (--out).

Principes :
  * inspection obligatoire en premier (rapport imprime + sauve) ;
  * courbes de puissance UNIQUEMENT si une vraie serie temporelle existe ;
  * aucune interpolation, aucune valeur inventee : metrique absente =
    "NON DISPONIBLE" et figure correspondante ignoree ;
  * chaque valeur utilisee est tracee jusqu'a son fichier/cle source.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
PHASES = ["encode", "decode", "yolo_orig", "yolo_recon"]
PHASE_LABEL = {
    "encode": "Encodage VCM",
    "decode": "Décodage VCM",
    "yolo_orig": "YOLO (images originales)",
    "yolo_recon": "YOLO (images reconstruites)",
}
DET = ["precision", "recall", "map50", "map50_95"]
DET_LABEL = {"precision": "Precision", "recall": "Recall",
             "map50": "mAP50", "map50_95": "mAP50-95"}
REP_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
PHASE_COLORS = {"encode": "#4c72b0", "decode": "#55a868",
                "yolo_orig": "#c44e52", "yolo_recon": "#8172b2"}

RX_REP_DIR = re.compile(
    r"(?:rep(?:etition)?|run|trial|iter(?:ation)?)[_\-\s]?0*([1-3])(?!\d)|^r0*([1-3])$",
    re.I)
RX_ORIG = re.compile(r"orig|raw|source|baseline|uncompressed", re.I)
RX_RECON = re.compile(r"recon|decoded|(?<!un)compressed", re.I)
RX_ENERGY = re.compile(r"energy|energie|joule|kwh|(?<![a-z])wh(?![a-z])|(?<![a-z])j(?![a-z])", re.I)
RX_TIME = re.compile(r"time|duration|elapsed|latency|temps|seconds|(?<![a-z])s$", re.I)
RX_NOTTIME = re.compile(r"timestamp|start|end|date|fps", re.I)
RX_PSNR = re.compile(r"psnr", re.I)
RX_MSSSIM = re.compile(r"ms[-_ ]?ssim", re.I)
RX_BITRATE = re.compile(r"bitrate|bit[-_ ]rate|kbps|mbps", re.I)
RX_MAP5095 = re.compile(r"map[-_ ]?50[-_ ]?95|map[-_ ]?0?\.5[-_:]0?\.95|(^|[._/ ])map(\(b\))?$", re.I)
RX_MAP50 = re.compile(r"map[-_ ]?50(?![-_ ]?\d)", re.I)
RX_PRECISION = re.compile(r"precision", re.I)
RX_RECALL = re.compile(r"recall", re.I)
RX_PENALTY = re.compile(r"per[-_ ]?(frame|image|img)|mean|avg|average|cpu|gpu|ram|idle|std", re.I)

RX_TCOL = re.compile(r"time|elapsed|^t$|^ts$|^sec(onds?)?$|^date", re.I)
RX_PCOL = re.compile(r"power|watt|puissance|(^|[_\W])w$", re.I)
RX_AGG = re.compile(r"avg|mean|average|moy|max|min|peak|total|energy|limit", re.I)
RX_LABEL_COL = re.compile(r"^(phase|stage|step|task|name|label|variant|mode|component|stage_name)$", re.I)


# --------------------------------------------------------------------------
# Lecture robuste (lecture seule)
# --------------------------------------------------------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    df = None
    for kw in ({"sep": None, "engine": "python"}, {"sep": ";"}, {}):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", **kw)
            if df.shape[1] >= 1:
                break
        except Exception:
            df = None
    if df is None:
        raise ValueError("CSV illisible")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            if len(s) and s.str.fullmatch(r"-?\d+,\d+").all():  # virgule decimale FR
                df[c] = pd.to_numeric(s.str.replace(",", "."))
    return df


def flatten_json(obj, prefix=""):
    """Retourne (scalaires {cle: valeur}, [(nom, DataFrame)])."""
    rec, frames = {}, []
    if isinstance(obj, dict):
        num_lists = defaultdict(dict)
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                r, f = flatten_json(v, key)
                rec.update(r)
                frames += f
            elif isinstance(v, list):
                if v and all(isinstance(x, dict) for x in v):
                    frames.append((key, pd.json_normalize(v)))
                elif v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                    num_lists[len(v)][str(k)] = v
            else:
                rec[key] = v
        for n, cols in num_lists.items():
            if n >= 3 and len(cols) >= 2:
                frames.append((prefix or "root", pd.DataFrame(cols)))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        frames.append((prefix or "root", pd.json_normalize(obj)))
    return rec, frames


def to_seconds(s: pd.Series, name: str):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.9:
        sec = num.astype(float)
        if re.search(r"(^|[_\W])ms$|millis", name, re.I):
            sec = sec / 1000.0
        return sec
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().mean() >= 0.9:
        return (dt - dt.min()).dt.total_seconds()
    return None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify(s: str, det: bool = False):
    """Retourne la phase d'une chaine (cle ou chemin) ou None."""
    s = s.lower()
    if "yolo" in s or det:
        o, r = bool(RX_ORIG.search(s)), bool(RX_RECON.search(s))
        if o and not r:
            return "yolo_orig"
        if r and not o:
            return "yolo_recon"
        return None
    e, d = "encod" in s, "decod" in s
    if e and not d:
        return "encode"
    if d and not e:
        return "decode"
    return None


def quantity(key_l: str):
    if RX_ENERGY.search(key_l) and "power" not in key_l.replace("energy", ""):
        return "energy"
    if RX_ENERGY.search(key_l):
        return "energy" if "energy" in key_l or "joule" in key_l else None
    if RX_TIME.search(key_l) and not RX_NOTTIME.search(key_l):
        return "time"
    return None


def auto_metric(path_l: str, key_l: str):
    if RX_MSSSIM.search(key_l):
        return "ms_ssim"
    if RX_PSNR.search(key_l):
        return "psnr"
    if RX_BITRATE.search(key_l):
        return "bitrate"
    det = None
    if RX_MAP5095.search(key_l):
        det = "map50_95"
    elif RX_MAP50.search(key_l):
        det = "map50"
    elif RX_PRECISION.search(key_l):
        det = "precision"
    elif RX_RECALL.search(key_l):
        det = "recall"
    if det:
        ph = classify(key_l, det=True) or classify(path_l, det=True)
        return f"{ph}_{det}" if ph else None
    q = quantity(key_l)
    if q:
        ph = classify(key_l) or classify(path_l)
        if ph:
            return f"{ph}_{q}"
        if q == "energy" and "total" in key_l:
            return "total_energy"
    return None


def convert_value(metric: str, key_l: str, val: float, default_energy_unit: str):
    """Conversion d'unites SEULEMENT si l'unite figure dans le nom de la cle."""
    note = ""
    if metric.endswith("_energy") or metric == "total_energy":
        if "kwh" in key_l:
            return val * 3.6e6, "kWh->J"
        if re.search(r"(?<![a-z])wh(?![a-z])", key_l):
            return val * 3600.0, "Wh->J"
        if re.search(r"(?<![a-z])mj(?![a-z])", key_l):
            return val / 1000.0, "mJ->J"
        if re.search(r"joule|(?<![a-z])j(?![a-z])", key_l):
            return val, "J"
        f = {"J": 1.0, "Wh": 3600.0, "kWh": 3.6e6}[default_energy_unit]
        return val * f, f"unite NON precisee dans la cle, supposee {default_energy_unit}"
    if metric.endswith("_time"):
        if re.search(r"(?<![a-z])ms(?![a-z])|millisec", key_l):
            return val / 1000.0, "ms->s"
        return val, "s (suppose)"
    return val, note


# --------------------------------------------------------------------------
# Scan d'une repetition
# --------------------------------------------------------------------------
def numeric_scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def extract_series(df: pd.DataFrame, rel: str):
    if len(df) < 3:
        return []
    tcols = [c for c in df.columns if RX_TCOL.search(str(c))]
    pcols = [c for c in df.columns if RX_PCOL.search(str(c)) and not RX_AGG.search(str(c))]
    if not tcols or not pcols:
        return []
    tcol = tcols[0]
    t = to_seconds(df[tcol], str(tcol))
    if t is None:
        return []
    out = []
    for pc in pcols:
        p = pd.to_numeric(df[pc], errors="coerce")
        m = t.notna() & p.notna()
        if m.sum() < 3:
            continue
        tt, pp = t[m].to_numpy(float), p[m].to_numpy(float)
        order = np.argsort(tt, kind="stable")
        tt, pp = tt[order], pp[order]
        if tt.max() - tt.min() <= 0:
            continue
        phase = classify(str(pc)) or classify(rel)
        out.append(dict(rel=rel, col=str(pc), tcol=str(tcol), t=tt - tt.min(), p=pp,
                        n=int(m.sum()), phase=phase or "global"))
    return out


def handle_frame(df, rel, prefix, entry, records, series):
    found = extract_series(df, rel)
    if found:
        series += found
        entry["series"] += len(found)
        entry["notes"].append(f"serie temporelle de puissance ({found[0]['n']} points, "
                              f"colonnes: {[s['col'] for s in found]})")
        return
    df = df.dropna(how="all")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(df) == 1:
        for c in df.columns:
            v = numeric_scalar(df.iloc[0][c])
            if v is not None:
                records.append((rel, f"{prefix}{c}", v))
        entry["notes"].append("resume (1 ligne) : energie/puissance moyenne, pas de serie")
        return
    lab = next((c for c in df.columns if RX_LABEL_COL.match(str(c))
                and not pd.api.types.is_numeric_dtype(df[c])), None)
    if lab is not None and df[lab].is_unique and len(df) <= 30:
        for _, row in df.iterrows():
            for c in num_cols:
                v = numeric_scalar(row[c])
                if v is not None:
                    records.append((rel, f"{prefix}{row[lab]}.{c}", v))
        entry["notes"].append(f"tableau 'long' une ligne par '{lab}' ({len(df)} lignes)")
        return
    agg = [c for c in num_cols if RX_PSNR.search(str(c)) or RX_MSSSIM.search(str(c))]
    for c in agg:
        v = numeric_scalar(df[c].mean())
        if v is not None:
            records.append((rel, f"{prefix}{c}[mean_of_{len(df)}_rows]", v))
    entry["notes"].append(f"tableau multi-lignes ({len(df)} lignes) sans serie de puissance"
                          + (f"; moyenne calculee pour {agg}" if agg else "; non agrege"))


def scan_rep(dirs, root: Path, out_dir: Path):
    info = dict(files=[], records=[], series=[], dirs=[str(d) for d in dirs])
    for d in dirs:
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in (".csv", ".json"):
                continue
            if out_dir in f.resolve().parents:
                continue
            rel = f.relative_to(root).as_posix()
            entry = dict(rel=rel, size=f.stat().st_size, series=0, notes=[], cols=[])
            info["files"].append(entry)
            if entry["size"] > 300 * 1024 * 1024:
                entry["notes"].append("fichier > 300 Mo ignore")
                continue
            try:
                if f.suffix.lower() == ".csv":
                    df = read_csv_robust(f)
                    entry["cols"] = [str(c) for c in df.columns][:14]
                    entry["rows"] = len(df)
                    handle_frame(df, rel, "", entry, info["records"], info["series"])
                else:
                    with open(f, "r", encoding="utf-8-sig") as fh:
                        obj = json.load(fh)
                    rec, frames = flatten_json(obj)
                    entry["cols"] = list(rec.keys())[:14]
                    for k, v in rec.items():
                        nv = numeric_scalar(v)
                        if nv is not None:
                            info["records"].append((rel, k, nv))
                    if rec:
                        entry["notes"].append(f"{len(rec)} valeurs scalaires")
                    for name, df in frames:
                        handle_frame(df, rel, f"{name}.", entry, info["records"], info["series"])
            except Exception as e:  # lecture seule : on signale et on continue
                entry["notes"].append(f"ERREUR de lecture: {type(e).__name__}: {e}")
    return info


def find_repetitions(root: Path, out_dir: Path):
    found = defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_dir() or out_dir == p.resolve() or out_dir in p.resolve().parents:
            continue
        m = RX_REP_DIR.search(p.name)
        if m:
            n = int(m.group(1) or m.group(2))
            found[n].append(p)
    pruned = {}
    for n, lst in found.items():
        pruned[n] = sorted([p for p in lst if not any(q != p and q in p.parents for q in lst)])
    return pruned


# --------------------------------------------------------------------------
# Extraction des metriques
# --------------------------------------------------------------------------
def extract_metrics(info, overrides, energy_unit):
    cands = defaultdict(list)
    for i, (path, key, val) in enumerate(info["records"]):
        pl, kl = path.lower(), key.lower()
        full = f"{pl}::{kl}"
        hits = set()
        for m, rx in overrides.items():
            if re.search(rx, full, re.I):
                hits.add(m)
        am = auto_metric(pl, kl)
        if am and am not in overrides:
            hits.add(am)
        for m in hits:
            v, note = convert_value(m, kl, val, energy_unit)
            pen = bool(RX_PENALTY.search(kl))
            tot = "total" in kl
            cands[m].append(((pen, not tot, i), v, f"{path} :: {key}", note))
    chosen, warns = {}, []
    for m, lst in cands.items():
        lst.sort(key=lambda x: x[0])
        best = lst[0]
        chosen[m] = dict(value=best[1], source=best[2], note=best[3])
        distinct = {round(c[1], 9) for c in lst}
        if len(distinct) > 1:
            warns.append(f"[AMBIGU] {m}: {len(lst)} candidats, retenu '{best[2]}'. "
                         f"Autres: " + "; ".join(f"{c[2]}={c[1]:.6g}" for c in lst[1:4]))
    return chosen, warns


def stats_of(values):
    v = np.array([x for x in values if x is not None], float)
    n = len(v)
    mean = float(v.mean()) if n else np.nan
    std = float(v.std(ddof=1)) if n > 1 else np.nan
    cv = std / mean * 100 if n > 1 and mean != 0 else np.nan
    return n, mean, std, cv


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def save(fig, out_dir, name, dpi):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    return name


def plot_power_curves(rep, series, out_dir, dpi):
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.9 * n + 0.8), squeeze=False)
    for ax, s in zip(axes[:, 0], series):
        ax.plot(s["t"], s["p"], lw=0.9, color=REP_COLORS.get(rep, "k"))
        ax.set_xlabel(f"Temps relatif (s) — colonne « {s['tcol']} »")
        ax.set_ylabel(f"Puissance — {s['col']}")
        ax.set_title(f"{PHASE_LABEL.get(s['phase'], 'Série globale')} · {s['rel']} · {s['n']} points",
                     fontsize=9)
        ax.grid(alpha=.3)
    fig.suptitle(f"Répétition {rep} — évolution de la puissance (données brutes)", fontweight="bold")
    return save(fig, out_dir, f"fig1_power_curve_rep{rep}", dpi)


def bar_ms(ax, x, vals, color):
    v = np.array([a for a in vals if a is not None], float)
    if not len(v):
        return
    m = v.mean()
    s = v.std(ddof=1) if len(v) > 1 else 0
    ax.bar(x, m, yerr=s if len(v) > 1 else None, capsize=5, color=color, alpha=.75,
           edgecolor="k", lw=.6)
    ax.scatter([x] * len(v), v, color="k", s=16, zorder=3)


def fig_total(M, reps, out_dir, dpi):
    vals = {r: M.get("total_energy", {}).get(r) for r in reps}
    vals = {r: v for r, v in vals.items() if v is not None}
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (r, v) in enumerate(sorted(vals.items())):
        ax.bar(i, v, color=REP_COLORS[r], alpha=.85, edgecolor="k", lw=.6)
        ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    n, mean, std, cv = stats_of(list(vals.values()))
    ax.axhline(mean, color="k", ls="--", lw=1)
    if n > 1:
        ax.axhspan(mean - std, mean + std, color="gray", alpha=.18)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([f"Rép. {r}" for r in sorted(vals)])
    ax.set_ylabel("Énergie totale (J)")
    ttl = f"Énergie totale par répétition (n={n})"
    if n > 1:
        ttl += f"\nmoyenne={mean:.4g} J · σ={std:.3g} · CV={cv:.2f} %"
    ax.set_title(ttl, fontsize=10)
    ax.grid(axis="y", alpha=.3)
    return save(fig, out_dir, "fig2a_total_energy_comparison", dpi)


def fig_phases(M, reps, out_dir, dpi):
    avail = [p for p in PHASES if any(M.get(f"{p}_energy", {}).get(r) is not None for r in reps)]
    if not avail:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.8 / len(reps)
    ax = axes[0]
    for j, r in enumerate(reps):
        ys = [M.get(f"{p}_energy", {}).get(r, np.nan) for p in avail]
        ax.bar(np.arange(len(avail)) + j * w - 0.4 + w / 2, ys, w, label=f"Rép. {r}",
               color=REP_COLORS[r], edgecolor="k", lw=.5)
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels([PHASE_LABEL[p].replace(" (", "\n(") for p in avail], fontsize=8)
    ax.set_ylabel("Énergie (J)")
    ax.set_title("Énergie par phase et par répétition")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    ax = axes[1]
    cvs, labels = [], []
    for p in avail:
        n, mean, std, cv = stats_of([M[f"{p}_energy"].get(r) for r in reps])
        cvs.append(cv)
        labels.append(PHASE_LABEL[p].replace(" (", "\n("))
    if any(M.get("total_energy", {}).get(r) is not None for r in reps):
        cvs.append(stats_of([M["total_energy"].get(r) for r in reps])[3])
        labels.append("Total")
    cv_plot = [0 if np.isnan(c) else c for c in cvs]
    ax.bar(range(len(cv_plot)), cv_plot, color="#999", edgecolor="k", lw=.5)
    for i, c in enumerate(cvs):
        ax.text(i, cv_plot[i], "n/a" if np.isnan(c) else f"{c:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Coefficient de variation (%)")
    ax.set_title("Variabilité inter-répétitions (CV = σ/μ, σ échantillon n-1)")
    ax.grid(axis="y", alpha=.3)
    return save(fig, out_dir, "fig2b_phase_energy_comparison", dpi)


def fig_time_energy(M, reps, out_dir, dpi):
    made = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, q, yl in ((axes[0], "time", "Temps (s)"), (axes[1], "energy", "Énergie (J)")):
        ps = [p for p in PHASES if any(M.get(f"{p}_{q}", {}).get(r) is not None for r in reps)]
        for i, p in enumerate(ps):
            bar_ms(ax, i, [M[f"{p}_{q}"].get(r) for r in reps], PHASE_COLORS[p])
            made = True
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([PHASE_LABEL[p].replace(" (", "\n(") for p in ps], fontsize=8)
        ax.set_ylabel(yl)
        ax.set_title(f"{yl.split(' ')[0]} par phase (moyenne ± σ, points = répétitions)")
        ax.grid(axis="y", alpha=.3)
    if not made:
        plt.close(fig)
        return None
    return save(fig, out_dir, "fig3a_time_energy_by_phase", dpi)


def fig_quality(M, reps, out_dir, dpi):
    items = [("bitrate", "Bitrate"), ("psnr", "PSNR (dB)"), ("ms_ssim", "MS-SSIM")]
    items = [(k, l) for k, l in items if any(M.get(k, {}).get(r) is not None for r in reps)]
    if not items:
        return None
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 4.2), squeeze=False)
    for ax, (k, l) in zip(axes[0], items):
        bar_ms(ax, 0, [M[k].get(r) for r in reps], "#4c72b0")
        ax.set_xticks([])
        ax.set_title(l)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Qualité / débit (moyenne ± σ sur les répétitions)", fontweight="bold")
    return save(fig, out_dir, "fig3b_bitrate_psnr_msssim", dpi)


def fig_detection(M, reps, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    made = False
    for j, (ph, col) in enumerate((("yolo_orig", "#c44e52"), ("yolo_recon", "#8172b2"))):
        for i, d in enumerate(DET):
            vals = [M.get(f"{ph}_{d}", {}).get(r) for r in reps]
            if any(v is not None for v in vals):
                bar_ms_offset(ax, i + (j - .5) * 0.38, vals, col, 0.36)
                made = True
    if not made:
        plt.close(fig)
        return None
    ax.set_xticks(range(len(DET)))
    ax.set_xticklabels([DET_LABEL[d] for d in DET])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Détection YOLO : originales vs reconstruites (moyenne ± σ)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c44e52", label=PHASE_LABEL["yolo_orig"]),
                       Patch(color="#8172b2", label=PHASE_LABEL["yolo_recon"])])
    ax.grid(axis="y", alpha=.3)
    return save(fig, out_dir, "fig3c_detection_metrics", dpi)


def bar_ms_offset(ax, x, vals, color, width):
    v = np.array([a for a in vals if a is not None], float)
    m = v.mean()
    s = v.std(ddof=1) if len(v) > 1 else 0
    ax.bar(x, m, width, yerr=s if len(v) > 1 else None, capsize=4, color=color, alpha=.8,
           edgecolor="k", lw=.5)
    ax.scatter([x] * len(v), v, color="k", s=12, zorder=3)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VCM-EnergyMap : figures de l'experience 1 (lecture seule).")
    ap.add_argument("--root", required=True, help="Dossier racine contenant les 3 repetitions.")
    ap.add_argument("--rep1"); ap.add_argument("--rep2"); ap.add_argument("--rep3")
    ap.add_argument("--out", default="experiment1_figures", help="Dossier de sortie (nouveau).")
    ap.add_argument("--mapping", help="JSON optionnel {metrique: regex} pour forcer l'association.")
    ap.add_argument("--energy-unit", default="J", choices=["J", "Wh", "kWh"],
                    help="Unite supposee si elle n'est pas dans le nom de la cle (defaut J).")
    ap.add_argument("--inspect-only", action="store_true", help="Inspection seule, aucune figure.")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    out_dir = Path(a.out).resolve()
    if not root.is_dir():
        sys.exit(f"Racine introuvable: {root}")

    explicit = {1: a.rep1, 2: a.rep2, 3: a.rep3}
    reps_dirs = {}
    auto = find_repetitions(root, out_dir)
    for n in (1, 2, 3):
        if explicit[n]:
            reps_dirs[n] = [Path(explicit[n]).resolve()]
        elif n in auto:
            reps_dirs[n] = auto[n]
    for n, ds in reps_dirs.items():
        for d in ds:
            if d == out_dir or d in out_dir.parents or out_dir in d.parents:
                sys.exit("Le dossier de sortie ne doit pas etre dans/avec un dossier de resultats.")
    if not reps_dirs:
        sys.exit("Aucun dossier de repetition detecte. Utilisez --rep1 --rep2 --rep3.")

    L = []
    def log(s=""):
        print(s)
        L.append(s)

    log("=" * 78)
    log("1) INSPECTION DES DONNEES (lecture seule)")
    log("=" * 78)
    log(f"Racine : {root}")
    infos = {}
    for n in sorted(reps_dirs):
        infos[n] = scan_rep(reps_dirs[n], root, out_dir)
    for n in (1, 2, 3):
        if n not in infos:
            log(f"\n[Répétition {n}] AUCUN DOSSIER TROUVE (utilisez --rep{n})")
            continue
        inf = infos[n]
        log(f"\n[Répétition {n}] dossiers: {', '.join(inf['dirs'])}")
        log(f"  {len(inf['files'])} fichier(s) CSV/JSON ; "
            f"{len(inf['series'])} serie(s) temporelle(s) de puissance ; "
            f"{len(inf['records'])} valeur(s) scalaire(s) numerique(s)")
        for e in inf["files"]:
            log(f"   - {e['rel']} ({e['size'] / 1024:.1f} Ko"
                + (f", {e['rows']} lignes" if 'rows' in e else "") + ")")
            if e["cols"]:
                log(f"       cles/colonnes: {e['cols']}")
            for nt in e["notes"]:
                log(f"       -> {nt}")

    log("\n--- Verdict séries temporelles ---")
    with_series = [n for n in sorted(infos) if infos[n]["series"]]
    if with_series:
        log(f"Series temporelles reelles de puissance trouvees pour la/les repetition(s): {with_series}")
        missing = [n for n in sorted(infos) if n not in with_series]
        if missing:
            log(f"PAS de serie temporelle pour: {missing} -> aucune courbe produite pour elles "
                f"(rien n'est interpole ni reconstruit).")
    else:
        log("AUCUNE serie temporelle de puissance trouvee (pas de colonne temps + colonne puissance "
            "avec >= 3 echantillons). Seuls des resumes (energie totale / puissance moyenne) "
            "sont disponibles : les courbes puissance=f(t) ne peuvent PAS etre tracees sans "
            "inventer des donnees. Des figures comparatives sont produites a la place.")

    if a.inspect_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "inspection_report.txt").write_text("\n".join(L), encoding="utf-8")
        print(f"\nRapport: {out_dir / 'inspection_report.txt'}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {}
    if a.mapping:
        overrides = json.loads(Path(a.mapping).read_text(encoding="utf-8"))

    # ---- metriques ----
    reps = sorted(infos)
    M = defaultdict(dict)
    SRC = defaultdict(dict)
    all_warns = []
    for n in reps:
        chosen, warns = extract_metrics(infos[n], overrides, a.energy_unit)
        all_warns += [f"[Rép {n}] {w}" for w in warns]
        for m, d in chosen.items():
            M[m][n] = d["value"]
            SRC[m][n] = d["source"] + (f" [{d['note']}]" if d["note"] else "")
    # total derive uniquement si les 3 phases de la chaine VCM existent
    for n in reps:
        if n not in M.get("total_energy", {}):
            parts = [M.get(f"{p}_energy", {}).get(n) for p in ("encode", "decode", "yolo_recon")]
            if all(p is not None for p in parts):
                M["total_energy"][n] = float(sum(parts))
                SRC["total_energy"][n] = "DERIVE = encode + decode + yolo_recon (aucune cle 'total' trouvee)"

    wanted = ["total_energy"] + [f"{p}_{q}" for p in PHASES for q in ("time", "energy")] \
        + ["bitrate", "psnr", "ms_ssim"] + [f"{p}_{d}" for p in ("yolo_orig", "yolo_recon") for d in DET]
    rows = []
    for m in wanted:
        vals = [M.get(m, {}).get(r) for r in reps]
        n, mean, std, cv = stats_of(vals)
        row = dict(metric=m)
        for r in reps:
            row[f"rep{r}"] = M.get(m, {}).get(r, np.nan)
        row.update(n=n, mean=mean, std_ddof1=std, cv_percent=cv,
                   sources=" | ".join(f"rep{r}: {SRC[m][r]}" for r in reps if r in SRC.get(m, {})))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics_stats_summary.csv", index=False, encoding="utf-8-sig")

    log("\n" + "=" * 78)
    log("2) METRIQUES EXTRAITES (valeur reelle, source tracee)")
    log("=" * 78)
    for _, r in table.iterrows():
        if r["n"] == 0:
            log(f"  {r['metric']:<22} NON DISPONIBLE")
        else:
            sd = "n/a" if np.isnan(r["std_ddof1"]) else f"{r['std_ddof1']:.4g}"
            cv = "n/a" if np.isnan(r["cv_percent"]) else f"{r['cv_percent']:.2f}%"
            log(f"  {r['metric']:<22} n={int(r['n'])} moy={r['mean']:.5g} sd={sd} CV={cv}")
    for w in all_warns:
        log(w)
    notes = {s.split('[')[-1] for m in SRC for s in SRC[m].values() if "supposee" in s}
    if notes:
        log("\nATTENTION: unite d'energie NON precisee dans certaines cles, supposee "+a.energy_unit+" (voir colonne sources du CSV ; corrigez avec --energy-unit).")
    log("Les temps sans unite dans la cle sont supposes en secondes.")

    # ---- figures ----
    log("\n" + "=" * 78)
    log("3) FIGURES")
    log("=" * 78)
    made = []
    for n in reps:
        if infos[n]["series"]:
            made.append(plot_power_curves(n, infos[n]["series"], out_dir, a.dpi))
        else:
            log(f"  fig1 rep{n}: IGNOREE (pas de serie temporelle reelle)")
    for fn, label in ((fig_total, "fig2a"), (fig_phases, "fig2b"), (fig_time_energy, "fig3a"),
                      (fig_quality, "fig3b"), (fig_detection, "fig3c")):
        r = fn(M, reps, out_dir, a.dpi)
        if r:
            made.append(r)
        else:
            log(f"  {label}: IGNOREE (donnees non disponibles)")
    for m in made:
        log(f"  OK  {m}.png / .pdf")
    log(f"\nSortie: {out_dir}")
    (out_dir / "report_experiment1.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
