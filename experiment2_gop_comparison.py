#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
experiment2_gop_comparison.py -- Experience 2 VCM-EnergyMap : influence du GOP (16 / 32 / 64) sur le Base Layer.

Un SEUL parametre change entre les configurations : --gop. Memes images, meme checkpoint, meme YOLOv5s, memes threads.

NE MODIFIE AUCUN script existant. Il appelle (sous-processus, sequentiellement, jamais en parallele) :
  test_base_profiled.py  encode / decode / evaluate / energy
  compare_detection_145.py   (via le worker de run_vcm_energymap.py)
  energy_yolo_145.py         (ses fonctions prepare_image / measured_inference sont IMPORTEES ; son main n'est pas lance)
et REUTILISE les fonctions de run_vcm_energymap.py (journal, reprise, quarantaine, lecture des JSON/CSV, calcul des
energies brute/nette). Les deux fichiers doivent donc rester dans le meme dossier (svc_code\).

SEMANTIQUE DU GOP (verifiee dans base_codec.py et test_base_profiled.py) :
  position i = 0,1,2,... dans la sequence ; I-frame si i % GOP == 0, sinon P-frame. Avec --start_index 1 :
  GOP=32 -> I aux fichiers frame_001, 033, 065, 097, 129 (5 I + 140 P sur 145 images).
  GOP=16 -> 10 I ; GOP=64 -> 3 I (001, 065, 129). Le GOP est ecrit dans l'en-tete .vcmb et relu par le decodeur.
  Le tampon d'images est remis a zero a chaque I-frame (_p_order=1) ; le tampon de flots MWNet n'est PAS remis a zero
  (comportement de l'original conserve, --closed_gop n'est PAS utilise) : un GOP n'est donc pas decodable isolement.

ORDRE D'EXECUTION (3 phases, pour regrouper les mesures de meme nature) :
  phase 1 (par GOP)  : encode -> decode -> evaluate -> structure (verifications) -> detection
  phase 2 (par GOP)  : energie du codec (blocs encode / decode sur TOUTES les images ; pas de --energy_frames ici)
  phase 3 (par GOP)  : temps + energie YOLO, images originales PUIS reconstruites (processus et compteur SEPARES du codec)
  puis agregation : comparison\gop_comparison.csv/.json, stage_breakdown.csv, experiment_log.txt, figures\

PILOTE 65 IMAGES (GOP 16/32/64 -> 5 / 3 / 2 I-frames), resultats dans results\experiment2_gop_pilot65\ :
  python experiment2_gop_comparison.py --pilot --dry_run
  python experiment2_gop_comparison.py --pilot
EXECUTION COMPLETE 145 IMAGES, resultats dans results\experiment2_gop\ :
  python experiment2_gop_comparison.py --dry_run
  python experiment2_gop_comparison.py
REPRISE : ajouter --resume. Rien n'est supprime ni ecrase ; les fichiers partiels d'une etape non terminee sont
DEPLACES dans gop_N\_partial_<horodatage>\.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NAN = float("nan")
REQUIRED = ["test_base_profiled.py", "energy_profiling.py", "profiling.py", "base_codec.py", "bitstream.py",
            "compare_detection_145.py", "energy_yolo_145.py", "run_vcm_energymap.py"]
RV_NAMES = ["run_cmd", "Status", "quarantine", "collect_checkpoint", "block_stats", "cpu_kind", "read_json", "read_csv",
            "jsonable", "isnum", "fnum", "mean_std", "sha256_file", "pkg_version", "log"]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
C_ENC, C_DEC, C_REC, C_ORIG, C_I = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#eb6834"


# ======================================================================================
# worker YOLO (temps + energie), processus separe, SANS modifier energy_yolo_145.py
# ======================================================================================
def yolo_energy_worker(cfg_path):
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    import energy_yolo_145 as ey       # NB : son import cree (sans rien effacer) energy_results\yolo_145 s'il manque
    from energy_profiling import EnergyMeter, dump
    need = ["prepare_image", "measured_inference", "DetectMultiBackend", "WEIGHTS", "DEVICE"]
    miss = [x for x in need if not hasattr(ey, x)]
    if miss:
        raise RuntimeError(f"energy_yolo_145.py ne fournit pas {miss} (version differente ?)")
    n = int(cfg["n"])
    model = ey.DetectMultiBackend(weights=str(ey.WEIGHTS), device=ey.DEVICE, dnn=False, fp16=False)
    model.eval()
    meter = EnergyMeter(country_iso_code=cfg["country"], measure_power_secs=float(cfg["measure_power_secs"]),
                        project_name="VCM-EnergyMap-YOLO")
    try:
        info = meter.method_info()
        print("[yolo-energy] method:", {k: info[k] for k in ("cpu_power_method", "energy_kind")}, flush=True)
        idle = meter.measure_idle(float(cfg["idle_seconds"]))
        out = {"meta": {**info, "images": n, "conf_thres": ey.CONF_THRES, "iou_nms": ey.IOU_NMS_THRES, "classes": ey.CLASSES,
                        "image_size": ey.IMAGE_SIZE, "threads": torch.get_num_threads(), "torch": torch.__version__,
                        "scope": "inference + NMS seulement ; lecture/pretraitement des images HORS bloc mesure ; "
                                 "energie du codec NON incluse (processus separe)"},
               "idle": idle, "blocks": []}
        for label, folder in (("yolo_original", cfg["original_dir"]), ("yolo_reconstructed", cfg["reconstructed_dir"])):
            files = sorted(Path(folder).glob("*.png"))
            if len(files) != n:
                raise RuntimeError(f"{folder}: {len(files)} images, {n} attendues")
            tensors = [ey.prepare_image(p) for p in files]            # hors mesure
            for t in tensors[:3]:                                      # echauffement hors mesure
                ey.measured_inference(model, t)
            ndet = [0]

            def block():
                tot = 0
                for t in tensors:
                    tot += len(ey.measured_inference(model, t))
                ndet[0] = tot
                return len(tensors)

            res = meter.measure_repeated(label, block, idle, min_seconds=float(cfg["min_seconds"]), warmup_iters=0,
                                         repeats=int(cfg["repeats"]))
            res["images"] = len(tensors)
            res["detections_last_pass"] = ndet[0]
            out["blocks"].append(res)
            g = res["aggregate"]
            print(f"[yolo-energy] {label}: {g['time_per_frame_s_mean']:.3f} s/img, gross {g['gross_j_per_frame_mean']:.2f} J/img, "
                  f"net {g['net_j_per_frame_mean']:.2f} J/img", flush=True)
            del tensors
        out["all_tasks"] = meter.tasks
    finally:
        meter.close()
    Path(cfg["output_json"]).parent.mkdir(parents=True, exist_ok=True)
    dump(cfg["output_json"], out)


if "--_yolo_worker" in sys.argv:
    yolo_energy_worker(sys.argv[sys.argv.index("--_yolo_worker") + 1])
    sys.exit(0)

try:
    import run_vcm_energymap as rv
except Exception as _e:
    sys.exit(f"ERREUR : impossible d'importer run_vcm_energymap.py depuis {ROOT} ({_e}). "
             f"Ce script le reutilise : gardez les deux fichiers dans le meme dossier.")
_missing = [x for x in RV_NAMES if not hasattr(rv, x)]
if _missing:
    sys.exit(f"ERREUR : run_vcm_energymap.py ne fournit pas {_missing} (version differente ?)")
log, isnum, fnum = rv.log, rv.isnum, rv.fnum


def expected_i(n, gop):
    return [i for i in range(n) if i % gop == 0]


# ======================================================================================
# verifications de structure : le GOP change-t-il REELLEMENT la position des I-frames ? bitstream valide ?
# ======================================================================================
def verify_structure(P, gop, n, k0, ckn, prefix, inp_dir):
    from bitstream import BitstreamReader          # stdlib seulement
    exp = expected_i(n, gop)
    errs, chk = [], {"gop": gop, "n_frames": n, "expected_I_positions": exp,
                     "expected_I_files": [f"{prefix}{k0 + i:03d}.png" for i in exp]}
    stem = f"base_{ckn}"
    fsize = os.path.getsize(P["bs"])
    rd = BitstreamReader(str(P["bs"]))
    h = rd.header
    chk["header"] = {"width": h.width, "height": h.height, "n_frames": h.n_frames, "gop": h.gop, "closed_gop": h.closed_gop,
                     "ckpt_id": h.ckpt_id, "header_bytes": rd.header_bytes, "shapes": {k: list(v) for k, v in h.shapes.items()}}
    if h.gop != gop:
        errs.append(f"GOP de l'en-tete {h.gop} != {gop}")
    if h.n_frames != n:
        errs.append(f"n_frames de l'en-tete {h.n_frames} != {n}")
    if h.closed_gop:
        errs.append("closed_gop actif (non attendu dans cette experience)")
    sizes = []
    try:
        for _ in range(h.n_frames):
            _, _, nb = rd.next_frame()
            sizes.append(nb)
        tail = rd.f.read(1)
    except Exception as e:
        errs.append(f"lecture des enregistrements impossible : {e}")
        tail = b""
    rd.close()
    if tail:
        errs.append("octets en trop apres le dernier enregistrement")
    if len(sizes) == n and rd.header_bytes + sum(sizes) != fsize:
        errs.append(f"en-tete+enregistrements = {rd.header_bytes + sum(sizes)} octets != taille du fichier {fsize}")
    chk["file_size_bytes"] = fsize
    chk["records_parsed"] = len(sizes)
    try:
        from PIL import Image
        w0, h0 = Image.open(Path(inp_dir) / f"{prefix}{k0:03d}.png").size
        chk["input_size_wh"] = [w0, h0]
        if (h.width, h.height) != (w0, h0):
            errs.append(f"dimensions de l'en-tete {(h.width, h.height)} != images d'entree {(w0, h0)}")
        chk["bpp_file_from_file_size"] = 8 * fsize / (n * w0 * h0)
        outs = sorted(P["recon"].glob("*.png"))
        chk["decoded_png_count"] = len(outs)
        if len(outs) != n:
            errs.append(f"{len(outs)} PNG decodes, {n} attendus")
        elif Image.open(outs[0]).size != (w0, h0) or Image.open(outs[-1]).size != (w0, h0):
            errs.append("dimensions des PNG decodes differentes de l'original")
    except ImportError:
        chk["note_pil"] = "Pillow absent : controle des dimensions saute"
    fr = rv.read_csv(P["metrics"] / f"{stem}_enc_frames.csv")
    obs = [int(r["frame"]) for r in fr if r["type"] == "I"]
    chk["observed_I_positions_encoder"] = obs
    if obs != exp:
        errs.append(f"positions des I-frames observees {obs} != attendues {exp}")
    csv_sizes = [int(r["record_bytes"]) for r in fr]
    if len(sizes) == n and csv_sizes != sizes:
        bad = next(i for i, (a, b) in enumerate(zip(csv_sizes, sizes)) if a != b)
        errs.append(f"taille d'enregistrement du CSV encodeur != taille relue dans le fichier (image {bad})")
    chk["record_sizes_match_csv"] = (csv_sizes == sizes)
    try:
        dj = rv.read_json(P["metrics"] / f"{stem}_dec_profile.json")
        nI = next((r["n_frames"] for r in dj["frame_summary"] if r["type"] == "I"), 0)
        chk["decoder_I_frames_profiled"] = nI
        if nI != len(exp):
            errs.append(f"le decodeur a traite {nI} I-frames, {len(exp)} attendues")
    except Exception as e:
        chk["decoder_profile"] = f"illisible ({e})"
    chk["errors"] = errs
    chk["ok"] = not errs
    return chk


# ======================================================================================
# collecte (reutilise run_vcm_energymap.collect_checkpoint) + colonnes propres a l'experience 2
# ======================================================================================
YOLO_KEYS = ("n_repeats", "time_s_per_frame_mean", "time_s_per_frame_std", "gross_j_per_frame_mean", "gross_j_per_frame_std",
             "net_j_per_frame_mean", "net_j_per_frame_std", "gross_cpu_j_per_frame_mean", "net_cpu_j_per_frame_mean")


def collect_gop(gop, P, ckpt, args, n):
    ns = Namespace(no_frames=n, gop=gop, start_index=args.start_index, threads=args.threads)
    row, srows, extra = rv.collect_checkpoint(args.checkpoint_number, P["dir"], ckpt, rv.Status(P["dir"] / "status.json"), ns)
    row = {"gop": gop, **row}
    for s in srows:
        s["gop"] = gop
    notes = [x for x in row.get("notes", "").split(" | ")
             if x and not x.startswith("energie YOLO : non mesuree par ce script")]
    stem = f"base_{args.checkpoint_number}"
    exp = expected_i(n, gop)
    row["n_I_frames_expected"], row["n_P_frames_expected"] = len(exp), n - len(exp)
    row["I_frame_files_expected"] = ";".join(f"{args.prefix}{args.start_index + i:03d}" for i in exp)
    sc = P["dir"] / "structure_check.json"
    if sc.exists():
        c = rv.read_json(sc)
        row["structure_ok"] = c.get("ok")
        row["structure_errors"] = "; ".join(c.get("errors", []))
        row["n_I_frames_observed"] = len(c.get("observed_I_positions_encoder", []))
        row["bpp_file_from_file_size"] = fnum(c.get("bpp_file_from_file_size"))
    else:
        row["structure_ok"] = "non verifie"
        notes.append("structure_check.json absent -> positions des I-frames/validite du bitstream non verifiees")
    # debits par type (octets reellement ecrits)
    pf = P["metrics"] / f"{stem}_enc_frames.csv"
    pfr = {"frame": [], "type": [], "bpp_payload": [], "psnr": []}
    if pf.exists():
        fr = rv.read_csv(pf)
        tot = sum(int(r["record_bytes"]) for r in fr)
        for t in ("I", "P"):
            v = [int(r["record_bytes"]) for r in fr if r["type"] == t]
            row[f"mean_record_bytes_{t}"] = sum(v) / len(v) if v else NAN
            row[f"bytes_share_{t}_pct"] = 100 * sum(v) / tot if tot else NAN
        pfr["frame"] = [int(r["frame"]) for r in fr]
        pfr["type"] = [r["type"] for r in fr]
        pfr["bpp_payload"] = [fnum(r["bpp_payload"]) for r in fr]
    pe = P["metrics"] / f"{stem}_eval_frames.csv"
    if pe.exists():
        er = rv.read_csv(pe)
        mse = [10 ** (-fnum(r["psnr_dec_png8"]) / 10) for r in er if isnum(fnum(r["psnr_dec_png8"]))]
        row["psnr_db_png8"] = 10 * math.log10(1 / (sum(mse) / len(mse))) if mse else NAN
        pfr["psnr"] = [fnum(r["psnr_dec_png8"]) for r in er]
        for t in ("I", "P"):
            v = [fnum(r["psnr_dec_png8"]) for r in er if r["type"] == t and isnum(fnum(r["psnr_dec_png8"]))]
            row[f"psnr_db_png8_mean_of_frames_{t}"] = sum(v) / len(v) if v else NAN
        # drift : PSNR moyen des P-frames selon leur rang dans le GOP (1er tiers / dernier tiers)
        ps = [(int(r["frame"]) % gop, fnum(r["psnr_dec_png8"])) for r in er if r["type"] == "P" and isnum(fnum(r["psnr_dec_png8"]))]
        if len(ps) >= 6 and gop > 3:
            a = [p for k, p in ps if k <= gop // 3]
            b = [p for k, p in ps if k > 2 * gop // 3]
            row["psnr_P_first_third_of_gop"] = sum(a) / len(a) if a else NAN
            row["psnr_P_last_third_of_gop"] = sum(b) / len(b) if b else NAN
    else:
        row["psnr_db_png8"] = NAN
    row["psnr_source_primary"] = "PNG 8 bits du decodeur, PSNR calcule depuis la MSE moyenne (toujours disponible)"
    row["msssim_png8_primary"] = row.get("msssim_png8_mean", NAN)
    # YOLO : temps + energie, mesures a part
    py = P["metrics"] / "yolo_energy.json"
    row["yolo_energy_frames"] = NAN
    if py.exists():
        ej = rv.read_json(py)
        idle = ej.get("idle")
        row["yolo_cpu_power_method"] = ej.get("meta", {}).get("cpu_power_method")
        row["yolo_energy_kind_corrected"] = ("MIXTE : CPU mesuree (RAPL via windows_emi) + RAM modelisee"
                                             if str(row["yolo_cpu_power_method"]).lower() in ("windows_emi", "intel_rapl")
                                             else "MODELISEE (proxy, pas une mesure)")
        row["yolo_energy_frames"] = ej.get("meta", {}).get("images", NAN)
        for label, pref in (("yolo_original", "yoloE_orig"), ("yolo_reconstructed", "yoloE_rec")):
            bs = rv.block_stats(ej.get("blocks", []), label, idle)
            if bs is None:
                notes.append(f"YOLO energie : bloc {label} absent")
                bs = {}
            for k in YOLO_KEYS:
                row[f"{pref}_{k}"] = bs.get(k, NAN)
            for k in ("gross_j_per_frame_mean", "net_j_per_frame_mean"):
                v = bs.get(k, NAN)
                row[f"{pref}_{k.replace('_j_', '_mwh_')}"] = v / 3.6 if isnum(v) else NAN
            blk = next((b for b in ej.get("blocks", []) if b.get("label") == label), {})
            row[f"{pref}_detections_last_pass"] = blk.get("detections_last_pass", NAN)
        notes.append("energie/temps YOLO : inference+NMS seuls, mesures dans un processus et avec un compteur SEPARES du codec ; "
                     "jamais additionnes a l'energie du codec")
    else:
        notes.append("yolo_energy.json absent -> temps et energie YOLO = NaN")
        for pref in ("yoloE_orig", "yoloE_rec"):
            for k in YOLO_KEYS:
                row[f"{pref}_{k}"] = NAN
    row["notes"] = " | ".join(notes)
    return row, srows, extra, pfr


REL_COLS = ["bpp_file", "bitstream_total_bytes", "enc_gross_j_per_frame_mean", "dec_gross_j_per_frame_mean",
            "enc_net_j_per_frame_mean", "dec_net_j_per_frame_mean", "enc_time_s_per_frame_mean", "dec_time_s_per_frame_mean",
            "psnr_db_png8", "msssim_png8_primary", "det_mAP50", "det_mAP50_95", "yoloE_rec_gross_j_per_frame_mean",
            "yoloE_rec_time_s_per_frame_mean"]


# ======================================================================================
# figures (matplotlib ; uniquement des valeurs mesurees)
# ======================================================================================
def make_figures(rows, perframe, outdir, footnote):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullLocator, ScalarFormatter
    except Exception as e:
        log(f"FIGURES NON GENEREES : matplotlib indisponible ({e}). pip install matplotlib")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False})
    rows = sorted(rows, key=lambda r: r["gop"])
    gops = [r["gop"] for r in rows]
    made = []

    def save(fig, name):
        import textwrap
        wrapped = "\n".join(textwrap.wrap(footnote, int(fig.get_size_inches()[0] * 15)))
        fig.text(0.01, 0.005, wrapped, fontsize=7, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.03 + 0.012 * wrapped.count("\n"), 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        made.append(name)
        log(f"figure generee : {name}.png/.pdf")

    def gop_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(gops)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(gops) / 1.3, max(gops) * 1.9)
        ax.set_xlabel("GOP (images entre deux I-frames ; échelle log₂)")

    def plot_series(ax, specs):
        any_ = 0
        for lab, col, err, color, mk, ls in specs:
            pts = [(r["gop"], r.get(col, NAN), r.get(err, NAN) if err else NAN) for r in rows]
            pts = [p for p in pts if isnum(p[1])]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            es = [p[2] if isnum(p[2]) else 0 for p in pts]
            ax.errorbar(xs, ys, yerr=es if any(es) else None, color=color, marker=mk, ls=ls, lw=1.8, ms=7, mec=SURF, mew=1.5,
                        capsize=3, ecolor=color, label=lab)
            ax.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8, color=INK)
            any_ += 1
        return any_

    def fig_panels(name, title, panels):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4), squeeze=False)
        ok = False
        for ax, (sub, ylab, specs, ref) in zip(axes[0], panels):
            got = plot_series(ax, specs)
            if got and ref:
                v = [r[ref[1]] for r in rows if isnum(r.get(ref[1]))]
                if v:
                    m = sum(v) / len(v)
                    ax.axhline(m, ls="--", lw=1, color=INK2)
                    ax.annotate(ref[0], (gops[0] / 1.25, m), textcoords="offset points", xytext=(2, 4), fontsize=8, color=INK2)
            if not got:
                ax.annotate("non mesuré", (0.5, 0.5), xycoords="axes fraction", ha="center", color=INK2)
            elif got > 1:
                ax.legend(loc="best", fontsize=8)
            ok |= bool(got)
            gop_axis(ax)
            ax.set_ylabel(ylab)
            ax.set_title(sub, fontsize=10)
        if not ok:
            plt.close(fig)
            log(f"figure NON generee ({name}) : aucune valeur mesuree")
            return
        fig.suptitle(title, fontweight="bold")
        save(fig, name)

    fig_panels("fig1_gop_vs_debit", "Débit réel vs GOP (octets écrits dans le .vcmb)", [
        ("Débit du fichier", "Débit (bits par pixel)", [("bpp fichier", "bpp_file", None, C_REC, "o", "-")], None),
        ("Taille du bitstream", "Octets", [("octets", "bitstream_total_bytes", None, C_REC, "s", "-")], None)])
    for name, pref, who, color in (("fig2_gop_vs_energie_encodage", "enc", "encodage", C_ENC),
                                   ("fig3_gop_vs_energie_decodage", "dec", "décodage", C_DEC)):
        fig_panels(name, f"Énergie de {who} par image vs GOP (CPU mesurée + RAM modélisée)", [
            ("Énergie brute (CPU+RAM)", "Énergie (J / image)", [(who, f"{pref}_gross_j_per_frame_mean", f"{pref}_gross_j_per_frame_std", color, "o", "-")], None),
            ("Énergie nette (brute − repos)", "Énergie (J / image)", [(who, f"{pref}_net_j_per_frame_mean", f"{pref}_net_j_per_frame_std", color, "o", "-")], None)])
    blocks_ok = all(isnum(r.get("enc_time_s_per_frame_mean")) and isnum(r.get("dec_time_s_per_frame_mean")) for r in rows)
    if blocks_ok:
        specs = [("encodage", "enc_time_s_per_frame_mean", "enc_time_s_per_frame_std", C_ENC, "o", "-"),
                 ("décodage", "dec_time_s_per_frame_mean", "dec_time_s_per_frame_std", C_DEC, "s", "-")]
        sub = "blocs répétés (moyenne ± écart-type)"
    else:
        specs = [("encodage", "enc_time_s_per_frame_profile", None, C_ENC, "o", "-"),
                 ("décodage", "dec_time_s_per_frame_profile", None, C_DEC, "s", "-")]
        sub = "profil, 1 passe, E/S incluses (blocs d'énergie absents)"
    fig_panels("fig4_gop_vs_temps", "Temps d'encodage et de décodage par image vs GOP", [(sub, "Temps (s / image)", specs, None)])
    fig_panels("fig5_gop_vs_psnr_msssim", "Qualité de reconstruction vs GOP (PNG 8 bits du décodeur)", [
        ("PSNR (depuis la MSE moyenne)", "PSNR (dB)", [("PSNR", "psnr_db_png8", None, C_REC, "o", "-")], None),
        ("MS-SSIM moyen", "MS-SSIM", [("MS-SSIM", "msssim_png8_primary", None, C_REC, "s", "-")], None)])
    fig_panels("fig6_gop_vs_map50", "Précision de détection (mAP@50, YOLOv5s) vs GOP", [
        ("images reconstruites", "mAP@50", [("reconstruites", "det_mAP50", None, C_REC, "o", "-")], ("images originales", "det_mAP50_original"))])
    fig_panels("fig7_gop_vs_map50_95", "Précision de détection (mAP@50:95, YOLOv5s) vs GOP", [
        ("images reconstruites", "mAP@50:95", [("reconstruites", "det_mAP50_95", None, C_REC, "o", "-")], ("images originales", "det_mAP50_95_original"))])
    yspec = lambda k, sd: [("originales", f"yoloE_orig_{k}", f"yoloE_orig_{sd}", C_ORIG, "s", "--"),
                           ("reconstruites", f"yoloE_rec_{k}", f"yoloE_rec_{sd}", C_REC, "o", "-")]
    fig_panels("fig8_gop_vs_energie_yolo", "Énergie d'inférence YOLOv5s par image vs GOP (hors énergie du codec)", [
        ("Énergie brute", "Énergie (J / image)", yspec("gross_j_per_frame_mean", "gross_j_per_frame_std"), None),
        ("Énergie nette", "Énergie (J / image)", yspec("net_j_per_frame_mean", "net_j_per_frame_std"), None)])

    # fig9 : energie du codec (enc+dec) vs debit
    rr = [r for r in rows if isnum(r.get("bpp_file")) and isnum(r.get("codec_gross_j_per_frame_enc_plus_dec"))]
    if rr:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for r, mk in zip(rr, ("o", "s", "^", "D", "v", "P")):
            ax.plot(r["bpp_file"], r["codec_gross_j_per_frame_enc_plus_dec"], mk, color=C_REC, ms=10, mec=SURF, mew=1.5)
            ax.annotate(f"GOP {r['gop']}", (r["bpp_file"], r["codec_gross_j_per_frame_enc_plus_dec"]), textcoords="offset points",
                        xytext=(9, 6), fontsize=9, color=INK)
        ax.set_xlabel("Débit réel (bpp, fichier .vcmb)")
        ax.set_ylabel("Énergie brute codec : encodage + décodage (J / image)")
        ax.set_title("Le GOP qui minimise le débit minimise-t-il l'énergie ?")
        save(fig, "fig9_energie_codec_vs_debit")
    else:
        log("figure NON generee (fig9) : debit ou energie encodage+decodage manquants")

    # fig10/11 : structure par image (preuve visuelle de la position des I-frames)
    def structure(name, key, ylab, title):
        gs = [g for g in gops if perframe.get(g, {}).get(key) and any(isnum(v) for v in perframe[g][key])]
        if not gs:
            log(f"figure NON generee ({name}) : aucune donnee par image")
            return
        fig, axes = plt.subplots(len(gs), 1, sharex=True, figsize=(9, 2.3 * len(gs) + 1.2), squeeze=False)
        for ax, g in zip(axes[:, 0], gs):
            d = perframe[g]
            x = d["frame"][:len(d[key])]
            y = d[key]
            ax.plot(x, y, color=INK2, lw=1)
            ix = [(a, b) for a, b, t in zip(x, y, d["type"]) if t == "I"]
            for a, _ in ix:
                ax.axvline(a, color=C_I, lw=0.6, ls=":")
            ax.plot([a for a, _ in ix], [b for _, b in ix], ls="none", marker="o", color=C_I, ms=6, mec=SURF, mew=1.2, label="I-frame")
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(f"GOP = {g}  ({len(ix)} I-frames)", loc="left", fontsize=9)
            if g == gs[0]:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("Position de l'image dans la séquence (0 = première image du fichier)")
        fig.suptitle(title, fontweight="bold")
        save(fig, name)

    structure("fig10_structure_debit_par_image", "bpp_payload", "bpp (payload)", "Débit par image : les I-frames suivent bien le GOP")
    structure("fig11_structure_psnr_par_image", "psnr", "PSNR (dB)", "PSNR par image (PNG 8 bits) : dérive entre deux I-frames")
    return made


# ======================================================================================
def build_parser():
    ap = argparse.ArgumentParser(description="Experience 2 : influence du GOP (voir l'en-tete du fichier)")
    ap.add_argument("--pilot", action="store_true", help="pilote : 65 images, resultats dans experiment2_gop_pilot65")
    ap.add_argument("--no_frames", type=int, default=None, help="defaut 145 (65 avec --pilot)")
    ap.add_argument("--gops", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--checkpoint_number", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="repetitions des blocs d'energie (codec ET YOLO)")
    ap.add_argument("--min_seconds", type=float, default=20.0)
    ap.add_argument("--warmup_iters", type=int, default=0,
                    help="passes completes d'echauffement avant chaque bloc d'energie du codec (defaut 0 : l'encodeur est deja "
                         "chauffe par la passe qui prepare le bitstream ; mettre 1 pour plus de rigueur, au prix de +2 passes/GOP)")
    ap.add_argument("--idle_seconds", type=float, default=30.0)
    ap.add_argument("--measure_power_secs", type=float, default=1.0)
    ap.add_argument("--country", default="MAR")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup_frames", type=int, default=2)
    ap.add_argument("--tensors", choices=["auto", "yes", "no"], default="auto",
                    help="--dump_tensors de test_base_profiled (test bit-exact + PSNR float). auto : oui si <= 65 images "
                         "(145 images = ~0,6 Go par fichier .pt et par GOP)")
    ap.add_argument("--prefix", default="frame_")
    ap.add_argument("--inp_path", default=str(ROOT / "videos_5" / "input_png"))
    ap.add_argument("--labels_dir", default=str(ROOT / "videos_5" / "labels"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "base"))
    ap.add_argument("--results_root", default=str(ROOT / "results"))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_energy", action="store_true", help="saute l'energie du codec (phase 2)")
    ap.add_argument("--skip_yolo_energy", action="store_true", help="saute temps/energie YOLO (phase 3)")
    ap.add_argument("--skip_detection", action="store_true")
    ap.add_argument("--no_iframe_block", action="store_true", help="ne pas mesurer le bloc 'I-frame seule'")
    ap.add_argument("--no_hash", action="store_true")
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--machine_label", default="Intel i5-6300U, CPU seul, Windows 10")
    ap.add_argument("--collect_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = build_parser().parse_args()
    py = sys.executable
    script = str(ROOT / "test_base_profiled.py")
    n = args.no_frames or (65 if args.pilot else 145)
    k0, ckn = args.start_index, args.checkpoint_number
    run_name = args.run_name or ("experiment2_gop_pilot65" if args.pilot and n == 65 else
                                 "experiment2_gop" if n == 145 else f"experiment2_gop_{n}f")
    run_dir = Path(args.results_root) / run_name
    comp = run_dir / "comparison"
    log.set_file(comp / "experiment_log.txt")
    gops = list(dict.fromkeys(args.gops))
    dump = args.tensors == "yes" or (args.tensors == "auto" and n <= 65)
    ckpt = Path(args.ckpt_dir) / f"checkpoint_base_{ckn}.pth.tar"
    inp = Path(args.inp_path)
    frames = [inp / f"{args.prefix}{i:03d}.png" for i in range(k0, k0 + n)]

    log("=" * 78)
    log(f"EXPERIENCE 2 (GOP) | run={run_name} | frames={n} (indices {k0}..{k0 + n - 1}) | GOP={gops} | checkpoint {ckn} | "
        f"threads={args.threads} | resume={args.resume} | dump_tensors={dump}")
    log(f"python {sys.version.split()[0]} | {platform.platform()} | torch {rv.pkg_version('torch')} | codecarbon "
        f"{rv.pkg_version('codecarbon')} | constriction {rv.pkg_version('constriction')} | matplotlib {rv.pkg_version('matplotlib')}")
    log("commande : " + subprocess.list2cmdline([py] + sys.argv))

    # ------------------------------------------------------------ verifications
    problems = []
    for s in REQUIRED:
        if not (ROOT / s).is_file():
            problems.append(f"script introuvable : {ROOT / s}")
    if not args.collect_only:
        for mod in ("torch", "codecarbon", "constriction"):
            if importlib.util.find_spec(mod) is None:
                msg = f"module manquant dans CET interpreteur : {mod} (activez svc_env_hp)"
                if args.dry_run:
                    log("AVERTISSEMENT : " + msg)
                else:
                    problems.append(msg)
        for f in frames:
            if not f.is_file():
                problems.append(f"image manquante : {f}")
        if not ckpt.is_file():
            problems.append(f"checkpoint introuvable : {ckpt}")
        if not args.skip_detection:
            for i in range(k0, k0 + n):
                lab = Path(args.labels_dir) / f"frame_{i:04d}.txt"
                if not lab.is_file():
                    problems.append(f"annotation manquante : {lab}")
            if not (ROOT / "yolov5s.pt").is_file():
                problems.append("yolov5s.pt introuvable")
    sets = {g: tuple(expected_i(n, g)) for g in gops}
    if len(set(sets.values())) != len(gops):
        problems.append(f"avec {n} images, des GOP differents donnent les MEMES positions d'I-frames : {sets} -> comparaison vide de sens")
    if problems:
        for p in problems:
            log("ERREUR : " + p)
        return 2

    log("positions d'I-frames ATTENDUES (position 0-based -> fichier) :")
    for g in gops:
        e = expected_i(n, g)
        log(f"  GOP {g:>3} : {len(e)} I + {n - len(e)} P ; positions {e} ; fichiers {[f'{args.prefix}{k0 + i:03d}' for i in e]}")

    def paths(g):
        d = run_dir / f"gop_{g}"
        return {"dir": d, "bs": d / "bitstream" / f"base_{ckn}.vcmb", "metrics": d / "metrics", "recon": d / "reconstructed",
                "det": d / "detection", "logs": d / "logs"}

    stem = f"base_{ckn}"
    yolo_orig, yolo_lab = run_dir / "_yolo_inputs" / "original", run_dir / "_yolo_inputs" / "labels"

    def codec_cmds(g):
        P = paths(g)
        common = ["--root", str(ROOT), "--checkpoint", str(ckpt), "--checkpoint_number", str(ckn), "--bitstream", str(P["bs"]),
                  "--prefix", args.prefix, "--start_index", str(k0), "--results", str(P["metrics"]),
                  "--threads", str(args.threads), "--warmup_frames", str(args.warmup_frames)]
        dmp = ["--dump_tensors"] if dump else []
        return {
            "encode": [py, script, "encode", *common, *dmp, "--inp_path", str(inp), "--no_frames", str(n), "--gop", str(g)],
            "decode": [py, script, "decode", *common, *dmp, "--out_path", str(P["recon"])],
            "evaluate": [py, script, "evaluate", *common, *dmp, "--inp_path", str(inp), "--out_path", str(P["recon"])],
            # energie sur TOUTES les images (--no_frames n) : sinon le nombre d'I-frames mesure ne serait pas celui du GOP
            "energy": [py, script, "energy", *common, "--inp_path", str(inp), "--no_frames", str(n), "--gop", str(g),
                       "--country", args.country, "--measure_power_secs", str(args.measure_power_secs),
                       "--idle_seconds", str(args.idle_seconds), "--min_seconds", str(args.min_seconds),
                       "--repeats", str(args.repeats), "--warmup_iters", str(args.warmup_iters),
                       *([] if args.no_iframe_block else ["--also_iframe"])]}

    def outputs(g, step):
        P, m = paths(g), paths(g)["metrics"]
        return {"encode": [P["bs"], m / f"{stem}_enc_profile.json", m / f"{stem}_enc_frames.csv"] + ([m / f"{stem}_enc_rec.pt"] if dump else []),
                "decode": [m / f"{stem}_dec_profile.json"] + ([m / f"{stem}_dec_rec.pt"] if dump else []),
                "evaluate": [m / f"{stem}_eval_summary.json", m / f"{stem}_eval_frames.csv"],
                "structure": [P["dir"] / "structure_check.json"],
                "detection": [P["det"] / "results.json", P["det"] / "metrics_summary.csv"],
                "energy": [m / f"{stem}_energy.json"],
                "yolo_energy": [m / "yolo_energy.json"]}[step]

    if args.dry_run:
        log("DRY RUN : rien n'est execute.")
        for g in gops:
            for step, c in codec_cmds(g).items():
                log(f"  [gop_{g}/{step}] " + subprocess.list2cmdline(c))
            log(f"  [gop_{g}/structure] verification en-tete .vcmb, positions des I, tailles d'enregistrements (in-process)")
            log(f"  [gop_{g}/detection] worker de run_vcm_energymap.py : original={yolo_orig} reconstruit={paths(g)['recon']}")
            log(f"  [gop_{g}/yolo_energy] worker interne : original={yolo_orig} reconstruit={paths(g)['recon']} -> {outputs(g, 'yolo_energy')[0]}")
        return 0

    # ------------------------------------------------------------ execution
    summary, statuses, active = {}, {}, []
    if not args.collect_only:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_detection or not args.skip_yolo_energy:
            yolo_orig.mkdir(parents=True, exist_ok=True)
            yolo_lab.mkdir(parents=True, exist_ok=True)
            for i, f in zip(range(k0, k0 + n), frames):
                if not (yolo_orig / f.name).exists():
                    shutil.copy2(f, yolo_orig / f.name)
                lab = Path(args.labels_dir) / f"frame_{i:04d}.txt"
                if not args.skip_detection and not (yolo_lab / lab.name).exists():
                    shutil.copy2(lab, yolo_lab / lab.name)
        fp_base = {"no_frames": n, "start_index": k0, "prefix": args.prefix, "inp_path": str(inp), "threads": args.threads,
                   "warmup_frames": args.warmup_frames, "repeats": args.repeats, "min_seconds": args.min_seconds,
                   "idle_seconds": args.idle_seconds, "warmup_iters": args.warmup_iters, "measure_power_secs": args.measure_power_secs,
                   "dump_tensors": dump, "checkpoint_number": ckn, "iframe_block": not args.no_iframe_block,
                   "labels_dir": args.labels_dir}
        sha = None
        for g in gops:
            P = paths(g)
            if P["dir"].exists() and not args.resume:
                log(f"[GOP {g}] {P['dir']} existe deja : LAISSE INTACT, configuration sautee (utilisez --resume ou --run_name <autre>).")
                summary[g] = {"all": "saute (dossier existant)"}
                continue
            P["dir"].mkdir(parents=True, exist_ok=True)
            st = rv.Status(P["dir"] / "status.json")
            fp = {**fp_base, "gop": g}
            if st.d["params"] and st.d["params"] != fp:
                diff = [k for k in fp if st.d["params"].get(k) != fp[k]]
                log(f"[GOP {g}] ERREUR : parametres differents de la serie existante ({diff}). Ne pas melanger : --run_name <autre>.")
                summary[g] = {"all": "parametres incoherents"}
                continue
            st.d["params"] = fp
            st.d["checkpoint_file"], st.d["checkpoint_size_bytes"] = str(ckpt), ckpt.stat().st_size
            if not args.no_hash and not st.d.get("checkpoint_sha256"):
                if sha is None:
                    log(f"calcul du SHA256 du checkpoint ({ckpt.stat().st_size / 1e6:.0f} Mo)...")
                    sha = rv.sha256_file(ckpt)
                st.d["checkpoint_sha256"] = sha
            st.save()
            statuses[g] = st
            summary[g] = {}
            active.append(g)

        def exec_step(g, step, deps, runner, extra_move=(), sleep=0.0):
            P, st, res = paths(g), statuses[g], summary[g]
            tag = f"[GOP {g}]"
            outs = outputs(g, step)
            if args.resume and st.step(step).get("state") == "done" and all(Path(o).exists() for o in outs):
                log(f"{tag} etape {step} : deja terminee, saut (--resume)")
                res[step] = "deja fait"
                return
            if any(res.get(d, "") not in ("ok", "deja fait") for d in deps):
                log(f"{tag} etape {step} : IGNOREE (dependance en echec : {deps})")
                res[step] = "ignore (dependance)"
                st.set(step, state="skipped_dependency")
                return
            rv.quarantine(list(outs) + list(extra_move), P["dir"])
            if sleep > 0:
                log(f"{tag} pause de {sleep:.0f} s avant la mesure (refroidissement). Fermez les autres applications.")
                time.sleep(sleep)
            t0 = time.perf_counter()
            st.set(step, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                rc, dt, cmdtxt = runner()
                if rc != 0:
                    raise RuntimeError(f"code de retour {rc}")
                missing = [str(o) for o in outs if not Path(o).exists()]
                if missing:
                    raise RuntimeError(f"sorties attendues absentes : {missing}")
                st.set(step, state="done", seconds=dt, finished=time.strftime("%Y-%m-%d %H:%M:%S"), cmd=cmdtxt,
                       outputs=[str(o) for o in outs])
                res[step] = "ok"
            except Exception as e:
                log(f"{tag} ERREUR etape {step} : {e}")
                log(traceback.format_exc(), console=False)
                st.set(step, state="failed", error=str(e), seconds=time.perf_counter() - t0)
                res[step] = f"ECHEC ({e})"

        def sub(g, step):
            P = paths(g)
            cmd = codec_cmds(g)[step]
            return lambda: (*rv.run_cmd(cmd, P["logs"] / f"{step}.log", f"gop_{g}/{step}"), subprocess.list2cmdline(cmd))

        def worker(g, step, flag, cfg):
            P = paths(g)
            cfgp = P["dir"] / f"{step}_worker_config.json"
            cfgp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            cmd = [py, str(Path(__file__).resolve() if flag == "--_yolo_worker" else ROOT / "run_vcm_energymap.py"), flag, str(cfgp)]

            def r():
                nrec = len(list(P["recon"].glob("*.png"))) if P["recon"].exists() else 0
                if nrec != n:
                    raise RuntimeError(f"{nrec} images reconstruites trouvees, {n} attendues")
                return (*rv.run_cmd(cmd, P["logs"] / f"{step}.log", f"gop_{g}/{step}"), subprocess.list2cmdline(cmd))
            return r

        def structure_runner(g):
            def r():
                t0 = time.perf_counter()
                chk = verify_structure(paths(g), g, n, k0, ckn, args.prefix, inp)
                outputs(g, "structure")[0].write_text(json.dumps(rv.jsonable(chk), indent=2, ensure_ascii=False), encoding="utf-8")
                for e in chk["errors"]:
                    log(f"[GOP {g}] STRUCTURE : {e}")
                log(f"[GOP {g}] structure {'OK' if chk['ok'] else 'ECHEC'} : I observees {chk['observed_I_positions_encoder']} | "
                    f"{chk['records_parsed']} enregistrements | {chk['file_size_bytes']} octets")
                return (0 if chk["ok"] else 1), time.perf_counter() - t0, "in-process"
            return r

        # ---------------- phase 1
        log("-" * 78)
        log("PHASE 1 : encodage, decodage, evaluation, verification de structure, detection")
        for g in active:
            log(f"[GOP {g}] debut -> {paths(g)['dir']}")
            P = paths(g)
            exec_step(g, "encode", [], sub(g, "encode"))
            exec_step(g, "decode", ["encode"], sub(g, "decode"), extra_move=[P["recon"]])
            exec_step(g, "evaluate", ["encode", "decode"], sub(g, "evaluate"))
            exec_step(g, "structure", ["encode", "decode"], structure_runner(g))
            if not args.skip_detection:
                cfg = {"original_dir": str(yolo_orig), "reconstructed_dir": str(P["recon"]), "labels_dir": str(yolo_lab),
                       "output_dir": str(P["det"]), "threads": args.threads}
                exec_step(g, "detection", ["decode"], worker(g, "detection", "--_detect_worker", cfg), extra_move=[P["det"]])
            log(f"[GOP {g}] phase 1 : {summary[g]}")

        # ---------------- estimation de duree de la phase 2
        try:
            tot = 0.0
            for g in active:
                m = paths(g)["metrics"]
                one = 0.0
                for d in ("enc", "dec"):
                    pj = rv.read_json(m / f"{stem}_{d}_profile.json")
                    one += sum(fnum(r["wall_mean_s"]) * r["n_frames"] for r in pj["frame_summary"])
                tot += (args.warmup_iters + args.repeats) * one + args.idle_seconds
                log(f"[GOP {g}] 1 passe encodage+decodage ~ {one / 60:.1f} min (mesure du profil)")
            log(f"ESTIMATION phase 2 (energie codec, {args.repeats} repetitions, hors bloc I-frame) : ~ {tot / 3600:.1f} h pour {len(active)} GOP. "
                f"Interruptible (Ctrl+C) puis --resume.")
        except Exception as e:
            log(f"estimation de duree impossible ({e})")

        # ---------------- phase 2
        if not args.skip_energy:
            log("-" * 78)
            log("PHASE 2 : energie du codec (blocs entiers encode / decode, toutes les images)")
            for g in active:
                exec_step(g, "energy", ["encode", "decode"], sub(g, "energy"), sleep=args.cooldown)
                log(f"[GOP {g}] phase 2 : {summary[g].get('energy')}")
        # ---------------- phase 3
        if not args.skip_yolo_energy:
            log("-" * 78)
            log("PHASE 3 : temps + energie YOLOv5s (processus separe, originales puis reconstruites)")
            for g in active:
                P = paths(g)
                cfg = {"original_dir": str(yolo_orig), "reconstructed_dir": str(P["recon"]), "n": n, "threads": args.threads,
                       "country": args.country, "measure_power_secs": args.measure_power_secs, "idle_seconds": args.idle_seconds,
                       "min_seconds": args.min_seconds, "repeats": args.repeats, "output_json": str(outputs(g, "yolo_energy")[0])}
                exec_step(g, "yolo_energy", ["decode"], worker(g, "yolo_energy", "--_yolo_worker", cfg), sleep=args.cooldown)
                log(f"[GOP {g}] phase 3 : {summary[g].get('yolo_energy')}")

    # ------------------------------------------------------------ agregation
    log("-" * 78)
    log("agregation des resultats reels dans " + str(comp))
    rows, srows_all, extras, perframe = [], [], {}, {}
    for g in gops:
        P = paths(g)
        if not P["dir"].exists():
            log(f"GOP {g} : aucun resultat ({P['dir']})")
            continue
        try:
            row, srows, extra, pfr = collect_gop(g, P, ckpt, args, n)
            rows.append(row)
            srows_all += srows
            extras[g] = extra
            perframe[g] = pfr
        except Exception as e:
            log(f"GOP {g} : agregation impossible ({e})")
            log(traceback.format_exc(), console=False)
    if not rows:
        log("rien a agreger.")
        return 1
    base = next((r for r in rows if r["gop"] == 32), None)
    for r in rows:
        for c in REL_COLS:
            b = base.get(c, NAN) if base else NAN
            v = r.get(c, NAN)
            r[f"{c}_rel_gop32_pct"] = 100 * (v - b) / b if isnum(v) and isnum(b) and b != 0 else NAN
    # coherence inter-configurations : les positions d'I-frames different reellement
    obs = {r["gop"]: tuple(rv.read_json(paths(r["gop"])["dir"] / "structure_check.json")["observed_I_positions_encoder"])
           for r in rows if (paths(r["gop"])["dir"] / "structure_check.json").exists()}
    distinct = len(set(obs.values())) == len(obs) if obs else "non verifie"
    log(f"positions d'I-frames observees : {({g: list(v) for g, v in obs.items()})} | toutes distinctes : {distinct}")
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    comp.mkdir(parents=True, exist_ok=True)
    with open(comp / "gop_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("NaN" if isinstance(r.get(c), float) and not math.isfinite(r[c]) else r.get(c, "NaN")) for c in cols})
    if srows_all:
        with open(comp / "stage_breakdown.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(srows_all[0].keys()))
            w.writeheader()
            w.writerows(srows_all)
    shas = {r["gop"]: r.get("checkpoint_sha256") for r in rows}
    meta = {"experiment": "experiment2_gop", "run_name": run_name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv[1:], "machine_label": args.machine_label, "platform": platform.platform(),
            "python": sys.version.split()[0],
            "versions": {p: rv.pkg_version(p) for p in ("torch", "torchvision", "numpy", "codecarbon", "constriction", "matplotlib")},
            "params": vars(args), "n_frames": n, "gops": gops, "checkpoint_number": ckn,
            "checkpoint_sha256_per_config": shas,
            "same_checkpoint_in_all_configs": len({v for v in shas.values() if v}) <= 1,
            "i_frame_positions_observed": {str(g): list(v) for g, v in obs.items()}, "i_frame_positions_all_distinct": distinct,
            "gop_semantics": "I-frame si position%GOP==0 (position 0-based dans la sequence) ; tampon d'images remis a zero a chaque I ; "
                             "tampon de flots MWNet non remis a zero (closed_gop non utilise)",
            "warnings": [
                "Une seule sequence de 145 images et un seul checkpoint : resultats non generalisables.",
                "Debit, PSNR, MS-SSIM et detection sont deterministes ; seules l'energie et le temps ont des repetitions (ecart-type).",
                "Energie = CPU mesuree (RAPL via windows_emi si confirme dans cpu_power_method) + RAM modelisee, machine entiere ; "
                "energy_kind du script d'origine mal etiquete, voir energy_kind_corrected.",
                "L'energie d'encodage inclut le calcul des bits estimes (est_bits), identique pour tous les GOP (biais commun, non quantifie).",
                "L'energie YOLO (inference+NMS) est mesuree a part et n'est jamais ajoutee a celle du codec.",
                "Derive thermique/OneDrive possibles : fermer les autres applications, secteur branche, dossier hors synchronisation.",
                "Verifier que les differences d'energie entre GOP depassent l'ecart-type des repetitions avant de conclure."],
            "column_units": rv.COLUMN_UNITS}
    (comp / "gop_comparison.json").write_text(json.dumps(rv.jsonable({"meta": meta, "rows": rows, "extra": extras}), indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    log(f"ecrit : gop_comparison.csv / .json ({len(rows)} configurations)")
    kind = next((r.get("energy_kind_corrected") for r in rows if r.get("energy_kind_corrected")), "energie codec non mesuree")
    foot = (f"{args.machine_label} | checkpoint {ckn} | N={n} images | énergie: {kind} | {args.repeats} répétitions | "
            f"brute = CPU+RAM bloc entier, nette = brute − repos | {len(rows)}/{len(gops)} GOP")
    make_figures(rows, perframe, comp / "figures", foot)
    log("RESUME DES ETAPES : " + json.dumps(rv.jsonable({str(k): v for k, v in summary.items()}), ensure_ascii=False))
    bad = [g for g, v in summary.items() for s, x in v.items() if str(x).startswith(("ECHEC", "parametres"))]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
