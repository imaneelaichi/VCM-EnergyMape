"""
test_base_profiled.py -- Base Layer with a REAL bitstream, per-stage profiling and metrics.

Sub-commands
------------
  encode    frames (PNG)  -> bitstream.vcmb  (+ encoder-side stats, profile)
  decode    bitstream.vcmb -> decoded frames (PNG) (+ profile).  Does NOT need the original frames
  evaluate  original frames + decoded frames + bitstream stats -> PSNR / MS-SSIM / BPP table
  roundtrip encode and decode in two separate Python processes, then evaluate

Frame naming: <inp_path>/<prefix>%03d.png, indices start at --start_index (default 0, as the
original test_base.py; with ffmpeg use `-start_number 0`).

Bit-rate definitions (all reported):
  bpp_est          ideal code length of the *same symbols* under the model likelihoods
                   (original estimate_bpp formula, lower bound 1e-9)            [theoretical]
  bpp_payload      8 * range-coder payload bytes / (H*W)                         [measured]
  bpp_frame        8 * (payload + per-frame header: lengths + alphabet ranges) / (H*W)   [measured]
  bpp_file         8 * total file size (incl. one-time sequence header) / (frames*H*W)    [measured]
  H*W = original (un-padded) frame size, as in the original estimate_bpp(input=coding_frame).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from math import log10
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from base_codec import BaseLayerCodec           # noqa: E402
from bitstream import BitstreamReader, BitstreamWriter, SequenceHeader  # noqa: E402
from profiling import Profiler                  # noqa: E402


def _frame_path(d, prefix, idx):
    return os.path.join(d, f"{prefix}{idx:03d}.png")


def _save_png(x: torch.Tensor, path: str):
    ToPILImage()(x.detach().clamp(0, 1).squeeze(0).cpu()).save(path)


def _env_meta(args):
    return {"torch": torch.__version__, "threads": torch.get_num_threads(), "python": sys.version.split()[0],
            "platform": sys.platform, "argv": sys.argv[1:], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# ======================================================================================
# ENCODE
# ======================================================================================
def run_encode(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    prof = Profiler()
    with prof.init_timer("model_build"):
        codec = BaseLayerCodec("enc", root=a.root, closed_gop=a.closed_gop)
    with prof.init_timer("checkpoint_load"):
        info = codec.load_checkpoint(a.checkpoint)
    first = Image.open(_frame_path(a.inp_path, a.prefix, a.start_index))
    W, H = first.size
    n = a.no_frames
    os.makedirs(os.path.dirname(os.path.abspath(a.bitstream)), exist_ok=True)

    # ---------------- warm-up (discarded): first k frames, then reset
    if a.warmup_frames > 0:
        with prof.phase("warmup"):
            for i in range(min(a.warmup_frames, n)):
                x = ToTensor()(Image.open(_frame_path(a.inp_path, a.prefix, a.start_index + i)).convert("RGB")).unsqueeze(0)
                with prof.frame(i, "I" if i % a.gop == 0 else "P", "enc"):
                    codec.encode_frame(x, i, a.gop, prof)
        codec.reset_sequence()

    hdr = SequenceHeader(W, H, n, a.gop, ckpt_id=a.checkpoint_number, closed_gop=a.closed_gop)
    writer = BitstreamWriter(a.bitstream, hdr)
    rows, recs = [], []
    for i in range(n):
        ftype = "I" if i % a.gop == 0 else "P"
        with prof.frame(i, ftype, "enc"):
            with prof.stage("image_read"):
                img = Image.open(_frame_path(a.inp_path, a.prefix, a.start_index + i)).convert("RGB")
            with prof.stage("preprocess_to_tensor"):
                x = ToTensor()(img).unsqueeze(0)
            r = codec.encode_frame(x, i, a.gop, prof)
            with prof.stage("bitstream_write"):
                rec_bytes = writer.add_frame(r["payload"], r["ranges"])
        hdr.shapes.update({k: tuple(v[1:]) for k, v in r["shapes"].items()})
        pix = H * W
        mse = torch.mean((r["rec"] - x) ** 2).item()
        rows.append({"frame": i, "type": ftype, "payload_bytes": len(r["payload"]), "record_bytes": rec_bytes,
                     "bits_est": sum(r["est_bits"].values()), "bpp_est": sum(r["est_bits"].values()) / pix,
                     "bpp_payload": 8 * len(r["payload"]) / pix, "bpp_frame": 8 * rec_bytes / pix,
                     "mse_enc_local": mse, "psnr_enc_local": 10 * log10(1 / mse) if mse > 0 else float("inf"),
                     **{f"est_bits_{k}": v for k, v in r["est_bits"].items()},
                     **{f"coded_bits_{k}": v for k, v in r["coded_bits"].items()}})
        if a.dump_tensors:
            recs.append(r["rec"].clone())
    t0 = time.perf_counter()
    sizes = writer.close()
    prof.init_times["bitstream_file_write_total"] = time.perf_counter() - t0
    sizes["bpp_file"] = 8 * sizes["total_bytes"] / (n * H * W)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.bitstream).stem
    prof.export(str(out / f"{stem}_enc_profile"), meta={**_env_meta(a), **sizes, "checkpoint": a.checkpoint, "ckpt_info": info})
    with open(out / f"{stem}_enc_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k)))
        w.writeheader()
        w.writerows(rows)
    if a.dump_tensors:
        torch.save(recs, str(out / f"{stem}_enc_rec.pt"))
    print(f"[encode] {n} frames -> {a.bitstream}: {sizes['total_bytes']} bytes, bpp_file={sizes['bpp_file']:.5f}")
    return sizes


# ======================================================================================
# DECODE  (never touches the original frames)
# ======================================================================================
def run_decode(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    prof = Profiler()
    with prof.init_timer("model_build"):
        codec = BaseLayerCodec("dec", root=a.root, closed_gop=False)
    with prof.init_timer("checkpoint_load"):
        info = codec.load_checkpoint(a.checkpoint)
    rd = BitstreamReader(a.bitstream)
    h = rd.header
    codec.closed_gop = h.closed_gop
    os.makedirs(a.out_path, exist_ok=True)

    if a.warmup_frames > 0:
        with prof.phase("warmup"):
            for i in range(min(a.warmup_frames, h.n_frames)):
                payload, ranges, _ = rd.next_frame()
                with prof.frame(i, "I" if i % h.gop == 0 else "P", "dec"):
                    codec.decode_frame(payload, ranges, i, h.gop, h.shapes, (h.height, h.width), prof)
        codec.reset_sequence()
        rd.close()
        rd = BitstreamReader(a.bitstream)

    recs = []
    for i in range(h.n_frames):
        ftype = "I" if i % h.gop == 0 else "P"
        with prof.frame(i, ftype, "dec"):
            with prof.stage("bitstream_read"):
                payload, ranges, _ = rd.next_frame()
            rec = codec.decode_frame(payload, ranges, i, h.gop, h.shapes, (h.height, h.width), prof)
            with prof.stage("image_write"):
                _save_png(rec, _frame_path(a.out_path, a.prefix, a.start_index + i))
        if a.dump_tensors:
            recs.append(rec.clone())
    rd.close()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.bitstream).stem
    prof.export(str(out / f"{stem}_dec_profile"), meta={**_env_meta(a), "checkpoint": a.checkpoint, "ckpt_info": info})
    if a.dump_tensors:
        torch.save(recs, str(out / f"{stem}_dec_rec.pt"))
    print(f"[decode] {h.n_frames} frames -> {a.out_path}")


# ======================================================================================
# EVALUATE (original vs independently decoded)
# ======================================================================================
def run_evaluate(a):
    from loss import MS_SSIM
    stem = Path(a.bitstream).stem
    out = Path(a.results)
    ms = MS_SSIM(reduction="mean", data_range=1.0).to("cpu")
    enc_rows = list(csv.DictReader(open(out / f"{stem}_enc_frames.csv")))
    meta = json.load(open(out / f"{stem}_enc_profile.json"))["meta"]
    dec_pt = out / f"{stem}_dec_rec.pt"
    enc_pt = out / f"{stem}_enc_rec.pt"
    dec_recs = torch.load(dec_pt) if dec_pt.exists() else None
    enc_recs = torch.load(enc_pt) if enc_pt.exists() else None
    rows = []
    for r in enc_rows:
        i = int(r["frame"])
        x = ToTensor()(Image.open(_frame_path(a.inp_path, a.prefix, a.start_index + i)).convert("RGB")).unsqueeze(0)
        d = ToTensor()(Image.open(_frame_path(a.out_path, a.prefix, a.start_index + i)).convert("RGB")).unsqueeze(0)
        row = {"frame": i, "type": r["type"], "payload_bytes": r["payload_bytes"], "bpp_est": float(r["bpp_est"]),
               "bpp_payload": float(r["bpp_payload"]), "bpp_frame": float(r["bpp_frame"])}
        # (a) metrics on the 8-bit PNG written by the decoder (what a user would get)
        mse8 = torch.mean((d - x) ** 2).item()
        row["psnr_dec_png8"] = 10 * log10(1 / mse8) if mse8 > 0 else float("inf")
        row["msssim_dec_png8"] = ms(d, x).mean().item()
        # (b) metrics on the decoder's float reconstruction (comparable with the original test_base.py)
        if dec_recs is not None:
            df = dec_recs[i]
            mse = torch.mean((df - x) ** 2).item()
            row["psnr_dec_float"] = 10 * log10(1 / mse) if mse > 0 else float("inf")
            row["msssim_dec_float"] = ms(df, x).mean().item()
            if enc_recs is not None:
                row["max_abs_diff_enc_vs_dec"] = (df - enc_recs[i]).abs().max().item()
                row["bit_exact_enc_vs_dec"] = bool(torch.equal(df, enc_recs[i]))
        rows.append(row)
    with open(out / f"{stem}_eval_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(k, t=None):
        v = [x[k] for x in rows if k in x and (t is None or x["type"] == t)]
        return sum(v) / len(v) if v else None

    n = len(rows)
    summ = {"n_frames": n, "bitstream": {k: meta[k] for k in ("header_bytes", "frames_bytes", "total_bytes", "bpp_file")},
            "bpp_est_mean": avg("bpp_est"), "bpp_payload_mean": avg("bpp_payload"), "bpp_frame_mean": avg("bpp_frame"),
            "bpp_real_over_est_ratio": avg("bpp_payload") / avg("bpp_est"),
            "psnr_dec_png8_mean": avg("psnr_dec_png8"), "msssim_dec_png8_mean": avg("msssim_dec_png8")}
    if dec_recs is not None:
        mse_f = np.mean([10 ** (-x["psnr_dec_float"] / 10) for x in rows])
        summ["psnr_dec_float_from_mean_mse"] = 10 * log10(1 / mse_f)   # same aggregation as original test_base.py
        summ["msssim_dec_float_mean"] = avg("msssim_dec_float")
        if enc_recs is not None:
            summ["max_abs_diff_enc_vs_dec"] = max(x["max_abs_diff_enc_vs_dec"] for x in rows)
            summ["all_frames_bit_exact"] = all(x["bit_exact_enc_vs_dec"] for x in rows)
    for t in ("I", "P"):
        if any(x["type"] == t for x in rows):
            summ[f"bpp_payload_mean_{t}"] = avg("bpp_payload", t)
            summ[f"bpp_est_mean_{t}"] = avg("bpp_est", t)
    json.dump(summ, open(out / f"{stem}_eval_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    return summ


# ======================================================================================
# ENERGY (whole-block measurements only; see energy_profiling.py for scope and limits)
# ======================================================================================
def run_energy(a):
    from energy_profiling import EnergyMeter, dump
    if a.threads:
        torch.set_num_threads(a.threads)
    n = a.no_frames
    frames = [ToTensor()(Image.open(_frame_path(a.inp_path, a.prefix, a.start_index + i)).convert("RGB")).unsqueeze(0)
              for i in range(n)]
    enc = BaseLayerCodec("enc", root=a.root, closed_gop=a.closed_gop); enc.load_checkpoint(a.checkpoint)
    dec = BaseLayerCodec("dec", root=a.root, closed_gop=a.closed_gop); dec.load_checkpoint(a.checkpoint)
    # bitstream produced once, OUTSIDE the measured blocks (decoder blocks read it from memory: no disk I/O)
    records = []
    for i, x in enumerate(frames):
        r = enc.encode_frame(x, i, a.gop); records.append((r["payload"], r["ranges"], {k: tuple(v[1:]) for k, v in r["shapes"].items()}))
    shapes = {}
    for _, _, sh in records:
        shapes.update(sh)
    H, W = frames[0].shape[-2:]

    def enc_block():
        enc.reset_sequence()
        for i, x in enumerate(frames):
            enc.encode_frame(x, i, a.gop)
        return n

    def dec_block():
        dec.reset_sequence()
        for i, (pl, rg, _) in enumerate(records):
            dec.decode_frame(pl, rg, i, a.gop, shapes, (H, W))
        return n

    def enc_I_block():                      # intra only: encode frame 0 repeatedly
        enc.reset_sequence(); enc.encode_frame(frames[0], 0, a.gop); return 1

    meter = EnergyMeter(country_iso_code=a.country, measure_power_secs=a.measure_power_secs)
    try:
        info = meter.method_info()
        print("[energy] method:", {k: info[k] for k in ("cpu_power_method", "energy_kind")})
        idle = meter.measure_idle(a.idle_seconds)
        out = {"meta": {**_env_meta(a), **info, "frames": n, "gop": a.gop, "size": [H, W], "checkpoint": a.checkpoint,
                        "note": "Energy of whole blocks only. Per-stage energy is NOT measured."},
               "idle": idle, "blocks": []}
        blocks = [("encode_sequence", enc_block), ("decode_sequence", dec_block)]
        if a.also_iframe:
            blocks.append(("encode_iframe_only", enc_I_block))
        for name, fn in blocks:
            res = meter.measure_repeated(name, fn, idle, min_seconds=a.min_seconds,
                                         warmup_iters=a.warmup_iters, repeats=a.repeats)
            out["blocks"].append(res)
            g = res["aggregate"]
            print(f"[energy] {name}: {g['time_per_frame_s_mean']:.3f} s/frame, gross {g['gross_j_per_frame_mean']:.2f} J/frame, "
                  f"net {g['net_j_per_frame_mean']:.2f} J/frame (std {g['net_j_per_frame_std']:.2f}), "
                  f"kind={info['energy_kind']}")
        out["all_tasks"] = meter.tasks
    finally:
        meter.close()
    Path(a.results).mkdir(parents=True, exist_ok=True)
    dump(str(Path(a.results) / f"{Path(a.bitstream).stem}_energy.json"), out)


# ======================================================================================
def _common(p):
    p.add_argument("--root", default=str(ROOT), help="repo root (for ./models, ./config)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint_number", type=int, default=1)
    p.add_argument("--bitstream", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--warmup_frames", type=int, default=0)
    p.add_argument("--dump_tensors", action="store_true", help="save float reconstructions (for bit-exactness checks)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("encode"); _common(e)
    e.add_argument("--inp_path", required=True); e.add_argument("--no_frames", type=int, required=True)
    e.add_argument("--gop", type=int, default=32); e.add_argument("--closed_gop", action="store_true")
    d = sp.add_parser("decode"); _common(d); d.add_argument("--out_path", required=True)
    v = sp.add_parser("evaluate"); _common(v)
    v.add_argument("--inp_path", required=True); v.add_argument("--out_path", required=True)
    en = sp.add_parser("energy"); _common(en)
    en.add_argument("--inp_path", required=True); en.add_argument("--no_frames", type=int, required=True)
    en.add_argument("--gop", type=int, default=32); en.add_argument("--closed_gop", action="store_true")
    en.add_argument("--country", default="MAR"); en.add_argument("--measure_power_secs", type=float, default=1.0)
    en.add_argument("--idle_seconds", type=float, default=30.0); en.add_argument("--min_seconds", type=float, default=20.0)
    en.add_argument("--repeats", type=int, default=3); en.add_argument("--warmup_iters", type=int, default=1)
    en.add_argument("--also_iframe", action="store_true")
    r = sp.add_parser("roundtrip"); _common(r)
    r.add_argument("--inp_path", required=True); r.add_argument("--out_path", required=True)
    r.add_argument("--no_frames", type=int, required=True); r.add_argument("--gop", type=int, default=32)
    r.add_argument("--closed_gop", action="store_true")
    a = ap.parse_args()
    if a.cmd == "encode":
        run_encode(a)
    elif a.cmd == "decode":
        run_decode(a)
    elif a.cmd == "evaluate":
        run_evaluate(a)
    elif a.cmd == "energy":
        run_energy(a)
    else:  # roundtrip: two independent processes
        base = [sys.executable, str(Path(__file__).resolve())]
        com = ["--root", a.root, "--checkpoint", a.checkpoint, "--checkpoint_number", str(a.checkpoint_number),
               "--bitstream", a.bitstream, "--prefix", a.prefix, "--start_index", str(a.start_index),
               "--results", a.results, "--threads", str(a.threads), "--warmup_frames", str(a.warmup_frames), "--dump_tensors"]
        enc = base + ["encode"] + com + ["--inp_path", a.inp_path, "--no_frames", str(a.no_frames), "--gop", str(a.gop)] + (["--closed_gop"] if a.closed_gop else [])
        dec = base + ["decode"] + com + ["--out_path", a.out_path]
        ev = base + ["evaluate"] + com + ["--inp_path", a.inp_path, "--out_path", a.out_path]
        for cmd in (enc, dec, ev):
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
