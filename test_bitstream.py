"""
test_bitstream.py -- validation of the real bitstream.

  T1  entropy backend: random tensors, extreme cases, bit-exact (factorized / gaussian / context+GMM)
  T2  container: varints, header, truncation detection
  T3  sequence: encoder instance A -> FILE -> decoder instance B (separate module objects,
      decoder has no MENet).  Compares, frame by frame and sub-stream by sub-stream, the
      de-quantised latents (encoder vs decoder) and the reconstructions (bit-exact).
  T4  equivalence with the ORIGINAL pipeline (Pframe.test logic: forward() + estimate_bpp):
      reconstruction difference and estimated-bpp difference.
  T5  corruption: flipping one payload byte must change the decoded output (proves the decoder
      really depends on the bits, not on hidden state).

Run (random weights, quick mechanical test):
    python test_bitstream.py --frames_dir /tmp/syn --prefix syn_ --no_frames 5 --gop 3
Run with real checkpoint on real frames (recommended before any experiment):
    python test_bitstream.py --checkpoint checkpoints/base/checkpoint_base_1.pth.tar \
        --frames_dir input/ParkScene --prefix ParkScene_ --no_frames 4 --gop 3
Exit code 0 only if every test passes.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bitstream as bs                                   # noqa: E402
import entropy_codec as ec                               # noqa: E402
from base_codec import BaseLayerCodec                    # noqa: E402
from context_model import ContextModel                   # noqa: E402
from entropy_models import (EntropyBottleneck, GaussianConditional,   # noqa: E402
                            GaussianMixtureModelConditional, estimate_bpp)
from Utils import Alignment                              # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


# ------------------------------------------------------------------------------ T1
def t1_backend():
    print("T1 entropy backend")
    torch.manual_seed(0)
    bn = EntropyBottleneck(16).eval()
    g = GaussianConditional(use_mean=True, quant_mode="round").eval()
    ctx = ContextModel(8, 16, GaussianMixtureModelConditional(K=3, quant_mode="round")).eval()
    cases = {
        "typical": (torch.randn(1, 16, 6, 9) * 3, torch.randn(1, 8, 6, 9) * 4, torch.randn(1, 8, 5, 7) * 3),
        "all-zero (lo==hi)": (torch.zeros(1, 16, 4, 4), torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 3, 3)),
        "wide range": (torch.randn(1, 16, 5, 5) * 40, torch.randn(1, 8, 5, 5) * 60, torch.randn(1, 8, 4, 4) * 30),
    }
    for name, (z, y, yc) in cases.items():
        cond = torch.cat([torch.randn_like(y) * 2, torch.rand_like(y) * 3 + 0.2], 1)
        phi = torch.randn(1, 16, *yc.shape[-2:])
        ec.TRACE = []
        fe = ec.FrameEncoder()
        zh = ec.encode_factorized(fe, bn, z, "z")
        yh = ec.encode_gaussian(fe, g, y, cond, "y")
        ch = ec.encode_context_gmm(fe, ctx, yc, phi, "c")
        payload, ranges = fe.finish(), fe.ranges
        fd = ec.FrameDecoder(payload, ranges)
        zd = ec.decode_factorized(fd, bn, z.shape, "z")
        yd = ec.decode_gaussian(fd, g, y.shape, cond, "y")
        cd = ec.decode_context_gmm(fd, ctx, yc.shape, phi, "c")
        check(f"T1 {name}", torch.equal(zh, zd) and torch.equal(yh, yd) and torch.equal(ch, cd) and fd.fully_consumed(),
              f"({len(payload)} bytes)")
    ec.TRACE = None


# ------------------------------------------------------------------------------ T2
def t2_container():
    print("T2 container")
    for v in (0, 1, 127, 128, 300, 2 ** 31, 2 ** 40):
        b = bytearray(); bs._w_varint(b, v)
        assert bs._r_varint(io.BytesIO(bytes(b))) == v
    check("T2 varint roundtrip", True)
    check("T2 zigzag", all(bs._unzz(bs._zz(v)) == v for v in (-5, -1, 0, 1, 7, -300, 300)))
    h = bs.SequenceHeader(224, 192, 5, 3, ckpt_id=2, closed_gop=True, shapes={"I_z": (192, 3, 4), "R_y": (128, 12, 14)})
    h2 = bs.SequenceHeader.from_file(io.BytesIO(h.to_bytes()))
    check("T2 header roundtrip", (h2.width, h2.height, h2.n_frames, h2.gop, h2.ckpt_id, h2.closed_gop, h2.shapes) ==
          (224, 192, 5, 3, 2, True, h.shapes))
    # payload that contains the magic separator of the original BitStreamIO must survive
    payload = b"\x46\xE2\x84\x92" * 5 + os.urandom(100)
    rec = bs.frame_record(payload, [(-3, 4), (0, 1)])
    p2, r2, n = bs.read_frame_record(io.BytesIO(rec))
    check("T2 payload with original magic bytes", p2 == payload and r2 == [(-3, 4), (0, 1)] and n == len(rec))
    try:
        bs.read_frame_record(io.BytesIO(rec[:-10]))
        check("T2 truncation detected", False)
    except EOFError:
        check("T2 truncation detected", True)


# ------------------------------------------------------------------------------ helpers
def load_frames(d, prefix, n, start=0):
    return [ToTensor()(Image.open(os.path.join(d, f"{prefix}{start + i:03d}.png")).convert("RGB")).unsqueeze(0) for i in range(n)]


def encode_sequence(codec, frames, gop, trace=False):
    outs = []
    H, W = frames[0].shape[-2:]
    hdr = bs.SequenceHeader(W, H, len(frames), gop)
    traces = []
    for i, x in enumerate(frames):
        if trace:
            ec.TRACE = []
        r = codec.encode_frame(x, i, gop)
        if trace:
            traces.append(ec.TRACE); ec.TRACE = None
        hdr.shapes.update({k: tuple(v[1:]) for k, v in r["shapes"].items()})
        outs.append(r)
    return hdr, outs, traces


def write_file(path, hdr, outs):
    w = bs.BitstreamWriter(path, hdr)
    for r in outs:
        w.add_frame(r["payload"], r["ranges"])
    return w.close()


def decode_file(codec, path, trace=False):
    rd = bs.BitstreamReader(path)
    h = rd.header
    recs, traces = [], []
    for i in range(h.n_frames):
        payload, ranges, _ = rd.next_frame()
        if trace:
            ec.TRACE = []
        recs.append(codec.decode_frame(payload, ranges, i, h.gop, h.shapes, (h.height, h.width)))
        if trace:
            traces.append(ec.TRACE); ec.TRACE = None
    rd.close()
    return recs, traces


@torch.no_grad()
def reference_sequence(codec, frames, gop):
    """Logic of the ORIGINAL Pframe.test() / motion_forward() / forward_pair() on CPU:
    uses the networks' forward() (likelihood-based rate, no bitstream)."""
    m, align = codec, Alignment()
    fb, ref, fidx = [], None, 1
    out = []
    for idx, x in enumerate(frames):
        if idx % gop == 0:
            rec, lik, _ = m.if_model(align.align(x))
            rec = align.resume(rec).clamp(0, 1)
            rate = estimate_bpp(lik, input=rec).mean().item()
            fidx = 1
        else:
            if fidx == 1:
                fb = [align.align(ref)]
            ref_al, x_al = align.align(ref), align.align(x)
            if fidx > 1:
                buf = [fb[0], fb[1], fb[2]] if len(fb) == 3 else [fb[0], fb[0], fb[1]]
                pf, pfl = m.MWNet(buf, None, True)
                flow = m.MENet(ref_al, x_al)
                flow_hat, lik_m, _, _ = m.CondMotion(flow, xc=pfl, x2_back=pfl, temporal_cond=pf)
            else:
                flow = m.MENet(ref_al, x_al)
                flow_hat, lik_m = m.Motion(flow)
            warped = m.Resampler(ref_al, flow_hat)
            mc = m.MCNet(ref_al, warped)
            m.MWNet.append_flow(flow_hat)
            rec_al, lik_r, _, _ = m.Residual(x_al, xc=mc, x2_back=mc, temporal_cond=mc)
            rec_al = rec_al.clamp(0, 1)
            fb.append(rec_al)
            rec = align.resume(rec_al)
            rate = estimate_bpp(lik_m + lik_r, input=x).mean().item()
            fidx += 1
        if len(fb) == 4:
            fb.pop(0)
        ref = rec
        out.append((rec, rate))
    return out


# ------------------------------------------------------------------------------ T3/T4/T5
def t3_t4_t5(a):
    print("T3 sequence: encoder -> file -> independent decoder")
    frames = load_frames(a.frames_dir, a.prefix, a.no_frames, a.start_index)
    ck = a.checkpoint
    if ck is None:
        torch.manual_seed(0)
        tmp = BaseLayerCodec("enc", root=str(ROOT))
        ck = os.path.join(tempfile.gettempdir(), "rand_ckpt.pth.tar")
        torch.save({"state_dict": {"module." + k: v for k, v in tmp.state_dict().items()}}, ck)
        print("  (no --checkpoint: RANDOM weights -> mechanics only, quality numbers are meaningless)")
    enc = BaseLayerCodec("enc", root=str(ROOT)); enc.load_checkpoint(ck)
    dec = BaseLayerCodec("dec", root=str(ROOT)); dec.load_checkpoint(ck)
    check("T3 decoder has no motion-estimation network", not hasattr(dec, "MENet"))
    hdr, outs, tr_e = encode_sequence(enc, frames, a.gop, trace=True)
    path = os.path.join(tempfile.gettempdir(), "test_seq.vcmb")
    sizes = write_file(path, hdr, outs)
    recs_d, tr_d = decode_file(dec, path, trace=True)
    lat_ok = True
    for i, (te, td) in enumerate(zip(tr_e, tr_d)):
        same = len(te) == len(td) and all(t1 == t2 and torch.equal(x, y) for (t1, x), (t2, y) in zip(te, td))
        lat_ok &= same
        if not same:
            print(f"    latent mismatch at frame {i}")
    check("T3 decoded latents == encoded (quantised) latents, all sub-streams", lat_ok,
          f"({sum(len(t) for t in tr_e)} sub-streams)")
    exact = all(torch.equal(r["rec"], d) for r, d in zip(outs, recs_d))
    mx = max((r["rec"] - d).abs().max().item() for r, d in zip(outs, recs_d))
    check("T3 decoder reconstruction bit-exact vs encoder-side reconstruction", exact, f"(max abs diff {mx:.2e})")
    real = sizes["total_bytes"]
    print(f"  file: {real} bytes, payload bytes per frame: {[len(r['payload']) for r in outs]}")

    print("T4 equivalence with the original pipeline (forward() + estimate_bpp)")
    ref_model = BaseLayerCodec("enc", root=str(ROOT)); ref_model.load_checkpoint(ck)
    ref = reference_sequence(ref_model, frames, a.gop)
    d_rec = max((rr - d).abs().max().item() for (rr, _), d in zip(ref, recs_d))
    check("T4 decoded frames vs original-pipeline reconstructions", d_rec < a.tol_rec, f"(max abs diff {d_rec:.2e}, tol {a.tol_rec})")
    H, W = frames[0].shape[-2:]
    for i, ((_, rate_ref), r) in enumerate(zip(ref, outs)):
        mine = sum(r["est_bits"].values()) / (H * W)
        rel = abs(mine - rate_ref) / max(rate_ref, 1e-12)
        print(f"    frame {i} ({r['type']}): bpp_est(original)={rate_ref:.5f}  bpp_est(mine)={mine:.5f}  "
              f"real payload={8 * len(r['payload']) / (H * W):.5f}")
        check(f"T4 frame {i} estimated bpp matches original estimate_bpp", rel < a.tol_bpp, f"(rel diff {rel:.2e})")

    print("T5 corruption: flip one payload byte of frame 0 -> output must change")
    raw = bytearray(open(path, "rb").read())
    rd = bs.BitstreamReader(path)
    off = rd.header_bytes
    pl, rg, n = rd.next_frame(); rd.close()
    pos = off + (n - len(pl)) + len(pl) // 2
    raw[pos] ^= 0x5A
    bad = path + ".bad"
    open(bad, "wb").write(bytes(raw))
    dec2 = BaseLayerCodec("dec", root=str(ROOT)); dec2.load_checkpoint(ck)
    try:
        bad_recs, _ = decode_file(dec2, bad)
        changed = any(not torch.equal(r["rec"], d) for r, d in zip(outs, bad_recs))
        check("T5 corrupted stream decodes to a different result", changed)
    except Exception as e:  # a crash on a corrupted stream is also acceptable evidence of dependence
        check("T5 corrupted stream is rejected / differs", True, f"({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--no_frames", type=int, default=4)
    ap.add_argument("--gop", type=int, default=3)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tol_rec", type=float, default=1e-4, help="max abs diff allowed vs original forward() path")
    ap.add_argument("--tol_bpp", type=float, default=1e-3, help="relative diff allowed on estimated bpp")
    ap.add_argument("--skip_sequence", action="store_true")
    a = ap.parse_args()
    t1_backend()
    t2_container()
    if not a.skip_sequence:
        if a.frames_dir is None:
            sys.exit("--frames_dir required (or --skip_sequence)")
        t3_t4_t5(a)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
