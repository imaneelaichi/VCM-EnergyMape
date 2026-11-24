
"""
entropy_codec.py -- real entropy coding for the Base Layer of hadipardis/svc.

Backend: constriction.
Diagnostic version: PMF validation for Gaussian entropy coding.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import List, Optional, Tuple

import constriction
import numpy as np
import torch

from functional import lower_bound


_FAMILY = constriction.stream.model.Categorical(perfect=False)

TRACE: Optional[list] = None

_MAX_BLOCK_ELEMS = 4_000_000


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _stage(prof, name):
    """Return a profiler context manager (or a no-op)."""
    return prof.stage(name) if prof is not None else nullcontext()


def _trace(tag, t):
    if TRACE is not None:
        TRACE.append((tag, t.detach().clone()))


def _alphabet(sym: torch.Tensor) -> Tuple[int, int]:
    lo = int(sym.min().item())
    hi = int(sym.max().item())

    if hi == lo:
        hi = lo + 1

    return lo, hi


def _bits_from_lik(lik: torch.Tensor, bound: float = 1e-9) -> float:
    """Ideal code length in bits."""
    return float(
        -(lower_bound(lik, bound).log().sum().item()) / np.log(2.0)
    )


def _diagnose_gaussian_pmf(
    probs: np.ndarray,
    scale_flat: torch.Tensor,
    sym_np: np.ndarray,
    a: int,
    b: int,
    lo: int,
    hi: int,
    tag: str,
):
    """
    Diagnostic only.
    Does not modify the probability distribution.
    """

    row_sums = probs.sum(axis=1)

    positive = probs[probs > 0]

    scale_block = scale_flat[a:b].detach().cpu().numpy()

    print("\n========== PMF DIAGNOSTIC ==========", flush=True)

    print("Tag:", tag, flush=True)
    print("Alphabet:", lo, hi, "K:", hi - lo + 1, flush=True)
    print("Block:", a, b, flush=True)

    print("Shape:", probs.shape, flush=True)
    print("Dtype:", probs.dtype, flush=True)
    print("Contiguous:", probs.flags["C_CONTIGUOUS"], flush=True)

    print("All finite:", np.isfinite(probs).all(), flush=True)
    print("Negative count:", np.sum(probs < 0), flush=True)
    print("Zero count:", np.sum(probs == 0), flush=True)
    print("Positive count:", positive.size, flush=True)

    print(
        "Minimum positive probability:",
        positive.min() if positive.size else None,
        flush=True,
    )

    print(
        "Maximum probability:",
        probs.max() if probs.size else None,
        flush=True,
    )

    print(
        "Minimum row sum:",
        row_sums.min() if row_sums.size else None,
        flush=True,
    )

    print(
        "Maximum row sum:",
        row_sums.max() if row_sums.size else None,
        flush=True,
    )

    print(
        "Rows with sum <= 0:",
        np.sum(row_sums <= 0),
        flush=True,
    )

    print(
        "Rows with non-finite sum:",
        np.sum(~np.isfinite(row_sums)),
        flush=True,
    )

    print(
        "Scale minimum:",
        scale_block.min() if scale_block.size else None,
        flush=True,
    )

    print(
        "Scale maximum:",
        scale_block.max() if scale_block.size else None,
        flush=True,
    )

    print(
        "Symbols min/max:",
        sym_np[a:b].min() if b > a else None,
        sym_np[a:b].max() if b > a else None,
        flush=True,
    )

    print("====================================\n", flush=True)

    # Stop before calling Constriction if the PMF is mathematically invalid.
    if not np.isfinite(probs).all():
        raise ValueError(
            f"[{tag}] PMF contains NaN or Inf."
        )

    if np.any(probs < 0):
        raise ValueError(
            f"[{tag}] PMF contains negative probabilities."
        )

    if np.any(~np.isfinite(row_sums)):
        raise ValueError(
            f"[{tag}] PMF contains non-finite row sums."
        )

    if np.any(row_sums <= 0):
        raise ValueError(
            f"[{tag}] At least one PMF row has a sum <= 0."
        )

    if sym_np.size and (
        sym_np[a:b].min() < 0
        or sym_np[a:b].max() >= probs.shape[1]
    ):
        raise ValueError(
            f"[{tag}] Encoded symbol outside PMF alphabet."
        )


# --------------------------------------------------------------------------------------
# Frame encoder / decoder
# --------------------------------------------------------------------------------------

class FrameEncoder:
    """
    One range-coder stream per frame.
    """

    def __init__(self):
        self.enc = constriction.stream.queue.RangeEncoder()

        self.ranges: List[Tuple[int, int]] = []
        self.est_bits: dict = {}
        self.coded_bits: dict = {}
        self.shapes: dict = {}

    def mark(self, tag: str, bits_before: int, est_bits: float, shape):
        self.coded_bits[tag] = self.enc.num_bits() - bits_before
        self.est_bits[tag] = est_bits
        self.shapes[tag] = tuple(int(s) for s in shape)

    def finish(self) -> bytes:
        words = self.enc.get_compressed()
        return np.asarray(words, dtype="<u4").tobytes()


class FrameDecoder:

    def __init__(self, payload: bytes, ranges: List[Tuple[int, int]]):

        words = np.frombuffer(payload, dtype="<u4").copy()

        self.dec = constriction.stream.queue.RangeDecoder(words)

        self._ranges = list(ranges)
        self._i = 0

    def next_range(self) -> Tuple[int, int]:

        r = self._ranges[self._i]
        self._i += 1

        return r

    def fully_consumed(self) -> bool:
        return self._i == len(self._ranges)


# --------------------------------------------------------------------------------------
# 1) Factorized prior
# --------------------------------------------------------------------------------------

@torch.no_grad()
def _factorized_probs(bn, lo: int, hi: int, C: int) -> np.ndarray:

    K = hi - lo + 1

    s = torch.arange(
        lo,
        hi + 1,
        dtype=torch.float32
    ).view(1, 1, K, 1)

    inp = s + bn.mean.to(torch.float32).view(1, C, 1, 1)

    lik = bn._likelihood(inp)

    return (
        lik.view(C, K)
        .clamp_min(0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


@torch.no_grad()
def encode_factorized(
    fe: FrameEncoder,
    bn,
    z: torch.Tensor,
    tag: str,
    prof=None
) -> torch.Tensor:

    assert z.dim() == 4 and z.size(0) == 1

    C = z.size(1)

    bits0 = fe.enc.num_bits()

    with _stage(prof, f"{tag}_quantize"):

        mean = bn.mean.view(1, C, 1, 1)

        sym = torch.round(z - mean)

        lo, hi = _alphabet(sym)

        z_hat = sym.float() + mean

    with _stage(prof, f"{tag}_pmf"):

        probs = _factorized_probs(bn, lo, hi, C)

        sym_np = (
            (sym - lo)
            .to(torch.int64)
            .view(C, -1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        est = _bits_from_lik(bn._likelihood(z_hat))

    with _stage(prof, f"{tag}_rangecode"):

        for c in range(C):

            model = constriction.stream.model.Categorical(
                probs[c],
                perfect=False
            )

            fe.enc.encode(sym_np[c], model)

    fe.ranges.append((lo, hi))

    fe.mark(tag, bits0, est, z.shape)

    _trace(tag, z_hat)

    return z_hat


@torch.no_grad()
def decode_factorized(
    fd: FrameDecoder,
    bn,
    shape,
    tag: str,
    prof=None
) -> torch.Tensor:

    _, C, H, W = shape

    lo, hi = fd.next_range()

    with _stage(prof, f"{tag}_pmf"):

        probs = _factorized_probs(bn, lo, hi, C)

    with _stage(prof, f"{tag}_rangedecode"):

        out = np.empty((C, H * W), dtype=np.int32)

        for c in range(C):

            model = constriction.stream.model.Categorical(
                probs[c],
                perfect=False
            )

            out[c] = fd.dec.decode(model, H * W)

    with _stage(prof, f"{tag}_dequantize"):

        sym = (
            torch.from_numpy(out.astype(np.int64) + lo)
            .view(1, C, H, W)
        )

        z_hat = sym.float() + bn.mean.view(1, C, 1, 1)

    _trace(tag, z_hat)

    return z_hat


# --------------------------------------------------------------------------------------
# 2) Gaussian conditional
# --------------------------------------------------------------------------------------

@torch.no_grad()
def _gauss_probs(
    cond_model,
    scale_flat: torch.Tensor,
    lo: int,
    hi: int
) -> np.ndarray:

    s = torch.arange(
        lo,
        hi + 1,
        dtype=torch.float32
    ).view(1, -1)

    values = -s.abs()

    sc = scale_flat.view(-1, 1)

    upper = cond_model._standardized_cumulative(
        (values + 0.5) / sc
    )

    lower = cond_model._standardized_cumulative(
        (values - 0.5) / sc
    )

    return (
        (upper - lower)
        .clamp_min(0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _blocks(n: int, K: int):

    step = max(1, _MAX_BLOCK_ELEMS // max(K, 1))

    for a in range(0, n, step):

        yield a, min(n, a + step)


@torch.no_grad()
def encode_gaussian(
    fe: FrameEncoder,
    cond_model,
    y: torch.Tensor,
    condition: torch.Tensor,
    tag: str,
    prof=None
) -> torch.Tensor:

    bits0 = fe.enc.num_bits()

    with _stage(prof, f"{tag}_quantize"):

        cond_model._set_condition(condition)

        mean, scale = cond_model.mean, cond_model.scale

        sym = torch.round(y - mean)

        lo, hi = _alphabet(sym)

        y_hat = sym.float() + mean

        sym_np = (
            (sym - lo)
            .to(torch.int64)
            .view(-1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        scale_flat = scale.reshape(-1)

        est = _bits_from_lik(cond_model._likelihood(y_hat))

    K = hi - lo + 1

    t_pmf = 0.0
    t_rc = 0.0

    for a, b in _blocks(sym_np.shape[0], K):

        t0 = time.perf_counter()

        probs = _gauss_probs(
            cond_model,
            scale_flat[a:b],
            lo,
            hi
        )

        # --------------------------------------------------------------
        # PMF DIAGNOSTIC
        # --------------------------------------------------------------

        _diagnose_gaussian_pmf(
            probs=probs,
            scale_flat=scale_flat,
            sym_np=sym_np,
            a=a,
            b=b,
            lo=lo,
            hi=hi,
            tag=tag,
        )

        t1 = time.perf_counter()

        # Keep the original distribution unchanged.
        fe.enc.encode(
            sym_np[a:b],
            _FAMILY,
            probs
        )

        t2 = time.perf_counter()

        t_pmf += t1 - t0
        t_rc += t2 - t1

    if prof is not None:

        prof.add(f"{tag}_pmf", t_pmf)

        prof.add(f"{tag}_rangecode", t_rc)

    fe.ranges.append((lo, hi))

    fe.mark(tag, bits0, est, y.shape)

    _trace(tag, y_hat)

    return y_hat


@torch.no_grad()
def decode_gaussian(
    fd: FrameDecoder,
    cond_model,
    shape,
    condition: torch.Tensor,
    tag: str,
    prof=None
) -> torch.Tensor:

    lo, hi = fd.next_range()

    cond_model._set_condition(condition)

    mean, scale = cond_model.mean, cond_model.scale

    scale_flat = scale.reshape(-1)

    n = scale_flat.numel()

    K = hi - lo + 1

    out = np.empty(n, dtype=np.int32)

    t_pmf = 0.0
    t_rc = 0.0

    for a, b in _blocks(n, K):

        t0 = time.perf_counter()

        probs = _gauss_probs(
            cond_model,
            scale_flat[a:b],
            lo,
            hi
        )

        t1 = time.perf_counter()

        out[a:b] = fd.dec.decode(
            _FAMILY,
            probs
        )

        t2 = time.perf_counter()

        t_pmf += t1 - t0
        t_rc += t2 - t1

    if prof is not None:

        prof.add(f"{tag}_pmf", t_pmf)

        prof.add(f"{tag}_rangedecode", t_rc)

    with _stage(prof, f"{tag}_dequantize"):

        sym = (
            torch.from_numpy(out.astype(np.int64) + lo)
            .view(*shape)
        )

        y_hat = sym.float() + mean

    _trace(tag, y_hat)

    return y_hat


# --------------------------------------------------------------------------------------
# 3) Autoregressive context + Gaussian mixture
# --------------------------------------------------------------------------------------

@torch.no_grad()
def _ctx_probs(
    ctx,
    padded: torch.Tensor,
    phi: torch.Tensor,
    h: int,
    w: int,
    samples: torch.Tensor,
    prof_acc: dict
) -> np.ndarray:

    t0 = time.perf_counter()

    patch = ctx.mask.crop(padded, (h, w))

    patch_phi = ctx.mask.crop(phi, (h, w), 1)

    ctx._set_condition(
        patch,
        patch_phi,
        padding=False
    )

    t1 = time.perf_counter()

    lik = ctx.entropy_model._likelihood(samples)

    probs = (
        lik.view(lik.size(0), -1)
        .t()
        .clamp_min(0)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    t2 = time.perf_counter()

    prof_acc["ctx_net"] += t1 - t0

    prof_acc["pmf"] += t2 - t1

    return probs


@torch.no_grad()
def encode_context_gmm(
    fe: FrameEncoder,
    ctx,
    y: torch.Tensor,
    phi: torch.Tensor,
    tag: str,
    prof=None
) -> torch.Tensor:

    assert y.size(0) == 1

    _, C, H, W = y.shape

    bits0 = fe.enc.num_bits()

    with _stage(prof, f"{tag}_quantize"):

        sym = ctx.entropy_model.quantize(
            y,
            "symbols"
        ).to(torch.int64)

        lo, hi = _alphabet(sym)

        sym_idx = (
            (sym - lo)
            .cpu()
            .numpy()
            .astype(np.int32)
        )

    K = hi - lo + 1

    samples = torch.arange(
        lo,
        hi + 1,
        dtype=torch.float32
    ).view(-1, 1, 1, 1)

    padded = ctx.mask.pad(
        torch.zeros(1, C, H, W)
    )

    pad = ctx.padding

    acc = {
        "ctx_net": 0.0,
        "pmf": 0.0,
        "rc": 0.0
    }

    est = 0.0

    for h in range(H):

        for w in range(W):

            probs = _ctx_probs(
                ctx,
                padded,
                phi,
                h,
                w,
                samples,
                acc
            )

            t0 = time.perf_counter()

            fe.enc.encode(
                sym_idx[0, :, h, w],
                _FAMILY,
                probs
            )

            acc["rc"] += time.perf_counter() - t0

            padded[
                0,
                :,
                h + pad,
                w + pad
            ] = torch.from_numpy(
                (
                    sym_idx[0, :, h, w] + lo
                ).astype(np.float32)
            )

            lik = ctx.entropy_model._likelihood(
                sym[0, :, h, w]
                .float()
                .view(1, C, 1, 1)
            ).view(-1)

            est += _bits_from_lik(lik)

    y_hat = padded[
        :,
        :,
        pad:pad + H,
        pad:pad + W
    ].clone()

    if prof is not None:

        prof.add(
            f"{tag}_context_net",
            acc["ctx_net"]
        )

        prof.add(
            f"{tag}_pmf",
            acc["pmf"]
        )

        prof.add(
            f"{tag}_rangecode",
            acc["rc"]
        )

    fe.ranges.append((lo, hi))

    fe.mark(tag, bits0, est, y.shape)

    out = ctx.entropy_model.dequantize(y_hat)

    _trace(tag, out)

    return out


@torch.no_grad()
def decode_context_gmm(
    fd: FrameDecoder,
    ctx,
    shape,
    phi: torch.Tensor,
    tag: str,
    prof=None
) -> torch.Tensor:

    _, C, H, W = shape

    lo, hi = fd.next_range()

    samples = torch.arange(
        lo,
        hi + 1,
        dtype=torch.float32
    ).view(-1, 1, 1, 1)

    padded = ctx.mask.pad(
        torch.zeros(1, C, H, W)
    )

    pad = ctx.padding

    acc = {
        "ctx_net": 0.0,
        "pmf": 0.0,
        "rc": 0.0
    }

    for h in range(H):

        for w in range(W):

            probs = _ctx_probs(
                ctx,
                padded,
                phi,
                h,
                w,
                samples,
                acc
            )

            t0 = time.perf_counter()

            col = fd.dec.decode(
                _FAMILY,
                probs
            )

            acc["rc"] += time.perf_counter() - t0

            padded[
                0,
                :,
                h + pad,
                w + pad
            ] = torch.from_numpy(
                (
                    col.astype(np.int64) + lo
                ).astype(np.float32)
            )

    if prof is not None:

        prof.add(
            f"{tag}_context_net",
            acc["ctx_net"]
        )

        prof.add(
            f"{tag}_pmf",
            acc["pmf"]
        )

        prof.add(
            f"{tag}_rangedecode",
            acc["rc"]
        )

    out = ctx.entropy_model.dequantize(
        padded[
            :,
            :,
            pad:pad + H,
            pad:pad + W
        ].clone()
    )

    _trace(tag, out)

    return out
