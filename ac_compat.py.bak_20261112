import warnings
import numpy as np
import torch
import torchac
import constriction

dtype = torch.int16
_SCALE = 32767          # le CDF entier doit tenir dans int16


# ---------- Partie 1 : bottleneck (torchac, par lot) ----------

def pmf2cdf(pmf):
    """pmf (R, L+1) avec la masse de queue incluse -> cdf int16 (R, L+2)."""
    pmf = pmf.float().clamp_min(0)
    pmf = pmf / pmf.sum(-1, keepdim=True)
    cdf = torch.cumsum(pmf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    cdf = torch.round(cdf * _SCALE)
    cdf[..., -1] = _SCALE
    return cdf.to(torch.int16)


def _rows(cdf, idx_flat):
    return cdf.cpu()[idx_flat].float() / _SCALE        # (N, L+2)


def range_index_encode(symbols, cdf, cdf_length, idx):
    sym = symbols.cpu().long()
    idx = idx.cpu().long().expand(sym.shape)
    lens = cdf_length.cpu().reshape(-1).long()[idx]
    hi = lens - 1
    bad = (sym < 0) | (sym > hi)
    if bad.any():
        warnings.warn(f"ac_compat: {int(bad.sum())} symboles hors intervalle (clampés)")
        sym = torch.minimum(sym.clamp_min(0), hi)
    rows = _rows(cdf, idx.reshape(-1))
    return torchac.encode_float_cdf(
        rows, sym.reshape(-1).to(torch.int16), check_input_bounds=True)


def range_index_decode(strings, cdf, cdf_length, idx):
    rows = _rows(cdf, idx.cpu().long().reshape(-1))
    sym = torchac.decode_float_cdf(rows, strings)
    return sym.to(torch.int16).reshape(idx.shape).to(idx.device)


# ---------- Partie 2 : modèle de contexte (constriction, en flux) ----------

_enc = None
_dec = None
_FAM = constriction.stream.model.Categorical(perfect=False)


def _cdf_to_pmf(cdf):
    """cdf (C, S+1) en short -> pmf float32 (C, S), minimum 1/65536."""
    c = cdf.reshape(-1, cdf.shape[-1]).to(torch.int32) & 0xFFFF
    c[:, -1] = 65536
    c = c[:, 1:]
    pmf = torch.diff(c, dim=1, prepend=torch.zeros_like(c[:, :1]))
    return (pmf.clamp_min(1).float() / 65536.0).numpy()


def range_encode(symbols, cdf):
    global _enc
    if _enc is None:
        _enc = constriction.stream.queue.RangeEncoder()
    sym = symbols.reshape(-1).long().numpy().astype(np.int32)
    _enc.encode(sym, _FAM, _cdf_to_pmf(cdf))


def flush():
    global _enc
    data = _enc.get_compressed().tobytes() if _enc is not None else b""
    _enc = None
    return data


def set_string(string):
    global _dec
    _dec = constriction.stream.queue.RangeDecoder(
        np.frombuffer(string, dtype=np.uint32).copy())


def range_decode(cdf):
    sym = _dec.decode(_FAM, _cdf_to_pmf(cdf))
    return torch.from_numpy(np.asarray(sym, dtype=np.int16))