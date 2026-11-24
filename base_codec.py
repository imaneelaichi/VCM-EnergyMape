"""
base_codec.py -- Base-Layer encoder / decoder with a REAL entropy-coded bitstream.

It re-uses the original network modules (same attribute names, so the original
checkpoints load unchanged) and reproduces the frame-level logic of
`Pframe.test()` / `motion_forward()` / `forward_pair()` of test_base.py, but
replaces "estimate the rate from likelihoods" by
    encode: quantise -> range-code -> bytes
    decode: bytes -> range-decode -> dequantise -> decoder networks.

Decoder-side information (what the decoder may use): the bitstream, the sequence
header, the decoded previous frames and previously decoded flows (MWNet buffers).
The decoder object does NOT contain the motion-estimation network (PWC-Net / MENet),
which is encoder-only, and never receives the original frames or any latent kept
in encoder memory.

Frame logic replicated from the original (including its quirks):
  * I-frame when idx % gop == 0 (ANF coder + autoregressive context + GMM).
  * first P-frame after an I-frame: unconditional motion coder (Motion);
    next P-frames: CondMotion conditioned on MWNet extrapolation.
  * the reference of frame t is the *cropped, clamped, float* reconstruction of
    frame t-1 (re-padded with replicate padding), the 3-frame buffer holds the
    *aligned* reconstructions.
  * MWNet._flow_buffer is NOT reset at I-frames in the original test script
    (flows of the previous GOP leak into the first extrapolation of the next
    GOP). This is decoder-reproducible but means a GOP is not independently
    decodable. closed_gop=True resets it (documented deviation, off by default).
"""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch import nn

import entropy_codec as ec
from entropy_codec import FrameDecoder, FrameEncoder
from flownets import PWCNet
from Models import Refinement
from SDCNet import MotionExtrapolationNet
from Utils import Alignment
from networks_canf import __CODER_TYPES__, AugmentedNormalizedFlowHyperPriorCoder
from util.sampler import Resampler

IGNORED_PREFIXES_ENC = ("feature_extractor.",)           # YOLO front-end clone: only needed for training
IGNORED_PREFIXES_DEC = ("feature_extractor.", "MENet.")  # decoder has no motion estimation


@contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _build_coder(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    return __CODER_TYPES__[cfg["model_architecture"]](**cfg["model_params"])


def _stage(prof, name):
    return prof.stage(name) if prof is not None else nullcontext()


class BaseLayerCodec(nn.Module):
    def __init__(self, mode: str, root: str = ".", closed_gop: bool = False):
        super().__init__()
        assert mode in ("enc", "dec")
        self.mode = mode
        self.closed_gop = closed_gop
        self.root = os.path.abspath(root)
        with _chdir(self.root):   # original modules load ./models/*.pth(.tar|.ckpt) with relative paths
            self.if_model = AugmentedNormalizedFlowHyperPriorCoder(
                128, 320, 192, num_layers=2, use_QE=True, use_affine=False, use_context=True,
                condition="GaussianMixtureModel", quant_mode="round")
            if mode == "enc":
                self.MENet = PWCNet(trainable=True)
            self.MWNet = MotionExtrapolationNet(sequence_length=3)
            self.MWNet.__delattr__("flownet")
            self.Motion = _build_coder("./config/DVC_motion.yml")
            self.CondMotion = _build_coder("./config/CANF_motion_predprior.yml")
            self.Resampler = Resampler()
            self.MCNet = Refinement(6, 64, out_channels=3)
            self.Residual = _build_coder("./config/CANF_inter_coder.yml")
        self.align = Alignment()
        self.eval()
        self.reset_sequence()

    # ------------------------------------------------------------------ checkpoint
    def load_checkpoint(self, path: str) -> Dict[str, List[str]]:
        ck = torch.load(path, map_location="cpu")
        sd = ck["state_dict"] if "state_dict" in ck else ck
        ign = IGNORED_PREFIXES_DEC if self.mode == "dec" else IGNORED_PREFIXES_ENC
        clean = {}
        ignored = []
        for k, v in sd.items():
            k2 = k[7:] if k.startswith("module.") else k
            if k2.startswith(ign):
                ignored.append(k2)
                continue
            clean[k2] = v
        missing, unexpected = self.load_state_dict(clean, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch. missing={missing[:5]}... ({len(missing)}) "
                               f"unexpected={unexpected[:5]}... ({len(unexpected)})")
        return {"ignored_prefix_keys": ignored}

    # ------------------------------------------------------------------ sequence state
    def reset_sequence(self):
        self._frame_buffer: List[torch.Tensor] = []   # aligned reconstructions (<=3)
        self._ref: Optional[torch.Tensor] = None      # cropped, clamped reconstruction of previous frame
        self._p_order = 0
        self.MWNet.clear_buffer(None)

    def _on_new_I(self):
        self._p_order = 1
        if self.closed_gop:
            self.MWNet.clear_buffer(None)

    def _mwnet_inputs(self):
        fb = self._frame_buffer
        assert len(fb) in (2, 3)
        return [fb[0], fb[1], fb[2]] if len(fb) == 3 else [fb[0], fb[0], fb[1]]

    def _push_recon(self, rec_aligned: torch.Tensor):
        self._frame_buffer.append(rec_aligned)
        if len(self._frame_buffer) == 4:
            self._frame_buffer.pop(0)

    # ------------------------------------------------------------------ coder wrappers (encode)
    @torch.no_grad()
    def _enc_hp(self, fe, m, x, tag, prof):
        """GoogleHyperPriorCoder (first P-frame motion)."""
        with _stage(prof, f"{tag}_analysis"):
            feats = m.analysis(x)
        with _stage(prof, f"{tag}_hyper_analysis"):
            hyper = m.hyper_analysis(feats.abs() if m.use_abs else feats)
        z_hat = ec.encode_factorized(fe, m.entropy_bottleneck, hyper, f"{tag}_z", prof)
        with _stage(prof, f"{tag}_cond_net"):
            cond = m.hyper_synthesis(z_hat)
        y_hat = ec.encode_gaussian(fe, m.conditional_bottleneck, feats, cond, f"{tag}_y", prof)
        with _stage(prof, f"{tag}_synthesis"):
            x_hat = m.synthesis(y_hat)
        return x_hat

    @torch.no_grad()
    def _enc_tp(self, fe, m, x, xc, x2_back, tcond, tag, prof):
        """CANFHyperPriorCoderWithTemporalPrior (CondMotion, Residual)."""
        with _stage(prof, f"{tag}_analysis"):
            _, feats, jac = m.encode(x, None, None, xc=xc)
        with _stage(prof, f"{tag}_hyper_analysis"):
            hyper = m.hyper_analysis(feats.abs() if m.use_abs else feats)
        z_hat = ec.encode_factorized(fe, m.entropy_bottleneck, hyper, f"{tag}_z", prof)
        with _stage(prof, f"{tag}_cond_net"):
            hp = m.hyper_synthesis(z_hat)
            tp = m.pred_prior(tcond)
            cond = m.PA(torch.cat([hp, tp], dim=1))
        y_hat = ec.encode_gaussian(fe, m.conditional_bottleneck, feats, cond, f"{tag}_y", prof)
        with _stage(prof, f"{tag}_synthesis"):
            x_hat, _, _ = m.decode(x2_back, y_hat, None, xc=xc)
            if m.use_QE:
                x_hat = m.DQ(x_hat)
        return x_hat

    # ------------------------------------------------------------------ coder wrappers (decode)
    @torch.no_grad()
    def _dec_hp(self, fd, m, shapes, tag, prof):
        z_hat = ec.decode_factorized(fd, m.entropy_bottleneck, (1,) + tuple(shapes[f"{tag}_z"]), f"{tag}_z", prof)
        with _stage(prof, f"{tag}_cond_net"):
            cond = m.hyper_synthesis(z_hat)
        y_hat = ec.decode_gaussian(fd, m.conditional_bottleneck, (1,) + tuple(shapes[f"{tag}_y"]), cond, f"{tag}_y", prof)
        with _stage(prof, f"{tag}_synthesis"):
            return m.synthesis(y_hat)

    @torch.no_grad()
    def _dec_tp(self, fd, m, shapes, xc, x2_back, tcond, tag, prof):
        z_hat = ec.decode_factorized(fd, m.entropy_bottleneck, (1,) + tuple(shapes[f"{tag}_z"]), f"{tag}_z", prof)
        with _stage(prof, f"{tag}_cond_net"):
            hp = m.hyper_synthesis(z_hat)
            tp = m.pred_prior(tcond)
            cond = m.PA(torch.cat([hp, tp], dim=1))
        y_hat = ec.decode_gaussian(fd, m.conditional_bottleneck, (1,) + tuple(shapes[f"{tag}_y"]), cond, f"{tag}_y", prof)
        with _stage(prof, f"{tag}_synthesis"):
            x_hat, _, _ = m.decode(x2_back, y_hat, None, xc=xc)
            if m.use_QE:
                x_hat = m.DQ(x_hat)
        return x_hat

    # ------------------------------------------------------------------ shared motion compensation
    @torch.no_grad()
    def _motion_compensation(self, ref_al, flow_hat, prof):
        with _stage(prof, "motion_compensation"):
            warped = self.Resampler(ref_al, flow_hat)
            mc = self.MCNet(ref_al, warped)
            self.MWNet.append_flow(flow_hat)
        return mc

    # ------------------------------------------------------------------ ENCODER
    @torch.no_grad()
    def encode_frame(self, x: torch.Tensor, idx: int, gop: int, prof=None) -> dict:
        """x: (1,3,H,W) float in [0,1]. Returns payload bytes + side info + local reconstruction."""
        assert self.mode == "enc"
        fe = FrameEncoder()
        if idx % gop == 0:
            with _stage(prof, "align"):
                x_al = self.align.align(x)
            m = self.if_model
            with _stage(prof, "I_analysis"):
                _, code, _ = m.encode(x_al, None, None)
            with _stage(prof, "I_hyper_analysis"):
                hyper = m.hyper_analysis(code.abs() if m.use_abs else code)
            z_hat = ec.encode_factorized(fe, m.entropy_bottleneck, hyper, "I_z", prof)
            with _stage(prof, "I_cond_net"):
                phi = m.hyper_synthesis(z_hat)
            y_hat = ec.encode_context_gmm(fe, m.conditional_bottleneck, code, phi, "I_y", prof)
            with _stage(prof, "I_synthesis"):
                x_hat = m.decode(None, y_hat, jac=None)[0]
                if m.use_QE:
                    x_hat = m.DQ(x_hat)
            with _stage(prof, "crop_clamp"):
                rec = self.align.resume(x_hat).clamp(0, 1)
            self._on_new_I()
            ftype = "I"
        else:
            with _stage(prof, "align"):
                x_al = self.align.align(x)
                if self._p_order == 1:
                    self._frame_buffer = [self.align.align(self._ref)]
                ref_al = self.align.align(self._ref)
            with _stage(prof, "motion_estimation"):
                flow = self.MENet(ref_al, x_al)
            if self._p_order > 1:
                with _stage(prof, "motion_prediction"):
                    pred_frame, pred_flow = self.MWNet(self._mwnet_inputs(), None, True)
                flow_hat = self._enc_tp(fe, self.CondMotion, flow, pred_flow, pred_flow, pred_frame, "M", prof)
            else:
                flow_hat = self._enc_hp(fe, self.Motion, flow, "M", prof)
            mc = self._motion_compensation(ref_al, flow_hat, prof)
            rec_al = self._enc_tp(fe, self.Residual, x_al, mc, mc, mc, "R", prof).clamp(0, 1)
            self._push_recon(rec_al)
            with _stage(prof, "crop_clamp"):
                rec = self.align.resume(rec_al)
            self._p_order += 1
            ftype = "P"
        self._ref = rec
        return {"type": ftype, "payload": fe.finish(), "ranges": fe.ranges, "rec": rec,
                "est_bits": fe.est_bits, "coded_bits": fe.coded_bits, "shapes": fe.shapes}

    # ------------------------------------------------------------------ DECODER
    @torch.no_grad()
    def decode_frame(self, payload: bytes, ranges, idx: int, gop: int, shapes: dict, hw, prof=None) -> torch.Tensor:
        """Uses only: bytes, ranges, shapes + (H,W) from the sequence header and the decoder's own state."""
        rshape = (1, 3, int(hw[0]), int(hw[1]))
        assert self.mode == "dec"
        fd = FrameDecoder(payload, ranges)
        if idx % gop == 0:
            m = self.if_model
            z_hat = ec.decode_factorized(fd, m.entropy_bottleneck, (1,) + tuple(shapes["I_z"]), "I_z", prof)
            with _stage(prof, "I_cond_net"):
                phi = m.hyper_synthesis(z_hat)
            y_hat = ec.decode_context_gmm(fd, m.conditional_bottleneck, (1,) + tuple(shapes["I_y"]), phi, "I_y", prof)
            with _stage(prof, "I_synthesis"):
                x_hat = m.decode(None, y_hat, jac=None)[0]
                if m.use_QE:
                    x_hat = m.DQ(x_hat)
            with _stage(prof, "crop_clamp"):
                rec = self.align.resume(x_hat, rshape).clamp(0, 1)
            self._on_new_I()
        else:
            with _stage(prof, "align"):
                if self._p_order == 1:
                    self._frame_buffer = [self.align.align(self._ref)]
                ref_al = self.align.align(self._ref)
            if self._p_order > 1:
                with _stage(prof, "motion_prediction"):
                    pred_frame, pred_flow = self.MWNet(self._mwnet_inputs(), None, True)
                flow_hat = self._dec_tp(fd, self.CondMotion, shapes, pred_flow, pred_flow, pred_frame, "M", prof)
            else:
                flow_hat = self._dec_hp(fd, self.Motion, shapes, "M", prof)
            mc = self._motion_compensation(ref_al, flow_hat, prof)
            rec_al = self._dec_tp(fd, self.Residual, shapes, mc, mc, mc, "R", prof).clamp(0, 1)
            self._push_recon(rec_al)
            with _stage(prof, "crop_clamp"):
                rec = self.align.resume(rec_al, rshape)
            self._p_order += 1
        assert fd.fully_consumed(), "decoder did not consume all sub-stream ranges"
        self._ref = rec
        return rec
