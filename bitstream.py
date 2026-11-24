"""
bitstream.py -- container format for the Base Layer bitstream (.vcmb).

Why not the original `Utils.BitStreamIO`?  It concatenates strings separated by a
4-byte "magic" sequence and parses them back with split()/readline(); arbitrary
binary payloads may contain that sequence, so reading is not guaranteed to be
correct.  Here every field is length-prefixed (LEB128 varints).

Layout (all integers little-endian / LEB128):

  SEQUENCE HEADER  (written once)
    magic      4 bytes  b'VCMB'
    version    u8
    flags      u8       bit0 = closed_gop (MWNet buffers reset at each I-frame)
    width      u16      original (un-padded) frame width
    height     u16      original frame height
    n_frames   varint
    gop        varint   intra period
    ckpt_id    u8       base-layer checkpoint index (informative)
    n_shapes   u8
    n_shapes x { tag_len u8, tag bytes, C varint, H varint, W varint }
               shapes of every latent sub-stream (needed to decode)

  PER FRAME
    payload_len varint
    n_ranges    varint
    n_ranges x { zigzag(lo) varint, (hi-lo) varint }   alphabet of each sub-stream
    payload     payload_len bytes (one range-coder stream, 32-bit words)

Frame type is NOT stored: it is implied by (frame_index % gop == 0 -> I-frame).
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Tuple

MAGIC = b"VCMB"
VERSION = 1


def _w_varint(buf: bytearray, v: int):
    assert v >= 0
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _r_varint(f: BinaryIO) -> int:
    shift = v = 0
    while True:
        c = f.read(1)
        if not c:
            raise EOFError("truncated bitstream")
        b = c[0]
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v
        shift += 7


def _zz(v: int) -> int:
    return (v << 1) ^ (v >> 63) if v < 0 else v << 1


def _unzz(u: int) -> int:
    return (u >> 1) ^ -(u & 1)


@dataclass
class SequenceHeader:
    width: int
    height: int
    n_frames: int
    gop: int
    ckpt_id: int = 0
    closed_gop: bool = False
    shapes: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)  # tag -> (C,H,W)

    def to_bytes(self) -> bytes:
        b = bytearray(MAGIC)
        b += struct.pack("<BBHH", VERSION, 1 if self.closed_gop else 0, self.width, self.height)
        _w_varint(b, self.n_frames)
        _w_varint(b, self.gop)
        b += struct.pack("<BB", self.ckpt_id, len(self.shapes))
        for tag, (c, h, w) in self.shapes.items():
            t = tag.encode()
            b += struct.pack("<B", len(t)) + t
            _w_varint(b, c); _w_varint(b, h); _w_varint(b, w)
        return bytes(b)

    @staticmethod
    def from_file(f: BinaryIO) -> "SequenceHeader":
        if f.read(4) != MAGIC:
            raise ValueError("not a VCMB bitstream")
        ver, flags, w, h = struct.unpack("<BBHH", f.read(6))
        if ver != VERSION:
            raise ValueError(f"unsupported version {ver}")
        n_frames = _r_varint(f)
        gop = _r_varint(f)
        ckpt, n_shapes = struct.unpack("<BB", f.read(2))
        shapes = {}
        for _ in range(n_shapes):
            tl = f.read(1)[0]
            tag = f.read(tl).decode()
            shapes[tag] = (_r_varint(f), _r_varint(f), _r_varint(f))
        return SequenceHeader(w, h, n_frames, gop, ckpt, bool(flags & 1), shapes)


def frame_record(payload: bytes, ranges: List[Tuple[int, int]]) -> bytes:
    b = bytearray()
    _w_varint(b, len(payload))
    _w_varint(b, len(ranges))
    for lo, hi in ranges:
        _w_varint(b, _zz(lo))
        _w_varint(b, hi - lo)
    return bytes(b) + payload


def read_frame_record(f: BinaryIO) -> Tuple[bytes, List[Tuple[int, int]], int]:
    """Returns (payload, ranges, total_record_bytes)."""
    start = f.tell()
    n = _r_varint(f)
    nr = _r_varint(f)
    ranges = []
    for _ in range(nr):
        lo = _unzz(_r_varint(f))
        ranges.append((lo, lo + _r_varint(f)))
    payload = f.read(n)
    if len(payload) != n:
        raise EOFError("truncated frame payload")
    return payload, ranges, f.tell() - start


class BitstreamWriter:
    """Writes the header lazily (shapes are known only after the first I and P frames)."""

    def __init__(self, path: str, header: SequenceHeader):
        self.path = path
        self.header = header
        self.records: List[bytes] = []

    def add_frame(self, payload: bytes, ranges) -> int:
        rec = frame_record(payload, ranges)
        self.records.append(rec)
        return len(rec)

    def close(self) -> Dict[str, int]:
        head = self.header.to_bytes()
        with open(self.path, "wb") as f:
            f.write(head)
            for r in self.records:
                f.write(r)
        return {"header_bytes": len(head), "frames_bytes": sum(map(len, self.records)),
                "total_bytes": len(head) + sum(map(len, self.records))}


class BitstreamReader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.header = SequenceHeader.from_file(self.f)
        self.header_bytes = self.f.tell()

    def next_frame(self):
        return read_frame_record(self.f)

    def close(self):
        self.f.close()
