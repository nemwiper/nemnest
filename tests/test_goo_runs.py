"""The run-length decoder of .goo layers: Python and compiled give the same runs."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from nemnest import goo


def encode(runs, rng):
    """Runs (length, grey value) to .goo layer bytes, using every kind of token."""
    out = bytearray()
    color = 0
    for length, value in runs:
        diff = value - color
        if value not in (0, 255) and 0 < abs(diff) <= 15 and length <= 255 and rng.random() < 0.6:
            head = 0x80 | (0x20 if diff < 0 else 0) | abs(diff)        # previous colour +/- a small difference
            if length == 1 and rng.random() < 0.5:
                out.append(head)
            else:
                out += bytes([head | 0x10, length])
        else:
            kind = 0 if value == 0 else 3 if value == 255 else 1
            size = 0 if length < 16 else 1 if length < 1 << 12 else 2 if length < 1 << 20 else 3
            out.append((kind << 6) | (size << 4) | (length & 0x0F))
            if kind == 1:
                out.append(value)
            if size == 1:
                out.append(length >> 4)
            elif size == 2:
                out += bytes([(length >> 12) & 0xFF, (length >> 4) & 0xFF])
            elif size == 3:
                out += bytes([(length >> 20) & 0xFF, (length >> 12) & 0xFF, (length >> 4) & 0xFF])
        color = value
    return bytes(out)


def samples(count=200):
    rng = random.Random(3)
    lengths = [1, 2, 7, 15, 16, 200, 255, 256, 4095, 4096, 70000, (1 << 20) + 5, 3000000]
    greys = [0, 255, 255, 0, 1, 7, 100, 128, 250, 254]
    for _ in range(count):
        runs = [(rng.choice(lengths), rng.choice(greys)) for _ in range(rng.randint(1, 400))]
        yield runs, encode(runs, rng)


def test_python_decoder_reads_every_token_kind():
    for runs, data in samples():
        lengths, values = goo.decode_runs_python(data)
        assert lengths.dtype == np.int64 and values.dtype == np.uint8
        assert lengths.tolist() == [r[0] for r in runs] and values.tolist() == [r[1] for r in runs]
    for cut in (b"\x10", b"\x40", b"\x90", b"\x30\x00", b"\x70\x05"):     # a token that needs more bytes than there are
        with pytest.raises(ValueError):
            goo.decode_runs_python(cut)


def test_compiled_decoder_matches_python():
    if not goo.COMPILED:
        if os.environ.get("NEMNEST_REQUIRE_COMPILED"):
            pytest.fail("the compiled run decoder was not built or cannot be loaded")
        pytest.skip("compiled run decoder not built on this machine")
    for _runs, data in samples():
        fast, slow = goo.decode_runs(data), goo.decode_runs_python(data)
        assert fast[0].dtype == slow[0].dtype and fast[1].dtype == slow[1].dtype
        assert fast[0].tolist() == slow[0].tolist() and fast[1].tolist() == slow[1].tolist()
    assert len(goo.decode_runs(b"")[0]) == 0
    for cut in (b"\x10", b"\x40", b"\x90", b"\x30\x00", b"\x70\x05"):     # a token that needs more bytes than there are
        with pytest.raises(ValueError):
            goo.decode_runs(cut)
