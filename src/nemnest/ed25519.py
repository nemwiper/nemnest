"""Ed25519 signature verification (RFC 8032, section 5.1), pure Python.

Used to check the signature of release lists in the standalone programs,
where a compiled crypto library is one more thing that can fail to load
(it did, on Intel Macs). Verifying one signature takes a few milliseconds.
Only verification is here: releases are signed on the maintainer's
computer with the `cryptography` package (packaging/sign_release.py).
The tests compare this code with `cryptography` and with the RFC's test
vectors.
"""
from __future__ import annotations

import hashlib

P = 2 ** 255 - 19
Q = 2 ** 252 + 27742317777372353535851937790883648493       # order of the base point
D = -121665 * pow(121666, P - 2, P) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)


def _add(a, b):
    """Sum of two points in extended coordinates (X, Y, Z, T)."""
    A = (a[1] - a[0]) * (b[1] - b[0]) % P
    B = (a[1] + a[0]) * (b[1] + b[0]) % P
    C = 2 * a[3] * b[3] * D % P
    Dd = 2 * a[2] * b[2] % P
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % P, G * H % P, F * G % P, E * H % P)


def _mul(s: int, point):
    result = (0, 1, 1, 0)                                     # neutral element
    while s > 0:
        if s & 1:
            result = _add(result, point)
        point = _add(point, point)
        s >>= 1
    return result


def _equal(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % P == 0 and (a[1] * b[2] - b[1] * a[2]) % P == 0


def _recover_x(y: int, sign: int):
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(D * y * y + 1, P - 2, P)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if (x & 1) != sign:
        x = P - x
    return x


def _decompress(data: bytes):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % P)


_GY = 4 * pow(5, P - 2, P) % P
G = (_recover_x(_GY, 0), _GY, 1, _recover_x(_GY, 0) * _GY % P)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """True if `signature` (64 bytes) is valid for `message` and the 32-byte public key."""
    if len(public) != 32 or len(signature) != 64:
        return False
    A = _decompress(public)
    R = _decompress(signature[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= Q:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public + message).digest(), "little") % Q
    return _equal(_mul(s, G), _add(R, _mul(h, A)))
