"""The pure-Python Ed25519 verifier: RFC 8032 test vectors, and agreement
with the `cryptography` package on random keys, messages and tampering."""
from __future__ import annotations

import os

import pytest

from nemnest import ed25519

# RFC 8032, section 7.1: (public key, message, signature)
RFC = [
    ("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
]


@pytest.mark.parametrize("public,message,signature", RFC)
def test_rfc_vectors(public, message, signature):
    pub, msg, sig = bytes.fromhex(public), bytes.fromhex(message), bytes.fromhex(signature)
    assert ed25519.verify(pub, msg, sig)
    assert not ed25519.verify(pub, msg + b"x", sig)
    bad = bytearray(sig)
    bad[5] ^= 1
    assert not ed25519.verify(pub, msg, bytes(bad))
    bad = bytearray(sig)
    bad[40] ^= 1
    assert not ed25519.verify(pub, msg, bytes(bad))


def test_malformed_inputs():
    pub, _, sig = (bytes.fromhex(x) for x in RFC[0])
    assert not ed25519.verify(pub[:31], b"", sig)
    assert not ed25519.verify(pub, b"", sig[:63])
    assert not ed25519.verify(b"\xff" * 32, b"", sig)
    # s not reduced modulo the group order: rejected (no malleability)
    s = int.from_bytes(sig[32:], "little") + ed25519.Q
    assert not ed25519.verify(pub, b"", sig[:32] + s.to_bytes(32, "little"))


def test_agrees_with_cryptography():
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    for _ in range(20):
        key = Ed25519PrivateKey.generate()
        pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        msg = os.urandom(int.from_bytes(os.urandom(1), "big"))
        sig = key.sign(msg)
        assert ed25519.verify(pub, msg, sig)
        assert not ed25519.verify(pub, msg + b"\0", sig)
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        assert not ed25519.verify(other, msg, sig)
