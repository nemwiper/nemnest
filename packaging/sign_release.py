"""Sign a GitHub release so that installed programs accept it as an update.

Run on the maintainer's computer after the release workflow has attached the
zips (needs the GitHub CLI `gh`, logged in, and `pip install cryptography`):

    python packaging/sign_release.py v0.2.3

It downloads the release's zips, writes SHA256SUMS with their hashes, signs it
with the private key and uploads SHA256SUMS and SHA256SUMS.sig to the release.
The private key never leaves this computer; default location
~/.config/nemnest-release/signing_key (override with NEMNEST_SIGNING_KEY).

    python packaging/sign_release.py --new-key

creates the key pair once and prints the public key to put in
src/nemnest/update.py (PUBLIC_KEY).
"""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = "nemwiper/nemnest"
KEY = Path(os.environ.get("NEMNEST_SIGNING_KEY") or Path.home() / ".config" / "nemnest-release" / "signing_key")


def public_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def new_key() -> None:
    if KEY.exists():
        sys.exit(f"{KEY} already exists: not overwriting it")
    KEY.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(KEY.parent, 0o700)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    print(f"private key: {KEY} (keep a backup somewhere safe and offline)")
    print(f"public key : {public_b64(key)}")


def load_key() -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(KEY.read_bytes(), password=None)


def sums_for(folder: Path) -> bytes:
    lines = []
    for z in sorted(folder.glob("*.zip")):
        lines.append(f"{hashlib.sha256(z.read_bytes()).hexdigest()}  {z.name}")
    if not lines:
        sys.exit("the release has no zips")
    return ("\n".join(lines) + "\n").encode()


def sign(tag: str) -> None:
    key = load_key()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        subprocess.run(["gh", "release", "download", tag, "--repo", REPO, "--pattern", "*.zip",
                        "--dir", str(folder)], check=True)
        sums = sums_for(folder)
        (folder / "SHA256SUMS").write_bytes(sums)
        (folder / "SHA256SUMS.sig").write_text(base64.b64encode(key.sign(sums)).decode() + "\n")
        print(sums.decode(), end="")
        subprocess.run(["gh", "release", "upload", tag, "--repo", REPO, "--clobber",
                        str(folder / "SHA256SUMS"), str(folder / "SHA256SUMS.sig")], check=True)
    print(f"{tag} signed with public key {public_b64(key)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--new-key"]:
        new_key()
    elif len(sys.argv) == 2 and sys.argv[1].startswith("v"):
        sign(sys.argv[1])
    else:
        sys.exit(__doc__)
