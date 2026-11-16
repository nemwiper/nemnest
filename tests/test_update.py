"""Update check, signature and download, against a local server that imitates
the GitHub API. Swapping the files is tested on the built programs by the
release workflow (`--selftest-update`)."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import threading
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from nemnest import update  # noqa: E402


def test_versions():
    assert update.parse_version("v0.2.10") == (0, 2, 10)
    assert update.parse_version("0.3") == (0, 3)
    assert update.parse_version("v1.0-beta") == ()
    assert update.is_newer("v0.2.10", "0.2.9")
    assert not update.is_newer("v0.2.2", "0.2.2")
    assert not update.is_newer("v0.2.1", "0.2.2")
    assert not update.is_newer("nightly", "0.2.2")


def test_platform_names():
    assert update.asset_suffix("win32", "AMD64") == "-windows.zip"
    assert update.asset_suffix("darwin", "arm64") == "-macos-arm64.zip"
    assert update.asset_suffix("darwin", "x86_64") == "-macos-intel.zip"
    assert update.asset_suffix("linux", "x86_64") is None
    app = "/Applications/nemnest.app/Contents/MacOS/nemnest"
    assert update.install_target(app, "darwin") == Path("/Applications/nemnest.app")
    assert update.install_target("/usr/local/bin/python3", "darwin") is None
    if os.name == "nt":
        assert update.install_target(r"C:\Tools\nemnest\nemnest.exe", "win32") == Path(r"C:\Tools\nemnest")


def keypair():
    key = Ed25519PrivateKey.generate()
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return key, base64.b64encode(raw).decode()


def test_signature():
    key, public = keypair()
    sums = b"%s  nemnest-v9.9.9-windows.zip\n" % (b"a" * 64)
    sig = base64.b64encode(key.sign(sums))
    assert update.verify_sums(sums, sig, public) == {"nemnest-v9.9.9-windows.zip": "a" * 64}
    with pytest.raises(update.UpdateError):
        update.verify_sums(sums.replace(b"a", b"b"), sig, public)       # list changed
    other, _ = keypair()
    with pytest.raises(update.UpdateError):
        update.verify_sums(sums, base64.b64encode(other.sign(sums)), public)   # someone else's key


@pytest.fixture
def fake_github(tmp_path, monkeypatch):
    """A local 'latest release' v9.9.9 with a signed zip for this computer."""
    key, public = keypair()
    suffix = update.asset_suffix() or "-windows.zip"
    name = f"nemnest-v9.9.9{suffix}"
    files = tmp_path / "files"
    files.mkdir()
    with zipfile.ZipFile(files / name, "w") as z:
        if "macos" in suffix:
            z.writestr("nemnest.app/Contents/MacOS/nemnest", "new program")
        else:
            z.writestr("nemnest/nemnest.exe", "new program")
    sums = f"{hashlib.sha256((files / name).read_bytes()).hexdigest()}  {name}\n".encode()
    (files / "SHA256SUMS").write_bytes(sums)
    (files / "SHA256SUMS.sig").write_bytes(base64.b64encode(key.sign(sums)))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(files), **kw)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    (files / "latest.json").write_text(json.dumps({
        "tag_name": "v9.9.9", "html_url": f"{base}/release", "body": "notes",
        "assets": [{"name": n, "browser_download_url": f"{base}/{n}"}
                   for n in (name, "SHA256SUMS", "SHA256SUMS.sig")]}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(update, "API_URL", f"{base}/latest.json")
    monkeypatch.setattr(update, "PUBLIC_KEY", public)
    yield {"files": files, "name": name, "key": key}
    server.shutdown()


def test_python_install_is_told_pip(fake_github):
    rel = update.check()
    assert rel["tag"] == "v9.9.9" and rel["action"] == "pip"


@pytest.mark.skipif(update.asset_suffix() is None, reason="no standalone build for this platform")
def test_download_and_unpack(fake_github, tmp_path, monkeypatch):
    target = tmp_path / ("install/nemnest.app" if "macos" in fake_github["name"] else "install/nemnest")
    target.mkdir(parents=True)
    monkeypatch.setattr(update, "frozen", lambda: True)
    monkeypatch.setattr(update, "install_target", lambda *a, **k: target)
    rel = update.check()
    assert rel["action"] == "install" and rel["zip"] == fake_github["name"]
    seen = []
    new, got_target, work = update.prepare(rel, lambda d, t: seen.append((d, t)))
    assert got_target == target and new.exists() and seen
    assert not list(work.glob("*.zip"))              # the zip is removed once unpacked

    # a zip that does not match the signed list is refused
    (fake_github["files"] / fake_github["name"]).write_bytes(b"something else")
    with pytest.raises(update.UpdateError, match="signed hash"):
        update.prepare(update.check())


def test_unsigned_release_is_not_installed(fake_github, monkeypatch, tmp_path):
    (fake_github["files"] / "latest.json").write_text(json.dumps({
        **json.loads((fake_github["files"] / "latest.json").read_text()),
        "assets": [a for a in json.loads((fake_github["files"] / "latest.json").read_text())["assets"]
                   if not a["name"].startswith("SHA256SUMS")]}))
    monkeypatch.setattr(update, "frozen", lambda: True)
    monkeypatch.setattr(update, "install_target", lambda *a, **k: tmp_path)
    assert update.check()["action"] == "download"
