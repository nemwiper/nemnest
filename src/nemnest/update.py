"""Updates of the standalone program from the GitHub releases.

The window asks GitHub for the latest release. If it is newer, the user can
install it: the zip for this platform is downloaded, checked against the
release's signed list of SHA-256 hashes, unpacked, and a small script swaps
the files once the program has closed, then starts the new version.

Trust: `SHA256SUMS` lists the hash of every zip of a release and
`SHA256SUMS.sig` is its Ed25519 signature, made on the maintainer's computer
with a key that never goes to GitHub (packaging/sign_release.py). The public
key is below. A release without a valid signature is never installed: the
user is sent to the release page instead. So a compromised GitHub account
alone cannot push code to existing installations.

Only the standalone programs update themselves; a Python install is told the
pip command instead.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import __version__

REPO = "nemwiper/nemnest"
# overridable for tests: a local server that imitates the GitHub API
API_URL = os.environ.get("NEMNEST_UPDATE_API", f"https://api.github.com/repos/{REPO}/releases/latest")
# Ed25519 public key of the release signatures (raw 32 bytes, base64)
PUBLIC_KEY = os.environ.get("NEMNEST_UPDATE_KEY") or "VVRne8s0rrYny2naSk8GMHlcwKjCnSeZ5I2yiR3gqpo="
SUMS, SIG = "SHA256SUMS", "SHA256SUMS.sig"
TIMEOUT = 10


class UpdateError(Exception):
    pass


def parse_version(text: str) -> tuple:
    """'v0.2.10' -> (0, 2, 10). Anything else -> ()."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", str(text).strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def is_newer(tag: str, current: str = __version__) -> bool:
    new, cur = parse_version(tag), parse_version(current)
    return bool(new) and bool(cur) and new > cur


def asset_suffix(system: str | None = None, machine: str | None = None) -> str | None:
    """End of the zip name for this platform, or None if there is no standalone build."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    if system.startswith("win"):
        return "-windows.zip"
    if system == "darwin":
        return "-macos-arm64.zip" if machine in ("arm64", "aarch64") else "-macos-intel.zip"
    return None


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_target(exe: str | None = None, system: str | None = None) -> Path | None:
    """What the update replaces: the program folder on Windows, nemnest.app on macOS."""
    exe_path = Path(exe or sys.executable)
    system = system or sys.platform
    if system.startswith("win"):
        return exe_path.parent
    if system == "darwin" and len(exe_path.parents) > 2 and exe_path.parents[2].suffix == ".app":
        return exe_path.parents[2]
    return None


def _get(url: str, accept: str = "application/octet-stream") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Accept": accept, "User-Agent": f"nemnest/{__version__}"})


def latest_release() -> dict:
    """{'tag', 'url', 'notes', 'assets': {name: download_url}} of the latest release."""
    try:
        with urllib.request.urlopen(_get(API_URL, "application/vnd.github+json"), timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from exc
    return {"tag": data.get("tag_name", ""), "url": data.get("html_url", ""),
            "notes": data.get("body") or "",
            "assets": {a["name"]: a["browser_download_url"] for a in data.get("assets", [])}}


def check() -> dict | None:
    """The latest release if it is newer than this version, with what this
    installation can do with it ('install', 'pip' or 'download'); else None."""
    rel = latest_release()
    if not is_newer(rel["tag"]):
        return None
    suffix = asset_suffix()
    zips = [n for n in rel["assets"] if suffix and n.endswith(suffix)]
    rel["zip"] = zips[0] if zips else None
    rel["signed"] = SUMS in rel["assets"] and SIG in rel["assets"]
    if not frozen():
        rel["action"] = "pip"
    elif rel["zip"] and rel["signed"] and install_target():
        rel["action"] = "install"
    else:
        rel["action"] = "download"
    return rel


def _read(url: str, limit: int = 1 << 20) -> bytes:
    with urllib.request.urlopen(_get(url), timeout=TIMEOUT) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise UpdateError(f"{url} is unexpectedly large")
    return data


def verify_sums(sums: bytes, sig: bytes, public_key: str | None = None) -> dict:
    """Check the signature of SHA256SUMS and return {file name: sha256 hex}."""
    from . import ed25519
    try:
        ok = ed25519.verify(base64.b64decode(public_key or PUBLIC_KEY), sums, base64.b64decode(sig.strip()))
    except ValueError:
        ok = False
    if not ok:
        raise UpdateError("the release signature is not valid: the update was not installed")
    table = {}
    for line in sums.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            table[parts[1].lstrip("*")] = parts[0]
    return table


def download(rel: dict, folder: Path, progress=None) -> Path:
    """Download and check the zip of `rel` into `folder`; returns its path."""
    name = rel.get("zip")
    if not name:
        raise UpdateError("this release has no package for this computer")
    table = verify_sums(_read(rel["assets"][SUMS]), _read(rel["assets"][SIG]))
    if name not in table:
        raise UpdateError(f"{name} is not in the signed list of the release")
    path = folder / name
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(_get(rel["assets"][name]), timeout=TIMEOUT) as r, open(path, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 16):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError(f"download failed: {exc}") from exc
    if digest.hexdigest() != table[name]:
        path.unlink(missing_ok=True)
        raise UpdateError("the downloaded file does not match the signed hash: the update was not installed")
    return path


def unpack(zip_path: Path, folder: Path, system: str | None = None) -> Path:
    """Unpack the release zip; returns the new nemnest.app or program folder."""
    system = system or sys.platform
    out = folder / "new"
    out.mkdir()
    if system == "darwin":     # ditto keeps the app's symlinks, permissions and signature
        subprocess.run(["ditto", "-x", "-k", str(zip_path), str(out)], check=True)
        new = out / "nemnest.app"
    else:
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():   # no paths that leave the folder
                target = (out / member).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise UpdateError(f"unsafe path in the update: {member}")
            z.extractall(out)
        new = out / "nemnest"
    if not new.exists():
        raise UpdateError("the update package does not contain the program")
    return new


def _ps(text) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _sh(text) -> str:
    return "'" + str(text).replace("'", "'\\''") + "'"


def write_script(new: Path, target: Path, work: Path, pid: int, system: str | None = None,
                 restart: bool = True) -> list:
    """Write the swap script into `work` and return the command that runs it.

    The script waits for process `pid` to end, replaces `target` with `new`
    (putting the old files back if the copy fails), starts the program and
    removes `work`. A log is left in `work`'s parent if something fails."""
    system = system or sys.platform
    log = work.parent / "nemnest-update.log"
    start_win = 'Start-Process -FilePath "$inst\\nemnest.exe"' if restart else ""
    start_mac = 'open "$APP"' if restart else ""
    if system.startswith("win"):
        script = work / "update.ps1"
        script.write_text(f"""\
$ErrorActionPreference = 'Stop'
$new = {_ps(new)}; $inst = {_ps(target)}; $work = {_ps(work)}; $log = {_ps(log)}
Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500
$moved = @()
try {{
  foreach ($name in @('_internal', 'nemnest.exe', 'LICENSE.txt')) {{
    if (Test-Path -LiteralPath "$inst\\$name") {{
      Remove-Item -LiteralPath "$inst\\$name.old" -Recurse -Force -ErrorAction SilentlyContinue
      Rename-Item -LiteralPath "$inst\\$name" -NewName "$name.old"
      $moved += $name
    }}
    if (Test-Path -LiteralPath "$new\\$name") {{ Copy-Item -LiteralPath "$new\\$name" -Destination "$inst\\$name" -Recurse -Force }}
  }}
  foreach ($name in $moved) {{ Remove-Item -LiteralPath "$inst\\$name.old" -Recurse -Force -ErrorAction SilentlyContinue }}
}} catch {{
  "$(Get-Date) update failed: $_" | Out-File -LiteralPath $log -Append
  foreach ($name in $moved) {{
    Remove-Item -LiteralPath "$inst\\$name" -Recurse -Force -ErrorAction SilentlyContinue
    Rename-Item -LiteralPath "$inst\\$name.old" -NewName $name -ErrorAction SilentlyContinue
  }}
}}
{start_win}
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
""", encoding="utf-8-sig")
        return ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden", "-File", str(script)]
    script = work / "update.sh"
    old = target.with_name(target.name + ".old-update")
    script.write_text(f"""\
#!/bin/sh
NEW={_sh(new)}; APP={_sh(target)}; OLD={_sh(old)}; WORK={_sh(work)}; LOG={_sh(log)}
while kill -0 {int(pid)} 2>/dev/null; do sleep 0.3; done
rm -rf "$OLD"
if mv "$APP" "$OLD"; then
  if ditto "$NEW" "$APP"; then
    rm -rf "$OLD"
  else
    echo "$(date) update failed while copying" >> "$LOG"
    rm -rf "$APP"; mv "$OLD" "$APP"
  fi
else
  echo "$(date) update failed: cannot move $APP" >> "$LOG"
fi
{start_mac}
rm -rf "$WORK"
""")
    script.chmod(0o755)
    return ["/bin/sh", str(script)]


def can_replace(target: Path) -> bool:
    """The update needs to write next to the program (rename it and copy the new one)."""
    return os.access(target, os.W_OK) and os.access(target.parent, os.W_OK)


def start_install(new: Path, target: Path, work: Path, pid: int | None = None, restart: bool = True) -> None:
    """Start the detached swap script; the caller must then exit the program."""
    cmd = write_script(new, target, work, pid or os.getpid(), restart=restart)
    if sys.platform.startswith("win"):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(cmd, start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare(rel: dict, progress=None) -> tuple:
    """Download, check and unpack `rel`. Returns (new, target, work) for start_install."""
    target = install_target()
    if target is None:
        raise UpdateError("this installation cannot update itself")
    if not can_replace(target):
        raise UpdateError(f"nemnest cannot write to {target.parent}: move the program to a folder "
                          "you can write to, or download the new version from GitHub")
    work = Path(tempfile.mkdtemp(prefix="nemnest-update-"))
    try:
        zip_path = download(rel, work, progress)
        new = unpack(zip_path, work)
        zip_path.unlink()
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return new, target, work
