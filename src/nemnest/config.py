"""User configuration, printer profiles and slicer discovery.

Nothing here is machine-specific: the user's settings live in a per-user
config file (see `config_path`), printer profiles ship with the package and
can be extended by dropping JSON files in the user's `printers` folder.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

DEFAULTS = {
    "printer": "elegoo-saturn-4-ultra-16k",
    "slicer_exe": None,
    "optimize_height": False,
    "attach_small_rafts": True,
    "small_raft_mm2": 1200.0,
    "attach_min_mm2": 10.0,
    "gui_import_summary": True,
    "slice_defaults": {},
    "goo_profile": {},
}


def config_dir() -> Path:
    if os.environ.get("NEMNEST_CONFIG"):
        return Path(os.environ["NEMNEST_CONFIG"]).parent
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "nemnest"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nemnest"


def config_path() -> Path:
    if os.environ.get("NEMNEST_CONFIG"):
        return Path(os.environ["NEMNEST_CONFIG"])
    return config_dir() / "config.json"


def load_config() -> dict:
    """Defaults overlaid with the user's config file, if there is one."""
    config = json.loads(json.dumps(DEFAULTS))
    path = config_path()
    if path.is_file():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(user, dict):
            raise ValueError(f"{path} must contain a JSON object")
        config.update(user)
    return config


def save_config(changes: dict) -> Path:
    """Merge `changes` into the user's config file (created if missing)."""
    path = config_path()
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- printers

def user_printers_dir() -> Path:
    return config_dir() / "printers"


def _check_profile(profile: dict, source: str) -> dict:
    size, px = profile.get("size_mm"), profile.get("resolution_px")
    if not (isinstance(size, list) and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size)):
        raise ValueError(f"{source}: \"size_mm\" must be [X, Y, Z] in mm")
    if px is not None and not (isinstance(px, list) and len(px) == 2 and all(isinstance(v, int) and v > 0 for v in px)):
        raise ValueError(f"{source}: \"resolution_px\" must be [X, Y] in pixels")
    return profile


def list_printers() -> dict:
    """All known profiles by id; the user's folder overrides the built-in ones."""
    found = {}
    builtin = resources.files("nemnest") / "printers"
    for entry in sorted(builtin.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            profile = json.loads(entry.read_text(encoding="utf-8"))
            profile.setdefault("id", entry.name[:-5])
            profile["source"] = "built-in"
            found[profile["id"]] = profile
    folder = user_printers_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            profile = json.loads(path.read_text(encoding="utf-8"))
            profile.setdefault("id", path.stem)
            profile["source"] = str(path)
            found[profile["id"]] = _check_profile(profile, str(path))
    return found


def resolve_printer(ref=None, config: dict | None = None) -> dict:
    """A full printer profile from an id, a display name, a path to a profile
    file, or an inline dict. `None` means the printer set in the config."""
    config = config if config is not None else load_config()
    ref = ref if ref is not None else config.get("printer", DEFAULTS["printer"])
    if isinstance(ref, dict):
        profile = dict(ref)
        profile.setdefault("id", "custom")
        profile.setdefault("name", "custom")
        return _check_profile(profile, "inline printer")
    printers = list_printers()
    key = str(ref).strip()
    for pid, profile in printers.items():
        if key.lower() in (pid.lower(), str(profile.get("name", "")).lower()):
            return dict(profile)
    if key.lower().endswith(".json") and Path(key).is_file():
        profile = json.loads(Path(key).read_text(encoding="utf-8"))
        profile.setdefault("id", Path(key).stem)
        profile.setdefault("name", profile["id"])
        return _check_profile(profile, key)
    raise ValueError(f"Unknown printer \"{ref}\". Known printers: {', '.join(sorted(printers)) or 'none'}. "
                     f"Add a profile in {user_printers_dir()} or pass the plate size explicitly.")


# ------------------------------------------------------------------ slicer

SLICER_NAMES = ("mslicer-cli", "slicer")


def find_slicer(config: dict | None = None) -> str | None:
    """Path of the mslicer command-line slicer: NEMNEST_SLICER, then
    `slicer_exe` in the config, then the PATH."""
    config = config if config is not None else load_config()
    for candidate in (os.environ.get("NEMNEST_SLICER"), config.get("slicer_exe")):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return str(path)
            found = shutil.which(str(candidate))
            if found:
                return found
    for name in SLICER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if getattr(sys, "frozen", False):      # standalone program: a slicer placed next to it
        for name in SLICER_NAMES:
            for suffix in ("", ".exe"):
                path = Path(sys.executable).parent / (name + suffix)
                if path.is_file():
                    return str(path)
    return None


SLICER_HELP = (
    "The mslicer command-line slicer was not found. nemnest does not bundle it: build it from "
    "https://github.com/connorslade/mslicer (cargo build --release --package slicer --bin slicer) "
    "and then run `nemnest config --slicer /path/to/slicer`, set the NEMNEST_SLICER environment variable, "
    "or (standalone Windows program) put slicer.exe in the same folder as nemnest.exe."
)
