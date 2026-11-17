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
    "resin": None,
    "resin_by_printer": {},
    "slicer_exe": None,
    "optimize_height": False,
    "attach_small_rafts": True,
    "center_plates": True,
    "gap_mm": 2.0,
    "small_raft_mm2": 1200.0,
    "attach_min_mm2": 10.0,
    "gui_import_summary": True,
    "check_updates": True,
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


MAX_GAP_MM = 20.0


def check_gap(value) -> float:
    """Minimum distance between parts, mm: a number from 0 to MAX_GAP_MM."""
    try:
        gap = float(str(value).replace(",", "."))
    except ValueError:
        raise ValueError(f"the gap must be a number of mm, got \"{value}\"") from None
    if not 0 <= gap <= MAX_GAP_MM:
        raise ValueError(f"the gap must be between 0 and {MAX_GAP_MM:g} mm, got {gap:g}")
    return gap


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


def profile_id(name: str, fallback: str = "profile") -> str:
    """File-name-safe id made from a display name."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or fallback


def _number(value, what: str, minimum=None, whole=False):
    try:
        number = float(str(value).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{what} must be a number, got \"{value}\"") from None
    if whole and number != int(number):
        raise ValueError(f"{what} must be a whole number, got \"{value}\"")
    if minimum is not None and number < minimum:
        raise ValueError(f"{what} must be at least {minimum:g}, got {number:g}")
    return int(number) if whole else number


SLICE_FORMATS = ("goo", "ctb", "nanodlp")


def save_printer(name, size_mm, resolution_px=None, slice_format=None, replace_id=None) -> dict:
    """Store a user printer profile and return it. `replace_id` is the profile
    being edited: its other settings are kept, and if the name changed the old
    user file is removed. A built-in profile is never touched: editing one
    stores a user profile with the same id, which takes its place."""
    name = str(name).strip()
    if not name:
        raise ValueError("Give the printer a name.")
    size = [_number(v, f"Plate {axis}", 0.001) for v, axis in zip(size_mm, "XYZ")]
    if len(size) != 3:
        raise ValueError("The build volume needs three numbers (X, Y, Z in mm).")
    px = None
    if resolution_px and any(str(v).strip() for v in resolution_px):
        if len(resolution_px) != 2 or not all(str(v).strip() for v in resolution_px):
            raise ValueError("The screen resolution needs both X and Y in pixels (or leave both empty).")
        px = [_number(v, f"Screen pixels {axis}", 1, whole=True) for v, axis in zip(resolution_px, "XY")]
    if slice_format and slice_format not in SLICE_FORMATS:
        raise ValueError(f"Unknown slicing format \"{slice_format}\": use one of {', '.join(SLICE_FORMATS)}.")
    printers = list_printers()
    old = dict(printers.get(replace_id) or {})
    new_id = replace_id if replace_id and str(old.get("name", "")).strip().lower() == name.lower() else profile_id(name, "printer")
    if new_id != replace_id and new_id in printers:
        raise ValueError(f"There is already a printer called \"{printers[new_id].get('name', new_id)}\".")
    profile = {k: v for k, v in old.items() if k not in ("source", "id", "name", "size_mm", "resolution_px")}
    profile = {"id": new_id, "name": name, **profile, "size_mm": size}
    if px:
        profile["resolution_px"] = px
    if old.get("tested") and (size != old.get("size_mm") or px != old.get("resolution_px")):
        profile["tested"] = False           # no longer the machine that was verified
    profile.setdefault("tested", False)
    if slice_format is not None or "formats" not in profile:
        profile["formats"] = [slice_format] if slice_format else []
    folder = user_printers_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{new_id}.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    if replace_id and replace_id != new_id and (folder / f"{replace_id}.json").is_file():
        (folder / f"{replace_id}.json").unlink()
    return profile


def delete_printer(pid: str) -> str:
    """Remove a user printer profile. Returns "deleted", or "restored" when a
    built-in profile with the same id shows through again."""
    path = user_printers_dir() / f"{pid}.json"
    if not path.is_file():
        raise ValueError("Built-in printers cannot be deleted." if pid in list_printers()
                         else f"Unknown printer \"{pid}\".")
    path.unlink()
    return "restored" if pid in list_printers() else "deleted"


# ------------------------------------------------------------------ resins

# setting -> (label, whole number, minimum)
RESIN_SETTINGS = {
    "layer_height_mm": ("Layer thickness (mm)", False, 0.001),
    "exposure_time_s": ("Exposure time (s)", False, 0.0),
    "lift_distance_mm": ("Lift distance (mm)", False, 0.0),
    "lift_speed_mm_min": ("Lift speed (mm/min)", False, 0.0),
    "retract_speed_mm_min": ("Retract speed (mm/min)", False, 0.0),
    "wait_before_cure_s": ("Wait before exposure (s)", False, 0.0),
    "first_layers": ("Burn-in layers", True, 0),
    "first_exposure_time_s": ("Burn-in exposure time (s)", False, 0.0),
    "transition_layers": ("Transition layers", True, 0),
    "first_wait_before_cure_s": ("Burn-in wait before exposure (s)", False, 0.0),
}
RESIN_EXTRAS = {
    "density_g_ml": ("Density (g/ml)", False, 0.1),
    "bottle_price": ("Bottle price", False, 0.0),
    "bottle_capacity_l": ("Bottle capacity (l)", False, 0.001),
}


def user_resins_dir() -> Path:
    return config_dir() / "resins"


def list_resins() -> dict:
    """The user's resin profiles by id. None ship with nemnest: exposure
    values depend on the resin and on the printer."""
    found = {}
    folder = user_resins_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            profile = json.loads(path.read_text(encoding="utf-8"))
            profile.setdefault("id", path.stem)
            profile.setdefault("name", profile["id"])
            profile.setdefault("settings", {})
            profile["source"] = str(path)
            found[profile["id"]] = profile
    return found


def resolve_resin(ref=None, config: dict | None = None, printer_id: str | None = None) -> dict | None:
    """A resin profile from an id, a display name or an inline dict. `None`
    means the resin remembered for this printer (exposure depends on the
    machine, so the choice is kept per printer), else the general default
    set in the config; either may be none at all."""
    config = config if config is not None else load_config()
    if ref is None:
        ref = (config.get("resin_by_printer") or {}).get(printer_id) or config.get("resin")
    if not ref:
        return None
    if isinstance(ref, dict):
        return dict(ref)
    resins = list_resins()
    key = str(ref).strip().lower()
    for rid, profile in resins.items():
        if key in (rid.lower(), str(profile.get("name", "")).lower()):
            return dict(profile)
    raise ValueError(f"Unknown resin profile \"{ref}\". Known profiles: {', '.join(sorted(resins)) or 'none'}. "
                     f"Add one in the window (Printers and resins) or as a file in {user_resins_dir()}.")


def save_resin(name, settings: dict, brand="", color="", notes="", extras: dict | None = None,
               replace_id=None) -> dict:
    """Store a resin profile and return it. Empty settings are left out, so the
    printer's own defaults apply to them."""
    name = str(name).strip()
    if not name:
        raise ValueError("Give the resin profile a name.")
    clean = {}
    for key, value in (settings or {}).items():
        if key not in RESIN_SETTINGS:
            raise ValueError(f"Unknown resin setting \"{key}\".")
        if value is None or str(value).strip() == "":
            continue
        label, whole, minimum = RESIN_SETTINGS[key]
        clean[key] = _number(value, label, minimum, whole)
    more = {}
    for key, value in (extras or {}).items():
        if key == "currency":
            if str(value).strip():
                more[key] = str(value).strip()
        elif key in RESIN_EXTRAS and value is not None and str(value).strip() != "":
            label, whole, minimum = RESIN_EXTRAS[key]
            more[key] = _number(value, label, minimum, whole)
    resins = list_resins()
    old = resins.get(replace_id) or {}
    new_id = replace_id if replace_id and str(old.get("name", "")).strip().lower() == name.lower() else profile_id(name, "resin")
    if new_id != replace_id and new_id in resins:
        raise ValueError(f"There is already a resin profile called \"{resins[new_id].get('name', new_id)}\".")
    profile = {"id": new_id, "name": name, "brand": str(brand).strip(), "color": str(color).strip(),
               "notes": str(notes).strip(), "settings": clean, **more}
    folder = user_resins_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{new_id}.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    if replace_id and replace_id != new_id and (folder / f"{replace_id}.json").is_file():
        (folder / f"{replace_id}.json").unlink()
    return profile


def delete_resin(rid: str) -> None:
    path = user_resins_dir() / f"{rid}.json"
    if not path.is_file():
        raise ValueError(f"Unknown resin profile \"{rid}\".")
    path.unlink()


def slice_settings(printer: dict, resin: dict | None, config: dict, call: dict | None = None) -> dict:
    """Slicing settings by increasing priority: the printer profile, the
    user's config, the chosen resin profile, then what this call passes."""
    return {**printer.get("slice_defaults", {}), **config.get("slice_defaults", {}),
            **((resin or {}).get("settings") or {}), **(call or {})}


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
        exe = Path(sys.executable)
        folders = [exe.parent]
        if len(exe.parents) > 3 and exe.parents[2].suffix == ".app":   # macOS: next to nemnest.app
            folders.append(exe.parents[3])
        for folder in folders:
            for name in SLICER_NAMES:
                for suffix in ("", ".exe"):
                    path = folder / (name + suffix)
                    if path.is_file():
                        return str(path)
    return None


SLICER_HELP = (
    "The mslicer command-line slicer was not found. nemnest does not bundle it: build it from "
    "https://github.com/connorslade/mslicer (cargo build --release --package slicer --bin slicer) "
    "and then run `nemnest config --slicer /path/to/slicer`, set the NEMNEST_SLICER environment variable, "
    "or (standalone program) put slicer.exe in the same folder as nemnest.exe on Windows, "
    "or the slicer binary in the same folder as nemnest.app on macOS."
)
