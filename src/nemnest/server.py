"""MCP server (stdio) exposing nemnest to AI assistants.

No dependency beyond numpy, scipy and pillow: the MCP protocol (JSON-RPC over
stdin/stdout, one message per line) is implemented here. The `tool_*`
functions are also what the command line calls.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import traceback
from pathlib import Path

from . import __version__, config as cfg, goo, nest


def printer_from(args: dict, config: dict | None = None) -> dict:
    """Full printer profile for a call: `printer` (id, name, file or inline
    dict) or the configured one, with optional explicit plate overrides."""
    profile = cfg.resolve_printer(args.get("printer"), config)
    if args.get("plate_size_mm"):
        profile["size_mm"] = [float(v) for v in args["plate_size_mm"]]
        profile["name"] = "custom"
        profile["id"] = "custom"
        for key in ("goo_profile", "formats", "tested", "notes"):   # they belong to the other machine
            profile.pop(key, None)
    if args.get("plate_resolution_px"):
        profile["resolution_px"] = [int(v) for v in args["plate_resolution_px"]]
    return profile


def plate_of(profile: dict) -> dict:
    """The part of a profile the nesting engine needs (and stores in layout.json)."""
    return {k: profile[k] for k in ("id", "name", "size_mm", "resolution_px") if k in profile}


FOLDER_PROPS = {
    "folder": {"description": "Folder with the STL files, or a list of folders to nest together.",
               "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "exclude": {"type": "array", "items": {"type": "string"},
                "description": "File name patterns to leave out, e.g. [\"*water*\"]. Case-insensitive."},
}

PLATE_PROPS = {
    "printer": {"type": "string",
                "description": "Printer profile id or name (see list_printers). "
                               "If omitted, the printer set in the user's config."},
    "plate_size_mm": {
        "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
        "description": "Explicit build volume [X, Y, Z] in mm, overriding the printer profile.",
    },
}

TOOLS = [
    {
        "name": "list_printers",
        "description": "Lists the available printer profiles (build volume, resolution, whether the "
                       "profile was verified on real prints) and the one currently configured.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "scan_folder",
        "description": "Lists the STL files of a folder with bounding box, height, footprint area "
                       "and whether each fits the plate. Writes nothing.",
        "inputSchema": {
            "type": "object",
            "properties": {**FOLDER_PROPS, **PLATE_PROPS},
            "required": ["folder"],
        },
    },
    {
        "name": "arrange_plates",
        "description": "Arranges the pre-supported STLs of a folder onto as few build plates as possible "
                       "(nesting on the real footprint, rotation around Z only, so supports stay valid). "
                       "Writes to output_dir one merged plate_NN.stl per plate, a plate_NN.png preview "
                       "and layout.json. Existing plate_NN.* files in output_dir are replaced.",
        "inputSchema": {
            "type": "object",
            "properties": {
                **FOLDER_PROPS,
                "output_dir": {"type": "string",
                               "description": "Where to write the plates. Default: <first folder>/plates."},
                "raft_height_mm": {"type": "number",
                                   "description": "Thickness below which geometry counts as raft: rafts may "
                                                  "overlap each other and sit under other parts, bodies may "
                                                  "not. Default 1; 0 = disabled."},
                "optimize_height": {"type": "boolean",
                                    "description": "true = with the same number of plates, group tall parts "
                                                   "together to reduce the sum of plate heights, i.e. total "
                                                   "print time. Never adds plates; takes longer to compute. "
                                                   "Default: the user's config (false)."},
                "attach_small_rafts": {"type": "boolean",
                                       "description": "true = small isolated parts are moved, on the same "
                                                      "plate, so their raft overlaps the raft of a larger "
                                                      "part (easier to remove from the plate). Changes "
                                                      "neither plate count nor heights. Default: the "
                                                      "user's config (true)."},
                "center_plates": {"type": "boolean",
                                  "description": "true = once arranged, the parts of each plate are "
                                                 "shifted together into the middle of the plate, so a "
                                                 "plate that is not full is not packed into a corner. "
                                                 "Changes neither plate count nor distances. Default: "
                                                 "the user's config (true)."},
                "small_raft_mm2": {"type": "number",
                                   "description": "Footprint area below which a part counts as small "
                                                  "(default 1200 mm2, about 35 x 35 mm)."},
                "attach_min_mm2": {"type": "number",
                                   "description": "Minimum raft overlap for a small part to count as "
                                                  "attached (default 10 mm2)."},
                "use_gpu": {"type": "boolean",
                            "description": "true = require the GPU, false = CPU only. If omitted the GPU is "
                                           "used when CuPy/CUDA is available."},
                "workers": {"type": "integer",
                            "description": "Processes used to read the STLs in parallel. "
                                           "Default: available cores minus 2, at most 6."},
                "write_stl": {"type": "boolean",
                              "description": "false = placement only (previews and layout.json), without "
                                             "writing the plate_NN.stl files. Default true."},
                "gap_mm": {"type": "number", "description": "Minimum distance between part bodies, mm, from 0 to 20 "
                                                         "(rafts may still overlap). Default: the user's "
                                                         "config (2)."},
                "edge_margin_mm": {"type": "number",
                                   "description": "Minimum distance from the plate edge (default 1)."},
                "rotation_step_deg": {"type": "number",
                                      "description": "Step of the Z rotations tried "
                                                     "(default 45; 0 = no rotation)."},
                "copies": {"type": "object", "additionalProperties": {"type": "integer"},
                           "description": "Copies per file, e.g. {\"orc.stl\": 3}."},
                "attempts": {"type": "integer",
                             "description": "Different orderings to try (default 12)."},
                "time_limit_s": {"type": "number", "description": "Maximum search time (default 60)."},
                **PLATE_PROPS,
            },
            "required": ["folder"],
        },
    },
    {
        "name": "export_plates",
        "description": "Writes the plate_NN.stl of some plates (or all) from a layout.json already "
                       "computed by arrange_plates, without redoing the placement.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plates_dir": {"type": "string", "description": "Folder containing layout.json."},
                "plates": {"type": "array", "items": {"type": "integer"},
                           "description": "Plate numbers to write. If omitted, all."},
            },
            "required": ["plates_dir"],
        },
    },
    {
        "name": "slice_plates",
        "description": "Slices the plate_NN.stl files of a folder with the mslicer command-line slicer "
                       "(installed separately). Writes one print file per plate and reports layers, "
                       "estimated print time and resin use. Exposure settings default to the printer "
                       "profile and the user's config: pass the values for your resin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plates_dir": {"type": "string", "description": "Folder with the plate_NN.stl files."},
                "plates": {"type": "array", "items": {"type": "integer"},
                           "description": "Plate numbers to slice. If omitted, all."},
                "format": {"type": "string", "enum": ["goo", "ctb", "nanodlp"],
                           "description": "Output format. Default: the printer profile's format. Only "
                                          ".goo files get corrected metadata (time, volume, weight)."},
                "layer_height_mm": {"type": "number"},
                "exposure_time_s": {"type": "number"},
                "first_exposure_time_s": {"type": "number"},
                "first_layers": {"type": "integer"},
                "transition_layers": {"type": "integer"},
                "lift_distance_mm": {"type": "number"},
                "lift_speed_mm_min": {"type": "number"},
                "retract_speed_mm_min": {"type": "number"},
                "supersample": {"type": "integer"},
                "wait_before_cure_s": {"type": "number"},
                "first_wait_before_cure_s": {"type": "number"},
                "plate_resolution_px": {"type": "array", "items": {"type": "integer"},
                                        "minItems": 2, "maxItems": 2},
                **PLATE_PROPS,
            },
            "required": ["plates_dir"],
        },
    },
]


def tool_list_printers(args):
    config = cfg.load_config()
    rows = [{"id": pid, "name": p.get("name", pid), "size_mm": p["size_mm"],
             "resolution_px": p.get("resolution_px"), "formats": p.get("formats", []),
             "tested": bool(p.get("tested", False)), "notes": p.get("notes", ""), "source": p["source"]}
            for pid, p in sorted(cfg.list_printers().items())]
    return {"configured": config.get("printer"), "printers": rows,
            "user_printers_dir": str(cfg.user_printers_dir()), "config_file": str(cfg.config_path())}


def tool_scan_folder(args):
    return nest.scan(args["folder"], plate_of(printer_from(args)), exclude=args.get("exclude"))


def tool_arrange_plates(args):
    config = cfg.load_config()
    folder = args["folder"]
    first = Path(folder if isinstance(folder, str) else folder[0])
    out = Path(args.get("output_dir") or first / "plates")
    result = nest.arrange(
        folder, out, plate_of(printer_from(args, config)),
        exclude=args.get("exclude"),
        write_stl=bool(args.get("write_stl", True)),
        raft_height=float(args.get("raft_height_mm", 1.0)),
        gpu=args["use_gpu"] if "use_gpu" in args else "auto",
        workers=int(args["workers"]) if "workers" in args else None,
        optimize_height=bool(args.get("optimize_height", config.get("optimize_height", False))),
        attach_small_rafts=bool(args.get("attach_small_rafts", config.get("attach_small_rafts", True))),
        center=bool(args.get("center_plates", config.get("center_plates", True))),
        small_raft_mm2=float(args.get("small_raft_mm2", config.get("small_raft_mm2", 1200.0))),
        attach_min_mm2=float(args.get("attach_min_mm2", config.get("attach_min_mm2", 10.0))),
        gap=cfg.check_gap(args.get("gap_mm", config.get("gap_mm", 2.0))),
        edge=float(args.get("edge_margin_mm", 1.0)),
        rotation_step=float(args.get("rotation_step_deg", 45.0)),
        copies=args.get("copies"),
        attempts=int(args.get("attempts", 12)),
        time_limit=float(args.get("time_limit_s", 60.0)),
    )
    result["output_dir"] = str(out)
    return result


def tool_export_plates(args):
    return nest.export_plates(args["plates_dir"], args.get("plates"))


SLICE_FLAGS = {
    "layer_height_mm": "--layer-height",
    "exposure_time_s": "--exposure-time",
    "first_exposure_time_s": "--first-exposure-time",
    "first_layers": "--first-layers",
    "transition_layers": "--transition-layers",
    "lift_distance_mm": "--lift-distance",
    "lift_speed_mm_min": "--lift-speed",
    "retract_speed_mm_min": "--retract-speed",
    "supersample": "--supersample",
    "wait_before_cure_s": "--exposure-delay",
    "first_wait_before_cure_s": "--first-exposure-delay",
}


def tool_slice_plates(args):
    config = cfg.load_config()
    printer = printer_from(args, config)
    # settings: printer profile < user's config < this call
    args = {**printer.get("slice_defaults", {}), **config.get("slice_defaults", {}), **args}
    if not args.get("format") and not printer.get("formats"):
        raise ValueError(
            f"The printer profile \"{printer.get('name', printer.get('id'))}\" is for arranging only: nemnest has no "
            "verified slicing settings for it. Slice the exported plate_NN.stl files with your usual slicer, "
            "or pass an explicit format (goo, ctb, nanodlp) to try mslicer at your own risk.")
    fmt = args.get("format") or printer["formats"][0]
    goo_profile = {**printer.get("goo_profile", {}), **config.get("goo_profile", {})}
    exe = cfg.find_slicer(config)
    if not exe:
        raise RuntimeError(cfg.SLICER_HELP)
    if not printer.get("resolution_px"):
        raise ValueError("Slicing needs the screen resolution: use a printer profile with "
                         "\"resolution_px\" or pass plate_resolution_px.")
    plates_dir = Path(args["plates_dir"])
    stls = sorted(plates_dir.glob("plate_*.stl"))
    if args.get("plates"):
        wanted = {int(n) for n in args["plates"]}
        stls = [s for s in stls if s.stem.split("_")[-1].isdigit() and int(s.stem.split("_")[-1]) in wanted]
    if not stls:
        raise ValueError(f"No plate_*.stl to slice in {plates_dir}")
    base = [exe,
            "--platform-size", ", ".join(str(v) for v in printer["size_mm"]),
            "--platform-resolution", ", ".join(str(v) for v in printer["resolution_px"])]
    for key, flag in SLICE_FLAGS.items():
        if key in args:
            base += [flag, str(args[key])]
    done = []
    # mslicer centres the plate by itself, so only the extent has to fit.
    # Its "extends outside of print volume" warning always fires with the
    # v0.10.1 CLI and carries no information: this check replaces it.
    spurious = "Model extends outsize of print volume"
    for stl in stls:
        flat = nest.load_stl(stl).reshape(-1, 3)
        extent = flat.max(0) - flat.min(0)
        if any(float(e) > float(s) + 1e-3 for e, s in zip(extent, printer["size_mm"])):
            raise ValueError(f"{stl.name} does not fit the build volume: extent "
                             f"{[round(float(e), 2) for e in extent]} mm")
        del flat
        out = stl.with_suffix("." + fmt)
        # mslicer samples each layer at its bottom plane (the first one at
        # Z = 0, where it only sees what touches the plate exactly). Sinking
        # the plate by half a layer puts the sample at mid thickness, as the
        # manufacturers' slicers do, so the first layer holds the whole raft.
        sink = -float(args.get("layer_height_mm", 0.05)) / 2
        proc = subprocess.run(base + ["--mesh", str(stl), "--position", f"0, 0, {sink}", "--quiet", str(out)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"mslicer failed on {stl.name}: {proc.stderr.strip() or proc.stdout.strip()}")
        fixed = None
        if fmt == "goo" and goo_profile:
            # metadata as the manufacturer's slicer writes it; images untouched
            fixed = goo.apply_profile(out, goo_profile)
        log = " | ".join(line.strip(" \\") for line in proc.stdout.replace("\x1b[0G\x1b[K", "").splitlines()
                         if line.strip() and spurious not in line)
        done.append({"plate": stl.name, "output": str(out), "bytes": out.stat().st_size,
                     "extent_mm": [round(float(e), 2) for e in extent], "profile_applied": fixed,
                     "log": log[-400:]})
    return {"printer": plate_of(printer), "format": fmt, "slicer": exe, "sliced": done}


HANDLERS = {
    "list_printers": tool_list_printers,
    "scan_folder": tool_scan_folder,
    "arrange_plates": tool_arrange_plates,
    "export_plates": tool_export_plates,
    "slice_plates": tool_slice_plates,
}


def handle(msg: dict):
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nemnest", "version": __version__},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        if name not in HANDLERS:
            raise KeyError(f"Unknown tool: {name}")
        try:
            result = HANDLERS[name](params.get("arguments") or {})
            text, is_error = json.dumps(result, indent=1, ensure_ascii=False), False
        except Exception as exc:  # a tool error, not a protocol error
            traceback.print_exc(file=sys.stderr)
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}
    raise KeyError(f"Unsupported method: {method}")


def main():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in msg:  # notification: no reply
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
        except KeyError as exc:
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32601, "message": str(exc.args[0]) if exc.args else str(exc)}}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32603, "message": str(exc)}}
        stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
