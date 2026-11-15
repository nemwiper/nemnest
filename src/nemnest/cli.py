"""Command line: `nemnest scan|arrange|export|slice|printers|config|gui|mcp`."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, config as cfg


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def _plate_args(p):
    p.add_argument("--printer", help="printer profile id or name (default: the configured one)")
    p.add_argument("--plate-size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="explicit build volume in mm, overriding the printer profile")


def _folder_args(p):
    p.add_argument("folder", nargs="+", help="folder(s) with the pre-supported STL files")
    p.add_argument("-x", "--exclude", action="append", default=[], metavar="PATTERN",
                   help="file name pattern to leave out, e.g. \"*water*\" (repeatable)")


def _common(ns, args: dict) -> dict:
    if getattr(ns, "printer", None):
        args["printer"] = ns.printer
    if getattr(ns, "plate_size", None):
        args["plate_size_mm"] = ns.plate_size
    return args


def _folders(ns):
    return ns.folder[0] if len(ns.folder) == 1 else ns.folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemnest",
        description="Auto-nesting of pre-supported STLs onto as few resin printer build plates as possible.")
    parser.add_argument("--version", action="version", version=f"nemnest {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("printers", help="list the printer profiles")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("scan", help="list the STLs of a folder with size and fit, without writing anything")
    _folder_args(p)
    _plate_args(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("arrange", help="arrange the STLs on as few plates as possible")
    _folder_args(p)
    _plate_args(p)
    p.add_argument("-o", "--output", help="output folder (default: <first folder>/plates)")
    p.add_argument("--no-stl", action="store_true", help="placement only: previews and layout.json")
    h = p.add_mutually_exclusive_group()
    h.add_argument("--optimize-height", dest="optimize_height", action="store_true", default=None,
                   help="group tall parts to cut total print time, never adding plates")
    h.add_argument("--no-optimize-height", dest="optimize_height", action="store_false")
    r = p.add_mutually_exclusive_group()
    r.add_argument("--attach-small-rafts", dest="attach_small_rafts", action="store_true", default=None,
                   help="move small parts so their raft overlaps a larger one")
    r.add_argument("--no-attach-small-rafts", dest="attach_small_rafts", action="store_false")
    p.add_argument("--gap", type=float, help="minimum distance between parts in mm (default 2)")
    p.add_argument("--edge", type=float, help="minimum distance from the plate edge in mm (default 1)")
    p.add_argument("--raft-height", type=float,
                   help="thickness in mm treated as raft, allowed to overlap (default 1; 0 = off)")
    p.add_argument("--rotation-step", type=float, help="Z rotation step in degrees (default 45; 0 = none)")
    p.add_argument("--copies", action="append", default=[], metavar="FILE=N",
                   help="number of copies of a file, e.g. orc.stl=3 (repeatable)")
    p.add_argument("--attempts", type=int, help="orderings to try (default 12)")
    p.add_argument("--time-limit", type=float, help="maximum search time in seconds (default 60)")
    p.add_argument("--workers", type=int, help="processes used to read STLs")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", dest="gpu", action="store_true", default=None, help="require the GPU (CuPy)")
    g.add_argument("--cpu", dest="gpu", action="store_false", help="do not use the GPU")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("export", help="write plate STLs from an existing layout.json")
    p.add_argument("plates_dir", help="folder containing layout.json")
    p.add_argument("--plates", nargs="+", type=int, help="plate numbers (default: all)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("slice", help="slice the plate STLs with mslicer")
    p.add_argument("plates_dir", help="folder with the plate_NN.stl files")
    _plate_args(p)
    p.add_argument("--plates", nargs="+", type=int, help="plate numbers (default: all)")
    p.add_argument("--format", choices=["goo", "ctb", "nanodlp"])
    p.add_argument("--layer-height", type=float, dest="layer_height_mm", metavar="MM")
    p.add_argument("--exposure", type=float, dest="exposure_time_s", metavar="S")
    p.add_argument("--first-exposure", type=float, dest="first_exposure_time_s", metavar="S")
    p.add_argument("--first-layers", type=int, dest="first_layers", metavar="N")
    p.add_argument("--transition-layers", type=int, dest="transition_layers", metavar="N")
    p.add_argument("--wait-before-cure", type=float, dest="wait_before_cure_s", metavar="S")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("config", help="show or change the user configuration")
    p.add_argument("--printer", help="set the default printer profile")
    p.add_argument("--slicer", help="set the path of the mslicer command-line slicer")
    p.add_argument("--optimize-height", choices=["on", "off"], help="set the default for height optimisation")

    p = sub.add_parser("gui", help="open the window: import STLs and see the plates from above")
    p.add_argument("paths", nargs="*", help="STL files or folders to load at start")

    sub.add_parser("mcp", help="run as an MCP server on stdin/stdout (for AI assistants)")
    return parser


def cmd_printers(ns, server):
    res = server.tool_list_printers({})
    if ns.json:
        return res
    for p in res["printers"]:
        mark = "*" if p["id"] == res["configured"] else " "
        size = " x ".join(f"{v:g}" for v in p["size_mm"])
        px = " x ".join(str(v) for v in p["resolution_px"]) if p["resolution_px"] else "-"
        print(f"{mark} {p['id']:<32} {size} mm   {px} px   {'tested' if p['tested'] else 'untested'}")
    print(f"\n* = default. Add your own profiles in {res['user_printers_dir']}")


def cmd_scan(ns, server):
    res = server.tool_scan_folder(_common(ns, {"folder": _folders(ns), "exclude": ns.exclude}))
    if ns.json:
        return res
    for m in res["models"]:
        size = " x ".join(f"{v:.1f}" for v in m["size_mm"])
        print(f"{'  ' if m['fits'] else 'NO'} {m['name']:<48} {size} mm   footprint {m['footprint_mm2']:.0f} mm2")
    print(f"\n{res['count']} models, total footprint {res['total_footprint_mm2']:.0f} mm2, "
          f"plate {res['plate_area_mm2']:.0f} mm2")


def cmd_arrange(ns, server):
    args = _common(ns, {"folder": _folders(ns), "exclude": ns.exclude, "write_stl": not ns.no_stl})
    for key, value in (("output_dir", ns.output), ("optimize_height", ns.optimize_height),
                       ("attach_small_rafts", ns.attach_small_rafts), ("gap_mm", ns.gap),
                       ("edge_margin_mm", ns.edge), ("raft_height_mm", ns.raft_height),
                       ("rotation_step_deg", ns.rotation_step), ("attempts", ns.attempts),
                       ("time_limit_s", ns.time_limit), ("workers", ns.workers), ("use_gpu", ns.gpu)):
        if value is not None:
            args[key] = value
    copies = {}
    for item in ns.copies:
        name, _, n = item.rpartition("=")
        if not name or not n.isdigit():
            raise ValueError(f"--copies expects FILE=N, got \"{item}\"")
        copies[name] = int(n)
    if copies:
        args["copies"] = copies
    res = server.tool_arrange_plates(args)
    if ns.json:
        return res
    for p in res["plates"]:
        print(f"plate {p['plate']:>2}: {len(p['parts']):>3} parts, fill {p['fill']:.0%}, "
              f"height {p['max_height_mm']:.1f} mm")
    for name in res["does_not_fit"]:
        print(f"DOES NOT FIT THE PLATE: {name}")
    for name in res["too_tall"]:
        print(f"TOO TALL FOR THE PRINTER: {name}")
    print(f"\n{res['placed']} parts on {res['plates_count']} plates "
          f"(sum of heights {res['height_sum_mm']:.1f} mm) -> {res['output_dir']}")


def cmd_export(ns, server):
    args = {"plates_dir": ns.plates_dir}
    if ns.plates:
        args["plates"] = ns.plates
    res = server.tool_export_plates(args)
    if ns.json:
        return res
    print(json.dumps(res, indent=1))


def cmd_slice(ns, server):
    args = _common(ns, {"plates_dir": ns.plates_dir})
    for key in ("plates", "format", "layer_height_mm", "exposure_time_s", "first_exposure_time_s",
                "first_layers", "transition_layers", "wait_before_cure_s"):
        if getattr(ns, key) is not None:
            args[key] = getattr(ns, key)
    res = server.tool_slice_plates(args)
    if ns.json:
        return res
    total_s = total_ml = 0.0
    for s in res["sliced"]:
        info = s["profile_applied"] or {}
        line = f"{s['plate']} -> {s['output']} ({s['bytes'] / 1e6:.0f} MB)"
        if info:
            line += f"   {info['layers']} layers, {_hms(info['printing_time_s'])}"
            total_s += info["printing_time_s"]
            if "volume_ml" in info:
                line += f", {info['volume_ml']:.1f} ml"
                total_ml += info["volume_ml"]
        print(line)
    if total_s:
        print(f"\ntotal: {_hms(total_s)}" + (f", {total_ml:.1f} ml of resin" if total_ml else ""))


def cmd_config(ns, server):
    changes = {}
    if ns.printer:
        changes["printer"] = cfg.resolve_printer(ns.printer)["id"]
    if ns.slicer:
        changes["slicer_exe"] = ns.slicer
    if ns.optimize_height:
        changes["optimize_height"] = ns.optimize_height == "on"
    if changes:
        print(f"saved to {cfg.save_config(changes)}")
    config = cfg.load_config()
    print(f"config file : {cfg.config_path()}{'' if cfg.config_path().is_file() else '  (not created yet)'}")
    print(f"printer     : {config.get('printer')}")
    print(f"slicer      : {cfg.find_slicer(config) or 'NOT FOUND'}")
    print(f"settings    : {json.dumps({k: v for k, v in config.items() if k not in ('printer', 'slicer_exe')})}")
    if changes.get("slicer_exe") and not cfg.find_slicer(config):
        print("warning: that slicer path does not exist", file=sys.stderr)


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.command == "gui":
        from . import gui
        return gui.main(ns.paths)
    from . import server   # imported late: pulls in numpy/scipy
    if ns.command == "mcp":
        server.main()
        return 0
    command = {"printers": cmd_printers, "scan": cmd_scan, "arrange": cmd_arrange,
               "export": cmd_export, "slice": cmd_slice, "config": cmd_config}[ns.command]
    try:
        result = command(ns, server)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"nemnest: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
