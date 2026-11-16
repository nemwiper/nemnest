"""Entry point of the standalone program for Windows and macOS (built with PyInstaller).

`nemnest --selftest result.json` checks the frozen program without a screen:
the drag-and-drop Tk extension, the bundled printer profiles, the worker
processes that read STL files and a real arrangement. The build workflows run
it after building; the result is written to a file because a windowed program
has no console on Windows.
"""
import json
import multiprocessing
import sys
import tempfile
import traceback
from pathlib import Path


def _pid(_):
    import os
    return os.getpid()


def selftest(out_path: str) -> int:
    report = {"ok": False}
    try:
        import numpy as np
        from nemnest import __version__, config as cfg, nest

        report["version"] = __version__
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()          # loads the tkdnd extension for this platform
        report["tk"] = str(root.tk.call("info", "patchlevel"))
        root.destroy()

        # nest.load_parts falls back to one process if the pool fails: check the pool itself
        import concurrent.futures
        import os
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            pids = set(pool.map(_pid, range(4)))
        report["worker_processes"] = len(pids - {os.getpid()})

        printer = cfg.resolve_printer("elegoo-saturn-4-ultra-16k", {})
        plate = {k: printer[k] for k in ("id", "name", "size_mm", "resolution_px")}
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "models", Path(tmp) / "plates"
            src.mkdir()
            for k in range(12):         # flat boxes on a 1 mm raft
                w, d, h = 30 + 3 * k, 20 + k, 10 + 2 * k
                box = np.array([[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
                                [0, 0, h], [w, 0, h], [w, d, h], [0, d, h]], dtype=np.float32)
                faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
                nest.save_stl(src / f"box_{k:02d}.stl", box[np.array(faces)])
            result = nest.arrange(src, out, plate, workers=2, attempts=2, time_limit=20)
            plates = result.get("plates") or []
            report["plates"] = len(plates)
            report["parts"] = sum(len(p["parts"]) for p in plates)
            report["files"] = sorted(p.name for p in out.iterdir())
        report["ok"] = report["worker_processes"] >= 1 and report["parts"] == 12 and "plate_01.stl" in report["files"]
    except Exception:
        report["error"] = traceback.format_exc()
    Path(out_path).write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()     # the STL readers run in worker processes
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    from nemnest import gui
    sys.exit(gui.main(sys.argv[1:]))
