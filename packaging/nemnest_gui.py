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

        # updates: the signature check must work in the built program
        import base64
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from nemnest import update
        key = Ed25519PrivateKey.generate()
        public = base64.b64encode(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        sums = b"0" * 64 + b"  test.zip\n"
        update.verify_sums(sums, base64.b64encode(key.sign(sums)), public)
        try:
            update.verify_sums(sums + b"x", base64.b64encode(key.sign(sums)), public)
            report["signature_check"] = "accepted a bad signature"
        except update.UpdateError:
            report["signature_check"] = "ok"
        report["update_key"] = update.PUBLIC_KEY[:8] + "…"

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
        report["ok"] = report["signature_check"] == "ok" and report["worker_processes"] >= 1 and report["parts"] == 12 and "plate_01.stl" in report["files"]
    except Exception:
        report["error"] = traceback.format_exc()
    Path(out_path).write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def selftest_update(folder: str) -> int:
    """Rehearse an update on copies of this program, with the real swap script.

    `folder`/inst gets a copy of this program plus a stale file that the
    update must remove (and, on Windows, a user's slicer.exe that it must
    keep); `folder`/new gets another copy. The detached script swaps them once
    this process has exited; the workflow then checks the result."""
    import os
    import shutil
    from nemnest import update

    target = update.install_target()
    if target is None:
        print("not a standalone program", file=sys.stderr)
        return 1
    root = Path(folder)
    inst, new, work = root / "inst" / target.name, root / "new" / target.name, root / "work"
    for path in (inst, new):
        path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":       # ditto keeps symlinks and the signature
            import subprocess
            subprocess.run(["ditto", str(target), str(path)], check=True)
        else:
            shutil.copytree(target, path, symlinks=True)
    stale = inst / ("Contents/Resources/stale-file" if sys.platform == "darwin" else "_internal/stale-file")
    stale.write_text("from the old version")
    if os.name == "nt":
        (inst / "slicer.exe").write_text("the user's slicer")
    work.mkdir()
    update.start_install(new, inst, work, os.getpid(), restart=False)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()     # the STL readers run in worker processes
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest-update":
        sys.exit(selftest_update(sys.argv[2]))
    from nemnest import gui
    sys.exit(gui.main(sys.argv[1:]))
