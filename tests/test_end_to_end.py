"""End to end: synthetic STLs are arranged through the MCP server (stdio) and
the command line, then distances are checked on the real, transformed geometry.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from nemnest import cli, config as cfg, nest

GAP, EDGE, STEP = 2.0, 1.0, 45.0


def prism(poly, height):
    """Extrudes a polygon that is convex or star-shaped around its centroid."""
    poly = np.asarray(poly, dtype=np.float32)
    c = poly.mean(0)
    tris = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        tris.append([[*c, 0], [*b, 0], [*a, 0]])
        tris.append([[*c, height], [*a, height], [*b, height]])
        tris.append([[*a, 0], [*b, 0], [*b, height]])
        tris.append([[*a, 0], [*b, height], [*a, height]])
    return np.array(tris, dtype=np.float32)


def circle(r, n=48):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def make_models(folder: Path, rng: random.Random) -> int:
    count = 0

    def save(name, tris, offset=(0, 0, 0)):
        nonlocal count
        nest.save_stl(folder / f"{name}.stl", tris + np.array(offset, dtype=np.float32))
        count += 1

    for k in range(10):   # miniatures: round raft
        save(f"mini_{k:02d}", prism(circle(rng.uniform(11, 19)), rng.uniform(30, 60)),
             offset=(rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(0, 3)))
    for k in range(6):    # rectangular blocks
        w, d = rng.uniform(20, 55), rng.uniform(12, 30)
        save(f"box_{k:02d}", prism([(0, 0), (w, 0), (w, d), (0, d)], rng.uniform(10, 40)))
    for k in range(4):    # long thin parts (swords, spears)
        w = rng.uniform(90, 130)
        save(f"long_{k:02d}", prism([(0, 0), (w, 0), (w, 7), (0, 7)], 12))
    for k in range(4):    # L brackets: two prisms in the same file
        a, t = rng.uniform(35, 55), 10
        leg1 = prism([(0, 0), (a, 0), (a, t), (0, t)], 20)
        leg2 = prism([(0, 0), (t, 0), (t, a), (0, a)], 20)
        save(f"ell_{k:02d}", np.concatenate([leg1, leg2]))
    for k in range(4):    # triangles
        s = rng.uniform(30, 50)
        save(f"tri_{k:02d}", prism([(0, 0), (s, 0), (0, s)], 15))
    return count


class Client:
    def __init__(self, env):
        self.proc = subprocess.Popen([sys.executable, "-m", "nemnest", "mcp"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", env=env)
        self.n = 0

    def call(self, method, params=None):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        assert reply["id"] == self.n, reply
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def tool(self, name, arguments):
        res = self.call("tools/call", {"name": name, "arguments": arguments})
        assert not res["isError"], res["content"][0]["text"]
        return json.loads(res["content"][0]["text"])

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def measure_layout(layout, fine=0.25):
    """Smallest part-to-part and part-to-edge distances of a layout.json,
    measured on the geometry moved exactly as the plate STL is written.
    The raft band is dropped: rafts are allowed to overlap."""
    w, d, _h = layout["printer"]["size_mm"]
    raft = layout["settings"]["raft_height_mm"]
    gw, gh = int(math.ceil(w / fine)), int(math.ceil(d / fine))
    worst_gap, worst_edge = float("inf"), float("inf")
    for plate in layout["plates"]:
        masks = []
        for e in plate["parts"]:
            tris = nest.move_tris(e["file"], e["rotation_z_deg"], e["pivot_mm"], e["center_mm"], e["z_min_mm"])
            xy = tris[:, :, :2].astype(np.float64)
            lo, hi = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
            worst_edge = min(worst_edge, lo[0] + w / 2, lo[1] + d / 2, w / 2 - hi[0], d / 2 - hi[1])
            body = xy[tris[:, :, 2].max(1) > raft] if raft > 0 else xy
            m, origin = nest.rasterize(body, fine)
            j0 = int(round((origin[0] + w / 2) / fine))
            i0 = int(round((origin[1] + d / 2) / fine))
            full = np.zeros((gh + 40, gw + 40), bool)   # 20 px margin
            full[i0 + 20:i0 + 20 + m.shape[0], j0 + 20:j0 + 20 + m.shape[1]] = m
            masks.append(full)
        for a in range(len(masks)):
            dist = ndimage.distance_transform_edt(~masks[a]) * fine
            for b in range(a + 1, len(masks)):
                worst_gap = min(worst_gap, float(dist[masks[b]].min()))
    return worst_gap, worst_edge


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    work = tmp_path_factory.mktemp("nemnest")
    models = work / "models"
    models.mkdir()
    count = make_models(models, random.Random(7))
    env = dict(os.environ, NEMNEST_CACHE=str(work / "cache"), NEMNEST_CONFIG=str(work / "config.json"))
    os.environ.update(NEMNEST_CACHE=env["NEMNEST_CACHE"], NEMNEST_CONFIG=env["NEMNEST_CONFIG"])
    return {"work": work, "models": models, "count": count, "env": env}


def test_printer_profiles():
    printers = cfg.list_printers()
    assert "elegoo-saturn-4-ultra-16k" in printers
    assert len(printers) > 30
    for pid, p in printers.items():
        assert len(p["size_mm"]) == 3 and p["size_mm"][0] >= p["size_mm"][1] > 0, pid
        assert p["id"] == pid and p["name"], pid
        if p.get("resolution_px"):
            assert len(p["resolution_px"]) == 2 and p["resolution_px"][0] > p["resolution_px"][1], pid
            pitch = [mm / px for mm, px in zip(p["size_mm"], p["resolution_px"])]
            assert all(0.012 < v < 0.06 for v in pitch), (pid, pitch)   # a typo in a size shows up here
        if not p.get("tested"):
            assert not p.get("formats") and not p.get("goo_profile"), pid   # arranging only
    assert cfg.resolve_printer("Elegoo Saturn 4 Ultra 16K")["id"] == "elegoo-saturn-4-ultra-16k"
    with pytest.raises(ValueError):
        cfg.resolve_printer("no-such-printer")


def test_mcp_arrange(workspace):
    models, n = workspace["models"], workspace["count"]
    client = Client(workspace["env"])
    info = client.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "0"}})
    assert info["serverInfo"]["name"] == "nemnest"
    tools = {t["name"] for t in client.call("tools/list")["tools"]}
    assert {"list_printers", "scan_folder", "arrange_plates", "export_plates", "slice_plates"} <= tools

    scan = client.tool("scan_folder", {"folder": str(models)})
    assert scan["count"] == n and all(m["fits"] for m in scan["models"])

    out = workspace["work"] / "plates_mcp"
    res = client.tool("arrange_plates", {"folder": str(models), "output_dir": str(out), "use_gpu": False,
                                         "gap_mm": GAP, "edge_margin_mm": EDGE, "rotation_step_deg": STEP})
    client.close()
    placed = sum(len(p["parts"]) for p in res["plates"])
    assert placed == n and not res["does_not_fit"] and not res["too_tall"]
    assert res["plates_count"] >= res["area_lower_bound"]
    for p in res["plates"]:
        flat = nest.load_stl(p["stl"]).reshape(-1, 3)
        lo, hi = flat.min(0), flat.max(0)
        assert abs(lo[2]) < 1e-3                       # everything sits on the plate
        w, d, h = res["printer"]["size_mm"]
        assert lo[0] >= -w / 2 and hi[0] <= w / 2 and lo[1] >= -d / 2 and hi[1] <= d / 2 and hi[2] <= h
        assert Path(p["preview"]).is_file()

    layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    worst_gap, worst_edge = measure_layout(layout)
    assert worst_gap >= GAP - 0.75, "parts too close"   # 0.75 = tolerance of the check raster
    assert worst_edge >= EDGE - 1e-6, "part outside the edge margin"


def test_cli_arrange_and_export(workspace, capsys):
    out = workspace["work"] / "plates_cli"
    keep = out / "plate_01_notes.txt"
    out.mkdir()
    keep.write_text("mine")
    assert cli.main(["arrange", str(workspace["models"]), "-o", str(out), "--no-stl", "--cpu",
                     "--optimize-height", "--exclude", "long_*", "--time-limit", "30"]) == 0
    assert "plates" in capsys.readouterr().out
    layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    assert layout["placed"] == workspace["count"] - 4
    assert not list(out.glob("plate_*.stl")) and keep.is_file()   # unrelated files are left alone
    assert cli.main(["export", str(out), "--plates", "1"]) == 0
    assert (out / "plate_01.stl").is_file()
    assert cli.main(["scan", str(workspace["work"] / "missing")]) == 1


def test_untested_printer_is_arranging_only(workspace):
    from nemnest import server
    with pytest.raises(ValueError, match="arranging only"):
        server.tool_slice_plates({"plates_dir": str(workspace["work"] / "plates_mcp"), "printer": "elegoo-saturn-3-ultra"})
    scan = server.tool_scan_folder({"folder": str(workspace["models"]), "printer": "Phrozen Sonic Mini 8K"})
    assert scan["plate_area_mm2"] == pytest.approx(165 * 71.28, abs=0.1)


def test_slice_without_slicer_explains(workspace, monkeypatch):
    monkeypatch.delenv("NEMNEST_SLICER", raising=False)
    monkeypatch.setattr(cfg.shutil, "which", lambda name: None)
    from nemnest import server
    with pytest.raises(RuntimeError, match="mslicer"):
        server.tool_slice_plates({"plates_dir": str(workspace["work"] / "plates_mcp")})


def test_window_session_places_in_arrival_order(workspace):
    from nemnest import gui
    files = sorted(str(p) for p in workspace["models"].iterdir())
    s = gui.Session()
    s.add_files(files[:10])
    s.layout()
    first = [(it.label, it.plate, it.i, it.j) for it in s.placements]
    s.add_files(files[10:])
    s.layout()
    now = [(it.label, it.plate, it.i, it.j) for it in s.placements]
    assert now[:10] == first                                    # adding parts never moves earlier ones
    assert [it.label for it in s.placements] == s.labels()       # arrival order
    assert {it.theta for it in s.placements} == {0.0}            # unrotated
    plates = [it.plate for it in s.placements]
    assert plates == sorted(plates) and s.n_plates > 1           # a new plate only when the last is full
    assert s.render(0, 3.0).size[0] > 100
    in_order = s.n_plates
    s.layout(optimise=True, time_limit=20, gpu=False)
    assert s.n_plates <= in_order and len(s.placements) == len(files)
    out = workspace["work"] / "plates_gui"
    result = s.export(out)
    assert result["placed"] == len(files) and (out / "plate_01.stl").is_file()
    assert measure_layout(json.loads((out / "layout.json").read_text(encoding="utf-8")))[0] >= GAP - 0.75
    s.remove([0, 1])
    s.layout()
    assert len(s.placements) == len(files) - 2


def test_window_add_printer(workspace):
    from nemnest import gui
    profile = gui.save_printer("My Test Printer 8K", ["218.88", "122.88", "250"], ["7680", "4320"])
    assert profile["id"] == "my-test-printer-8k"
    assert cfg.resolve_printer("my-test-printer-8k")["size_mm"] == [218.88, 122.88, 250.0]
    with pytest.raises(ValueError):
        gui.save_printer("", [1, 2, 3])
    with pytest.raises(ValueError):
        gui.save_printer("bad", ["x", 2, 3])


def test_window_optimise_reports_progress_and_can_be_cancelled(workspace):
    from nemnest import gui
    s = gui.Session()
    s.add_folder(workspace["models"])
    s.layout()
    before = [(it.label, it.plate, it.i, it.j) for it in s.placements]
    seen = []
    s.layout(optimise=True, time_limit=20, gpu=False, progress=seen.append)
    fractions = [gui.progress_fraction(info) for info in seen]
    assert seen and all(0.0 <= f <= 1.0 for f in fractions) and seen[-1]["plates"] == s.n_plates
    assert s.optimised

    s.layout()                                   # back to arrival order

    def cancel(_info):
        raise gui.Cancelled()
    with pytest.raises(gui.Cancelled):
        s.layout(optimise=True, time_limit=20, gpu=False, progress=cancel)
    assert [(it.label, it.plate, it.i, it.j) for it in s.placements] == before and not s.optimised


def test_part_number_lands_on_the_part():
    ring = np.zeros((60, 80), bool)
    ring[5:55, 5:75] = True
    ring[15:45, 15:75] = False            # a horseshoe open to the right
    i, j = nest.label_anchor(ring)
    assert ring[i, j] and not ring[30, 40]   # the bounding-box centre is in the hole
    assert min(i, 59 - i, j, 79 - j) >= 5


def test_window_accepts_mixed_files_and_folders(workspace):
    from nemnest import gui
    models = workspace["models"]
    other = workspace["work"] / "notes.txt"
    other.write_text("not a model")
    one = sorted(models.glob("box_*.stl"))[0]
    s = gui.Session()
    added = s.add_paths([str(one), str(other), str(models), str(workspace["work"] / "missing.stl")])
    assert added == 1 + workspace["count"]        # the single file, then the whole folder; the rest ignored
    s.layout()
    assert len(s.placements) == added and sum(label.startswith(one.name) for label in s.labels()) == 2


def test_partial_plate_is_centered(workspace):
    """A plate that is not full sits in the middle of the plate, not in a
    corner; the shift keeps the distances between the parts."""
    files = sorted(workspace["models"].glob("*.stl"))[:3]
    printer = cfg.resolve_printer("elegoo-saturn-4-ultra-16k")
    w, d, _h = printer["size_mm"]

    def group_box(center):
        out = workspace["work"] / f"plates_center_{center}"
        nest.arrange(workspace["models"], out, printer, gap=GAP, edge=EDGE, files=files, write_stl=False,
                     gpu=False, time_limit=10, center=center)
        layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
        assert layout["plates_count"] == 1
        pts = np.concatenate([nest.move_tris(e["file"], e["rotation_z_deg"], e["pivot_mm"], e["center_mm"],
                                             e["z_min_mm"]).reshape(-1, 3)[:, :2]
                              for e in layout["plates"][0]["parts"]])
        return pts.min(0), pts.max(0), measure_layout(layout)

    lo, hi, (gap, edge) = group_box(True)
    mid = (lo + hi) / 2
    assert abs(mid[0]) < 1.0 and abs(mid[1]) < 1.0, mid        # group in the middle, within 1 mm
    assert gap >= GAP - 0.75 and edge >= EDGE - 1e-6
    lo0, hi0, (gap0, _e) = group_box(False)
    assert lo0[0] < lo[0] - 10                                   # without it, the group hugs one side
    assert np.allclose(hi0 - lo0, hi - lo, atol=1e-3)            # same group, only shifted


def test_gap_setting(workspace, capsys):
    """The gap is checked, remembered in the config, used by the window and
    by the command line, and the parts keep at least that distance."""
    for bad in (-1, 25, "abc"):
        with pytest.raises(ValueError):
            cfg.check_gap(bad)
    assert cfg.check_gap("1,5") == 1.5
    from nemnest import gui
    assert cli.main(["config", "--gap", "4"]) == 0
    assert cfg.load_config()["gap_mm"] == 4.0
    s = gui.Session()
    assert s.gap == 4.0                                          # the window starts from the config
    s.add_files(sorted(str(p) for p in workspace["models"].iterdir())[:8])
    assert not s.set_gap("4") and s.set_gap(3) and cfg.load_config()["gap_mm"] == 3.0
    s.layout(optimise=True, time_limit=10, gpu=False)
    out = workspace["work"] / "plates_gap_window"
    s.export(out)
    layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    assert layout["settings"]["gap_mm"] == 3.0 and measure_layout(layout)[0] >= 3.0 - 0.75
    out = workspace["work"] / "plates_gap_cli"
    assert cli.main(["arrange", str(workspace["models"]), "-o", str(out), "--no-stl", "--cpu",
                     "--time-limit", "10"]) == 0                 # no --gap: the config value
    layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    assert layout["settings"]["gap_mm"] == 3.0 and measure_layout(layout)[0] >= 3.0 - 0.75
    cfg.save_config({"gap_mm": 2.0})
    capsys.readouterr()


def test_resin_profiles_and_slicing_priority(workspace):
    from nemnest import server
    resin = cfg.save_resin("Test Clear Blue 50 um", {"layer_height_mm": "0,05", "exposure_time_s": "3",
                                                     "first_layers": "5", "first_exposure_time_s": 35,
                                                     "transition_layers": 0, "wait_before_cure_s": "",
                                                     "lift_speed_mm_min": None},
                           brand="Elegoo", color="Clear Blue", extras={"bottle_price": "38,87", "currency": "EUR"})
    assert resin["id"] == "test-clear-blue-50-um"
    assert resin["settings"] == {"layer_height_mm": 0.05, "exposure_time_s": 3.0, "first_layers": 5,
                                 "first_exposure_time_s": 35.0, "transition_layers": 0}   # empty ones left out
    assert resin["bottle_price"] == 38.87 and cfg.resolve_resin("Test Clear Blue 50 um")["brand"] == "Elegoo"
    assert any(r["id"] == resin["id"] for r in server.tool_list_resins({})["resins"])

    printer = cfg.resolve_printer("elegoo-saturn-4-ultra-16k")
    config = cfg.load_config()
    plain = cfg.slice_settings(printer, None, config)
    assert plain["exposure_time_s"] == 2.2 and plain["wait_before_cure_s"] == 1.2
    with_resin = cfg.slice_settings(printer, resin, config)
    assert with_resin["exposure_time_s"] == 3.0 and with_resin["first_exposure_time_s"] == 35.0
    assert with_resin["wait_before_cure_s"] == 1.2                 # not in the resin: printer default kept
    assert cfg.slice_settings(printer, resin, config, {"exposure_time_s": 2.5})["exposure_time_s"] == 2.5

    with pytest.raises(ValueError, match="already"):
        cfg.save_resin("test clear blue 50 um", {})
    with pytest.raises(ValueError, match="number"):
        cfg.save_resin("Broken", {"exposure_time_s": "three"})
    with pytest.raises(ValueError, match="whole"):
        cfg.save_resin("Broken", {"first_layers": "2.5"})
    renamed = cfg.save_resin("Test Clear Blue", resin["settings"], replace_id=resin["id"])
    assert renamed["id"] == "test-clear-blue" and resin["id"] not in cfg.list_resins()
    cfg.delete_resin("test-clear-blue")
    assert "test-clear-blue" not in cfg.list_resins()
    with pytest.raises(ValueError):
        cfg.resolve_resin("test-clear-blue")
    assert cfg.resolve_resin(None, {"resin": None}) is None


def test_printers_can_be_edited_and_deleted(workspace):
    mars = cfg.resolve_printer("Elegoo Mars 2")
    assert mars["size_mm"] == [130.56, 82.62, 150.0] and mars["resolution_px"] == [2560, 1620]
    made = cfg.save_printer("Shop Printer", ["200", "120,5", "240"], ["", ""])
    assert made["id"] == "shop-printer" and made["size_mm"] == [200.0, 120.5, 240.0] and "resolution_px" not in made
    edited = cfg.save_printer("Shop Printer", [201, 120.5, 240], ["7680", "4320"], slice_format="ctb",
                              replace_id="shop-printer")
    assert edited["id"] == "shop-printer" and edited["formats"] == ["ctb"] and edited["resolution_px"] == [7680, 4320]
    renamed = cfg.save_printer("Shop Printer 2", [201, 120.5, 240], ["7680", "4320"], replace_id="shop-printer")
    assert renamed["id"] == "shop-printer-2" and renamed["formats"] == ["ctb"]      # kept from before
    assert "shop-printer" not in cfg.list_printers()
    assert cfg.delete_printer("shop-printer-2") == "deleted"
    with pytest.raises(ValueError, match="already"):
        cfg.save_printer("Elegoo Mars 2", [1, 1, 1])
    with pytest.raises(ValueError, match="both"):
        cfg.save_printer("Half", [1, 1, 1], ["100", ""])

    # editing a built-in stores the user's own copy under the same id; deleting it restores the original
    mine = cfg.save_printer("Elegoo Mars 2", [130, 82, 150], [2560, 1620], replace_id="elegoo-mars-2")
    assert mine["id"] == "elegoo-mars-2" and cfg.resolve_printer("elegoo-mars-2")["size_mm"] == [130.0, 82.0, 150.0]
    assert cfg.list_printers()["elegoo-mars-2"]["source"] != "built-in"
    assert cfg.delete_printer("elegoo-mars-2") == "restored"
    assert cfg.resolve_printer("elegoo-mars-2")["size_mm"] == [130.56, 82.62, 150.0]
    with pytest.raises(ValueError, match="Built-in"):
        cfg.delete_printer("elegoo-mars-2")

    # a tested profile whose plate is changed is no longer the verified machine
    saturn = cfg.resolve_printer("elegoo-saturn-4-ultra-16k")
    changed = cfg.save_printer(saturn["name"], [210, 118, 220], saturn["resolution_px"], replace_id=saturn["id"])
    assert changed["tested"] is False
    cfg.delete_printer(saturn["id"])
    assert cfg.resolve_printer(saturn["id"])["tested"] is True


def test_window_session_remembers_the_resin(workspace):
    from nemnest import gui
    resin = cfg.save_resin("Session Resin", {"exposure_time_s": 2.8})
    s = gui.Session()
    assert s.resin is None
    s.set_resin(resin["id"])
    assert gui.Session().resin["name"] == "Session Resin"        # remembered for next time
    s.set_printer("elegoo-mars-2")
    assert s.resin is None                                        # ... but only for the printer it was chosen on
    s.set_printer("elegoo-saturn-4-ultra-16k")
    assert s.resin["name"] == "Session Resin"
    from nemnest import server
    assert cfg.resolve_resin(None, cfg.load_config(), "elegoo-mars-2") is None
    cfg.delete_resin(resin["id"])
    assert gui.Session().resin is None                            # a deleted profile is not an error
    s.set_resin(None)
    assert cfg.load_config()["resin_by_printer"]["elegoo-saturn-4-ultra-16k"] is None
