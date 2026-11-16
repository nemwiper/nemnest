"""A small desktop window for nemnest (Tkinter, no extra dependency).

Import STLs and see them from above, one plate per screen. Parts are placed as
they arrive, unrotated; when one no longer fits, a new plate is opened. The
"Optimise" button runs the full search for the fewest plates.

`Session` holds the state and has no GUI in it, so it can be tested headless.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
from pathlib import Path

from . import __version__, config as cfg, nest

HINT = "Add STL files to start"
HINT_DROP = "Drop STL files or folders here"
PHASES = {"placing": "Trying different orders", "pairing": "Pairing large parts",
          "height": "Grouping tall parts", "rafts": "Joining small rafts"}


class Cancelled(Exception):
    """Raised inside the search when the user cancels it."""


def progress_fraction(info: dict) -> float:
    """0..1 for a progress dict of `nest.pack`: the orderings fill the first
    part of the bar, the later phases advance with their own clock."""
    share = min(1.0, info["phase_elapsed"] / max(info["phase_limit"], 1e-6))
    return {"placing": 0.45 * info["attempt"] / info["attempts"], "pairing": 0.45 + 0.25 * share,
            "height": 0.70 + 0.25 * share, "rafts": 0.97}.get(info["phase"], 0.0)


class Session:
    """The parts imported so far and their current layout."""

    def __init__(self, printer=None, gap=None, edge=1.0, raft_height=1.0, res=0.5):
        self.config = cfg.load_config()
        self.printer = cfg.resolve_printer(printer, self.config)
        gap = self.config.get("gap_mm", 2.0) if gap is None else gap
        self.gap, self.edge, self.raft_height, self.res = cfg.check_gap(gap), edge, raft_height, res
        self.files: list[str] = []          # arrival order; a file listed twice is two copies
        self.parts: dict[str, nest.Part] = {}
        self.label_index: dict[str, int] = {}
        self.settings: dict = {}
        self.clear_layout()

    # ---- parts
    def clear_layout(self):
        self.placements, self.n_plates, self.grid = [], 0, None
        self.unplaceable, self.too_tall = [], []
        self.optimised = False
        self.stats = {}

    def add_files(self, paths, workers=None) -> int:
        paths = [str(Path(p)) for p in paths if str(p).lower().endswith(".stl")]
        new = [p for p in dict.fromkeys(paths) if p not in self.parts]
        if new:
            loaded = nest.load_parts([(p, self.res, None, self.raft_height) for p in new], workers)
            self.parts.update(zip(new, loaded))
        self.files.extend(paths)
        return len(paths)

    def add_folder(self, folder, workers=None) -> int:
        return self.add_files(nest.list_stls(folder), workers)

    def add_paths(self, paths, workers=None) -> int:
        """Files and folders mixed, as they come from a drop: the STLs among
        the files, and every STL directly inside each folder. Anything else
        is ignored."""
        files = []
        for path in paths:
            path = Path(path)
            if path.is_dir():
                files += nest.list_stls(path)
            elif path.is_file() and path.suffix.lower() == ".stl":
                files.append(path)
        return self.add_files(files, workers)

    def remove(self, indices):
        drop = set(indices)
        self.files = [f for i, f in enumerate(self.files) if i not in drop]

    def clear(self):
        self.files = []
        self.clear_layout()

    def labels(self) -> list[str]:
        """One unique label per entry of `files`: the name, numbered for copies
        and prefixed with the folder for equal names in different folders."""
        names = {f: self.parts[f].name for f in dict.fromkeys(self.files)}
        clash = {n for n in names.values() if list(names.values()).count(n) > 1}
        seen: dict[str, int] = {}
        out = []
        for f in self.files:
            base = f"{Path(f).parent.name}/{names[f]}" if names[f] in clash else names[f]
            seen[f] = seen.get(f, 0) + 1
            out.append(base if self.files.count(f) == 1 else f"{base} #{seen[f]}")
        return out

    def set_printer(self, ref):
        self.printer = cfg.resolve_printer(ref, self.config)

    def set_gap(self, value) -> bool:
        """Change the minimum distance between parts and remember it.
        Returns whether it changed; the caller lays the plates out again."""
        gap = cfg.check_gap(value)
        if gap == self.gap:
            return False
        self.gap = gap
        self.config["gap_mm"] = gap
        cfg.save_config({"gap_mm": gap})
        return True

    # ---- layout
    def layout(self, optimise=False, rotate=True, optimize_height=False, time_limit=60.0, attempts=24, gpu="auto",
               progress=None):
        """Place everything. Default: arrival order, no rotation, no search.
        If `progress` raises, the search stops and the previous layout is kept."""
        plate_w, plate_d, plate_h = self.printer["size_mm"]
        labels = self.labels()
        self.label_index = {label: i for i, label in enumerate(labels)}
        todo = [(self.parts[f], label) for f, label in zip(self.files, labels)]
        too_tall = [label for p, label in todo if p.size[2] > plate_h]
        todo = [(p, label) for p, label in todo if p.size[2] <= plate_h]
        if not todo:
            self.clear_layout()
            self.too_tall = too_tall
            return
        angles = [a * 45.0 for a in range(8)] if optimise and rotate else [0.0]
        result = nest.pack(
            [p for p, _ in todo], [label for _, label in todo], plate_w, plate_d, self.res, self.gap, self.edge,
            angles, attempts=attempts, time_limit=time_limit, gpu=gpu if optimise else False,
            optimize_height=bool(optimise and optimize_height),
            attach_small=bool(optimise and self.raft_height > 0 and self.config.get("attach_small_rafts", True)),
            small_raft_mm2=float(self.config.get("small_raft_mm2", 1200.0)),
            attach_min_mm2=float(self.config.get("attach_min_mm2", 10.0)),
            fixed_order=not optimise, center=bool(self.config.get("center_plates", True)),
            progress=progress)
        self.clear_layout()
        self.too_tall = too_tall
        self.placements, self.n_plates, self.unplaceable, self.grid = result
        self.optimised = bool(optimise)
        self.stats = self.grid.get("stats") or {}
        self.settings = {"gap_mm": self.gap, "edge_margin_mm": self.edge,
                         "rotation_step_deg": 45.0 if len(angles) > 1 else 0.0, "resolution_mm": self.res,
                         "raft_height_mm": self.raft_height, "optimize_height": bool(optimise and optimize_height),
                         "attach_small_rafts": bool(optimise), "fixed_order": not optimise,
                         "center": bool(optimise and self.config.get("center_plates", True))}

    def plate_items(self, k: int) -> list:
        return [it for it in self.placements if it.plate == k]

    def plate_info(self, k: int) -> dict:
        items = self.plate_items(k)
        plate_w, plate_d, _h = self.printer["size_mm"]
        return {"parts": len(items),
                "fill": sum(it.part.body_area for it in items) / (plate_w * plate_d) if items else 0.0,
                "height_mm": max((it.part.size[2] for it in items), default=0.0)}

    def render(self, k: int, scale: float):
        """PIL image of plate k (0-based) at `scale` pixels per mm."""
        plate_w, plate_d, _h = self.printer["size_mm"]
        grid = self.grid or {"res": self.res}
        return nest.render_preview(self.plate_items(k) if self.grid else [], grid, plate_w, plate_d, scale)

    def plate(self) -> dict:
        return {k: self.printer[k] for k in ("id", "name", "size_mm", "resolution_px") if k in self.printer}

    def export(self, output_dir, write_stl=True) -> dict:
        """Write previews, layout.json and plate STLs of the layout on screen."""
        if not self.placements:
            raise ValueError("Nothing to export: add some STL files first.")
        return nest.write_layout(output_dir, self.plate(), self.placements, self.n_plates, self.grid,
                                 self.settings, models=len(self.files), unplaceable=self.unplaceable,
                                 too_tall=self.too_tall, write_stl=write_stl)


def save_printer(name: str, size_mm, resolution_px=None) -> dict:
    """Store a user printer profile and return it."""
    name = name.strip()
    if not name:
        raise ValueError("Give the printer a name.")
    size = [float(v) for v in size_mm]
    if len(size) != 3 or any(v <= 0 for v in size):
        raise ValueError("The build volume needs three positive numbers (X, Y, Z in mm).")
    pid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "printer"
    profile = {"id": pid, "name": name, "tested": False, "size_mm": size, "formats": []}
    if resolution_px:
        px = [int(v) for v in resolution_px]
        if len(px) != 2 or any(v <= 0 for v in px):
            raise ValueError("The screen resolution needs two positive whole numbers (X, Y in pixels).")
        profile["resolution_px"] = px
    folder = cfg.user_printers_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{pid}.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


# ---------------------------------------------------------------- window

def main(argv=None) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except ImportError as exc:
        print(f"nemnest: the window needs Tkinter and Pillow's ImageTk ({exc}). "
              "On Debian/Ubuntu: sudo apt install python3-tk python3-pil.imagetk")
        return 1

    session = Session()
    try:                       # optional: drag and drop of files onto the window
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        can_drop = True
    except Exception:          # not installed, or its Tk extension does not load here
        root = tk.Tk()
        can_drop = False
    hint = HINT_DROP if can_drop else HINT
    root.title(f"nemnest {__version__}")
    root.geometry("1180x720")
    root.minsize(820, 520)
    state = {"plate": 0, "busy": False, "photo": None, "resize": None,
             "progress": None, "cancel": False, "progress_win": None, "summary_win": None}
    jobs: queue.Queue = queue.Queue()

    # ---- toolbar
    bar = ttk.Frame(root, padding=(8, 6))
    bar.pack(side="top", fill="x")
    buttons = {}

    def button(key, text, command, parent=bar):
        buttons[key] = ttk.Button(parent, text=text, command=command)
        buttons[key].pack(side="left", padx=(0, 6))

    # ---- body: plate on the left, part list on the right
    body = ttk.Frame(root)
    body.pack(side="top", fill="both", expand=True)
    side = ttk.Frame(body, padding=(0, 0, 8, 0))
    side.pack(side="right", fill="y")
    ttk.Label(side, text="Parts on this plate").pack(anchor="w")
    tree = ttk.Treeview(side, columns=("n", "name", "h"), show="headings", height=20, selectmode="extended")
    for col, text, width, anchor in (("n", "#", 34, "e"), ("name", "File", 230, "w"), ("h", "Height", 64, "e")):
        tree.heading(col, text=text)
        tree.column(col, width=width, anchor=anchor, stretch=col == "name")
    tree.pack(fill="y", expand=True)
    notes = tk.StringVar()
    ttk.Label(side, textvariable=notes, wraplength=320, foreground="#b03030", justify="left").pack(anchor="w", pady=4)
    canvas = tk.Canvas(body, background="#1c1e22", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True, padx=8)

    # ---- bottom: plate navigation and status
    foot = ttk.Frame(root, padding=(8, 6))
    foot.pack(side="bottom", fill="x")
    plate_text = tk.StringVar(value="No plates yet")
    try:
        from tkinter import font as tkfont
        strong = tkfont.nametofont("TkDefaultFont").copy()
        strong.configure(weight="bold", size=strong.cget("size") + 2)
    except Exception:
        strong = None
    status = tk.StringVar(value=hint)
    prev_b = ttk.Button(foot, text="◀", width=3, command=lambda: go(-1))
    prev_b.pack(side="left")
    ttk.Label(foot, textvariable=plate_text, width=58, anchor="center", font=strong).pack(side="left")
    next_b = ttk.Button(foot, text="▶", width=3, command=lambda: go(+1))
    next_b.pack(side="left")
    ttk.Label(foot, textvariable=status, anchor="e").pack(side="right")

    # ---- drawing
    def draw():
        if state["busy"]:      # the layout is being rebuilt: redrawn when it is done
            return
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        plate_w, plate_d, _h = session.printer["size_mm"]
        scale = min((cw - 24) / plate_w, (ch - 24) / plate_d)
        image = session.render(state["plate"], scale)
        state["photo"] = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=state["photo"])
        if not session.placements:
            canvas.create_text(cw // 2, ch // 2, text=hint, fill="#8a8f98", font=("TkDefaultFont", 14))

    def refresh():
        n = session.n_plates
        state["plate"] = min(max(state["plate"], 0), max(n - 1, 0))
        tree.delete(*tree.get_children())
        if n:
            info = session.plate_info(state["plate"])
            plate_text.set(f"Plate {state['plate'] + 1} of {n}   ·   {info['parts']} parts   ·   "
                           f"fill {info['fill']:.0%}   ·   height {info['height_mm']:.1f} mm")
            for k, it in enumerate(session.plate_items(state["plate"])):
                tree.insert("", "end", iid=it.label, values=(k + 1, it.label, f"{it.part.size[2]:.1f}"))
            mode = "optimised" if session.optimised else "in order of arrival"
            status.set(f"{len(session.placements)} parts on {n} plates, {mode}   ·   {session.printer['name']}")
        else:
            plate_text.set("No plates yet")
            status.set(hint if not session.files else "No part could be placed")
        problems = [f"Does not fit the plate: {u['name']}" for u in session.unplaceable]
        problems += [f"Too tall for the printer: {name}" for name in session.too_tall]
        notes.set("\n".join(problems))
        prev_b.state(["!disabled"] if state["plate"] > 0 else ["disabled"])
        next_b.state(["!disabled"] if state["plate"] < n - 1 else ["disabled"])
        draw()

    def go(step, absolute=None):
        if not session.n_plates:
            return
        target = absolute if absolute is not None else state["plate"] + step
        target = min(max(target, 0), session.n_plates - 1)
        if target != state["plate"]:
            state["plate"] = target
            refresh()

    # ---- background work: one job at a time, the window stays responsive
    def run(label, work, done=None, jump_to_last=False):
        if state["busy"]:
            return
        state["busy"] = True
        status.set(label)
        for b in buttons.values():
            b.state(["disabled"])

        def target():
            try:
                jobs.put((True, work(), done, jump_to_last))
            except Exception as exc:   # shown to the user, never fatal
                jobs.put((False, exc, None, False))
        threading.Thread(target=target, daemon=True).start()

    def poll():
        try:
            ok, value, done, jump = jobs.get_nowait()
        except queue.Empty:
            update_progress()
            root.after(80, poll)
            return
        state["busy"] = False
        close_progress()
        for b in buttons.values():
            b.state(["!disabled"])
        if jump:
            state["plate"] = max(session.n_plates - 1, 0)
        refresh()
        if not ok and isinstance(value, Cancelled):
            status.set("Optimisation cancelled: the previous layout is unchanged")
        elif not ok:
            messagebox.showerror("nemnest", str(value))
        elif done:
            done(value)
        root.after(80, poll)

    def centre(win):
        win.update_idletasks()
        win.geometry(f"+{root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2}"
                     f"+{root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 2}")

    # ---- progress of a long search: bar, best result so far, cancel
    def open_progress():
        win = tk.Toplevel(root)
        win.title("Optimising")
        win.transient(root)
        win.resizable(False, False)
        phase, best = tk.StringVar(value="Preparing the parts…"), tk.StringVar(value=" ")

        def cancel():
            state["cancel"] = True
            phase.set("Cancelling…")
        win.protocol("WM_DELETE_WINDOW", cancel)
        frame = ttk.Frame(win, padding=14)
        frame.pack()
        ttk.Label(frame, textvariable=phase, width=44).pack(anchor="w")
        bar_ = ttk.Progressbar(frame, length=360, maximum=1000, mode="determinate")
        bar_.pack(pady=8)
        ttk.Label(frame, textvariable=best, font=strong).pack(anchor="w")
        ttk.Button(frame, text="Cancel", command=cancel).pack(anchor="e", pady=(8, 0))
        centre(win)
        state.update(progress=None, cancel=False,
                     progress_win={"win": win, "phase": phase, "best": best, "bar": bar_, "value": 0.0})

    def update_progress():
        box, info = state["progress_win"], state["progress"]
        if not box or not info:
            return
        box["value"] = max(box["value"], progress_fraction(info))      # the bar never goes back
        box["bar"]["value"] = int(1000 * box["value"])
        if not state["cancel"]:
            text = PHASES.get(info["phase"], info["phase"])
            if info["phase"] == "placing":
                text += f" ({info['attempt']} of {info['attempts']})"
            box["phase"].set(text + "…")
        if info["plates"]:
            box["best"].set(f"Best so far: {info['plates']} plates")

    def close_progress():
        if state["progress_win"]:
            state["progress_win"]["win"].destroy()
            state["progress_win"] = None

    def on_progress(info):                 # runs in the worker thread
        if state["cancel"]:
            raise Cancelled()
        state["progress"] = info

    # ---- summary after an import
    def import_summary(value):
        added, plates_before = value
        new = session.n_plates - plates_before
        status.set(f"Imported {added} parts   ·   {session.n_plates} plates"
                   + (f" ({new} new)" if plates_before and new else ""))
        if not session.config.get("gui_import_summary", True) or not added:
            return
        win = tk.Toplevel(root)
        win.title("Import")
        win.transient(root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=18)
        frame.pack()
        rows = [("Imported", f"{added} parts"), ("Plates", str(session.n_plates))]
        if plates_before:
            rows.append(("New plates", str(new)))
        skipped = len(session.unplaceable) + len(session.too_tall)
        if skipped:
            rows.append(("Not placed", f"{skipped} (too large for this printer)"))
        for r, (name, text) in enumerate(rows):
            ttk.Label(frame, text=name).grid(row=r, column=0, sticky="w", padx=(0, 24), pady=2)
            ttk.Label(frame, text=text, font=strong).grid(row=r, column=1, sticky="w", pady=2)
        hide = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Don't show this again", variable=hide).grid(
            row=len(rows), column=0, columnspan=2, sticky="w", pady=(12, 8))

        def close(_event=None):
            if hide.get():
                session.config["gui_import_summary"] = False
                cfg.save_config({"gui_import_summary": False})
            state["summary_win"] = None
            win.destroy()
        ok = ttk.Button(frame, text="OK", command=close)
        ok.grid(row=len(rows) + 1, column=1, sticky="e")
        win.bind("<Return>", close)
        win.bind("<Escape>", close)
        win.protocol("WM_DELETE_WINDOW", close)
        centre(win)
        ok.focus_set()
        win.grab_set()
        state["summary_win"] = win

    def importing(add):
        def work():
            before = session.n_plates
            added = add()
            session.layout()
            return added, before
        run("Reading models…", work, import_summary, jump_to_last=True)

    # ---- actions
    def add_stl():
        paths = filedialog.askopenfilenames(title="Add STL files", filetypes=[("STL", "*.stl"), ("All files", "*.*")])
        if paths:
            importing(lambda: session.add_files(paths))

    def add_folder():
        folder = filedialog.askdirectory(title="Add all STL files of a folder")
        if folder:
            importing(lambda: session.add_folder(folder))

    def remove_selected():
        picked = [session.label_index[iid] for iid in tree.selection() if iid in session.label_index]
        if picked:
            run("Rearranging…", lambda: (session.remove(picked), session.layout()))

    def clear_all():
        if session.files and messagebox.askyesno("nemnest", "Remove all parts?"):
            session.clear()
            state["plate"] = 0
            refresh()

    def optimise():
        if not session.files:
            return
        win = tk.Toplevel(root)
        win.title("Optimise")
        win.transient(root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=14)
        frame.pack()
        rotate, height, seconds = tk.BooleanVar(value=True), tk.BooleanVar(value=False), tk.IntVar(value=120)
        ttk.Label(frame, text="Rearrange all parts on as few plates as possible.").pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(frame, text="Allow rotating parts on the plate (supports stay valid)",
                        variable=rotate).pack(anchor="w")
        ttk.Checkbutton(frame, text="Also group tall parts to shorten total print time",
                        variable=height).pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(anchor="w", pady=8)
        ttk.Label(row, text="Search for at most").pack(side="left")
        ttk.Spinbox(row, from_=10, to=900, increment=10, width=5, textvariable=seconds).pack(side="left", padx=6)
        ttk.Label(row, text="seconds").pack(side="left")
        gap = tk.StringVar(value=f"{session.gap:g}")
        row = ttk.Frame(frame)
        row.pack(anchor="w", pady=(0, 8))
        ttk.Label(row, text="Minimum gap between parts").pack(side="left")
        ttk.Spinbox(row, from_=0, to=cfg.MAX_GAP_MM, increment=0.5, width=5, textvariable=gap).pack(side="left", padx=6)
        ttk.Label(row, text=f"mm (0 to {cfg.MAX_GAP_MM:g}; rafts may still touch)").pack(side="left")

        def start():
            try:
                session.set_gap(gap.get())      # remembered for next time
            except ValueError as exc:
                messagebox.showerror("nemnest", str(exc).capitalize(), parent=win)
                return
            opts = dict(optimise=True, rotate=rotate.get(), optimize_height=height.get(),
                        time_limit=float(seconds.get()))
            win.destroy()
            state["plate"] = 0
            open_progress()
            run("Optimising…", lambda: session.layout(progress=on_progress, **opts))
        ttk.Button(frame, text="Optimise", command=start).pack(side="right")
        ttk.Button(frame, text="Cancel", command=win.destroy).pack(side="right", padx=6)
        win.grab_set()

    def export(slice_too=False):
        if not session.placements:
            return
        folder = filedialog.askdirectory(title="Folder for the plate files (existing plate_NN files are replaced)")
        if not folder:
            return

        def work():
            result = session.export(folder)
            sliced = None
            if slice_too:
                from . import server
                sliced = server.tool_slice_plates({"plates_dir": folder, "printer": session.printer})
            return result, sliced

        def done(value):
            result, sliced = value
            text = f"{result['plates_count']} plates written to\n{folder}"
            if sliced:
                total = sum((s["profile_applied"] or {}).get("printing_time_s", 0) for s in sliced["sliced"])
                text += f"\n\n{len(sliced['sliced'])} print files ({sliced['format']})"
                if total:
                    text += f", about {total // 3600} h {total % 3600 // 60:02d} min of printing"
                text += ".\nCheck the first layers in your usual slicer before the first print."
            messagebox.showinfo("nemnest", text)
        run("Slicing… this takes a few minutes" if slice_too else "Writing the plates…", work, done)

    printer_var = tk.StringVar()

    def printer_names():
        return {p.get("name", pid): pid for pid, p in sorted(cfg.list_printers().items())}

    def pick_printer(_event=None):
        pid = printer_names().get(printer_var.get())
        if pid and pid != session.printer.get("id"):
            session.set_printer(pid)
            state["plate"] = 0
            run("Rearranging…", session.layout)

    def add_printer():
        win = tk.Toplevel(root)
        win.title("Add a printer")
        win.transient(root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=14)
        frame.pack()
        fields = {}
        rows = (("name", "Name", 28), ("x", "Plate width X (mm)", 10), ("y", "Plate depth Y (mm)", 10),
                ("z", "Maximum height Z (mm)", 10), ("px", "Screen pixels X (optional)", 10),
                ("py", "Screen pixels Y (optional)", 10))
        for r, (key, text, width) in enumerate(rows):
            ttk.Label(frame, text=text).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 10))
            fields[key] = ttk.Entry(frame, width=width)
            fields[key].grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(frame, text="The pixel resolution is only needed for slicing.",
                  foreground="#666").grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(6, 10))

        def save():
            try:
                px = [fields["px"].get().strip(), fields["py"].get().strip()]
                profile = save_printer(fields["name"].get(), [fields[k].get().replace(",", ".") for k in "xyz"],
                                       px if all(px) else None)
            except ValueError as exc:
                messagebox.showerror("nemnest", f"Please check the values: {exc}", parent=win)
                return
            win.destroy()
            combo["values"] = list(printer_names())
            printer_var.set(profile["name"])
            pick_printer()
        ttk.Button(frame, text="Save", command=save).grid(row=len(rows) + 1, column=1, sticky="e")
        ttk.Button(frame, text="Cancel", command=win.destroy).grid(row=len(rows) + 1, column=0, sticky="e", padx=6)
        fields["name"].focus_set()
        win.grab_set()

    button("add", "Add STL…", add_stl)
    button("folder", "Add folder…", add_folder)
    button("remove", "Remove selected", remove_selected)
    button("clear", "Clear", clear_all)
    ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
    button("optimise", "Optimise…", optimise)
    button("export", "Export plates…", export)
    button("slice", "Export and slice…", lambda: export(slice_too=True))
    button("add_printer", "Add printer…", add_printer, parent=bar)
    buttons["add_printer"].pack_forget()
    buttons["add_printer"].pack(side="right")
    combo = ttk.Combobox(bar, textvariable=printer_var, values=list(printer_names()), state="readonly", width=30)
    combo.pack(side="right", padx=6)
    ttk.Label(bar, text="Printer").pack(side="right")
    printer_var.set(session.printer.get("name", ""))
    combo.bind("<<ComboboxSelected>>", pick_printer)


    # ---- plate navigation: wheel, arrows, page keys
    def wheel(event):
        go(-1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4) else +1)
    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind(sequence, wheel)
    for keys, step in ((("<Left>", "<Up>", "<Prior>"), -1), (("<Right>", "<Down>", "<Next>"), +1)):
        for key in keys:
            root.bind(key, lambda _e, s=step: go(s) if not isinstance(root.focus_get(), (ttk.Entry, ttk.Combobox, ttk.Treeview)) else None)
    root.bind("<Home>", lambda _e: go(0, absolute=0))
    root.bind("<End>", lambda _e: go(0, absolute=session.n_plates - 1))
    tree.bind("<Delete>", lambda _e: remove_selected())

    def dropped(event):
        paths = root.tk.splitlist(event.data)
        if not paths:
            return
        if state["busy"]:
            status.set("Busy: drop the files again when this is done")
            return
        if not any(Path(p).is_dir() or str(p).lower().endswith(".stl") for p in paths):
            status.set("Nothing to import: drop STL files or folders")
            return
        importing(lambda: session.add_paths(paths))
    state["dropped"] = dropped        # also reachable without a real drag (tests)
    if can_drop:
        for widget in (root, canvas, tree):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", dropped)

    def resized(_event):
        if state["resize"]:
            root.after_cancel(state["resize"])
        state["resize"] = root.after(120, draw)
    canvas.bind("<Configure>", resized)

    setup_updates(root, tk, ttk, messagebox, body, state, centre)

    session.add_paths(argv or [])
    if session.files:
        session.layout()
    root.after(50, refresh)
    root.after(80, poll)
    root.mainloop()
    return 0


def setup_updates(root, tk, ttk, messagebox, body, state, centre):
    """Look for a newer release in the background and offer it in a banner
    above the plate. Installing asks first, downloads with a progress bar,
    then closes the window; a script swaps the files and reopens nemnest."""
    import webbrowser
    from . import update

    config = cfg.load_config()
    if not config.get("check_updates", True) or os.environ.get("NEMNEST_NO_UPDATE_CHECK"):
        return
    found: queue.Queue = queue.Queue()

    def look():
        try:
            found.put(("check", update.check()))
        except Exception:
            found.put(("check", None))          # offline or GitHub unreachable: say nothing
    threading.Thread(target=look, daemon=True).start()

    banner = ttk.Frame(root, padding=(10, 6))
    message = tk.StringVar()
    ttk.Label(banner, textvariable=message).pack(side="left")
    buttons = ttk.Frame(banner)
    buttons.pack(side="right")

    def show(rel):
        tag = rel["tag"]
        for child in buttons.winfo_children():
            child.destroy()
        if rel["action"] == "install":
            message.set(f"nemnest {tag} is available (you have {__version__}).")
            ttk.Button(buttons, text="Update and restart…", command=lambda: install(rel)).pack(side="left", padx=4)
        elif rel["action"] == "pip":
            message.set(f"nemnest {tag} is available. Update with: "
                        f"pip install --upgrade git+https://github.com/{update.REPO}@{tag}")
        else:
            message.set(f"nemnest {tag} is available (you have {__version__}).")
        ttk.Button(buttons, text="What's new", command=lambda: webbrowser.open(rel["url"])).pack(side="left", padx=4)
        ttk.Button(buttons, text="Not now", command=banner.pack_forget).pack(side="left", padx=4)
        banner.pack(side="top", fill="x", before=body)

    def install(rel):
        if state["busy"]:
            messagebox.showinfo("nemnest", "Wait for the current task to finish, then update.")
            return
        if not messagebox.askyesno(
                "Update nemnest",
                f"Download nemnest {rel['tag']} and restart?\n\nParts in the window that are not "
                "exported will be lost. Exported plates and your settings are kept."):
            return
        win = tk.Toplevel(root)
        win.title("Updating nemnest")
        win.transient(root)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)      # nothing to cancel halfway safely
        frame = ttk.Frame(win, padding=14)
        frame.pack()
        text = tk.StringVar(value=f"Downloading nemnest {rel['tag']}…")
        ttk.Label(frame, textvariable=text, width=44).pack(anchor="w")
        bar_ = ttk.Progressbar(frame, length=360, maximum=1000, mode="determinate")
        bar_.pack(pady=8)
        centre(win)
        state["busy"] = True
        share = {"done": 0, "total": 0}

        def work():
            try:
                found.put(("ready", update.prepare(
                    rel, lambda d, t: share.update(done=d, total=t))))
            except Exception as exc:
                found.put(("failed", exc))
        threading.Thread(target=work, daemon=True).start()
        state["update_win"] = (win, bar_, text, share)
        root.after(150, poll_updates)       # the check's polling has stopped: restart it

    def poll_updates():
        try:
            kind, value = found.get_nowait()
        except queue.Empty:
            if state.get("update_win"):
                _, bar_, _, share = state["update_win"]
                if share["total"]:
                    bar_["value"] = int(1000 * share["done"] / share["total"])
            root.after(150, poll_updates)
            return
        if kind == "check":
            if value:
                show(value)
            return                          # nothing else will come unless the user installs
        win = state.pop("update_win")[0]
        win.destroy()
        state["busy"] = False
        if kind == "failed":
            messagebox.showerror("nemnest", f"The update was not installed.\n\n{value}")
            return
        new, target, work = value
        try:
            update.start_install(new, target, work)
        except Exception as exc:
            messagebox.showerror("nemnest", f"The update was not installed.\n\n{exc}")
            return
        root.destroy()

    root.after(150, poll_updates)


if __name__ == "__main__":
    raise SystemExit(main())
