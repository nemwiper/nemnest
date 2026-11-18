"""Window to look at the layers of a sliced file and at the islands found in it.

Opened from the main window ("Check slices…"). The list on the right holds the
islands, largest first; choosing one jumps to its layer and zooms on it. The
layer is drawn with what has something underneath in grey and what has nothing
under it in orange, so an island is an orange patch standing on its own.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from . import check, goo

COLOURS = np.array([(28, 30, 34), (190, 192, 198), (255, 140, 0)], dtype=np.uint8)   # empty, held, nothing below


def picture(code, pixel_mm, size):
    """RGB PIL image of a layer view, scaled to fit `size` (w, h) with square millimetres."""
    from PIL import Image
    h, w = code.shape
    mm_w, mm_h = w * pixel_mm[0], h * pixel_mm[1]
    scale = min(size[0] / mm_w, size[1] / mm_h)
    out = (max(1, int(mm_w * scale)), max(1, int(mm_h * scale)))
    return Image.fromarray(COLOURS[code], "RGB").resize(out, Image.NEAREST), scale


def open_checker(root, path=None, start_dir=None):
    """Open the window; with `path` the check starts at once, otherwise a file is asked for."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    if path is None:
        path = filedialog.askopenfilename(parent=root, title="Sliced file to check",
                                          initialdir=start_dir or None,
                                          filetypes=[("Elegoo .goo", "*.goo"), ("All files", "*.*")])
        if not path:
            return None
    if Path(path).suffix.lower() != ".goo":
        messagebox.showinfo("nemnest", "Only .goo files can be checked for now.", parent=root)
        return None

    win = tk.Toplevel(root)
    win.title(f"Check slices: {Path(path).name}")
    win.geometry("1240x760")
    win.minsize(900, 560)
    st = {"path": str(path), "goo": None, "result": None, "layer": 0, "box": None, "item": None,
          "photo": None, "busy": True, "redraw": None, "closed": False, "scale": 1.0, "origin": (0, 0), "shown": None}
    events: queue.Queue = queue.Queue()

    # ---- top: what is being checked, threshold
    top = ttk.Frame(win, padding=(8, 6))
    top.pack(side="top", fill="x")
    summary = tk.StringVar(value="Checking…")
    ttk.Label(top, textvariable=summary).pack(side="left")
    bar = ttk.Progressbar(top, length=220, maximum=1000, mode="determinate")
    bar.pack(side="left", padx=10)
    min_px = tk.IntVar(value=10)
    ttk.Label(top, text="px").pack(side="right")
    spin = ttk.Spinbox(top, from_=1, to=100000, increment=5, width=7, textvariable=min_px)
    spin.pack(side="right", padx=(6, 4))
    ttk.Label(top, text="Hide islands smaller than").pack(side="right")

    # ---- right: the islands
    side = ttk.Frame(win, padding=(0, 0, 8, 8))
    side.pack(side="right", fill="y")
    ttk.Label(side, text="Islands, largest first").pack(anchor="w")
    box_frame = ttk.Frame(side)
    box_frame.pack(fill="y", expand=True)
    tree = ttk.Treeview(box_frame, columns=("layer", "z", "px", "mm2"), show="headings", height=22, selectmode="browse")
    for col, text, width in (("layer", "Layer", 60), ("z", "Height", 84), ("px", "Pixels", 76), ("mm2", "mm²", 76)):
        tree.heading(col, text=text)
        tree.column(col, width=width, anchor="e", stretch=False)
    scroll = ttk.Scrollbar(box_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="y", expand=True)
    scroll.pack(side="left", fill="y")
    legend = ttk.Frame(side)
    legend.pack(anchor="w", pady=(8, 0))
    for colour, text in (("#bec0c6", "has something under it"), ("#ff8c00", "nothing under it")):
        row = ttk.Frame(legend)
        row.pack(anchor="w")
        tk.Label(row, background=colour, width=2).pack(side="left", padx=(0, 6), pady=1)
        ttk.Label(row, text=text).pack(side="left")

    # ---- bottom: layer navigation
    foot = ttk.Frame(win, padding=(8, 6))
    foot.pack(side="bottom", fill="x")
    where = tk.StringVar(value="")
    prev_b = ttk.Button(foot, text="◀", width=3, command=lambda: step(-1))
    prev_b.pack(side="left")
    next_b = ttk.Button(foot, text="▶", width=3, command=lambda: step(+1))
    next_b.pack(side="left", padx=(4, 10))
    slider = ttk.Scale(foot, from_=0, to=1, orient="horizontal", length=320)
    slider.pack(side="left")
    ttk.Label(foot, textvariable=where, width=44).pack(side="left", padx=10)
    whole_b = ttk.Button(foot, text="Whole plate", command=lambda: zoom(None))
    whole_b.pack(side="right")

    canvas = tk.Canvas(win, background="#1c1e22", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True, padx=8, pady=(0, 8))

    # ---- drawing
    def draw():
        st["redraw"] = None
        g = st["goo"]
        if g is None or st["closed"]:
            return
        cw, ch = max(canvas.winfo_width(), 60), max(canvas.winfo_height(), 60)
        code, shown = check.view(g, st["layer"], st["box"], max_side=max(cw, ch))
        res = st["result"]
        pixel = res["pixel_mm"] if res else (g.header["x_size"] / g.header["x_resolution"],
                                             g.header["y_size"] / g.header["y_resolution"])
        block = (shown[1] - shown[0]) / max(code.shape[1], 1)          # file pixels per picture cell
        image, scale = picture(code, (pixel[0] * block, pixel[1] * block), (cw - 16, ch - 16))
        st["photo"] = ImageTk.PhotoImage(image)
        canvas.delete("all")
        left, topy = (cw - image.width) // 2, (ch - image.height) // 2
        canvas.create_image(left, topy, image=st["photo"], anchor="nw")
        canvas.create_rectangle(left - 1, topy - 1, left + image.width, topy + image.height, outline="#8a8f98")
        st.update(scale=scale, origin=(left, topy), shown=shown)
        item = st["item"]
        if item is not None and item["layer"] == st["layer"]:
            sx = image.width / max(shown[1] - shown[0], 1)
            sy = image.height / max(shown[3] - shown[2], 1)
            x0, x1 = left + (item["x"][0] - shown[0]) * sx, left + (item["x"][1] - shown[0]) * sx
            y0, y1 = topy + (item["y"][0] - shown[2]) * sy, topy + (item["y"][1] - shown[2]) * sy
            pad = 6
            canvas.create_rectangle(x0 - pad, y0 - pad, x1 + pad, y1 + pad, outline="#ff3030", width=2)
        z = st["layer"] * g.header["layer_thickness"]
        where.set(f"Layer {st['layer']} of {g.header['layer_count'] - 1}   ·   {z:.2f} mm"
                  + ("   ·   zoomed" if st["box"] else ""))

    def schedule():
        if st["redraw"]:
            win.after_cancel(st["redraw"])
        st["redraw"] = win.after(40, draw)

    def set_layer(n, from_slider=False):
        g = st["goo"]
        if g is None:
            return
        n = min(max(int(round(n)), 0), g.header["layer_count"] - 1)
        if n != st["layer"]:
            st["layer"] = n
            schedule()
        if not from_slider:
            slider.set(n)

    def step(d):
        set_layer(st["layer"] + d)

    def zoom(item):
        """Zoom on an island (with some of what is around it), or back to the whole plate."""
        g = st["goo"]
        if item is None or g is None:
            st["box"] = None
        else:
            w, h = g.header["x_resolution"], g.header["y_resolution"]
            cx, cy = (item["x"][0] + item["x"][1]) / 2, (item["y"][0] + item["y"][1]) / 2
            half_w = max(360, 3 * (item["x"][1] - item["x"][0]))
            half_h = max(260, 3 * (item["y"][1] - item["y"][0]))
            st["box"] = (max(0, cx - half_w), min(w, cx + half_w), max(0, cy - half_h), min(h, cy + half_h))
        schedule()

    def fill():
        res = st["result"]
        tree.delete(*tree.get_children())
        if not res:
            return
        try:
            limit = max(1, int(min_px.get()))
        except (tk.TclError, ValueError):                 # the box is empty while the user types
            return
        shown = [i for i in res["items"] if i["area_px"] >= limit]
        for k, i in enumerate(shown[:2000]):
            tree.insert("", "end", iid=str(k), values=(i["layer"], f"{i['z_mm']:.2f} mm", i["area_px"], f"{i['area_mm2']:.3f}"))
        st["listed"] = shown
        hidden = len(res["items"]) - len(shown)
        text = f"{len(shown)} islands of {limit} px or more" if shown else f"No islands of {limit} px or more"
        summary.set(text + (f"   ·   {hidden} smaller ones hidden" if hidden else "")
                    + f"   ·   {res['layers']} layers")

    def picked(_event=None):
        sel = tree.selection()
        if not sel:
            return
        item = st["listed"][int(sel[0])]
        st["item"] = item
        st["layer"] = -1                       # force the redraw even on the same layer
        zoom(item)
        set_layer(item["layer"])

    # ---- the check runs in the background
    def work():
        try:
            def progress(done, total):
                if st["closed"]:
                    raise InterruptedError()
                events.put(("progress", done / max(total, 1)))
            events.put(("done", check.check_file(st["path"], progress=progress)))
        except InterruptedError:
            pass
        except Exception as exc:               # shown to the user
            events.put(("error", exc))

    def poll():
        if st["closed"]:
            return
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "progress":
                    bar["value"] = int(1000 * value)
                elif kind == "error":
                    st["busy"] = False
                    bar.pack_forget()
                    summary.set("The check failed")
                    messagebox.showerror("nemnest", str(value), parent=win)
                else:
                    st["busy"] = False
                    st["result"] = value
                    bar.pack_forget()
                    fill()
                    if st.get("listed"):
                        tree.selection_set("0")
                        tree.focus("0")
                        picked()
        except queue.Empty:
            pass
        win.after(80, poll)

    def close():
        st["closed"] = True
        if st["goo"] is not None:
            st["goo"].f.close()
        win.destroy()

    try:
        st["goo"] = goo.GooFile(st["path"])
    except Exception as exc:
        win.destroy()
        messagebox.showerror("nemnest", f"Cannot read {Path(path).name}: {exc}", parent=root)
        return None
    slider.configure(to=st["goo"].header["layer_count"] - 1, command=lambda v: set_layer(float(v), from_slider=True))
    tree.bind("<<TreeviewSelect>>", picked)
    min_px.trace_add("write", lambda *_: fill())          # typing, the arrows and pasting all land here

    def wheel(event):
        step(-1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4) else +1)
    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind(sequence, wheel)
    for key, d in (("<Left>", -1), ("<Right>", +1), ("<Prior>", -10), ("<Next>", +10)):
        win.bind(key, lambda _e, d=d: step(d) if not isinstance(win.focus_get(), (ttk.Entry, ttk.Spinbox, ttk.Treeview)) else None)
    canvas.bind("<Configure>", lambda _e: schedule())
    win.protocol("WM_DELETE_WINDOW", close)
    threading.Thread(target=work, daemon=True).start()
    win.after(60, draw)
    win.after(80, poll)
    return win
