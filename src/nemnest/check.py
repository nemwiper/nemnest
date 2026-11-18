"""Checks on a sliced file, done directly on the runs of each layer.

A layer is a list of lit segments (row, x0, x1). Nothing here ever builds the
layer image, which is what makes a 16K plate take seconds instead of minutes.

What is looked for, layer by layer against the layer below:

island    a region with nothing under it. It cures in mid-resin and ends up on
          the film, or as a blob on the print.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import goo

EMPTY = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))


# ---------------------------------------------------------------- segments

def segments_from_runs(lengths, values, width: int):
    """Lit segments (row, x0, x1), x1 exclusive, sorted by row then x."""
    lengths = np.asarray(lengths, dtype=np.int64)
    lit = np.asarray(values) > 0
    end = np.cumsum(lengths)
    start = (end - lengths)[lit]
    end = end[lit]
    if len(start) == 0:
        return EMPTY
    row0, row1 = start // width, (end - 1) // width
    same = row0 == row1
    if same.all():
        return row0, start - row0 * width, end - row0 * width
    rows, x0, x1 = [row0[same]], [start[same] - row0[same] * width], [end[same] - row0[same] * width]
    for s, e, r0, r1 in zip(start[~same], end[~same], row0[~same], row1[~same]):   # a run wrapping over rows
        r = np.arange(int(r0), int(r1) + 1)
        rows.append(r)
        x0.append(np.maximum(s, r * width) - r * width)
        x1.append(np.minimum(e, (r + 1) * width) - r * width)
    rows, x0, x1 = np.concatenate(rows), np.concatenate(x0), np.concatenate(x1)
    order = np.lexsort((x0, rows))
    return rows[order], x0[order], x1[order]


def segments_from_image(image):
    """Segments of a 2D array (non-zero = lit). For tests and small images."""
    image = np.asarray(image) > 0
    height, width = image.shape
    flat = np.concatenate([image, np.zeros((height, 1), bool)], axis=1).ravel()
    change = np.flatnonzero(np.diff(np.concatenate([[False], flat])))
    start, end = change[0::2], change[1::2]
    rows = start // (width + 1)
    return rows, start - rows * (width + 1), end - rows * (width + 1)


def _ranges(seg_a, seg_b, width, row_shift=0, touch=0):
    """For each segment of A, the range [lo, hi) of the segments of B on row
    (row + row_shift) whose x range overlaps it, widened by `touch` pixels."""
    k = width + 2
    rows_b, b0, b1 = seg_b
    base = (seg_a[0] + row_shift) * k
    lo = np.searchsorted(rows_b * k + b1, base + seg_a[1] - touch, side="right")
    hi = np.searchsorted(rows_b * k + b0, base + seg_a[2] + touch, side="left")
    return lo, np.maximum(hi, lo)


def regions(seg, width):
    """(label of the connected region of each segment, number of regions), 8-connectivity."""
    m = len(seg[0])
    if m == 0:
        return np.zeros(0, np.int64), 0
    lo, hi = _ranges(seg, seg, width, row_shift=1, touch=1)
    count = hi - lo
    total = int(count.sum())
    a = np.repeat(np.arange(m), count)
    b = np.arange(total) - np.repeat(np.cumsum(count) - count, count) + np.repeat(lo, count)
    graph = coo_matrix((np.ones(total, np.int8), (a, b)), shape=(m, m))
    n, label = connected_components(graph, directed=False)
    return label, n


def _stats(seg, label, n):
    """Per region: area in pixels and bounding box (x0, x1, y0, y1), x1 and y1 exclusive."""
    rows, x0, x1 = seg
    area = np.bincount(label, weights=(x1 - x0), minlength=n).astype(np.int64)
    bx0 = np.full(n, np.iinfo(np.int64).max)
    bx1 = np.zeros(n, np.int64)
    by0 = np.full(n, np.iinfo(np.int64).max)
    by1 = np.zeros(n, np.int64)
    np.minimum.at(bx0, label, x0)
    np.maximum.at(bx1, label, x1)
    np.minimum.at(by0, label, rows)
    np.maximum.at(by1, label, rows + 1)
    return area, bx0, bx1, by0, by1


# ------------------------------------------------------------------ checks

def check_layer(cur, cur_regions, below, width, pixel_mm):
    """Islands of one layer: regions with no pixel directly over a lit pixel
    of the layer below. `cur_regions` is the result of `regions(cur, width)`,
    `pixel_mm` is (x pitch, y pitch). Returns a list of dicts."""
    label, n = cur_regions
    if n == 0:
        return []
    px, py = pixel_mm
    held = np.zeros(n, bool)
    if len(below[0]):
        lo, hi = _ranges(cur, below, width)
        held[label[hi > lo]] = True
    free = np.flatnonzero(~held)
    if len(free) == 0:
        return []
    area, bx0, bx1, by0, by1 = _stats(cur, label, n)
    return [{"kind": "island", "area_px": int(area[r]), "area_mm2": float(area[r] * px * py),
             "x": [int(bx0[r]), int(bx1[r])], "y": [int(by0[r]), int(by1[r])]} for r in free]


def _chunk(job):
    path, first, last = job
    g = goo.GooFile(path)
    h = g.header
    width = h["x_resolution"]
    pixel = (h["x_size"] / width, h["y_size"] / h["y_resolution"])
    below = segments_from_runs(*g._runs(first - 1), width)
    out = []
    for n in range(first, last):
        cur = segments_from_runs(*g._runs(n), width)
        for item in check_layer(cur, regions(cur, width), below, width, pixel):
            item["layer"] = n
            item["z_mm"] = round(n * h["layer_thickness"], 3)
            out.append(item)
        below = cur
    g.f.close()
    return out


def check_file(path, workers=None, progress=None, chunk=40) -> dict:
    """All islands of a sliced .goo file, largest first.
    `progress(done_layers, total_layers)` is called as chunks finish; if it
    raises, the check stops and the exception propagates."""
    path = str(path)
    g = goo.GooFile(path)
    h = g.header
    g.f.close()
    count = h["layer_count"]
    jobs = [(path, a, min(a + chunk, count)) for a in range(1, count, chunk)]
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    items, done = [], 0
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            items += _chunk(job)
            done += job[2] - job[1]
            if progress:
                progress(done, count - 1)
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(_chunk, job): job for job in jobs}
            try:
                for fut in as_completed(futures):
                    items += fut.result()
                    job = futures[fut]
                    done += job[2] - job[1]
                    if progress:
                        progress(done, count - 1)
            except BaseException:
                for fut in futures:
                    fut.cancel()
                raise
    items.sort(key=lambda i: (-i["area_px"], i["layer"]))
    return {
        "file": path,
        "layers": count,
        "layer_thickness_mm": h["layer_thickness"],
        "resolution_px": [h["x_resolution"], h["y_resolution"]],
        "pixel_mm": [h["x_size"] / h["x_resolution"], h["y_size"] / h["y_resolution"]],
        "compiled_reader": goo.COMPILED,
        "counts": {"island": len(items)},
        "items": items,
    }


def view(path_or_file, layer: int, box=None, max_side=1400):
    """Picture of a layer for display: an (h, w) uint8 array with 0 = empty,
    1 = lit and also lit in the layer below, 2 = lit with nothing below.
    `box` = (x0, x1, y0, y1) in pixels crops before scaling. Returns the array
    and the box actually shown."""
    g = path_or_file if isinstance(path_or_file, goo.GooFile) else goo.GooFile(path_or_file)
    h = g.header
    width, height = h["x_resolution"], h["y_resolution"]
    x0, x1, y0, y1 = box if box else (0, width, 0, height)
    x0, x1, y0, y1 = max(0, int(x0)), min(width, int(x1)), max(0, int(y0)), min(height, int(y1))
    cur = g.image(layer)[y0:y1, x0:x1] > 0
    below = g.image(layer - 1)[y0:y1, x0:x1] > 0 if layer > 0 else np.ones_like(cur)
    code = cur.astype(np.uint8) + (cur & ~below)
    step = max(1, int(np.ceil(max((x1 - x0), (y1 - y0)) / max_side)))
    if step > 1:                                          # shrink keeping the strongest value of each block
        hh, ww = (code.shape[0] // step) * step, (code.shape[1] // step) * step
        if hh and ww:
            code = code[:hh, :ww].reshape(hh // step, step, ww // step, step).max(axis=(1, 3))
    return code, (x0, x1, y0, y1)
