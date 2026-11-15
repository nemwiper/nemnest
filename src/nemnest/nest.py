"""2D nesting of pre-supported STLs onto as few plates as possible.

Each model is projected onto the XY plane (true footprint, not the bounding
box), rotated only about Z (so the supports stay valid) and nested onto the
plate. Plate coordinates: origin at the centre, mm.
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft as sfft
from scipy import ndimage
from scipy.spatial import ConvexHull



def _cache_dir() -> Path:
    """Per-user cache folder for computed footprints (override: NEMNEST_CACHE)."""
    if os.environ.get("NEMNEST_CACHE"):
        return Path(os.environ["NEMNEST_CACHE"])
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "nemnest" / "cache"
    if (Path.home() / "Library" / "Caches").is_dir():
        return Path.home() / "Library" / "Caches" / "nemnest"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nemnest"


CACHE_DIR = _cache_dir()
CACHE_VERSION = "1"

FLUSH_TOL = 0.05   # mm of overhang tolerated for parts flush against the plate edge


def gpu_module():
    """CuPy if installed and a usable GPU is present, otherwise None."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy
            cupy.fft.rfft2(cupy.zeros((8, 8), dtype=cupy.float32))
        return cupy
    except Exception:
        return None


STL_REC = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


# ---------------------------------------------------------------- STL I/O

def load_stl(path) -> np.ndarray:
    """Return the triangles as an (n, 3, 3) float32 array."""
    path = str(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        if len(head) == 84:
            n = struct.unpack("<I", head[80:84])[0]
            expected = 84 + 50 * n
            if n > 0 and (size == expected or (size > expected and not head.lstrip().startswith(b"solid"))):
                data = np.fromfile(f, dtype=STL_REC, count=n)
                return np.ascontiguousarray(data["v"])
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    arr = np.array(nums, dtype=np.float32)
    if arr.size == 0 or len(arr) % 3:
        raise ValueError(f"Unreadable STL: {path}")
    return arr.reshape(-1, 3, 3)


def _records(tris: np.ndarray) -> np.ndarray:
    rec = np.zeros(len(tris), dtype=STL_REC)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    rec["n"] = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec["v"] = tris
    return rec


def save_stl(path, tris: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"nemnest".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        _records(tris.astype(np.float32)).tofile(f)


# ---------------------------------------------------------------- footprints

@dataclass
class Part:
    name: str
    path: str
    ntris: int
    size: tuple          # (dx, dy, dz) mm
    zmin: float
    mask: np.ndarray     # XY footprint, rows = increasing y
    origin: np.ndarray   # model coordinates of the corner of pixel (0, 0)
    res: float
    area: float          # footprint area in mm^2
    hull: np.ndarray     # exact convex hull in XY (model coordinates)
    body: np.ndarray     # footprint without the raft (same grid as mask)
    body_area: float

    @property
    def center(self) -> np.ndarray:
        h, w = self.mask.shape
        return self.origin + np.array([w, h]) * self.res / 2


def rasterize(xy: np.ndarray, res: float, frame=None):
    """Conservative footprint of 2D triangles (n, 3, 2). Returns (mask, origin).

    `frame` = (origin, shape) of an already computed footprint: the new one is
    drawn on the same grid (used for the body without the raft)."""
    pad = 2
    if frame is not None:
        mn = np.asarray(frame[0]) + pad * res
        h, w = frame[1]
        if len(xy) == 0:
            return np.zeros((h, w), dtype=bool), mn - pad * res
    else:
        flat = xy.reshape(-1, 2)
        mn = flat.min(0)
        mx = flat.max(0)
        w = int(math.floor((mx[0] - mn[0]) / res)) + 1 + 2 * pad
        h = int(math.floor((mx[1] - mn[1]) / res)) + 1 + 2 * pad
        w += w % 2  # even dimensions: 90-degree rotations stay exact
        h += h % 2
    px = np.floor((xy - mn) / res).astype(np.int32) + pad
    mask = np.zeros((h, w), dtype=bool)
    mask[px[..., 1].ravel(), px[..., 0].ravel()] = True
    span = px.max(1) - px.min(1)
    big = span.max(1) > 1
    if big.any():
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        for t in px[big].reshape(-1, 6).tolist():
            draw.polygon(t, fill=1, outline=1)
        mask |= np.asarray(img, dtype=bool)
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool))
    return mask, mn - pad * res


def cache_file(path, res: float, raft_height: float) -> Path:
    path = str(Path(path).resolve())
    st = os.stat(path)
    key = hashlib.sha1("|".join(map(str, (path, st.st_size, st.st_mtime_ns, res, raft_height,
                                          CACHE_VERSION))).encode()).hexdigest()
    return CACHE_DIR / f"{key}.npz"


def _load_job(job):
    return load_part(*job)


def load_parts(jobs, workers: int | None = None) -> list:
    """Load several footprints; `jobs` = [(path, res, name, raft_height), ...].

    Files not yet in the cache are processed in parallel across several
    processes (largest first, to balance the load); if the pool fails to start
    we fall back to loading them sequentially. The output order matches
    `jobs`."""
    out: list = [None] * len(jobs)
    cold = [k for k, j in enumerate(jobs) if not cache_file(j[0], j[1], j[3]).is_file()]
    if workers is None:
        workers = max(1, min(6, (os.cpu_count() or 2) - 2))
    workers = min(workers, len(cold))
    if workers > 1:
        cold.sort(key=lambda k: -os.path.getsize(jobs[k][0]))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                for k, part in zip(cold, pool.map(_load_job, [jobs[k] for k in cold])):
                    out[k] = part
        except Exception:
            pass   # pool unavailable or interrupted: the missing ones are loaded below
    for k, job in enumerate(jobs):
        if out[k] is None:
            out[k] = load_part(*job)
    return out


def load_part(path, res: float, name: str | None = None, raft_height: float = 0.0,
              use_cache: bool = True) -> Part:
    """Footprint of an STL. The result is saved to disk: as long as the file
    does not change (size and date), later calls do not re-read the STL."""
    path = str(Path(path).resolve())
    cached = cache_file(path, res, raft_height)
    if use_cache and cached.is_file():
        try:
            with np.load(cached) as z:
                meta = z["meta"]
                mask = z["mask"]
                return Part(name=name or Path(path).name, path=path, ntris=int(meta[0]),
                            size=tuple(float(v) for v in meta[1:4]), zmin=float(meta[4]),
                            mask=mask, origin=z["origin"], res=res, area=float(mask.sum()) * res * res,
                            hull=z["hull"], body=z["body"], body_area=float(z["body"].sum()) * res * res)
        except Exception:
            pass   # unreadable cache: recompute
    part = _compute_part(path, res, name, raft_height)
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, mask=part.mask, body=part.body, origin=part.origin, hull=part.hull,
                                meta=np.array([part.ntris, *part.size, part.zmin], dtype=np.float64))
            os.replace(tmp, cached)
        except OSError:
            pass
    return part


def _compute_part(path, res: float, name: str | None = None, raft_height: float = 0.0) -> Part:
    tris = load_stl(path)
    flat = tris.reshape(-1, 3)
    mn = flat.min(0)
    mx = flat.max(0)
    xy = tris[:, :, :2].astype(np.float64)
    mask, origin = rasterize(xy, res)
    pts = xy.reshape(-1, 2)
    hull = pts[ConvexHull(pts).vertices]
    if raft_height > 0:
        # body = triangles that rise above the raft (supports and model)
        keep = tris[:, :, 2].max(1) > mn[2] + raft_height
        body, _ = rasterize(xy[keep], res, frame=(origin, mask.shape))
        body &= mask
    else:
        body = mask
    return Part(
        name=name or Path(path).name,
        path=str(path),
        ntris=len(tris),
        size=tuple(float(v) for v in (mx - mn)),
        zmin=float(mn[2]),
        mask=mask,
        origin=origin,
        res=res,
        area=float(mask.sum()) * res * res,
        hull=hull,
        body=body,
        body_area=float(body.sum()) * res * res,
    )


def hull_extent(part: Part, theta: float):
    """Exact extent (min, max) in mm, relative to the centre of rotation,
    of the part rotated by theta degrees."""
    a = math.radians(theta)
    c, s = math.cos(a), math.sin(a)
    q = part.hull - part.center
    x = q[:, 0] * c - q[:, 1] * s
    y = q[:, 0] * s + q[:, 1] * c
    return np.array([x.min(), y.min()]), np.array([x.max(), y.max()])


def narrowest_angle(part: Part, plate_w: float, step: float = 0.25):
    """Angle that minimises the depth (Y) while keeping the length within the plate.
    Returns (angle, depth, length)."""
    best = None
    for a in np.arange(0.0, 180.0, step):
        lo, hi = hull_extent(part, float(a))
        w, d = hi - lo
        key = (w > plate_w, d)
        if best is None or key < best[0]:
            best = (key, float(a), float(d), float(w))
    return best[1], best[2], best[3]


def variant(part: Part, theta: float, grow: int):
    """Footprints rotated by theta degrees about the centre and grown by `grow` px.

    Returns (full, body, k0): `full` includes the raft and determines the plate
    limits, `body` is the portion that must not overlap other parts; they have
    the same shape. k0 = offset in pixels (x, y) of the corner of pixel (0, 0)
    relative to the centre of rotation.
    """
    h, w = part.mask.shape
    theta = theta % 360
    if theta == 0:
        full, body, k0 = part.mask, part.body, (-w // 2, -h // 2)
    else:
        ys, xs = np.nonzero(part.mask)
        isb = part.body[ys, xs]
        qx = xs + 0.5 - w / 2
        qy = ys + 0.5 - h / 2
        right = theta % 90 == 0
        if right:
            c, s = {90: (0, 1), 180: (-1, 0), 270: (0, -1)}[int(theta)]
        else:
            c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        kx = np.floor(c * qx - s * qy).astype(np.int64)
        ky = np.floor(s * qx + c * qy).astype(np.int64)
        e = 0 if right else 1
        k0 = (int(kx.min()) - e, int(ky.min()) - e)
        shape = (int(ky.max()) - k0[1] + 1 + e, int(kx.max()) - k0[0] + 1 + e)
        full = np.zeros(shape, bool)
        body = np.zeros(shape, bool)
        full[ky - k0[1], kx - k0[0]] = True
        body[(ky - k0[1])[isb], (kx - k0[0])[isb]] = True
        if not right:
            full = ndimage.binary_dilation(full, structure=np.ones((3, 3), bool))
            body = ndimage.binary_dilation(body, structure=np.ones((3, 3), bool))
    if grow > 0:
        full = np.pad(full, grow)
        body = np.pad(body, grow)
        k0 = (k0[0] - grow, k0[1] - grow)
        full = ndimage.distance_transform_edt(~full) <= grow
        if body.any():
            body = ndimage.distance_transform_edt(~body) <= grow
    ys = np.flatnonzero(full.any(1))
    xs = np.flatnonzero(full.any(0))
    win = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    return (np.ascontiguousarray(full[win]), np.ascontiguousarray(body[win]),
            (k0[0] + int(xs[0]), k0[1] + int(ys[0])))


# ---------------------------------------------------------------- packing

@dataclass
class Placement:
    part: Part
    label: str
    plate: int
    theta: float
    i: int
    j: int
    k0: tuple
    shape: tuple
    shift: tuple = (0.0, 0.0)   # sub-pixel correction for flush parts
    tight: bool = False
    m: object = None            # footprint used for the plate occupancy


class _Plate:
    def __init__(self, hg, wg, fshape):
        self.occ = np.zeros((hg, wg), dtype=np.float64)
        self.items: list[Placement] = []
        self.used = 0.0
        self._fshape = fshape
        self._F = None

    def spectrum(self, rfft):
        if self._F is None:
            self._F = rfft(self.occ, self._fshape)
        return self._F

    def copy(self):
        other = _Plate.__new__(_Plate)
        other.occ = self.occ.copy()
        other.items = list(self.items)
        other.used = self.used
        other._fshape = self._fshape
        other._F = self._F
        return other

    def add(self, m, i, j):
        h, w = m.shape
        np.maximum(self.occ[i:i + h, j:j + w], m, out=self.occ[i:i + h, j:j + w])
        self.used += float(m.sum())
        self._F = None


def pack(parts, labels, plate_w, plate_d, res, gap, edge, angles,
         attempts=12, time_limit=60.0, seed=0, improve=True, gpu="auto",
         optimize_height=False, attach_small=False, small_raft_mm2=1200.0, attach_min_mm2=10.0,
         fixed_order=False, progress=None):
    """Arrange the parts onto the smallest number of plates found.

    With `fixed_order` nothing is searched: parts are placed one after the
    other in the given order, each on the current plate, opening a new plate
    when one does not fit. Adding parts at the end never moves the earlier ones.

    `progress`, if given, is called a few times per second with a dict (phase,
    attempt, attempts, plates, phase_elapsed, phase_limit). It may raise to
    abort the search: the exception propagates to the caller.

    Returns (placements, n_plates, unplaceable_labels, grid) where grid describes
    the grid used (origin in plate coordinates, pitch, growth).
    """
    g = int(math.ceil(gap / 2 / res - 1e-9))
    inset = edge - g * res
    wg = int(math.floor((plate_w - 2 * inset) / res + 1e-9))
    hg = int(math.floor((plate_d - 2 * inset) / res + 1e-9))
    gx0 = -plate_w / 2 + inset
    gy0 = -plate_d / 2 + inset
    fshape = (sfft.next_fast_len(hg, real=True), sfft.next_fast_len(wg, real=True))

    # The correlations (FFT) run on the GPU with CuPy when available.
    cp = gpu_module() if gpu in ("auto", True) else None
    if gpu is True and cp is None:
        raise RuntimeError("GPU requested but CuPy/CUDA is not usable")
    if cp is not None:
        xp = cp

        def rfft(a, shape):
            return cp.fft.rfft2(cp.asarray(a, dtype=cp.float32), shape)

        def irfft(f, shape):
            return cp.fft.irfft2(f, shape)
    else:
        xp = np

        def rfft(a, shape):
            return sfft.rfft2(a, shape, workers=-1)

        def irfft(f, shape):
            return sfft.irfft2(f, shape, workers=-1)

    spec_cache: dict = {}

    def conj_spec(m):
        """Conjugate spectrum of a footprint at plate size.
        On the GPU it is kept in video memory; on the CPU it is recomputed (too much RAM)."""
        if cp is None:
            return np.conj(rfft(m, fshape))
        key = id(m)
        if key not in spec_cache:
            spec_cache[key] = (m, cp.conj(rfft(m, fshape)))
        return spec_cache[key][1]

    def tight_variant(p, th):
        """Part that only fits flush: the conservative footprint overhangs the
        grid but the exact geometry lies within the plate. The footprint is
        cropped and the position locked on the flush axis, centring the geometry."""
        lo, hi = hull_extent(p, th)
        if hi[0] - lo[0] > plate_w + FLUSH_TOL or hi[1] - lo[1] > plate_d + FLUSH_TOL:
            return None
        _full, m, k0 = variant(p, th, g)
        k0 = list(k0)
        fix = [None, None]          # [fixed j, fixed i]
        shift = [0.0, 0.0]
        for ax, (n_grid, g0) in enumerate(((wg, gx0), (hg, gy0))):
            n = m.shape[1 - ax]
            if n <= n_grid:
                continue
            mid = (lo[ax] + hi[ax]) / 2
            r0 = int(round((g0 + mid) / res - k0[ax]))
            r0 = min(max(r0, 0), n - n_grid)
            m = m[:, r0:r0 + n_grid] if ax == 0 else m[r0:r0 + n_grid, :]
            k0[ax] += r0
            shift[ax] = -mid - g0 + k0[ax] * res
            fix[ax] = 0
        m = np.ascontiguousarray(m)
        return (th, m.astype(np.float64), tuple(k0), float(m.sum()), fix[1], fix[0], tuple(shift), True)

    cache: dict = {}
    oversize: dict = {}
    for p in parts:
        if p.path in cache:
            continue
        best_a, best_d, best_w = narrowest_angle(p, plate_w)
        own = [best_a, best_a + 180.0] if len(angles) > 1 else []
        vs = []
        seen = set()
        for th in list(angles) + own:
            _full, m, k0 = variant(p, th, g)
            if m.shape[0] > hg or m.shape[1] > wg:
                continue
            key = (m.shape, m.tobytes())
            if key in seen:
                continue
            seen.add(key)
            vs.append((th, m.astype(np.float64), k0, float(m.sum()), None, None, (0.0, 0.0), False))
        if not vs:
            for th in own or list(angles):
                tv = tight_variant(p, th)
                if tv is not None:
                    vs.append(tv)
        if not vs:
            oversize[p.path] = {"min_depth_mm": round(best_d, 2), "length_mm": round(best_w, 2),
                                "over_by_mm": round(max(best_d - plate_d, best_w - plate_w), 2)}
        cache[p.path] = vs

    todo = [(p, l) for p, l in zip(parts, labels) if cache[p.path]]
    unplaceable = [{"name": l, **oversize[p.path]} for p, l in zip(parts, labels) if not cache[p.path]]
    if not todo:
        return [], 0, unplaceable, dict(gx0=gx0, gy0=gy0, res=res, grow=g, hg=hg, wg=wg)

    min_area = {p.path: min(v[3] for v in cache[p.path]) for p, _ in todo}
    lower = max(1, math.ceil(sum(min_area[p.path] for p, _ in todo) / (hg * wg)))

    track = {"phase": "placing", "attempt": 0, "plates": None, "t0": time.time(), "limit": time_limit, "last": 0.0}

    def tick(force=False, **changes):
        """Report progress (throttled); a change of phase restarts its clock."""
        if changes.get("phase", track["phase"]) != track["phase"]:
            track["t0"] = time.time()
        track.update(changes)
        if progress is None:
            return
        now = time.time()
        if force or now - track["last"] >= 0.1:
            track["last"] = now
            progress({"phase": track["phase"], "attempt": track["attempt"], "attempts": max(1, attempts),
                      "plates": track["plates"], "phase_elapsed": now - track["t0"],
                      "phase_limit": track["limit"]})

    def try_place(pl, p, label):
        """Place p on the plate, leaving the parts already there untouched."""
        tick()
        if pl.used + min_area[p.path] > hg * wg:
            return False
        best = None
        for th, m, k0, _a, fi, fj, shift, tight in cache[p.path]:
            h, w = m.shape
            if not pl.items:
                i = j = 0
            else:
                corr = irfft(pl.spectrum(rfft) * conj_spec(m), fshape)
                valid = corr[:hg - h + 1, :wg - w + 1] < 0.5
                if fi is not None:
                    valid[fi + 1:, :] = False
                if fj is not None:
                    valid[:, fj + 1:] = False
                cols = valid.any(0)
                if not bool(cols.any()):
                    continue
                j = int(xp.argmax(cols))
                i = int(xp.argmax(valid[:, j]))
            score = (j + w, i + h)
            if best is None or score < best[0]:
                best = (score, th, m, k0, i, j, shift, tight)
        if best is None:
            return False
        _s, th, m, k0, i, j, shift, tight = best
        pl.add(m, i, j)
        pl.items.append(Placement(p, label, 0, th, i, j, k0, m.shape, shift, tight, m))
        return True

    def new_plate(items=()):
        pl = _Plate(hg, wg, fshape)
        for p, label in items:
            if not try_place(pl, p, label):
                return None
        return pl

    def run(order):
        plates: list[_Plate] = []
        for p, label in order:
            if not any(try_place(pl, p, label) for pl in plates):
                plates.append(new_plate([(p, label)]))
        return plates

    def quality(plates):
        return (len(plates), min(pl.used for pl in plates))

    def bbox_area(pl):
        return pl[0].size[0] * pl[0].size[1]

    def hull_area(pl):
        # convex hull area: measures the true extent of hollow or ring-shaped
        # parts, which have little footprint but take up a lot of plate
        x, y = pl[0].hull[:, 0], pl[0].hull[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    if fixed_order:
        plates = []
        for p, label in todo:
            if not plates or not try_place(plates[-1], p, label):
                plates.append(new_plate([(p, label)]))
        for k, pl in enumerate(plates):
            for it in pl.items:
                it.plate = k
        grid = dict(gx0=gx0, gy0=gy0, res=res, grow=g, hg=hg, wg=wg, lower_bound=lower, attempts=1,
                    stats={"backend": "GPU (CuPy)" if cp is not None else "CPU", "fixed_order": True})
        return [it for pl in plates for it in pl.items], len(plates), unplaceable, grid

    orders = [
        sorted(todo, key=lambda t: -t[0].area),
        sorted(todo, key=lambda t: -hull_area(t)),
        sorted(todo, key=lambda t: -max(t[0].size[0], t[0].size[1])),
        sorted(todo, key=lambda t: -bbox_area(t)),
    ]
    rng = random.Random(seed)
    best = None
    start = time.time()
    tried = 0
    for n in range(max(1, attempts)):
        if n < len(orders):
            order = orders[n]
        else:
            size_of = hull_area if n % 2 else (lambda t: t[0].area)
            order = sorted(todo, key=lambda t: -size_of(t) * rng.uniform(0.5, 1.5))
        plates = run(order)
        tried += 1
        if best is None or quality(plates) < quality(best):
            best = plates
        tick(force=True, attempt=n + 1, plates=len(best))
        if len(best) <= lower or time.time() - start > time_limit:
            break
    greedy_count = len(best)
    stats = {"backend": "GPU (CuPy)" if cp is not None else "CPU", "greedy_plates": greedy_count,
             "greedy_s": round(time.time() - start, 1)}

    # ---- pairwise search: two parts may share a plate even if neither of
    # ---- them is already fixed in a corner
    def offsets(n, na, nb):
        idx = xp.arange(n)
        d = xp.where(idx <= na, idx, idx - n)
        d[(idx > na) & (idx < n - nb)] = 10 ** 6
        return d

    def union(d, na, nb):
        return xp.maximum(na, d + nb) - xp.minimum(0, d)

    def pair_place(a, b):
        """Plate with a and b in the most compact arrangement, or None."""
        va_list, vb_list = cache[a[0].path], cache[b[0].path]
        fs = (sfft.next_fast_len(max(v[1].shape[0] for v in va_list) + max(v[1].shape[0] for v in vb_list), real=True),
              sfft.next_fast_len(max(v[1].shape[1] for v in va_list) + max(v[1].shape[1] for v in vb_list), real=True))
        fbs = [xp.conj(rfft(v[1], fs)) for v in vb_list]
        found = None
        for va in va_list:
            ha, wa = va[1].shape
            fa = rfft(va[1], fs)
            for vb, fb in zip(vb_list, fbs):
                hb, wb = vb[1].shape
                c = irfft(fa * fb, fs)
                dy, dx = offsets(fs[0], ha, hb), offsets(fs[1], wa, wb)
                uh, uw = union(dy, ha, hb), union(dx, wa, wb)
                ok = (c < 0.5) & (uh[:, None] <= hg) & (uw[None, :] <= wg)
                if not bool(ok.any()):
                    continue
                score = xp.where(ok, uw[None, :] * (hg + 1.0) + uh[:, None], xp.inf)
                yi, xi = divmod(int(xp.argmin(score)), fs[1])
                best_score = float(score[yi, xi])
                if found is None or best_score < found[0]:
                    found = (best_score, va, vb, int(dy[yi]), int(dx[xi]))
        if found is None:
            return None
        _s, va, vb, dy, dx = found
        ia, ja = -min(0, dy), -min(0, dx)
        pl = _Plate(hg, wg, fshape)
        for (part, label), v, i, j in ((a, va, ia, ja), (b, vb, ia + dy, ja + dx)):
            th, m, k0, _ar, _fi, _fj, shift, tight = v
            pl.add(m, i, j)
            pl.items.append(Placement(part, label, 0, th, i, j, k0, m.shape, shift, tight, m))
        return pl

    # fast filter at half resolution: discards impossible pairs
    hgc, wgc = (hg + 1) // 2, (wg + 1) // 2
    cs = (sfft.next_fast_len(2 * hgc + 2, real=True), sfft.next_fast_len(2 * wgc + 2, real=True))
    coarse: dict = {}

    def coarse_of(path):
        if path not in coarse:
            rows = []
            for v in cache[path]:
                m = v[1]
                h, w = m.shape
                m2 = np.pad(m, ((0, h % 2), (0, w % 2))).reshape((h + 1) // 2, 2, (w + 1) // 2, 2).max((1, 3))
                rows.append((rfft(m2.astype(np.float32), cs).astype(xp.complex64), m2.shape[0], m2.shape[1]))
            coarse[path] = (rows, xp.stack([xp.conj(r[0]) for r in rows]))
        return coarse[path]

    pair_cache: dict = {}

    def may_pair(pa, pb):
        key = tuple(sorted((pa.path, pb.path)))
        if key not in pair_cache:
            ok = False
            if min_area[pa.path] + min_area[pb.path] <= hg * wg:
                rows_a, _ = coarse_of(pa.path)
                rows_b, conj_b = coarse_of(pb.path)
                for fa, ha, wa in rows_a:
                    c = irfft(fa[None] * conj_b, cs)
                    for k, (_fb, hb, wb) in enumerate(rows_b):
                        uh = union(offsets(cs[0], ha, hb), ha, hb)
                        uw = union(offsets(cs[1], wa, wb), wa, wb)
                        if bool(((c[k] < 0.5) & (uh[:, None] <= hgc) & (uw[None, :] <= wgc)).any()):
                            ok = True
                            break
                    if ok:
                        break
            pair_cache[key] = ok
        return pair_cache[key]

    memo: dict = {}

    def fits_together(items):
        """A plate holding all of `items`, or None."""
        key = frozenset(label for _p, label in items)
        if key in memo:
            return memo[key]
        items = sorted(items, key=lambda t: -t[0].area)
        result = None
        feasible = sum(min_area[pp.path] for pp, _l in items) <= hg * wg and all(
            may_pair(items[x][0], items[y][0]) for x in range(len(items)) for y in range(x + 1, len(items)))
        if feasible:
            if len(items) == 1:
                result = new_plate(items)
            elif len(items) == 2:
                result = pair_place(items[0], items[1])
            else:
                seed_plate = pair_place(items[0], items[1])
                if seed_plate is not None and all(try_place(seed_plate, pp, l) for pp, l in items[2:]):
                    result = seed_plate
                if result is None:
                    for order in (items, sorted(items, key=lambda t: -max(t[0].size[0], t[0].size[1])), items[::-1]):
                        result = new_plate(order)
                        if result is not None:
                            break
        memo[key] = result
        return result

    def out_of_time():
        return time.time() - start > 3 * time_limit

    def eliminate(plates, by_height=False):
        """Try to empty one plate at a time by moving its parts onto the others.
        With by_height, plates already at least as tall as the part are preferred."""
        def target_key(cur, it):
            if by_height:
                tallest = max(i.part.size[2] for i in cur.items)
                return (max(0.0, it.part.size[2] - tallest), -cur.used)
            return (-cur.used,)

        changed = True
        while changed and not out_of_time():
            changed = False
            for victim in sorted(plates, key=lambda pl: pl.used):
                others = [q for q in plates if q is not victim]
                repl: dict = {}
                moved = True
                for it in sorted(victim.items, key=lambda it: -it.part.area):
                    x = (it.part, it.label)
                    target = None
                    for q in sorted(others, key=lambda q: target_key(repl.get(id(q), q), it)):
                        cur = repl.get(id(q), q)
                        trial = cur.copy()
                        if try_place(trial, *x):
                            target = (q, trial)
                            break
                        cand = fits_together([(i.part, i.label) for i in cur.items] + [x])
                        if cand is not None:
                            target = (q, cand.copy())
                            break
                        if out_of_time():
                            break
                    if target is None:
                        moved = False
                        break
                    repl[id(target[0])] = target[1]
                if moved:
                    plates = [repl.get(id(q), q) for q in others]
                    changed = True
                    break
        return plates

    if improve and len(best) > lower and len(todo) > 1:
        tick(force=True, phase="pairing", limit=max(1.0, time_limit - (time.time() - start)))
        # matching: as many compatible pairs as possible
        edges = [(x, y) for x in range(len(todo)) for y in range(x + 1, len(todo))
                 if may_pair(todo[x][0], todo[y][0])]
        stats["compatible_pairs"] = len(edges)
        pairs_done: dict = {}
        banned: set = set()
        matching: list = []
        for _round in range(6):
            usable = [e for e in edges if e not in banned]
            match = []
            for _try in range(3000):
                rng.shuffle(usable)
                taken, cur = set(), []
                for x, y in usable:
                    if x not in taken and y not in taken:
                        taken.update((x, y))
                        cur.append((x, y))
                if len(cur) > len(match):
                    match = cur
            bad = False
            for e in match:
                if e not in pairs_done:
                    pairs_done[e] = pair_place(todo[e[0]], todo[e[1]])
                if pairs_done[e] is None:
                    banned.add(e)
                    bad = True
            matching = [e for e in match if pairs_done[e] is not None]
            if not bad or out_of_time():
                break
        used_idx = {x for e in matching for x in e}
        paired = [pairs_done[e].copy() for e in matching]
        paired += [new_plate([todo[x]]) for x in range(len(todo)) if x not in used_idx]
        stats["matched_pairs"] = len(matching)
        candidates = [eliminate(paired)]
        if not out_of_time():
            candidates.append(eliminate(best))
        best = min(candidates, key=quality)

        # Whether a plate can be emptied depends on the arrangement the search
        # starts from. While there is time left, start again from new random
        # orders; give up after several restarts in a row that save no plate.
        restarts = stale = 0
        while len(best) > lower and stale < 8 and time.time() - start < time_limit:
            size_of = hull_area if restarts % 2 else (lambda t: t[0].area)
            cand = eliminate(run(sorted(todo, key=lambda t: -size_of(t) * rng.uniform(0.5, 1.5))))
            restarts += 1
            stale = 0 if len(cand) < len(best) else stale + 1
            if quality(cand) < quality(best):
                best = cand
            tick(force=True, plates=len(best))
        stats["restarts"] = restarts

    # ---- optional: for the same number of plates, group the tall parts to reduce
    # ---- the sum of the plate heights (i.e. the total print time)
    def plate_h(pl):
        return max(it.part.size[2] for it in pl.items)

    def total_h(plates):
        return sum(plate_h(pl) for pl in plates)

    def without(pl, removed):
        """Copy of the plate without some parts; the others stay where they are."""
        out = _Plate(hg, wg, fshape)
        for it in pl.items:
            if it not in removed:
                out.add(it.m, it.i, it.j)
                out.items.append(it)
        return out

    def with_added(pl, it):
        """Plate with the part `it` added, or None if it does not fit."""
        trial = pl.copy()
        if try_place(trial, it.part, it.label):
            return trial
        cand = fits_together([(i.part, i.label) for i in pl.items] + [(it.part, it.label)])
        return None if cand is None else cand.copy()

    def polish(plates, deadline):
        """Moves and swaps of single parts that lower the sum of the heights
        without ever increasing the number of plates."""
        moves_done = 0
        while time.time() < deadline:
            heights = [plate_h(pl) for pl in plates]
            moves = []
            for a, pa in enumerate(plates):
                for b, pb in enumerate(plates):
                    if a == b:
                        continue
                    for x in pa.items:
                        rest_a = [i.part.size[2] for i in pa.items if i is not x]
                        d = max(rest_a, default=0.0) + max(heights[b], x.part.size[2]) - heights[a] - heights[b]
                        if d < -0.5:
                            moves.append((d, a, b, x, None))
                        if a < b:
                            for y in pb.items:
                                rest_b = [i.part.size[2] for i in pb.items if i is not y]
                                d = (max(rest_a + [y.part.size[2]]) + max(rest_b + [x.part.size[2]])
                                     - heights[a] - heights[b])
                                if d < -0.5:
                                    moves.append((d, a, b, x, y))
            moves.sort(key=lambda mv: mv[0])
            applied = False
            for _d, a, b, x, y in moves[:80]:
                if time.time() >= deadline:
                    break
                new_a = without(plates[a], [x])
                new_b = without(plates[b], [y] if y is not None else [])
                new_b = with_added(new_b, x)
                if new_b is None:
                    continue
                if y is not None:
                    new_a = with_added(new_a, y)
                    if new_a is None:
                        continue
                plates = [new_a if k == a else new_b if k == b else pl for k, pl in enumerate(plates)]
                plates = [pl for pl in plates if pl.items]
                moves_done += 1
                applied = True
                break
            if not applied:
                break
        return plates, moves_done

    def build_by_height(order):
        """Fill one plate at a time starting from the tallest remaining part;
        the second part on each plate uses the exact pairwise test."""
        remaining = list(order)
        plates = []
        while remaining:
            pl = new_plate([remaining.pop(0)])
            k = 0
            while k < len(remaining):
                part, label = remaining[k]
                ok = try_place(pl, part, label)
                if not ok and len(pl.items) == 1:
                    cand = fits_together([(pl.items[0].part, pl.items[0].label), (part, label)])
                    if cand is not None:
                        pl, ok = cand.copy(), True
                if ok:
                    remaining.pop(k)
                else:
                    k += 1
            plates.append(pl)
        return plates

    if optimize_height and len(best) > 1:
        tick(force=True, phase="height", plates=len(best), limit=max(60.0, time_limit))
        t_h = time.time()
        deadline = t_h + max(60.0, time_limit)
        stats["height_sum_before_mm"] = round(total_h(best), 1)
        target = len(best)
        cands = [best]
        # restarts: parts sorted by height band, tallest first
        for band in (1.0, 10.0, 25.0, 50.0, 100.0):
            for size_of in (hull_area, lambda t: t[0].area):
                if time.time() >= deadline:
                    break
                order = sorted(todo, key=lambda t: (-(t[0].size[2] // band), -size_of(t)))
                plates = build_by_height(order)
                if len(plates) > target and improve and time.time() < deadline:
                    plates = eliminate(plates, by_height=True)
                if len(plates) < target:
                    target, cands = len(plates), [plates]
                elif len(plates) == target:
                    cands.append(plates)
        stats["height_candidates"] = len(cands)
        best = min(cands, key=total_h)
        best, stats["height_moves"] = polish(best, deadline)
        stats["height_sum_after_mm"] = round(total_h(best), 1)
        stats["height_s"] = round(time.time() - t_h, 1)

    # ---- optional: small isolated parts are moved, on the same plate, so
    # ---- that their raft overlaps the raft of a larger part: a single
    # ---- merged raft comes off the plate more easily
    plain_cache: dict = {}

    def plain_in_frame(part, theta, k0g, shape):
        """Full footprint, NOT grown, drawn in the frame of the grown
        footprint used for the placement (same angle)."""
        key = (part.path, theta)
        if key not in plain_cache:
            full, _body, k0u = variant(part, theta, 0)
            plain_cache[key] = (full, k0u)
        full, k0u = plain_cache[key]
        out = np.zeros(shape, dtype=np.float64)
        dy, dx = k0u[1] - k0g[1], k0u[0] - k0g[0]
        y0, x0 = max(dy, 0), max(dx, 0)
        y1, x1 = min(dy + full.shape[0], shape[0]), min(dx + full.shape[1], shape[1])
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = full[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        return out

    def attach_rafts(plates):
        small_px = small_raft_mm2 / (res * res)
        report = {"small_parts": 0, "already_attached": 0, "moved": 0, "left_isolated": 0}
        out = []
        for pl in plates:
            def is_small(it):
                return (not it.tight) and it.part.area / (res * res) <= small_px
            if not any(is_small(it) for it in pl.items) or all(is_small(it) for it in pl.items):
                out.append(pl)
                continue
            for label in [it.label for it in sorted(pl.items, key=lambda it: it.part.area) if is_small(it)]:
                it = next(i for i in pl.items if i.label == label)
                report["small_parts"] += 1
                # rafts of the large parts (small parts already settled do not count: attach to the large ones)
                rafts = np.zeros((hg, wg), dtype=np.float64)
                for o in pl.items:
                    if o is it or is_small(o):
                        continue
                    f = plain_in_frame(o.part, o.theta, o.k0, o.shape)
                    np.maximum(rafts[o.i:o.i + o.shape[0], o.j:o.j + o.shape[1]], f,
                               out=rafts[o.i:o.i + o.shape[0], o.j:o.j + o.shape[1]])
                mine = plain_in_frame(it.part, it.theta, it.k0, it.shape)
                need = min(attach_min_mm2 / (res * res), 0.2 * float(mine.sum()))
                now = float((rafts[it.i:it.i + it.shape[0], it.j:it.j + it.shape[1]] * mine).sum())
                if now >= need:
                    report["already_attached"] += 1
                    continue
                rest = without(pl, [it])
                f_occ = rest.spectrum(rfft)
                f_raft = rfft(rafts, fshape)
                found = None
                for th, m, k0, _a, fi, fj, shift, tight in cache[it.part.path]:
                    if tight:
                        continue
                    h, w = m.shape
                    plain = plain_in_frame(it.part, th, k0, m.shape)
                    free = irfft(f_occ * conj_spec(m), fshape)[:hg - h + 1, :wg - w + 1] < 0.5
                    overlap = irfft(f_raft * xp.conj(rfft(plain, fshape)), fshape)[:hg - h + 1, :wg - w + 1]
                    score = xp.where(free, overlap, -1.0)
                    k = int(xp.argmax(score))
                    i, j = divmod(k, score.shape[1])
                    value = float(score[i, j])
                    if value >= need and (found is None or value > found[0]):
                        found = (value, th, m, k0, i, j, shift)
                if found is None:
                    report["left_isolated"] += 1
                    continue
                _v, th, m, k0, i, j, shift = found
                rest.add(m, i, j)
                rest.items.append(Placement(it.part, it.label, 0, th, i, j, k0, m.shape, shift, False, m))
                pl = rest
                report["moved"] += 1
            out.append(pl)
        return out, report

    if attach_small:
        tick(force=True, phase="rafts", plates=len(best), limit=5.0)
        best, stats["raft_attach"] = attach_rafts(best)

    stats["search_s"] = round(time.time() - start, 1)
    best = sorted(best, key=lambda pl: -pl.used)
    for k, pl in enumerate(best):
        for it in pl.items:
            it.plate = k
    placements = [it for pl in best for it in pl.items]
    grid = dict(gx0=gx0, gy0=gy0, res=res, grow=g, hg=hg, wg=wg,
                lower_bound=lower, attempts=tried, stats=stats)
    return placements, len(best), unplaceable, grid


# ---------------------------------------------------------------- output

def transform_of(pl: Placement, grid) -> tuple:
    """(theta, centre of rotation in the model, its position on the plate)."""
    res = grid["res"]
    cx = grid["gx0"] + (pl.j - pl.k0[0]) * res + pl.shift[0]
    cy = grid["gy0"] + (pl.i - pl.k0[1]) * res + pl.shift[1]
    return pl.theta, pl.part.center, np.array([cx, cy])


def move_tris(path, theta, pivot, dest, zmin) -> np.ndarray:
    """Triangles of the STL rotated by theta about `pivot` (model XY),
    moved so the pivot lands at `dest` (plate XY) and resting on Z = 0."""
    tris = load_stl(path).astype(np.float64)
    a = math.radians(theta)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    tris[:, :, :2] = (tris[:, :, :2] - np.asarray(pivot)) @ rot.T + np.asarray(dest)
    tris[:, :, 2] -= zmin
    return tris.astype(np.float32)


def transformed_tris(pl: Placement, grid) -> np.ndarray:
    theta, c, dest = transform_of(pl, grid)
    return move_tris(pl.part.path, theta, c, dest, pl.part.zmin)


def write_plate_entries(path, entries) -> int:
    """Write the single merged STL of a plate from the layout.json entries."""
    chunks = (move_tris(e["file"], e["rotation_z_deg"], e["pivot_mm"], e["center_mm"], e["z_min_mm"])
              for e in entries)
    total = 0
    with open(path, "wb") as f:
        f.write(b"nemnest".ljust(80, b" "))
        f.write(struct.pack("<I", 0))
        for tris in chunks:
            _records(tris).tofile(f)
            total += len(tris)
        f.seek(80)
        f.write(struct.pack("<I", total))
    return total


def export_plates(output_dir, plates=None) -> dict:
    """Write the plate_NN.stl files of the given plates (all if None) using the
    already computed layout.json, without redoing the placement."""
    out = Path(output_dir)
    layout_path = out / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    wanted = None if plates is None else {int(n) for n in plates}
    missing = sorted((wanted or set()) - {pl["plate"] for pl in layout["plates"]})
    if missing:
        raise ValueError(f"Plates not present in the layout: {missing}")
    written = []
    for pl in layout["plates"]:
        if wanted is not None and pl["plate"] not in wanted:
            continue
        stl = out / f"plate_{pl['plate']:02d}.stl"
        tris = write_plate_entries(stl, pl["parts"])
        pl["stl"] = str(stl)
        written.append({"plate": pl["plate"], "stl": str(stl), "triangles": tris,
                        "bytes": stl.stat().st_size, "parts": [e["name"] for e in pl["parts"]]})
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    return {"output_dir": str(out), "written": written}


PALETTE = [(230, 97, 80), (86, 156, 214), (120, 190, 110), (232, 180, 70), (160, 120, 210),
           (70, 190, 190), (225, 130, 180), (150, 150, 90), (240, 140, 60), (110, 130, 230)]


def render_preview(items, grid, plate_w, plate_d, scale=4):
    """Top view of a plate as a PIL image: rafts dimmed, bodies solid, parts numbered."""
    res = grid["res"]
    w = int(round(plate_w * scale))
    h = int(round(plate_d * scale))
    img = Image.new("RGB", (w, h), (28, 30, 34))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    labels = []
    for n, it in enumerate(items):
        m, body, k0 = variant(it.part, it.theta, 0)
        _t, _c, dest = transform_of(it, grid)
        x0 = dest[0] + k0[0] * res + plate_w / 2
        y0 = dest[1] + k0[1] * res + plate_d / 2
        size = (max(1, int(round(m.shape[1] * res * scale))), max(1, int(round(m.shape[0] * res * scale))))
        left = int(round(x0 * scale))
        top = int(round(h - (y0 + m.shape[0] * res) * scale))
        color = PALETTE[n % len(PALETTE)]
        raft = Image.fromarray((m[::-1] * 110).astype(np.uint8)).resize(size, Image.NEAREST)
        img.paste(tuple(v // 2 for v in color), (left, top), raft)   # raft: semi-transparent
        solid = Image.fromarray((body[::-1] * 255).astype(np.uint8)).resize(size, Image.NEAREST)
        img.paste(color, (left, top), solid)
        labels.append((left + size[0] // 2, top + size[1] // 2, str(n + 1)))
    for x, y, t in labels:
        draw.text((x, y), t, fill=(255, 255, 255), font=font, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    draw.rectangle([0, 0, w - 1, h - 1], outline=(200, 200, 200))
    return img


def write_preview(path, items, grid, plate_w, plate_d, scale=4) -> None:
    render_preview(items, grid, plate_w, plate_d, scale).save(path)


def list_stls(folder, exclude=None) -> list:
    """STLs in one or more folders, excluding names that match `exclude`
    (patterns like *water*, case-insensitive)."""
    folders = [folder] if isinstance(folder, (str, Path)) else list(folder)
    patterns = [e.lower() for e in (exclude or [])]
    files = []
    for fo in folders:
        fo = Path(fo)
        if not fo.is_dir():
            raise FileNotFoundError(f"Folder not found: {fo}")
        for p in sorted(fo.iterdir()):
            if not (p.is_file() and p.suffix.lower() == ".stl"):
                continue
            if any(fnmatch.fnmatch(p.name.lower(), pat) for pat in patterns):
                continue
            files.append(p)
    return files


def arrange(folder, output_dir, printer, gap=2.0, edge=1.0, rotation_step=45.0,
            res=0.5, copies=None, attempts=12, time_limit=60.0, exclude=None,
            write_stl=True, raft_height=1.0, gpu="auto", workers=None, optimize_height=False,
            attach_small_rafts=False, small_raft_mm2=1200.0, attach_min_mm2=10.0,
            files=None, fixed_order=False) -> dict:
    """Read the STLs in the folder, nest them and write plates, previews and layout.json."""
    copies = copies or {}
    files = [Path(f) for f in files] if files is not None else list_stls(folder, exclude)
    if not files:
        raise ValueError(f"No .stl files in {folder}")
    plate_w, plate_d, plate_h = printer["size_mm"]
    names = [f.name for f in files]
    parts, labels, too_tall = [], [], []
    t_load = time.time()
    # identical names in different folders: disambiguate with the folder name
    loaded = load_parts([(str(f), res, f.name if names.count(f.name) == 1 else f"{f.parent.name}/{f.name}",
                          raft_height) for f in files], workers)
    t_load = time.time() - t_load
    for f, p in zip(files, loaded):
        if p.size[2] > plate_h:
            too_tall.append(p.name)
            continue
        n = int(copies.get(p.name, copies.get(f.stem, 1)))
        for c in range(n):
            parts.append(p)
            labels.append(p.name if n == 1 else f"{p.name} #{c + 1}")

    step = float(rotation_step)
    angles = [0.0] if step <= 0 else [a * step for a in range(int(round(360 / step)))]
    placements, n_plates, unplaceable, grid = pack(
        parts, labels, plate_w, plate_d, res, gap, edge, angles, attempts, time_limit, gpu=gpu,
        optimize_height=optimize_height,
        attach_small=bool(attach_small_rafts) and raft_height > 0, small_raft_mm2=small_raft_mm2,
        attach_min_mm2=attach_min_mm2, fixed_order=fixed_order)
    settings = {"gap_mm": gap, "edge_margin_mm": edge, "rotation_step_deg": step,
                "resolution_mm": res, "raft_height_mm": raft_height,
                "optimize_height": bool(optimize_height),
                "attach_small_rafts": bool(attach_small_rafts), "small_raft_mm2": small_raft_mm2,
                "fixed_order": bool(fixed_order)}
    return write_layout(output_dir, printer, placements, n_plates, grid, settings, models=len(parts),
                        unplaceable=unplaceable, too_tall=too_tall, load_s=t_load, write_stl=write_stl)


def write_layout(output_dir, printer, placements, n_plates, grid, settings, models=None,
                 unplaceable=(), too_tall=(), load_s=0.0, write_stl=True) -> dict:
    """Write previews, layout.json and (optionally) the merged plate STLs of a
    placement computed by `pack`."""
    plate_w, plate_d, _plate_h = printer["size_mm"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # only files this tool writes (plate_NN.ext): anything else is left alone
    for old in out.glob("plate_*.*"):
        if re.fullmatch(r"plate_\d+\.(stl|png|goo|ctb|nanodlp)", old.name):
            old.unlink()
    def clearance(it):
        theta, _c, dest = transform_of(it, grid)
        lo, hi = hull_extent(it.part, theta)
        return round(float(min(lo[0] + dest[0] + plate_w / 2, plate_w / 2 - hi[0] - dest[0],
                               lo[1] + dest[1] + plate_d / 2, plate_d / 2 - hi[1] - dest[1])), 2)

    plates = []
    for k in range(n_plates):
        items = [it for it in placements if it.plate == k]
        stl = out / f"plate_{k + 1:02d}.stl"
        png = out / f"plate_{k + 1:02d}.png"
        write_preview(png, items, grid, plate_w, plate_d)
        entries = []
        for n, it in enumerate(items):
            theta, pivot, dest = transform_of(it, grid)
            entries.append({
                "n": n + 1,
                "name": it.label,
                "file": it.part.path,
                "rotation_z_deg": theta,
                "center_mm": [round(float(dest[0]), 4), round(float(dest[1]), 4)],
                "pivot_mm": [float(pivot[0]), float(pivot[1])],
                "z_min_mm": float(it.part.zmin),
                "height_mm": round(it.part.size[2], 2),
                "edge_clearance_mm": clearance(it),
                "tight": it.tight,
            })
        if write_stl:
            write_plate_entries(stl, entries)
        fill = sum(it.part.body_area for it in items) / (plate_w * plate_d)
        plates.append({
            "plate": k + 1,
            "stl": str(stl) if write_stl else None,
            "preview": str(png),
            "parts": entries,
            "fill": round(fill, 3),
            "max_height_mm": round(max(it.part.size[2] for it in items), 2),
        })
    result = {
        "printer": printer,
        "settings": settings,
        "models": len(placements) if models is None else models,
        "plates_count": n_plates,
        "height_sum_mm": round(sum(max(it.part.size[2] for it in placements if it.plate == k)
                                   for k in range(n_plates)), 1),
        "placed": len(placements),
        "area_lower_bound": grid.get("lower_bound"),
        "attempts": grid.get("attempts"),
        "search": grid.get("stats"),
        "load_s": round(load_s, 1),
        "plates": plates,
        "does_not_fit": list(unplaceable),
        "too_tall": list(too_tall),
    }
    (out / "layout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def scan(folder, printer, res=0.5, exclude=None, workers=None) -> dict:
    plate_w, plate_d, plate_h = printer["size_mm"]
    rows = []
    files = list_stls(folder, exclude)
    for p in load_parts([(str(f), res, None, 0.0) for f in files], workers):
        dx, dy, dz = p.size
        fits_xy = min(dx, dy) <= min(plate_w, plate_d) and max(dx, dy) <= max(plate_w, plate_d)
        rows.append({
            "name": p.name,
            "size_mm": [round(dx, 2), round(dy, 2), round(dz, 2)],
            "triangles": p.ntris,
            "footprint_mm2": round(p.area, 1),
            "z_min_mm": round(p.zmin, 3),
            "fits": bool(fits_xy and dz <= plate_h),
        })
    total = sum(r["footprint_mm2"] for r in rows)
    return {"folder": str(folder), "count": len(rows), "models": rows,
            "total_footprint_mm2": round(total, 1),
            "plate_area_mm2": round(plate_w * plate_d, 1)}
