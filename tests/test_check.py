"""Island check on sliced files: the run-based search against scipy.ndimage on
the full images, on random layers and on a small .goo written for the test."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy import ndimage

from nemnest import check, cli, goo

EIGHT = np.ones((3, 3), bool)


def reference_islands(cur, below):
    """Areas of the regions of `cur` with no lit pixel directly under them."""
    lab, n = ndimage.label(cur, structure=EIGHT)
    if n == 0:
        return []
    held = np.unique(lab[cur & below])
    free = np.setdiff1d(np.arange(1, n + 1), held)
    return sorted(int(v) for v in ndimage.sum(cur, lab, free)) if len(free) else []


def random_layer(rng, shape, blobs):
    im = np.zeros(shape, bool)
    for _ in range(blobs):
        h, w = min(rng.integers(1, 9), shape[0]), min(rng.integers(1, 14), shape[1])
        y, x = rng.integers(0, shape[0] - h + 1), rng.integers(0, shape[1] - w + 1)
        im[y:y + h, x:x + w] = True
    return im


def runs_of(image):
    """Run lengths and values of an image read row after row, as the file stores them."""
    flat = (np.asarray(image) > 0).ravel()
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    return (bounds[1:] - bounds[:-1]).tolist(), [255 if flat[b] else 0 for b in bounds[:-1]]


def test_segments_match_the_image():
    rng = np.random.default_rng(5)
    for _ in range(40):
        im = random_layer(rng, (37, 53), rng.integers(0, 25))
        im[:, -1] |= rng.random(37) < 0.3            # lit pixels at the end of a row ...
        im[:, 0] |= rng.random(37) < 0.3             # ... and at the start of the next: runs wrap over rows
        for seg in (check.segments_from_image(im), check.segments_from_runs(*runs_of(im), 53)):
            back = np.zeros_like(im)
            for r, a, b in zip(*seg):
                assert not back[r, a:b].any()
                back[r, a:b] = True
            assert (back == im).all()
            assert (np.diff(seg[0] * 100 + seg[1]) > 0).all()        # sorted by row, then x


def test_regions_and_islands_match_ndimage():
    rng = np.random.default_rng(11)
    for _ in range(120):
        shape = (int(rng.integers(8, 40)), int(rng.integers(8, 60)))
        cur = random_layer(rng, shape, int(rng.integers(0, 30)))
        below = random_layer(rng, shape, int(rng.integers(0, 12)))
        seg, seg_below = check.segments_from_image(cur), check.segments_from_image(below)
        label, n = check.regions(seg, shape[1])
        assert n == ndimage.label(cur, structure=EIGHT)[1]
        found = check.check_layer(seg, (label, n), seg_below, shape[1], (0.02, 0.03))
        assert sorted(i["area_px"] for i in found) == reference_islands(cur, below)
        for i in found:
            box = cur[i["y"][0]:i["y"][1], i["x"][0]:i["x"][1]]
            assert box.any() and i["area_px"] <= box.size
            assert i["area_mm2"] == pytest.approx(i["area_px"] * 0.02 * 0.03)


def test_touching_only_at_a_corner_is_one_region_but_not_support():
    cur = np.zeros((4, 6), bool)
    cur[0, 0:2] = True
    cur[1, 2:4] = True                               # touches the first only diagonally: same region
    below = np.zeros((4, 6), bool)
    below[1, 4] = True                               # next to the region, not under it
    seg = check.segments_from_image(cur)
    label, n = check.regions(seg, 6)
    assert n == 1
    assert [i["area_px"] for i in check.check_layer(seg, (label, n), check.segments_from_image(below), 6, (1, 1))] == [4]
    below[1, 3] = True                               # now one pixel is right under it
    assert check.check_layer(seg, (label, n), check.segments_from_image(below), 6, (1, 1)) == []


def write_goo(path, layers, pixel_mm=(0.05, 0.04), thickness=0.05):
    """A minimal .goo holding the given layer images (non-zero = lit)."""
    height, width = layers[0].shape
    values = {"layer_count": len(layers), "x_resolution": width, "y_resolution": height,
              "x_size": width * pixel_mm[0], "y_size": height * pixel_mm[1], "z_size": 100.0,
              "layer_thickness": thickness, "exposure_time": 2.5, "bottom_exposure_time": 30.0, "bottom_layers": 2}
    header = bytearray()
    header += b"V3.0" + bytes(8)
    for n in (32, 24, 24, 32, 32, 32):
        header += b"test".ljust(n, bytes(1))
    header += struct.pack(">3H", 1, 0, 0)
    header += bytes(goo.PREVIEWS)
    for key, fmt in goo.HEADER_FIELDS:
        header += struct.pack(">" + fmt, values.get(key, 0))
    header += b"$".ljust(8, bytes(1))
    offset_at = len(header)
    header += struct.pack(">I", 0) + struct.pack(">B", 0) + struct.pack(">H", 0)
    header[offset_at:offset_at + 4] = struct.pack(">I", len(header))
    body = bytearray()
    for image in layers:
        data = bytearray()
        for length, value in zip(*runs_of(image)):
            while length:                                # runs of at most 20 bits, black or white
                part = min(length, (1 << 20) - 1)
                size = 0 if part < 16 else 1 if part < 1 << 12 else 2
                data.append(((3 if value else 0) << 6) | (size << 4) | (part & 0x0F))
                if size == 1:
                    data.append(part >> 4)
                elif size == 2:
                    data += bytes([(part >> 12) & 0xFF, (part >> 4) & 0xFF])
                length -= part
        body += struct.pack(">H15fH", 0, *([0.0] * 15), 255) + bytes(2) + struct.pack(">I", len(data) + 2) + b"\x55"
        body += data + bytes(3)
    path.write_bytes(bytes(header) + bytes(body))
    return path


@pytest.fixture(scope="module")
def sliced(tmp_path_factory):
    rng = np.random.default_rng(23)
    shape = (60, 90)
    layers = [random_layer(rng, shape, 30)]
    for _ in range(9):
        nxt = layers[-1].copy()
        nxt &= rng.random(shape) > 0.03                  # some pixels drop out
        nxt |= random_layer(rng, shape, 3)               # and new patches appear, some in the air
        layers.append(nxt)
    return write_goo(tmp_path_factory.mktemp("goo") / "plate_01.goo", layers), layers


def test_check_file_against_the_images(sliced):
    path, layers = sliced
    g = goo.GooFile(path)
    assert g.header["layer_count"] == len(layers)
    for n, image in enumerate(layers):
        assert ((g.image(n) > 0) == image).all()
    g.f.close()
    expected = sorted((n, a) for n in range(1, len(layers)) for a in reference_islands(layers[n], layers[n - 1]))
    assert expected, "the test file should contain islands"
    seen = []
    one = check.check_file(path, workers=1, progress=lambda done, total: seen.append((done, total)), chunk=4)
    assert sorted((i["layer"], i["area_px"]) for i in one["items"]) == expected
    assert seen[-1] == (len(layers) - 1, len(layers) - 1) and one["counts"]["island"] == len(expected)
    assert [i["area_px"] for i in one["items"]] == sorted((a for _n, a in expected), reverse=True)
    assert one["items"][0]["z_mm"] == pytest.approx(one["items"][0]["layer"] * 0.05)
    many = check.check_file(path, workers=2, chunk=3)
    assert sorted((i["layer"], i["area_px"]) for i in many["items"]) == expected

    def stop(_done, _total):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        check.check_file(path, workers=1, progress=stop, chunk=2)


def test_view_marks_what_has_nothing_under_it(sliced):
    path, layers = sliced
    code, box = check.view(path, 4)
    assert box == (0, 90, 0, 60) and code.shape == (60, 90)
    assert ((code > 0) == layers[4]).all() and ((code == 2) == (layers[4] & ~layers[3])).all()
    code, box = check.view(path, 4, box=(10, 50, 5, 45))
    assert code.shape == (40, 40) and ((code > 0) == layers[4][5:45, 10:50]).all()
    small, _ = check.view(path, 4, max_side=30)
    assert max(small.shape) <= 30 and (small > 0).any()
    first, _ = check.view(path, 0)
    assert first.max() <= 1                              # the first layer rests on the plate


def test_cli_and_tool_check(sliced, capsys):
    from nemnest import server
    path, _layers = sliced
    assert cli.main(["check", str(path), "--min-px", "1"]) == 0
    out = capsys.readouterr().out
    assert "islands" in out and "layer" in out
    assert cli.main(["check", str(path.parent), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["checked"][0]["file"] == str(path) and data["min_area_px"] == 10
    res = server.tool_check_slices({"file": str(path), "min_area_px": 1, "max_items": 2})
    assert len(res["checked"][0]["items"]) <= 2 and res["checked"][0]["islands"] >= len(res["checked"][0]["items"])
    assert cli.main(["check", str(path.parent / "missing.goo")]) == 1
    with pytest.raises(ValueError):
        server.tool_check_slices({"file": str(path.with_suffix(".ctb"))} if path.with_suffix(".ctb").write_bytes(b"x") else {})
