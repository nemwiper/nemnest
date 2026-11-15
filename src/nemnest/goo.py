"""Reader for .goo files (Elegoo): header, per-layer parameters and images."""
from __future__ import annotations

import struct

import numpy as np

PREVIEWS = 116 * 116 * 2 + 2 + 290 * 290 * 2 + 2
HEADER_FIELDS = [
    ("layer_count", "I"), ("x_resolution", "H"), ("y_resolution", "H"),
    ("x_mirror", "?"), ("y_mirror", "?"),
    ("x_size", "f"), ("y_size", "f"), ("z_size", "f"), ("layer_thickness", "f"),
    ("exposure_time", "f"), ("exposure_delay_mode", "B"), ("turn_off_time", "f"),
    ("bottom_before_lift_time", "f"), ("bottom_after_lift_time", "f"), ("bottom_after_retract_time", "f"),
    ("before_lift_time", "f"), ("after_lift_time", "f"), ("after_retract_time", "f"),
    ("bottom_exposure_time", "f"), ("bottom_layers", "I"),
    ("bottom_lift_distance", "f"), ("bottom_lift_speed", "f"), ("lift_distance", "f"), ("lift_speed", "f"),
    ("bottom_retract_distance", "f"), ("bottom_retract_speed", "f"),
    ("retract_distance", "f"), ("retract_speed", "f"),
    ("bottom_second_lift_distance", "f"), ("bottom_second_lift_speed", "f"),
    ("second_lift_distance", "f"), ("second_lift_speed", "f"),
    ("bottom_second_retract_distance", "f"), ("bottom_second_retract_speed", "f"),
    ("second_retract_distance", "f"), ("second_retract_speed", "f"),
    ("bottom_light_pwm", "H"), ("light_pwm", "H"), ("per_layer_settings", "?"),
    ("printing_time", "I"), ("total_volume", "f"), ("total_weight", "f"), ("total_price", "f"),
]
LAYER_FIELDS = ["pause_position_z", "layer_position_z", "layer_exposure_time", "layer_off_time",
                "before_lift_time", "after_lift_time", "after_retract_time", "lift_distance", "lift_speed",
                "second_lift_distance", "second_lift_speed", "retract_distance", "retract_speed",
                "second_retract_distance", "second_retract_speed"]


def _text(b: bytes) -> str:
    return b.split(b"\0")[0].decode("latin-1").strip()


class GooFile:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        self.path = str(path)
        self.offsets = {}          # field -> (offset, struct format or string length)
        h = {"version": _text(f.read(4))}
        f.read(8)   # magic
        for key, n in (("software_info", 32), ("software_version", 24), ("file_time", 24),
                       ("printer_name", 32), ("printer_type", 32), ("profile_name", 32)):
            self.offsets[key] = (f.tell(), n)
            h[key] = _text(f.read(n))
        h["anti_aliasing_level"], h["grey_level"], h["blur_level"] = struct.unpack(">3H", f.read(6))
        f.seek(PREVIEWS, 1)
        for key, fmt in HEADER_FIELDS:
            size = struct.calcsize(">" + fmt)
            self.offsets[key] = (f.tell(), fmt)
            h[key] = struct.unpack(">" + fmt, f.read(size))[0]
        h["price_unit"] = _text(f.read(8))
        h["layer_offset"] = struct.unpack(">I", f.read(4))[0]
        self.offsets["grey_scale_level"] = (f.tell(), "B")
        h["grey_scale_level"] = struct.unpack(">B", f.read(1))[0]
        h["transition_layers"] = struct.unpack(">H", f.read(2))[0]
        self.header = h
        # layer index
        self.layers = []
        pos = h["layer_offset"]
        for _ in range(h["layer_count"]):
            f.seek(pos)
            raw = f.read(2 + 4 * 15 + 2 + 2 + 4 + 1)
            vals = struct.unpack(">H15fH", raw[:64])
            info = dict(zip(LAYER_FIELDS, vals[1:16]))
            info["pause"] = vals[0]
            info["light_pwm"] = vals[16]
            length = struct.unpack(">I", raw[66:70])[0] - 2
            info["offset"] = pos
            info["data_offset"] = pos + 71
            info["data_length"] = length
            self.layers.append(info)
            pos += 71 + length + 1 + 2

    def lit_pixels(self, n: int) -> float:
        """Lit pixels of layer n (grey pixels count proportionally)."""
        lengths, values = self._runs(n)
        return float(np.dot(np.array(lengths, dtype=np.float64), np.array(values, dtype=np.float64)) / 255.0)

    def volume_mm3(self, max_samples: int = 200) -> float:
        """Volume of cured resin. The first layers are all counted, the rest
        are sampled (one every `step`), interpolating between samples."""
        h = self.header
        count = h["layer_count"]
        step = max(1, count // max_samples)
        idx = sorted(set(range(min(20, count))) | set(range(0, count, step)) | {count - 1})
        lit = np.array([self.lit_pixels(n) for n in idx])
        total = float(np.interp(np.arange(count), idx, lit).sum())
        voxel = (h["x_size"] / h["x_resolution"]) * (h["y_size"] / h["y_resolution"]) * h["layer_thickness"]
        return total * voxel

    def image(self, n: int) -> np.ndarray:
        """Layer n as a uint8 array of shape (y_resolution, x_resolution)."""
        lengths, values = self._runs(n)
        h = self.header
        flat = np.repeat(np.array(values, dtype=np.uint8), np.array(lengths, dtype=np.int64))
        want = h["x_resolution"] * h["y_resolution"]
        if flat.size != want:
            raise ValueError(f"layer {n}: {flat.size} pixels decoded, expected {want}")
        return flat.reshape(h["y_resolution"], h["x_resolution"])

    def _runs(self, n: int):
        info = self.layers[n]
        self.f.seek(info["data_offset"])
        data = self.f.read(info["data_length"])
        lengths, values = [], []
        color = 0
        k = 0
        end = len(data)
        while k < end:
            head = data[k]
            kind = head >> 6
            size = (head >> 4) & 3
            if kind == 2:
                diff = head & 0x0F
                if size & 1:
                    k += 1
                    length = data[k]
                else:
                    length = 1
                color = (color - diff if size & 2 else color + diff) & 0xFF
            else:
                if kind == 0:
                    color = 0
                elif kind == 3:
                    color = 255
                else:
                    k += 1
                    color = data[k]
                length = head & 0x0F
                if size == 1:
                    length |= data[k + 1] << 4
                    k += 1
                elif size == 2:
                    length |= (data[k + 1] << 12) | (data[k + 2] << 4)
                    k += 2
                elif size == 3:
                    length |= (data[k + 1] << 20) | (data[k + 2] << 12) | (data[k + 3] << 4)
                    k += 3
            k += 1
            lengths.append(length)
            values.append(color)
        return lengths, values


def exposure_schedule(count, bottom_layers, bottom_time, normal_time, transition_layers):
    """Per-layer exposure as computed by ELEGOO SatelLite: `bottom_layers` at
    full time, then `transition_layers` stepping down to the normal time."""
    out = []
    for n in range(count):
        if n < bottom_layers:
            out.append(bottom_time)
        elif n < bottom_layers + transition_layers:
            step = (bottom_time - normal_time) / transition_layers
            out.append(bottom_time - (n - bottom_layers + 1) * step)
        else:
            out.append(normal_time)
    return out


def apply_profile(path, profile: dict) -> dict:
    """Align the metadata of a .goo file with what the manufacturer's slicer writes.

    The layer images are left untouched. `profile` (config.json, key
    goo_profile) contains: printer_name, profile_name, x_mirror, y_mirror,
    per_layer_settings, grey_scale_level, motion_value (value written to the
    lift/retract fields, which tilting-vat printers ignore),
    pause_position_z, motion_s_per_layer (for the print-time estimate)."""
    g = GooFile(path)
    h = g.header
    volume = g.volume_mm3() if profile.get("resin_density_g_ml") else None
    g.f.close()
    times = exposure_schedule(h["layer_count"], h["bottom_layers"], h["bottom_exposure_time"],
                              h["exposure_time"], h["transition_layers"])
    wait = h["after_retract_time"]
    total = sum(times) + h["layer_count"] * (wait + profile.get("motion_s_per_layer", 0.0))
    motion = profile.get("motion_value")
    with open(path, "r+b") as f:
        def put(key, value):
            off, fmt = g.offsets[key]
            f.seek(off)
            if isinstance(fmt, int):
                f.write(str(value).encode("latin-1")[:fmt - 1].ljust(fmt, bytes(1)))
            else:
                f.write(struct.pack(">" + fmt, value))

        if profile.get("printer_name"):
            put("printer_name", profile["printer_name"])
            put("printer_type", profile["printer_name"])
        if profile.get("profile_name"):
            put("profile_name", profile["profile_name"])
        for key in ("x_mirror", "y_mirror", "per_layer_settings", "grey_scale_level"):
            if key in profile:
                put(key, profile[key])
        if motion is not None:
            for key in ("bottom_lift_distance", "bottom_lift_speed", "lift_distance", "lift_speed",
                        "bottom_retract_distance", "bottom_retract_speed", "retract_distance", "retract_speed"):
                put(key, motion)
        put("printing_time", int(round(total)))
        if volume is not None:
            # mslicer leaves fixed placeholder values: write the real volume (mm3) and weight (g)
            put("total_volume", volume)
            put("total_weight", volume / 1000.0 * profile["resin_density_g_ml"])
        for info, t in zip(g.layers, times):
            base = info["offset"] + 2          # after the pause field
            fields = dict(zip(LAYER_FIELDS, struct.unpack(">15f", _read(f, base, 60))))
            fields["layer_exposure_time"] = t
            if "pause_position_z" in profile:
                fields["pause_position_z"] = profile["pause_position_z"]
            if motion is not None:
                for key in LAYER_FIELDS[7:]:
                    fields[key] = motion
            f.seek(base)
            f.write(struct.pack(">15f", *(fields[k] for k in LAYER_FIELDS)))
    out = {"printing_time_s": int(round(total)), "layers": h["layer_count"]}
    if volume is not None:
        out["volume_ml"] = round(volume / 1000.0, 1)
        out["weight_g"] = round(volume / 1000.0 * profile["resin_density_g_ml"], 1)
    return out


def _read(f, offset, n):
    f.seek(offset)
    return f.read(n)
