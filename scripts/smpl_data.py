"""Read numeric WHAM/joblib outputs and validate the shared motion NPZ format."""
from collections import defaultdict
import math
import hashlib
from pathlib import Path
import pickle

import numpy as np

SMPL_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
              (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
              (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
              (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)]
# WHAM world coordinates have +Y up; our common format is right-handed +Z up.
Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class ArrayWrapper:
    pass


class NumericUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("joblib.numpy_pickle", "NumpyArrayWrapper"): ArrayWrapper,
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("collections", "defaultdict"): defaultdict, ("builtins", "dict"): dict,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported serialized global: {module}.{name}")
        return allowed[module, name]

    def load_build(self):
        super().load_build()
        wrapper = self.stack[-1]
        if not isinstance(wrapper, ArrayWrapper):
            return
        dtype = wrapper.dtype
        if dtype.hasobject or dtype.kind not in "biufc":
            raise ValueError(f"Only numeric arrays are supported: {dtype}")
        if any(not isinstance(n, int) or n < 0 for n in wrapper.shape):
            raise ValueError("Invalid array shape")
        count = math.prod(wrapper.shape)
        if count * dtype.itemsize > 1024**3:
            raise ValueError("Array exceeds the 1 GiB per-array limit")
        if getattr(wrapper, "numpy_array_alignment_bytes", None):
            size = self.read(1)
            if not size:
                raise ValueError("Truncated array padding")
            self.read(size[0])
        raw = self.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise ValueError("Truncated array")
        self.stack[-1] = np.frombuffer(raw, dtype=dtype).reshape(
            wrapper.shape, order=wrapper.order).copy()

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.BUILD[0]] = load_build


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numeric(path):
    with Path(path).open("rb") as stream:
        return NumericUnpickler(stream).load()


def select_person(value, person_id=None):
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected a nonempty WHAM person dictionary")
    if "pose_world" in value:
        if person_id is not None:
            raise ValueError("This file is already a single-person record")
        return "single", value
    if person_id is None:
        if len(value) != 1:
            raise ValueError(f"Multiple people found: {list(value)}. Select --person-id explicitly.")
        key = next(iter(value))
    else:
        matches = [key for key in value if str(key) == str(person_id)]
        if len(matches) != 1:
            raise ValueError(f"Person {person_id!r} not found; available: {list(value)}")
        key = matches[0]
    return str(key), value[key]


def validate_frames(frame_ids, fps, count=None):
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or len(ids) < 2 or ids.dtype.kind not in "iu":
        raise ValueError("frame_ids must contain at least two integer video-frame indices")
    if ids[0] < 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("frame_ids must be nonnegative and strictly increasing")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if count is not None and ids[-1] >= count:
        raise ValueError("SMPL frame_ids extend beyond the source video")
    return ids.astype(np.int64)


def load_motion(path):
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    if str(data.get("coordinate_system", "")) != "world_z_up":
        raise ValueError("Expected coordinate_system='world_z_up'; export WHAM first")
    if np.asarray(data.get("fps")).size != 1:
        raise ValueError("Expected one video fps value")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    ids = validate_frames(data.get("frame_ids"), fps)
    points = data.get("joints_world")
    if points is None or points.shape != (len(ids), 24, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite joints_world with shape (frames, 24, 3)")
    if "vertices_world" in data:
        vertices = data["vertices_world"]
        if (vertices.ndim != 3 or vertices.shape[0] != len(ids)
                or vertices.shape[2] != 3 or not np.isfinite(vertices).all()):
            raise ValueError("Invalid vertices_world")
    data.update(fps=fps, frame_ids=ids)
    return data
