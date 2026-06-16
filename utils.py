from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_video_id(path: Path, used_ids: set[str]) -> str:
    try:
        digest = file_sha1(path).upper()
    except OSError:
        digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest().upper()
    for length in (6, 8, 10, 12):
        candidate = f"V_{digest[:length]}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
    suffix = 1
    while True:
        candidate = f"V_{digest[:10]}_{suffix}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        suffix += 1


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "item"


def format_local_player_id(video_id: str, raw_track_id: int | str) -> str:
    try:
        track_number = int(raw_track_id)
        suffix = f"{track_number:04d}"
    except (TypeError, ValueError):
        suffix = sanitize_name(str(raw_track_id))
    return f"{video_id}_track_{suffix}"


def path_to_posix(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix()


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)
        file.write("\n")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1_i = int(math.floor(clamp(x1, 0, width - 1)))
    y1_i = int(math.floor(clamp(y1, 0, height - 1)))
    x2_i = int(math.ceil(clamp(x2, 1, width)))
    y2_i = int(math.ceil(clamp(y2, 1, height)))
    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def pad_bbox(bbox: Sequence[float], padding_ratio: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def mean_pairwise(values: Iterable[float]) -> float | None:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        return None
    return total / count
