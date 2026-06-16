from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils import bbox_area, clamp


@dataclass(frozen=True)
class CropQuality:
    score: float
    blur_score: float
    size_score: float
    full_body_score: float
    edge_score: float
    confidence_score: float
    mask_score: float
    person_shape_score: float
    aspect_ratio: float
    edge_touch_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "blur_score": self.blur_score,
            "size_score": self.size_score,
            "full_body_score": self.full_body_score,
            "edge_score": self.edge_score,
            "confidence_score": self.confidence_score,
            "mask_score": self.mask_score,
            "person_shape_score": self.person_shape_score,
            "aspect_ratio": self.aspect_ratio,
            "edge_touch_ratio": self.edge_touch_ratio,
        }


def laplacian_blur_score(image: np.ndarray) -> float:
    import cv2

    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_crop(
    crop: np.ndarray,
    bbox: Sequence[float],
    frame_shape: tuple[int, int, int] | tuple[int, int],
    confidence: float,
    mask_area_ratio: float | None,
) -> CropQuality:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    crop_h, crop_w = crop.shape[:2]
    blur = laplacian_blur_score(crop)

    size_score = clamp((crop_h / max(1.0, frame_h * 0.35)), 0.0, 1.0)
    sharpness_score = clamp(blur / 250.0, 0.0, 1.0)
    aspect = crop_h / max(1.0, crop_w)
    if 1.4 <= aspect <= 4.2:
        full_body_score = 1.0
    elif 1.0 <= aspect <= 5.2:
        full_body_score = 0.7
    elif 0.75 <= aspect <= 6.5:
        full_body_score = 0.4
    else:
        full_body_score = 0.15

    x1, y1, x2, y2 = bbox
    margin_x = max(2.0, frame_w * 0.015)
    margin_y = max(2.0, frame_h * 0.015)
    edge_hits = (
        int(x1 <= margin_x)
        + int(y1 <= margin_y)
        + int(x2 >= frame_w - margin_x)
        + int(y2 >= frame_h - margin_y)
    )
    edge_touch_ratio = edge_hits / 4.0
    edge_score = clamp(1.0 - edge_hits * 0.22, 0.0, 1.0)
    confidence_score = clamp(float(confidence), 0.0, 1.0)
    if mask_area_ratio is None:
        mask_score = 1.0
    else:
        mask_score = clamp(mask_area_ratio / 0.35, 0.25, 1.0)
    person_shape_score = clamp(
        0.42 * full_body_score
        + 0.24 * edge_score
        + 0.22 * size_score
        + 0.12 * mask_score,
        0.0,
        1.0,
    )

    score = (
        0.30 * size_score
        + 0.25 * sharpness_score
        + 0.20 * person_shape_score
        + 0.15 * edge_score
        + 0.10 * confidence_score
    ) * mask_score
    return CropQuality(
        score=float(score),
        blur_score=float(blur),
        size_score=float(size_score),
        full_body_score=float(full_body_score),
        edge_score=float(edge_score),
        confidence_score=float(confidence_score),
        mask_score=float(mask_score),
        person_shape_score=float(person_shape_score),
        aspect_ratio=float(aspect),
        edge_touch_ratio=float(edge_touch_ratio),
    )


def crop_is_valid(
    crop: np.ndarray,
    crop_bbox: Sequence[float],
    min_crop_height: int,
    blur_threshold: float,
    quality: CropQuality,
    min_person_shape_score: float = 0.0,
    max_edge_touch_ratio: float = 1.0,
) -> bool:
    if crop.size == 0:
        return False
    if crop.shape[0] < min_crop_height:
        return False
    if bbox_area(crop_bbox) <= 0:
        return False
    if blur_threshold > 0 and quality.blur_score < blur_threshold:
        return False
    if quality.person_shape_score < min_person_shape_score:
        return False
    if quality.edge_touch_ratio > max_edge_touch_ratio:
        return False
    return True
