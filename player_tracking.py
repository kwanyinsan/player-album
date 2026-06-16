from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import FilterConfig
from utils import bbox_area, bbox_center, clamp, ensure_dir, format_local_player_id
from video_io import VideoInfo, create_video_writer, iter_video_frames
from yoloe_player_detection import YOLOEPlayerDetector


@dataclass
class TrackFrame:
    frame_index: int
    local_track_id: str
    raw_track_id: int
    bbox: list[float]
    confidence: float
    label: str
    mask_polygon: list[list[float]] | None = None
    mask_area_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "local_track_id": self.local_track_id,
            "raw_track_id": self.raw_track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "label": self.label,
            "mask_polygon": self.mask_polygon,
            "mask_area_ratio": self.mask_area_ratio,
        }


@dataclass
class LocalTrack:
    local_player_id: str
    video_id: str
    raw_track_id: int
    source_video_name: str
    frames: list[TrackFrame] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0
    duration_sec: float = 0.0
    average_confidence: float = 0.0
    average_bbox_height: float = 0.0
    raw_movement_score: float = 0.0
    movement_score: float = 0.0
    center_position_score: float = 0.0
    activity_zone_score: float = 0.0
    average_bbox_size_score: float = 0.0
    active_score: float = 0.0
    crop_quality_score: float = 0.0
    crop_person_shape_score: float = 0.0
    crop_edge_score: float = 0.0
    crop_count: int = 0
    is_active: bool = True
    ignored_reason: str | None = None
    crop_paths: list[str] = field(default_factory=list)
    representative_crop: str | None = None

    def to_dict(self, include_frames: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "local_player_id": self.local_player_id,
            "video_id": self.video_id,
            "raw_track_id": self.raw_track_id,
            "source_video_name": self.source_video_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "average_confidence": self.average_confidence,
            "average_bbox_height": self.average_bbox_height,
            "raw_movement_score": self.raw_movement_score,
            "movement_score": self.movement_score,
            "center_position_score": self.center_position_score,
            "activity_zone_score": self.activity_zone_score,
            "average_bbox_size_score": self.average_bbox_size_score,
            "active_score": self.active_score,
            "crop_quality_score": self.crop_quality_score,
            "crop_person_shape_score": self.crop_person_shape_score,
            "crop_edge_score": self.crop_edge_score,
            "crop_count": self.crop_count,
            "is_active": self.is_active,
            "ignored_reason": self.ignored_reason,
            "crop_paths": self.crop_paths,
            "representative_crop": self.representative_crop,
        }
        if include_frames:
            payload["frames"] = [frame.to_dict() for frame in self.frames]
        return payload


def process_video_tracking(
    video_info: VideoInfo,
    detector: YOLOEPlayerDetector,
    filter_config: FilterConfig,
    min_crop_height: int,
) -> list[LocalTrack]:
    try:
        return _process_video_tracking_once(video_info, detector, filter_config, min_crop_height)
    except RuntimeError as exc:
        if not _is_half_precision_dtype_error(exc) or detector.force_full_precision:
            raise
        detector.disable_half_precision(str(exc))
        return _process_video_tracking_once(video_info, detector, filter_config, min_crop_height)


def _process_video_tracking_once(
    video_info: VideoInfo,
    detector: YOLOEPlayerDetector,
    filter_config: FilterConfig,
    min_crop_height: int,
) -> list[LocalTrack]:
    tracks_by_id: dict[str, LocalTrack] = {}
    for frame_index, result in enumerate(detector.track_video(video_info.path)):
        for track_frame in _extract_track_frames(result, frame_index, video_info.video_id, detector.config.prompt):
            track = tracks_by_id.get(track_frame.local_track_id)
            if track is None:
                track = LocalTrack(
                    local_player_id=track_frame.local_track_id,
                    video_id=video_info.video_id,
                    raw_track_id=track_frame.raw_track_id,
                    source_video_name=video_info.filename,
                )
                tracks_by_id[track_frame.local_track_id] = track
            track.frames.append(track_frame)
    tracks = sorted(tracks_by_id.values(), key=lambda item: item.local_player_id)
    finalize_track_stats(tracks, video_info, filter_config, min_crop_height)
    return tracks


def _is_half_precision_dtype_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "half" in message and "float" in message and "dtype" in message


def finalize_track_stats(
    tracks: list[LocalTrack],
    video_info: VideoInfo,
    filter_config: FilterConfig,
    min_crop_height: int,
) -> None:
    if not tracks:
        return

    raw_movement: dict[str, float] = {}
    frame_area = max(1.0, float(video_info.width * video_info.height))
    diagonal = max(1.0, math.hypot(video_info.width, video_info.height))
    max_center_distance = max(1.0, diagonal * 0.5)

    for track in tracks:
        track.frames.sort(key=lambda item: item.frame_index)
        track.frame_count = len(track.frames)
        track.start_frame = track.frames[0].frame_index if track.frames else 0
        track.end_frame = track.frames[-1].frame_index if track.frames else 0
        track.duration_sec = (
            (track.end_frame - track.start_frame + 1) / video_info.fps if video_info.fps > 0 else 0.0
        )
        confidences = [frame.confidence for frame in track.frames]
        heights = [max(0.0, frame.bbox[3] - frame.bbox[1]) for frame in track.frames]
        areas = [bbox_area(frame.bbox) for frame in track.frames]
        centers = [bbox_center(frame.bbox) for frame in track.frames]

        track.average_confidence = float(np.mean(confidences)) if confidences else 0.0
        track.average_bbox_height = float(np.mean(heights)) if heights else 0.0
        size_scores = [clamp(math.sqrt(area / frame_area) * 4.0, 0.0, 1.0) for area in areas]
        track.average_bbox_size_score = float(np.mean(size_scores)) if size_scores else 0.0

        movement = 0.0
        for previous, current in zip(centers, centers[1:]):
            movement += math.hypot(current[0] - previous[0], current[1] - previous[1])
        raw_movement[track.local_player_id] = movement / diagonal
        track.raw_movement_score = float(raw_movement[track.local_player_id])

        center_scores = []
        for x, y in centers:
            distance = math.hypot(x - video_info.width * 0.5, y - video_info.height * 0.5)
            center_scores.append(clamp(1.0 - distance / max_center_distance, 0.0, 1.0))
        track.center_position_score = float(np.mean(center_scores)) if center_scores else 0.0

    max_duration = max((track.duration_sec for track in tracks), default=0.0)
    max_movement = max(raw_movement.values(), default=0.0)

    for track in tracks:
        duration_norm = track.duration_sec / max_duration if max_duration > 0 else 0.0
        movement_norm = raw_movement[track.local_player_id] / max_movement if max_movement > 0 else 0.0
        track.movement_score = clamp(movement_norm, 0.0, 1.0)
        track.active_score = (
            0.35 * clamp(duration_norm, 0.0, 1.0)
            + 0.30 * track.movement_score
            + 0.20 * track.average_bbox_size_score
            + 0.15 * track.center_position_score
        )
    _estimate_activity_zone_scores(tracks, video_info)

    for track in tracks:
        _apply_track_filter(track, filter_config, min_crop_height)


def _estimate_activity_zone_scores(tracks: list[LocalTrack], video_info: VideoInfo) -> None:
    moving_tracks = [
        track
        for track in tracks
        if track.raw_movement_score >= 0.035 and track.average_bbox_size_score >= 0.08 and track.frames
    ]
    if len(moving_tracks) < 2:
        for track in tracks:
            track.activity_zone_score = track.center_position_score
        return

    weighted_points: list[tuple[float, float]] = []
    weights: list[float] = []
    for track in moving_tracks:
        weight = max(0.01, track.raw_movement_score) * max(0.05, track.average_bbox_size_score)
        stride = max(1, len(track.frames) // 20)
        for frame in track.frames[::stride]:
            weighted_points.append(bbox_center(frame.bbox))
            weights.append(weight)

    if not weighted_points:
        for track in tracks:
            track.activity_zone_score = track.center_position_score
        return

    points = np.asarray(weighted_points, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    center = np.average(points, axis=0, weights=weight_array)
    variance = np.average((points - center) ** 2, axis=0, weights=weight_array)
    spread = max(float(np.sqrt(np.sum(variance))), math.hypot(video_info.width, video_info.height) * 0.18)
    scale = max(spread * 1.65, math.hypot(video_info.width, video_info.height) * 0.16)

    for track in tracks:
        if not track.frames:
            track.activity_zone_score = 0.0
            continue
        centers = np.asarray([bbox_center(frame.bbox) for frame in track.frames], dtype=np.float32)
        distances = np.linalg.norm(centers - center, axis=1)
        mean_distance = float(np.mean(distances))
        zone_score = math.exp(-((mean_distance / scale) ** 2))
        track.activity_zone_score = clamp(0.75 * zone_score + 0.25 * track.center_position_score, 0.0, 1.0)


def render_tracking_debug_video(
    video_info: VideoInfo,
    tracks: list[LocalTrack],
    output_path: Path,
    prompt: str,
) -> None:
    import cv2

    ensure_dir(output_path.parent)
    writer = create_video_writer(output_path, video_info.fps, video_info.width, video_info.height)
    records_by_frame: dict[int, list[TrackFrame]] = {}
    tracks_by_id = {track.local_player_id: track for track in tracks}
    for track in tracks:
        for frame in track.frames:
            records_by_frame.setdefault(frame.frame_index, []).append(frame)

    try:
        for frame_index, frame in iter_video_frames(video_info.path):
            for record in records_by_frame.get(frame_index, []):
                track = tracks_by_id[record.local_track_id]
                _draw_track(frame, record, track, prompt)
            writer.write(frame)
    finally:
        writer.release()


def _apply_track_filter(track: LocalTrack, filter_config: FilterConfig, min_crop_height: int) -> None:
    reasons: list[str] = []
    if track.frame_count < filter_config.min_track_frames:
        reasons.append("too_short")

    if filter_config.active_player_filter:
        if filter_config.min_track_duration_sec > 0 and track.duration_sec < filter_config.min_track_duration_sec:
            reasons.append("too_short_duration")
        if track.movement_score < filter_config.min_movement_score:
            reasons.append("low_movement")
        if track.average_bbox_height < min_crop_height:
            reasons.append("too_small")
        if track.active_score < 0.20:
            reasons.append("low_active_score")

    track.is_active = not reasons
    track.ignored_reason = ",".join(reasons) if reasons else None


def _extract_track_frames(result: object, frame_index: int, video_id: str, default_label: str) -> list[TrackFrame]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []

    ids = _to_numpy(boxes.id)
    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    if ids is None or xyxy is None:
        return []

    ids = ids.reshape(-1)
    xyxy = xyxy.reshape(-1, 4)
    count = min(len(ids), len(xyxy))
    if count == 0:
        return []

    confidences = _optional_vector(getattr(boxes, "conf", None), count, default=1.0)
    classes = _optional_vector(getattr(boxes, "cls", None), count, default=0.0)
    mask_polygons, mask_ratios = _extract_masks(result, xyxy[:count])

    frames: list[TrackFrame] = []
    for index in range(count):
        raw_track_id = int(ids[index])
        bbox = [round(float(value), 3) for value in xyxy[index].tolist()]
        local_track_id = format_local_player_id(video_id, raw_track_id)
        frames.append(
            TrackFrame(
                frame_index=frame_index,
                local_track_id=local_track_id,
                raw_track_id=raw_track_id,
                bbox=bbox,
                confidence=float(confidences[index]),
                label=_label_for_class(result, classes[index], default_label),
                mask_polygon=mask_polygons[index],
                mask_area_ratio=mask_ratios[index],
            )
        )
    return frames


def _extract_masks(result: object, boxes: np.ndarray) -> tuple[list[list[list[float]] | None], list[float | None]]:
    count = len(boxes)
    polygons: list[list[list[float]] | None] = [None] * count
    ratios: list[float | None] = [None] * count
    masks = getattr(result, "masks", None)
    if masks is None:
        return polygons, ratios
    xy_list = getattr(masks, "xy", None)
    if xy_list is None:
        return polygons, ratios
    for index, points_raw in enumerate(xy_list[:count]):
        points = np.asarray(points_raw, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons[index] = _simplify_polygon(points)
        area = _polygon_area(points)
        ratios[index] = float(area / max(1.0, bbox_area(boxes[index])))
    return polygons, ratios


def _simplify_polygon(points: np.ndarray, max_points: int = 120) -> list[list[float]]:
    if len(points) > max_points:
        stride = int(math.ceil(len(points) / max_points))
        points = points[::stride]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _to_numpy(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _optional_vector(value: object | None, count: int, default: float) -> np.ndarray:
    array = _to_numpy(value)
    if array is None:
        return np.full(count, default, dtype=np.float32)
    array = array.reshape(-1)
    if len(array) < count:
        padded = np.full(count, default, dtype=np.float32)
        padded[: len(array)] = array
        return padded
    return array[:count]


def _label_for_class(result: object, class_value: float, default_label: str) -> str:
    names = getattr(result, "names", None)
    class_index = int(class_value)
    if isinstance(names, dict):
        return str(names.get(class_index, default_label))
    if isinstance(names, Sequence) and 0 <= class_index < len(names):
        return str(names[class_index])
    return default_label


def _draw_track(frame: np.ndarray, record: TrackFrame, track: LocalTrack, prompt: str) -> None:
    import cv2

    color = _color_for_id(record.local_track_id) if track.is_active else (120, 120, 120)
    if record.mask_polygon:
        points = np.asarray(record.mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

    x1, y1, x2, y2 = [int(round(value)) for value in record.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    short_id = record.local_track_id.split("_track_")[-1]
    status = "active" if track.is_active else "ignored"
    label = f"T{short_id} {record.confidence:.2f} {prompt} {status}"
    text_y = max(18, y1 - 6)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _color_for_id(value: str) -> tuple[int, int, int]:
    seed = abs(hash(value))
    return (
        80 + seed % 150,
        80 + (seed // 7) % 150,
        80 + (seed // 17) % 150,
    )
