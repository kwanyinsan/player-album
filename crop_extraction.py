from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import CropConfig
from crop_quality import CropQuality, crop_is_valid, evaluate_crop
from player_tracking import LocalTrack, TrackFrame
from utils import bbox_iou, ensure_dir, pad_bbox, path_to_posix, write_json
from video_io import VideoInfo, iter_video_frames


@dataclass
class CropCandidate:
    local_player_id: str
    frame_index: int
    source_bbox: list[float]
    crop_bbox: tuple[int, int, int, int]
    image: np.ndarray
    masked_image: np.ndarray | None
    quality: CropQuality


def extract_crops_for_video(
    video_info: VideoInfo,
    tracks: list[LocalTrack],
    video_output_dir: Path,
    output_root: Path,
    crop_config: CropConfig,
) -> dict[str, Any]:
    crops_root = ensure_dir(video_output_dir / "crops")
    active_tracks = {track.local_player_id: track for track in tracks if track.is_active}
    records_by_frame: dict[int, list[TrackFrame]] = {}
    for track in active_tracks.values():
        for frame in track.frames:
            records_by_frame.setdefault(frame.frame_index, []).append(frame)

    candidates_by_track: dict[str, list[CropCandidate]] = {track_id: [] for track_id in active_tracks}
    for frame_index, frame in iter_video_frames(video_info.path):
        for record in records_by_frame.get(frame_index, []):
            candidate = _candidate_from_frame(frame, record, crop_config)
            if candidate is not None:
                candidates_by_track[record.local_track_id].append(candidate)

    saved_total = 0
    no_crop_tracks: list[str] = []
    for local_track_id, track in active_tracks.items():
        selected = _select_best_crops(candidates_by_track.get(local_track_id, []), crop_config.max_crops_per_track)
        if not selected:
            track.is_active = False
            track.ignored_reason = track.ignored_reason or "no_valid_crops"
            no_crop_tracks.append(local_track_id)
            continue
        _save_track_crops(track, selected, crops_root, output_root, crop_config)
        saved_total += len(selected)

    return {
        "tracks_with_crops": sum(1 for track in tracks if track.crop_paths),
        "saved_crops": saved_total,
        "tracks_without_valid_crops": no_crop_tracks,
    }


def _candidate_from_frame(frame: np.ndarray, record: TrackFrame, crop_config: CropConfig) -> CropCandidate | None:
    frame_h, frame_w = frame.shape[:2]
    crop_bbox = pad_bbox(record.bbox, crop_config.crop_padding, frame_w, frame_h)
    x1, y1, x2, y2 = crop_bbox
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    if record.mask_area_ratio is not None and record.mask_area_ratio < crop_config.min_mask_area_ratio:
        return None

    masked_crop = _masked_crop(crop, crop_bbox, record.mask_polygon) if record.mask_polygon else None
    image_for_score = masked_crop if crop_config.mask_crop_mode == "masked" and masked_crop is not None else crop
    quality = evaluate_crop(image_for_score, crop_bbox, frame.shape, record.confidence, record.mask_area_ratio)
    if not crop_is_valid(
        image_for_score,
        crop_bbox,
        crop_config.min_crop_height,
        crop_config.blur_threshold,
        quality,
        crop_config.min_person_shape_score,
        crop_config.max_edge_touch_ratio,
    ):
        return None

    output_image = crop
    if crop_config.mask_crop_mode == "masked" and masked_crop is not None:
        output_image = masked_crop

    return CropCandidate(
        local_player_id=record.local_track_id,
        frame_index=record.frame_index,
        source_bbox=record.bbox,
        crop_bbox=crop_bbox,
        image=output_image,
        masked_image=masked_crop,
        quality=quality,
    )


def _select_best_crops(candidates: list[CropCandidate], limit: int) -> list[CropCandidate]:
    ordered = sorted(candidates, key=lambda item: item.quality.score, reverse=True)
    selected: list[CropCandidate] = []
    for candidate in ordered:
        if len(selected) >= limit:
            break
        if _is_near_duplicate(candidate, selected):
            continue
        selected.append(candidate)

    if len(selected) < limit:
        selected_ids = {id(item) for item in selected}
        for candidate in ordered:
            if len(selected) >= limit:
                break
            if id(candidate) not in selected_ids:
                selected.append(candidate)
                selected_ids.add(id(candidate))
    return sorted(selected, key=lambda item: item.frame_index)


def _is_near_duplicate(candidate: CropCandidate, selected: list[CropCandidate]) -> bool:
    for existing in selected:
        close_in_time = abs(candidate.frame_index - existing.frame_index) <= 3
        high_overlap = bbox_iou(candidate.source_bbox, existing.source_bbox) >= 0.92
        if close_in_time and high_overlap:
            return True
    return False


def _save_track_crops(
    track: LocalTrack,
    selected: list[CropCandidate],
    crops_root: Path,
    output_root: Path,
    crop_config: CropConfig,
) -> None:
    import cv2

    track_dir = ensure_dir(crops_root / track.local_player_id)
    metadata: dict[str, Any] = {
        "local_player_id": track.local_player_id,
        "video_id": track.video_id,
        "source_video_name": track.source_video_name,
        "crops": [],
    }
    track.crop_paths.clear()
    quality_scores = [candidate.quality.score for candidate in selected]
    shape_scores = [candidate.quality.person_shape_score for candidate in selected]
    edge_scores = [candidate.quality.edge_score for candidate in selected]
    track.crop_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0
    track.crop_person_shape_score = float(np.mean(shape_scores)) if shape_scores else 0.0
    track.crop_edge_score = float(np.mean(edge_scores)) if edge_scores else 0.0
    track.crop_count = len(selected)
    metadata["summary"] = {
        "crop_quality_score": track.crop_quality_score,
        "crop_person_shape_score": track.crop_person_shape_score,
        "crop_edge_score": track.crop_edge_score,
        "crop_count": track.crop_count,
    }
    for index, candidate in enumerate(selected, start=1):
        crop_path = track_dir / f"crop_{index:04d}.jpg"
        cv2.imwrite(str(crop_path), candidate.image)
        rel_crop = path_to_posix(crop_path, output_root)
        track.crop_paths.append(rel_crop)

        crop_entry: dict[str, Any] = {
            "path": rel_crop,
            "frame_index": candidate.frame_index,
            "source_bbox": candidate.source_bbox,
            "crop_bbox": list(candidate.crop_bbox),
            "quality": candidate.quality.to_dict(),
        }
        if crop_config.mask_crop_mode == "both" and candidate.masked_image is not None:
            masked_path = track_dir / f"crop_{index:04d}_masked.jpg"
            cv2.imwrite(str(masked_path), candidate.masked_image)
            crop_entry["masked_path"] = path_to_posix(masked_path, output_root)
        metadata["crops"].append(crop_entry)

    track.representative_crop = track.crop_paths[0] if track.crop_paths else None
    metadata["representative_crop"] = track.representative_crop
    write_json(track_dir / "metadata.json", metadata)


def _masked_crop(crop: np.ndarray, crop_bbox: tuple[int, int, int, int], polygon: list[list[float]]) -> np.ndarray:
    import cv2

    x1, y1, _, _ = crop_bbox
    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    points[:, 0] -= x1
    points[:, 1] -= y1
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    masked = crop.copy()
    masked[mask == 0] = 0
    return masked
