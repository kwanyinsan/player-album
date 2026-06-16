from __future__ import annotations

from typing import Any

import numpy as np

from config import ClusteringConfig, ExportConfig
from player_tracking import LocalTrack
from reid_embedding import EmbeddingRecord
from utils import clamp


def score_album_group(
    indices: list[int],
    records: list[EmbeddingRecord],
    tracks_by_id: dict[str, LocalTrack],
    similarity_matrix: np.ndarray,
    conflicts: list[dict[str, Any]],
    export_config: ExportConfig,
    clustering_config: ClusteringConfig,
    identity_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track_scores = [
        _score_track(records[index], tracks_by_id.get(records[index].local_player_id))
        for index in indices
    ]
    valid_scores = [item for item in track_scores if item["track_found"]]
    if not valid_scores:
        return _quality_payload(0.0, "rejected", ["no_valid_tracks"], track_scores)

    track_quality_values = [float(item["track_quality_score"]) for item in valid_scores]
    motion_values = [float(item["motion_score"]) for item in valid_scores]
    activity_values = [float(item["activity_zone_score"]) for item in valid_scores]
    static_fraction = _fraction(valid_scores, lambda item: "static_track" in item["reasons"])
    partial_fraction = _fraction(valid_scores, lambda item: "partial_or_edge_crop" in item["reasons"])
    low_crop_fraction = _fraction(valid_scores, lambda item: "low_crop_count" in item["reasons"])

    avg_track_quality = float(np.mean(track_quality_values))
    best_track_quality = float(np.max(track_quality_values))
    avg_motion = float(np.mean(motion_values))
    avg_activity = float(np.mean(activity_values))
    cluster_consistency = _cluster_consistency_score(indices, similarity_matrix)
    identity_score = float((identity_info or {}).get("identity_consistency_score", cluster_consistency))
    identity_reasons = list((identity_info or {}).get("reasons") or [])
    combined_consistency = clamp(0.60 * cluster_consistency + 0.40 * identity_score, 0.0, 1.0)

    score = (
        0.45 * avg_track_quality
        + 0.18 * best_track_quality
        + 0.17 * combined_consistency
        + 0.12 * avg_motion
        + 0.08 * avg_activity
    )

    reasons: list[str] = []
    penalty = 0.0
    if conflicts:
        penalty += 0.22
        reasons.append("same_video_conflict")
    if static_fraction >= 0.45:
        penalty += 0.18
        reasons.append("static_tracks")
    if partial_fraction >= 0.35:
        penalty += 0.10
        reasons.append("partial_or_edge_crops")
    if low_crop_fraction >= 0.50:
        penalty += 0.06
        reasons.append("low_crop_count")
    if len(indices) >= 35 and avg_motion < 0.42:
        penalty += 0.18
        reasons.append("large_low_motion_cluster")
    if cluster_consistency < 0.45:
        penalty += 0.10
        reasons.append("weak_cluster_consistency")
    if len(indices) >= 8 and identity_score < clustering_config.identity_album_min_consistency:
        penalty += 0.16
        reasons.append("weak_identity_consistency")
    elif len(indices) >= 16 and identity_score < clustering_config.identity_album_large_min_consistency:
        penalty += 0.10
        reasons.append("weak_identity_consistency")
    if "weak_identity_consistency" in identity_reasons:
        penalty += 0.12
        reasons.append("weak_identity_consistency")
    if "outfit_color_diversity" in identity_reasons:
        penalty += 0.10
        reasons.append("outfit_color_diversity")
    if "low_min_pair_similarity" in identity_reasons:
        penalty += 0.08
        reasons.append("low_min_pair_similarity")
    if (
        len(indices) >= 4
        and float((identity_info or {}).get("color_mean", 1.0))
        < clustering_config.identity_album_min_color_consistency
    ):
        penalty += 0.08
        reasons.append("outfit_color_diversity")
    if avg_track_quality < 0.52:
        reasons.append("low_track_quality")
    if avg_motion < 0.30:
        reasons.append("low_motion")

    score = clamp(score - penalty, 0.0, 1.0)
    if not export_config.quality_filter_export:
        status = "accepted"
    elif score >= export_config.album_accept_threshold:
        status = "accepted"
    elif score >= export_config.album_review_threshold:
        status = "review"
    else:
        status = "rejected"

    return _quality_payload(
        score,
        status,
        reasons,
        track_scores,
        {
            "average_track_quality": avg_track_quality,
            "best_track_quality": best_track_quality,
            "cluster_consistency_score": cluster_consistency,
            "identity_consistency_score": identity_score,
            "average_motion_score": avg_motion,
            "average_activity_zone_score": avg_activity,
            "static_track_fraction": static_fraction,
            "partial_crop_fraction": partial_fraction,
            "low_crop_count_fraction": low_crop_fraction,
            **_identity_summary(identity_info),
        },
    )


def _score_track(record: EmbeddingRecord, track: LocalTrack | None) -> dict[str, Any]:
    if track is None:
        return {
            "local_player_id": record.local_player_id,
            "track_found": False,
            "track_quality_score": 0.0,
            "reasons": ["missing_track"],
        }

    crop_quality = clamp(track.crop_quality_score, 0.0, 1.0)
    shape = clamp(track.crop_person_shape_score, 0.0, 1.0)
    absolute_motion = clamp(track.raw_movement_score / 0.18, 0.0, 1.0)
    motion = clamp(0.65 * absolute_motion + 0.35 * track.movement_score, 0.0, 1.0)
    duration = clamp(track.duration_sec / 2.0, 0.0, 1.0)
    size = clamp(track.average_bbox_size_score, 0.0, 1.0)
    activity = clamp(track.activity_zone_score, 0.0, 1.0)
    embedding = clamp(record.embedding_consistency_score, 0.0, 1.0)
    crop_count = clamp(track.crop_count / 6.0, 0.0, 1.0)

    score = (
        0.25 * crop_quality
        + 0.18 * shape
        + 0.17 * motion
        + 0.12 * duration
        + 0.10 * size
        + 0.08 * activity
        + 0.06 * embedding
        + 0.04 * crop_count
    )

    reasons: list[str] = []
    penalty = 0.0
    if track.raw_movement_score < 0.025 and track.duration_sec >= 1.0:
        penalty += 0.16
        reasons.append("static_track")
    if shape < 0.45 or track.crop_edge_score < 0.55:
        penalty += 0.10
        reasons.append("partial_or_edge_crop")
    if track.crop_count < 2:
        penalty += 0.04
        reasons.append("low_crop_count")
    if record.embedding_consistency_score < 0.45:
        penalty += 0.08
        reasons.append("inconsistent_track_embedding")

    return {
        "local_player_id": record.local_player_id,
        "track_found": True,
        "track_quality_score": clamp(score - penalty, 0.0, 1.0),
        "crop_quality_score": crop_quality,
        "person_shape_score": shape,
        "motion_score": motion,
        "raw_movement_score": track.raw_movement_score,
        "duration_score": duration,
        "bbox_size_score": size,
        "activity_zone_score": activity,
        "embedding_consistency_score": embedding,
        "crop_count_score": crop_count,
        "crop_count": track.crop_count,
        "reasons": reasons,
    }


def _cluster_consistency_score(indices: list[int], similarity_matrix: np.ndarray) -> float:
    if len(indices) <= 1 or similarity_matrix.size == 0:
        return 0.70
    values: list[float] = []
    for left_pos, left in enumerate(indices):
        for right in indices[left_pos + 1 :]:
            values.append(float(similarity_matrix[left, right]))
    if not values:
        return 0.70
    mean_similarity = float(np.mean(values))
    return clamp((mean_similarity - 0.76) / 0.20, 0.0, 1.0)


def _fraction(items: list[dict[str, Any]], predicate: Any) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def _quality_payload(
    score: float,
    status: str,
    reasons: list[str],
    track_scores: list[dict[str, Any]],
    summary: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "album_quality_score": float(score),
        "export_status": status,
        "quality_reasons": sorted(set(reasons)),
        "quality_summary": summary or {},
        "track_quality": track_scores,
    }


def _identity_summary(identity_info: dict[str, Any] | None) -> dict[str, float]:
    if not identity_info:
        return {}
    output: dict[str, float] = {}
    for key in (
        "reid_mean",
        "reid_min",
        "combined_mean",
        "combined_min",
        "combined_p10",
        "combined_std",
        "color_mean",
    ):
        if key in identity_info:
            output[f"identity_{key}"] = float(identity_info[key])
    return output
