from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config import ClusteringConfig
from reid_embedding import EmbeddingRecord
from utils import clamp, normalize_vector


def split_identity_inconsistent_clusters(
    labels: list[int],
    records: list[EmbeddingRecord],
    output_root: Path,
    similarity_matrix: np.ndarray,
    config: ClusteringConfig,
) -> tuple[list[int], dict[str, Any]]:
    if not labels or not config.identity_split_enabled:
        return labels, {"enabled": False, "split_clusters": [], "final_clusters": {}}

    fingerprints = _load_track_fingerprints(records, output_root)
    combined_similarity = _combined_similarity_matrix(
        similarity_matrix,
        fingerprints,
        config.identity_color_weight,
    )

    resolved_labels = list(labels)
    next_label = max(resolved_labels, default=-1) + 1
    report: dict[str, Any] = {
        "enabled": True,
        "color_weight": config.identity_color_weight,
        "split_distance_threshold": config.identity_split_distance_threshold,
        "split_min_cluster_size": config.identity_split_min_cluster_size,
        "split_clusters": [],
    }

    clusters = _indices_by_label(resolved_labels)
    for label, indices in sorted(clusters.items()):
        stats = _identity_stats(indices, similarity_matrix, combined_similarity, config)
        if not _should_split(indices, stats, config):
            continue

        sublabels = _cluster_combined_distances(
            indices,
            combined_similarity,
            config.identity_split_distance_threshold,
        )
        subgroups = _indices_by_sublabel(indices, sublabels)
        if len(subgroups) <= 1:
            continue

        assigned_labels: list[int] = []
        for sub_index, subgroup in enumerate(subgroups):
            new_label = label if sub_index == 0 else next_label
            if sub_index > 0:
                next_label += 1
            assigned_labels.append(new_label)
            for record_index in subgroup:
                resolved_labels[record_index] = new_label

        report["split_clusters"].append(
            {
                "original_label": int(label),
                "original_size": len(indices),
                "new_labels": assigned_labels,
                "new_sizes": [len(group) for group in subgroups],
                "reason": stats["reasons"],
                "before": stats,
            }
        )

    final_clusters = _indices_by_label(resolved_labels)
    report["final_clusters"] = {
        str(label): _identity_stats(indices, similarity_matrix, combined_similarity, config)
        for label, indices in sorted(final_clusters.items())
    }
    return resolved_labels, report


def identity_stats_by_player_id(
    identity_report: dict[str, Any],
    player_id_by_label: dict[int, str],
) -> dict[str, dict[str, Any]]:
    final_clusters = identity_report.get("final_clusters") or {}
    output: dict[str, dict[str, Any]] = {}
    for label_text, stats in final_clusters.items():
        try:
            label = int(label_text)
        except (TypeError, ValueError):
            continue
        player_id = player_id_by_label.get(label)
        if player_id is not None:
            output[player_id] = dict(stats)
    return output


def _load_track_fingerprints(records: list[EmbeddingRecord], output_root: Path) -> list[np.ndarray | None]:
    return [_track_fingerprint(record, output_root) for record in records]


def _track_fingerprint(record: EmbeddingRecord, output_root: Path) -> np.ndarray | None:
    paths = _fingerprint_crop_paths(record)
    crop_fingerprints: list[np.ndarray] = []
    for rel_path in paths:
        fingerprint = _crop_color_fingerprint(output_root / rel_path)
        if fingerprint is not None:
            crop_fingerprints.append(fingerprint)
    if not crop_fingerprints:
        return None
    return normalize_vector(np.mean(np.vstack(crop_fingerprints), axis=0))


def _fingerprint_crop_paths(record: EmbeddingRecord, limit: int = 4) -> list[str]:
    paths = list(record.crop_paths)
    if record.representative_crop:
        paths = [record.representative_crop, *[path for path in paths if path != record.representative_crop]]
    if len(paths) <= limit:
        return paths
    positions = np.linspace(0, len(paths) - 1, num=limit)
    selected: list[str] = []
    for position in positions:
        path = paths[int(round(float(position)))]
        if path not in selected:
            selected.append(path)
    return selected


def _crop_color_fingerprint(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None

    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    x1 = int(width * 0.18)
    x2 = max(x1 + 1, int(width * 0.82))
    y1 = int(height * 0.08)
    y2 = max(y1 + 1, int(height * 0.96))
    body = image[y1:y2, x1:x2]
    if body.size == 0:
        body = image

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    features: list[float] = []
    for start, end in ((0.12, 0.42), (0.42, 0.78), (0.78, 1.0)):
        part_y1 = int(hsv.shape[0] * start)
        part_y2 = max(part_y1 + 1, int(hsv.shape[0] * end))
        part = hsv[part_y1:part_y2, :]
        hs_hist = cv2.calcHist([part], [0, 1], None, [16, 4], [0, 180, 0, 256]).astype("float32").reshape(-1)
        v_hist = cv2.calcHist([part], [2], None, [8], [0, 256]).astype("float32").reshape(-1)
        hs_hist = hs_hist / (float(hs_hist.sum()) + 1e-12)
        v_hist = v_hist / (float(v_hist.sum()) + 1e-12)
        features.extend(float(value) for value in hs_hist)
        features.extend(float(value) for value in v_hist)

    return normalize_vector(np.asarray(features, dtype=np.float32))


def _combined_similarity_matrix(
    similarity_matrix: np.ndarray,
    fingerprints: list[np.ndarray | None],
    color_weight: float,
) -> np.ndarray:
    if similarity_matrix.size == 0:
        return similarity_matrix.astype(np.float32)

    combined = np.array(similarity_matrix, dtype=np.float32, copy=True)
    reid_weight = 1.0 - color_weight
    for left in range(len(fingerprints)):
        combined[left, left] = 1.0
        for right in range(left + 1, len(fingerprints)):
            color_score = _color_similarity(fingerprints[left], fingerprints[right])
            value = reid_weight * float(similarity_matrix[left, right]) + color_weight * color_score
            combined[left, right] = combined[right, left] = float(clamp(value, -1.0, 1.0))
    return combined


def _color_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.5
    raw = float(np.dot(left, right))
    return clamp((raw - 0.45) / 0.45, 0.0, 1.0)


def _should_split(indices: list[int], stats: dict[str, Any], config: ClusteringConfig) -> bool:
    if len(indices) < config.identity_split_min_cluster_size:
        return False
    if float(stats["combined_mean"]) < config.identity_split_mean_threshold:
        return True
    if float(stats["color_mean"]) < config.identity_split_color_threshold:
        return True
    if (
        float(stats["combined_p10"]) < config.identity_split_p10_threshold
        and float(stats["combined_std"]) > config.identity_split_std_threshold
    ):
        return True
    if len(indices) >= 24 and float(stats["combined_min"]) < config.identity_split_large_min_similarity:
        return True
    return False


def _identity_stats(
    indices: list[int],
    similarity_matrix: np.ndarray,
    combined_similarity: np.ndarray,
    config: ClusteringConfig,
) -> dict[str, Any]:
    color_weight = config.identity_color_weight
    if len(indices) <= 1 or similarity_matrix.size == 0:
        return {
            "size": len(indices),
            "reid_mean": 1.0,
            "reid_min": 1.0,
            "combined_mean": 1.0,
            "combined_min": 1.0,
            "combined_p10": 1.0,
            "combined_std": 0.0,
            "color_mean": 1.0,
            "identity_consistency_score": 1.0,
            "reasons": [],
        }

    reid_values: list[float] = []
    combined_values: list[float] = []
    color_values: list[float] = []
    for left_pos, left in enumerate(indices):
        for right in indices[left_pos + 1 :]:
            reid_value = float(similarity_matrix[left, right])
            combined_value = float(combined_similarity[left, right])
            reid_values.append(reid_value)
            combined_values.append(combined_value)
            if color_weight > 0:
                color_value = (combined_value - (1.0 - color_weight) * reid_value) / color_weight
                color_values.append(float(clamp(color_value, 0.0, 1.0)))

    combined_mean = float(np.mean(combined_values))
    combined_min = float(np.min(combined_values))
    combined_p10 = float(np.percentile(combined_values, 10))
    combined_std = float(np.std(combined_values))
    color_mean = float(np.mean(color_values)) if color_values else 0.5
    score = clamp((combined_mean - 0.68) / 0.22, 0.0, 1.0)

    reasons: list[str] = []
    if combined_mean < config.identity_min_consistency:
        reasons.append("weak_identity_consistency")
    if color_mean < config.identity_min_color_consistency:
        reasons.append("outfit_color_diversity")
    if combined_min < config.identity_split_large_min_similarity and len(indices) >= 3:
        reasons.append("low_min_pair_similarity")

    return {
        "size": len(indices),
        "reid_mean": float(np.mean(reid_values)),
        "reid_min": float(np.min(reid_values)),
        "combined_mean": combined_mean,
        "combined_min": combined_min,
        "combined_p10": combined_p10,
        "combined_std": combined_std,
        "color_mean": color_mean,
        "identity_consistency_score": float(score),
        "reasons": reasons,
    }


def _cluster_combined_distances(
    indices: list[int],
    combined_similarity: np.ndarray,
    threshold: float,
) -> list[int]:
    if len(indices) <= 1:
        return [0]
    distances = np.clip(1.0 - combined_similarity[np.ix_(indices, indices)], 0.0, 2.0)
    try:
        from sklearn.cluster import AgglomerativeClustering

        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
        return [int(label) for label in model.fit_predict(distances)]
    except ImportError:
        return _fallback_average_link_labels(distances, threshold)


def _fallback_average_link_labels(distances: np.ndarray, threshold: float) -> list[int]:
    clusters: list[list[int]] = [[index] for index in range(len(distances))]
    while True:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                pair_distance = float(np.mean(distances[np.ix_(clusters[left], clusters[right])]))
                if pair_distance < best_distance:
                    best_distance = pair_distance
                    best_pair = (left, right)
        if best_pair is None or best_distance > threshold:
            break
        left, right = best_pair
        clusters[left].extend(clusters[right])
        del clusters[right]

    labels = [0] * len(distances)
    for label, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = label
    return labels


def _indices_by_label(labels: list[int]) -> dict[int, list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        output[int(label)].append(index)
    return dict(output)


def _indices_by_sublabel(indices: list[int], sublabels: list[int]) -> list[list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for record_index, sublabel in zip(indices, sublabels):
        output[int(sublabel)].append(record_index)
    return [output[label] for label in sorted(output, key=lambda item: min(output[item]))]
