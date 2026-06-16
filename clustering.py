from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClusteringResult:
    labels: list[int]
    player_ids_by_index: list[str]
    player_id_by_label: dict[int, str]
    similarity_matrix: np.ndarray


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    distance_threshold: float,
) -> ClusteringResult:
    if method.lower() != "agglomerative":
        raise ValueError(
            f"Unsupported clustering method {method!r}. "
            "The interface is ready for more methods, but this version implements agglomerative clustering."
        )
    if embeddings.size == 0 or len(embeddings) == 0:
        return ClusteringResult([], [], {}, np.zeros((0, 0), dtype=np.float32))

    normalized = _normalize_rows(embeddings)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = np.clip(1.0 - similarity, 0.0, 2.0)
    if len(embeddings) == 1:
        labels = [0]
    else:
        labels = _agglomerative_labels(distances, distance_threshold)
    player_id_by_label = player_ids_for_labels(labels)
    player_ids_by_index = [player_id_by_label[label] for label in labels]
    return ClusteringResult(
        labels=labels,
        player_ids_by_index=player_ids_by_index,
        player_id_by_label=player_id_by_label,
        similarity_matrix=similarity.astype(np.float32),
    )


def save_similarity_matrix_csv(path: Path, local_track_ids: list[str], similarity_matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["local_track_id", *local_track_ids])
        for local_track_id, row in zip(local_track_ids, similarity_matrix):
            writer.writerow([local_track_id, *[f"{float(value):.6f}" for value in row]])


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _agglomerative_labels(distances: np.ndarray, threshold: float) -> list[int]:
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


def player_ids_for_labels(labels: list[int]) -> dict[int, str]:
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(label, []).append(index)
    sorted_labels = sorted(clusters, key=lambda label: min(clusters[label]))
    return {label: f"P{position:04d}" for position, label in enumerate(sorted_labels, start=1)}
