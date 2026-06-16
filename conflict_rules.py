from __future__ import annotations

from typing import Any

import numpy as np


def build_cannot_link_pairs(records: list[object]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if _value(records[left], "video_id") != _value(records[right], "video_id"):
                continue
            if intervals_overlap(
                int(_value(records[left], "start_frame")),
                int(_value(records[left], "end_frame")),
                int(_value(records[right], "start_frame")),
                int(_value(records[right], "end_frame")),
            ):
                pairs.append((left, right))
    return pairs


def find_cluster_conflicts(labels: list[int], records: list[object]) -> dict[int, list[dict[str, Any]]]:
    conflicts: dict[int, list[dict[str, Any]]] = {}
    for left, right in build_cannot_link_pairs(records):
        if labels[left] != labels[right]:
            continue
        label = labels[left]
        conflicts.setdefault(label, []).append(
            {
                "reason": "same_video_temporal_overlap",
                "video_id": _value(records[left], "video_id"),
                "source_video_name": _value(records[left], "source_video_name"),
                "local_tracks": [
                    _value(records[left], "local_player_id"),
                    _value(records[right], "local_player_id"),
                ],
                "frame_ranges": [
                    [int(_value(records[left], "start_frame")), int(_value(records[left], "end_frame"))],
                    [int(_value(records[right], "start_frame")), int(_value(records[right], "end_frame"))],
                ],
            }
        )
    return conflicts


def conflicts_by_player_id(
    conflicts_by_label: dict[int, list[dict[str, Any]]],
    player_id_by_label: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        player_id_by_label[label]: conflicts
        for label, conflicts in conflicts_by_label.items()
        if label in player_id_by_label
    }


def split_conflicting_clusters(
    labels: list[int],
    records: list[object],
    similarity_matrix: np.ndarray | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Split same-video temporal conflicts that clustering cannot model."""
    if not labels:
        return labels, {"split_clusters": [], "resolved_cluster_count": 0}

    cannot_link = {frozenset(pair) for pair in build_cannot_link_pairs(records)}
    members_by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        members_by_label.setdefault(label, []).append(index)

    resolved = [-1] * len(labels)
    next_label = 0
    split_events: list[dict[str, Any]] = []

    for original_label in sorted(members_by_label, key=lambda label: min(members_by_label[label])):
        members = sorted(
            members_by_label[original_label],
            key=lambda index: (
                -int(_value(records[index], "frame_count")),
                int(_value(records[index], "start_frame")),
                str(_value(records[index], "local_player_id")),
            ),
        )
        subclusters: list[list[int]] = []
        for member in members:
            compatible = [
                subgroup
                for subgroup in subclusters
                if all(frozenset((member, existing)) not in cannot_link for existing in subgroup)
            ]
            if compatible:
                _best_subcluster(member, compatible, similarity_matrix).append(member)
            else:
                subclusters.append([member])

        if len(subclusters) > 1:
            split_events.append(
                {
                    "original_label": original_label,
                    "original_local_tracks": [_value(records[index], "local_player_id") for index in members],
                    "split_sizes": [len(subgroup) for subgroup in subclusters],
                }
            )

        for subgroup in subclusters:
            for index in subgroup:
                resolved[index] = next_label
            next_label += 1

    return resolved, {
        "split_clusters": split_events,
        "resolved_cluster_count": next_label,
    }


def intervals_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) <= min(left_end, right_end)


def _best_subcluster(
    member: int,
    subclusters: list[list[int]],
    similarity_matrix: np.ndarray | None,
) -> list[int]:
    if similarity_matrix is None or similarity_matrix.size == 0:
        return subclusters[0]
    return max(
        subclusters,
        key=lambda subgroup: float(np.mean([similarity_matrix[member, existing] for existing in subgroup])),
    )


def _value(record: object, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    return getattr(record, key)
