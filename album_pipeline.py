"""Shared grouping and export stage, independent of model inference."""
from pathlib import Path
from typing import Any

import numpy as np

from clustering import cluster_embeddings, player_ids_for_labels, save_similarity_matrix_csv
from config import ClusteringConfig, ExportConfig
from conflict_rules import conflicts_by_player_id, find_cluster_conflicts, split_conflicting_clusters
from identity_validation import identity_stats_by_player_id, split_identity_inconsistent_clusters
from json_export import export_player_albums
from player_tracking import LocalTrack
from reid_embedding import EmbeddingRecord
from player_identity import reconcile_player_ids
from video_io import VideoInfo
from player_video_sorting import SORTING_VERSION


def group_and_export(
    output_root: Path,
    records: list[EmbeddingRecord],
    embeddings: np.ndarray,
    tracks_by_id: dict[str, LocalTrack],
    clustering: ClusteringConfig,
    export: ExportConfig,
    previous_groups: list[dict[str, Any]] | None = None,
    next_player_number: int = 1,
    source_root: Path | None = None,
    video_metadata: dict[str, VideoInfo] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = cluster_embeddings(embeddings, clustering.cluster_method, clustering.distance_threshold)
    save_similarity_matrix_csv(
        output_root / "global" / "similarity_matrix.csv",
        [record.local_player_id for record in records], result.similarity_matrix,
    )
    labels, conflicts_report = split_conflicting_clusters(result.labels, records, result.similarity_matrix)
    labels, identity_report = split_identity_inconsistent_clusters(
        labels, records, output_root, result.similarity_matrix, clustering, source_root,
    )
    player_ids = player_ids_for_labels(labels)
    continuity = {}
    if previous_groups is not None:
        player_ids, continuity = reconcile_player_ids(labels, records, previous_groups, next_player_number)
    conflicts = conflicts_by_player_id(find_cluster_conflicts(labels, records), player_ids)
    groups = export_player_albums(
        output_root, records, tracks_by_id, [player_ids[label] for label in labels],
        result.similarity_matrix, conflicts, export, clustering,
        identity_stats_by_player_id(identity_report, player_ids),
        source_root=source_root,
        video_metadata=video_metadata,
    )
    return groups, {
        "cluster_conflict_resolution": conflicts_report,
        "identity_cluster_resolution": identity_report,
        "player_identity_continuity": continuity,
        "video_sorting": {"version": SORTING_VERSION,
            "ranked_video_entries": sum(len(g["video_rankings"]) for g in groups),
            "unavailable_coverage": [{"player_id": g["player_id"], "video_id": r["video_id"],
                                      "reason": r["coverage_status"]}
                                     for g in groups for r in g["video_rankings"] if r["coverage_status"] != "valid"]},
    }
