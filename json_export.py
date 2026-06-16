from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from album_quality import score_album_group
from config import ClusteringConfig, ExportConfig
from player_tracking import LocalTrack
from reid_embedding import EmbeddingRecord
from utils import ensure_dir, path_to_posix, write_json
from video_io import VideoInfo


def write_local_tracks_json(video_info: VideoInfo, tracks: list[LocalTrack], path: Path, error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "video": video_info.to_dict(),
        "tracks": [track.to_dict(include_frames=True) for track in tracks],
    }
    if error:
        payload["error"] = error
    write_json(path, payload)


def export_player_albums(
    output_root: Path,
    records: list[EmbeddingRecord],
    tracks_by_id: dict[str, LocalTrack],
    player_ids_by_index: list[str],
    similarity_matrix: np.ndarray,
    conflicts_by_player: dict[str, list[dict[str, Any]]],
    export_config: ExportConfig,
    clustering_config: ClusteringConfig,
    identity_by_player: dict[str, dict[str, Any]] | None = None,
    source_root: Path | None = None,
) -> list[dict[str, Any]]:
    players_root = ensure_dir(output_root / "players")
    review_root = output_root / "review"
    rejected_root = output_root / "rejected"
    if export_config.export_review_rejected_folders:
        ensure_dir(review_root)
        ensure_dir(rejected_root)
    groups: dict[str, list[int]] = {}
    for index, player_id in enumerate(player_ids_by_index):
        groups.setdefault(player_id, []).append(index)

    player_groups: list[dict[str, Any]] = []
    for player_id in sorted(groups):
        indices = groups[player_id]
        quality = score_album_group(
            indices,
            records,
            tracks_by_id,
            similarity_matrix,
            conflicts_by_player.get(player_id, []),
            export_config,
            clustering_config,
            (identity_by_player or {}).get(player_id),
        )
        export_status = str(quality["export_status"])
        local_track_ids = [records[index].local_player_id for index in indices]
        videos = sorted({records[index].source_video_name for index in indices})
        video_ids = sorted({records[index].video_id for index in indices})
        copied_best_paths = []

        representative_crop = None
        should_copy_album = export_status == "accepted" or export_config.export_review_rejected_folders
        player_dir: Path | None = None
        if should_copy_album:
            album_root = {
                "accepted": players_root,
                "review": review_root,
                "rejected": rejected_root,
            }.get(export_status, players_root)
            player_dir = ensure_dir(album_root / player_id)
            for index in indices:
                record = records[index]
                if not record.representative_crop:
                    continue
                source_path = (source_root or output_root) / record.representative_crop
                if not source_path.exists():
                    continue
                target_path = player_dir / f"{record.local_player_id}_best.jpg"
                shutil.copy2(source_path, target_path)
                copied_best_paths.append(path_to_posix(target_path, output_root))
                if representative_crop is None:
                    representative_path = player_dir / "representative.jpg"
                    shutil.copy2(source_path, representative_path)
                    representative_crop = path_to_posix(representative_path, output_root)
        else:
            representative_crop = _first_representative_crop(indices, records)

        average_similarity = _average_group_similarity(indices, similarity_matrix)
        needs_review = player_id in conflicts_by_player
        metadata = {
            "player_id": player_id,
            "local_tracks": local_track_ids,
            "videos": videos,
            "video_ids": video_ids,
            "representative_crop": representative_crop,
            "best_crops": copied_best_paths,
            "average_similarity_score": average_similarity,
            "needs_review": needs_review,
            "conflicts": conflicts_by_player.get(player_id, []),
            "identity_validation": (identity_by_player or {}).get(player_id, {}),
            **quality,
        }
        if player_dir is not None:
            write_json(player_dir / "metadata.json", metadata)

        player_groups.append(
            {
                "player_id": player_id,
                "representative_crop": representative_crop,
                "videos": videos,
                "video_ids": video_ids,
                "local_tracks": local_track_ids,
                "cluster_size": len(local_track_ids),
                "average_similarity_score": average_similarity,
                "needs_review": needs_review,
                "conflicts": conflicts_by_player.get(player_id, []),
                "identity_validation": (identity_by_player or {}).get(player_id, {}),
                **quality,
            }
        )
    return player_groups


def build_video_index(
    videos: list[VideoInfo],
    player_groups: list[dict[str, Any]],
    tracks_by_id: dict[str, LocalTrack],
) -> list[dict[str, Any]]:
    players_by_video_id: dict[str, set[str]] = {video.video_id: set() for video in videos}
    for group in player_groups:
        if group.get("export_status") != "accepted":
            continue
        player_id = str(group["player_id"])
        for local_track_id in group["local_tracks"]:
            track = tracks_by_id.get(local_track_id)
            if track is not None:
                players_by_video_id.setdefault(track.video_id, set()).add(player_id)

    return [
        {
            "video_id": video.video_id,
            "original_filename": video.filename,
            "path": video.path.as_posix(),
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "duration_sec": video.duration_sec,
            "players": sorted(players_by_video_id.get(video.video_id, set())),
        }
        for video in videos
    ]


def write_global_indexes(global_dir: Path, player_groups: list[dict[str, Any]], video_index: list[dict[str, Any]]) -> None:
    ensure_dir(global_dir)
    write_json(global_dir / "player_groups.json", player_groups)
    write_json(global_dir / "video_index.json", video_index)


def _average_group_similarity(indices: list[int], similarity_matrix: np.ndarray) -> float:
    if len(indices) <= 1 or similarity_matrix.size == 0:
        return 1.0
    values: list[float] = []
    for left_pos, left in enumerate(indices):
        for right in indices[left_pos + 1 :]:
            values.append(float(similarity_matrix[left, right]))
    return float(np.mean(values)) if values else 1.0


def _first_representative_crop(indices: list[int], records: list[EmbeddingRecord]) -> str | None:
    for index in indices:
        if records[index].representative_crop:
            return records[index].representative_crop
    return None
