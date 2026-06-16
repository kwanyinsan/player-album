from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from clustering import cluster_embeddings, player_ids_for_labels, save_similarity_matrix_csv
from config import PipelineConfig, parse_config, validate_config
from conflict_rules import conflicts_by_player_id, find_cluster_conflicts, split_conflicting_clusters
from crop_extraction import extract_crops_for_video
from gallery_export import export_gallery_html
from identity_validation import identity_stats_by_player_id, split_identity_inconsistent_clusters
from json_export import build_video_index, export_player_albums, write_global_indexes, write_local_tracks_json
from player_tracking import LocalTrack, process_video_tracking, render_tracking_debug_video
from reid_embedding import EmbeddingOutput, extract_track_embeddings
from timing_utils import TimerRegistry, format_duration
from utils import ensure_dir, write_json
from video_io import VideoInfo, scan_videos
from yoloe_player_detection import YOLOEPlayerDetector


T = TypeVar("T")


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    timers = TimerRegistry()
    output_root = ensure_dir(config.output_dir)
    videos_root = ensure_dir(output_root / "videos")
    global_dir = ensure_dir(output_root / "global")
    ensure_dir(output_root / "players")
    if config.export.export_review_rejected_folders:
        ensure_dir(output_root / "review")
        ensure_dir(output_root / "rejected")

    logs = _base_logs(config)
    errors: list[dict[str, str]] = []
    videos: list[VideoInfo] = []
    all_tracks: list[LocalTrack] = []

    try:
        validate_config(config)
        videos, elapsed = _timed(
            timers,
            "scan_videos",
            lambda: scan_videos(config.input_dir, config.video_extensions, config.max_videos),
        )
        print(f"Loaded {len(videos)} video(s) in {format_duration(elapsed)}")
    except Exception as exc:
        errors.append(_error_payload("setup", exc))
        logs["errors"] = errors
        logs["processing_times"] = timers.summary()
        write_json(output_root / "logs.json", logs)
        print(f"Setup failed: {exc}")
        return 2

    detector = YOLOEPlayerDetector(config.detector)
    prompt_warning_logged = False

    for video in videos:
        print(f"\nProcessing video: {video.filename}")
        video_output_dir = ensure_dir(videos_root / video.video_id)
        tracks: list[LocalTrack] = []
        video_started = time.perf_counter()

        try:
            tracks, elapsed = _timed(
                timers,
                "yoloe_detection_tracking",
                lambda video=video: process_video_tracking(
                    video,
                    detector,
                    config.filtering,
                    config.crops.min_crop_height,
                ),
                video_id=video.video_id,
                filename=video.filename,
            )
            print(f"YOLOE detection/tracking time: {format_duration(elapsed)}")
            if detector.prompt_warning and not prompt_warning_logged:
                errors.append({"stage": "yoloe_prompt", "error": detector.prompt_warning})
                print(f"Prompt warning: {detector.prompt_warning}")
                prompt_warning_logged = True
        except Exception as exc:
            errors.append(_error_payload("yoloe_detection_tracking", exc, video))
            write_local_tracks_json(video, tracks, video_output_dir / "local_tracks.json", error=str(exc))
            print(f"Video failed during YOLOE tracking: {exc}")
            continue

        try:
            _, elapsed = _timed(
                timers,
                "crop_extraction",
                lambda video=video, tracks=tracks: extract_crops_for_video(
                    video,
                    tracks,
                    video_output_dir,
                    output_root,
                    config.crops,
                ),
                video_id=video.video_id,
                filename=video.filename,
            )
            print(f"Crop extraction time: {format_duration(elapsed)}")
        except Exception as exc:
            errors.append(_error_payload("crop_extraction", exc, video))
            print(f"Crop extraction failed for {video.filename}: {exc}")

        if config.export.save_debug_video:
            try:
                _, elapsed = _timed(
                    timers,
                    "debug_video_rendering",
                    lambda video=video, tracks=tracks: render_tracking_debug_video(
                        video,
                        tracks,
                        video_output_dir / "tracking_debug.mp4",
                        config.detector.prompt,
                    ),
                    video_id=video.video_id,
                    filename=video.filename,
                )
                print(f"Debug video rendering time: {format_duration(elapsed)}")
            except Exception as exc:
                errors.append(_error_payload("debug_video_rendering", exc, video))
                print(f"Debug video rendering failed for {video.filename}: {exc}")

        write_local_tracks_json(video, tracks, video_output_dir / "local_tracks.json")
        all_tracks.extend(tracks)
        print(f"Total video time: {format_duration(time.perf_counter() - video_started)}")

    embedding_output: EmbeddingOutput | None = None
    try:
        embedding_output, elapsed = _timed(
            timers,
            "reid_embedding_extraction",
            lambda: extract_track_embeddings(
                all_tracks,
                output_root,
                global_dir,
                config.reid,
                config.detector.device,
            ),
        )
        print(f"\nReID embedding extraction time: {format_duration(elapsed)}")
        errors.extend({"stage": "reid_crop", **error} for error in embedding_output.errors)
    except Exception as exc:
        errors.append(_error_payload("reid_embedding_extraction", exc))
        embedding_output = _empty_embedding_output(global_dir)
        print(f"\nReID embedding extraction failed: {exc}")

    records = embedding_output.records if embedding_output else []
    embeddings = embedding_output.embeddings if embedding_output else np.zeros((0, 0), dtype=np.float32)
    tracks_by_id = {track.local_player_id: track for track in all_tracks}

    player_groups: list[dict[str, Any]] = []
    try:
        cluster_result, elapsed = _timed(
            timers,
            "clustering",
            lambda: cluster_embeddings(
                embeddings,
                config.clustering.cluster_method,
                config.clustering.distance_threshold,
            ),
        )
        print(f"Clustering time: {format_duration(elapsed)}")
        local_track_ids = [record.local_player_id for record in records]
        save_similarity_matrix_csv(global_dir / "similarity_matrix.csv", local_track_ids, cluster_result.similarity_matrix)
        resolved_labels, conflict_resolution = split_conflicting_clusters(
            cluster_result.labels,
            records,
            cluster_result.similarity_matrix,
        )
        if conflict_resolution["split_clusters"]:
            logs["cluster_conflict_resolution"] = conflict_resolution
            print(
                "Same-video conflict resolution: "
                f"split {len(conflict_resolution['split_clusters'])} cluster(s)"
            )
        resolved_labels, identity_resolution = split_identity_inconsistent_clusters(
            resolved_labels,
            records,
            output_root,
            cluster_result.similarity_matrix,
            config.clustering,
        )
        if identity_resolution.get("split_clusters"):
            logs["identity_cluster_resolution"] = identity_resolution
            print(
                "Identity consistency resolution: "
                f"split {len(identity_resolution['split_clusters'])} cluster(s)"
            )
        player_id_by_label = player_ids_for_labels(resolved_labels)
        player_ids_by_index = [player_id_by_label[label] for label in resolved_labels]
        conflicts_by_label = find_cluster_conflicts(resolved_labels, records)
        conflicts = conflicts_by_player_id(conflicts_by_label, player_id_by_label)
        identity_by_player = identity_stats_by_player_id(identity_resolution, player_id_by_label)
        player_groups = export_player_albums(
            output_root,
            records,
            tracks_by_id,
            player_ids_by_index,
            cluster_result.similarity_matrix,
            conflicts,
            config.export,
            config.clustering,
            identity_by_player,
        )
    except Exception as exc:
        errors.append(_error_payload("clustering", exc))
        save_similarity_matrix_csv(global_dir / "similarity_matrix.csv", [], np.zeros((0, 0), dtype=np.float32))
        print(f"Clustering failed: {exc}")

    video_index = build_video_index(videos, player_groups, tracks_by_id)
    write_global_indexes(global_dir, player_groups, video_index)

    if config.export.save_gallery:
        try:
            _, elapsed = _timed(
                timers,
                "gallery_export",
                lambda: export_gallery_html(
                    output_root,
                    player_groups,
                    tracks_by_id,
                    include_review_rejected=config.export.gallery_include_review_rejected,
                ),
            )
            print(f"Gallery export time: {format_duration(elapsed)}")
        except Exception as exc:
            errors.append(_error_payload("gallery_export", exc))
            print(f"Gallery export failed: {exc}")

    logs.update(
        {
            "number_of_videos": len(videos),
            "number_of_local_tracks": len(all_tracks),
            "number_of_global_players": len(player_groups),
            "number_of_accepted_players": sum(
                1 for group in player_groups if group.get("export_status") == "accepted"
            ),
            "number_of_review_players": sum(1 for group in player_groups if group.get("export_status") == "review"),
            "number_of_rejected_players": sum(
                1 for group in player_groups if group.get("export_status") == "rejected"
            ),
            "processing_times": timers.summary(),
            "warnings": detector.runtime_warnings,
            "errors": errors,
        }
    )
    write_json(output_root / "logs.json", logs)
    total_time = sum(event.elapsed_sec for event in timers.events if event.name != "scan_videos")
    print(f"Total folder time: {format_duration(total_time)}")
    print(f"Output: {output_root}")
    return 0


def _base_logs(config: PipelineConfig) -> dict[str, Any]:
    return {
        "input_dir": config.input_dir.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "yoloe_model": config.detector.yoloe_model.as_posix(),
        "prompt": config.detector.prompt,
        "tracker": config.detector.tracker,
        "imgsz": config.detector.imgsz,
        "conf": config.detector.conf,
        "iou": config.detector.iou,
        "device": config.detector.device,
        "half": config.detector.half,
        "reid_model": config.reid.reid_model,
        "reid_model_path": config.reid.reid_model_path.as_posix() if config.reid.reid_model_path else None,
        "reid_distance": config.reid.distance_metric,
        "clustering_method": config.clustering.cluster_method,
        "distance_threshold": config.clustering.distance_threshold,
        "video_id_strategy": "sha1_content",
        "video_extensions": list(config.video_extensions),
        "max_videos": config.max_videos,
        "crop_config": asdict(config.crops),
        "filter_config": asdict(config.filtering),
        "clustering_config": asdict(config.clustering),
        "export_config": asdict(config.export),
    }


def _timed(
    timers: TimerRegistry,
    name: str,
    func: Callable[[], T],
    **metadata: Any,
) -> tuple[T, float]:
    start = time.perf_counter()
    with timers.timed(name, **metadata):
        result = func()
    return result, time.perf_counter() - start


def _error_payload(stage: str, exc: Exception, video: VideoInfo | None = None) -> dict[str, str]:
    payload = {
        "stage": stage,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    if video is not None:
        payload["video_id"] = video.video_id
        payload["filename"] = video.filename
    return payload


def _empty_embedding_output(global_dir: Path) -> EmbeddingOutput:
    from reid_embedding import EmbeddingRecord

    embeddings = np.zeros((0, 0), dtype=np.float32)
    np.save(global_dir / "local_track_embeddings.npy", embeddings)
    write_json(global_dir / "local_track_index.json", [])
    return EmbeddingOutput(embeddings=embeddings, records=[], errors=[], device_used="none")


if __name__ == "__main__":
    raise SystemExit(main())
