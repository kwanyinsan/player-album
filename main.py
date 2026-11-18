from __future__ import annotations

import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from album_pipeline import group_and_export
from embedding_state import track_states
from collection_cache import CollectionCache, write_cache_outputs
from inference_provenance import model_signature
from clustering import save_similarity_matrix_csv
from config import PipelineConfig, parse_config, validate_config
from gallery_export import export_gallery_html
from json_export import build_video_index, write_global_indexes, write_local_tracks_json
from player_tracking import LocalTrack
from reid_embedding import EmbeddingOutput, extract_track_embeddings
from timing_utils import TimerRegistry, format_duration
from utils import ensure_dir, write_json
from saved_run import checksum
from video_io import VideoInfo, scan_videos
from video_processing import analyze_video
from yoloe_player_detection import YOLOEPlayerDetector


T = TypeVar("T")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"inspect", "recluster", "add", "recover", "delete", "sort"}:
        from collection_cli import main as collection_main

        return collection_main(arguments)
    config = parse_config(argv)
    if any((config.output_dir / name).exists() for name in (
        "project.db", ".collection-update.json", ".collection.lock", "global/collection_state.json",
    )):
        print("This output is a saved collection. Use add, or select a new output folder.")
        return 2
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
        if config.reid.reid_model_path:
            logs["inference_signature"] = model_signature(config)
        videos, elapsed = _timed(
            timers,
            "scan_videos",
            lambda: scan_videos(config.input_dir, config.video_extensions, config.max_videos),
        )
        print(f"Loaded {len(videos)} video(s) in {format_duration(elapsed)}")
        video_fingerprints = {video.video_id: checksum(video.path) for video in videos}
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
            with timers.timed("video_analysis", video_id=video.video_id, filename=video.filename):
                tracks, stage_times = analyze_video(
                    video, detector, config, output_root,
                    on_optional_error=lambda stage, exc: errors.append(_error_payload(stage, exc, video)),
                )
                if checksum(video.path) != video_fingerprints[video.video_id]:
                    raise RuntimeError("Input video changed during analysis")
            for stage, elapsed in stage_times.items():
                print(f"{stage}: {format_duration(elapsed)}")
            if detector.prompt_warning and not prompt_warning_logged:
                errors.append({"stage": "yoloe_prompt", "error": detector.prompt_warning})
                print(f"Prompt warning: {detector.prompt_warning}")
                prompt_warning_logged = True
        except Exception as exc:
            errors.append(_error_payload("video_analysis", exc, video))
            write_local_tracks_json(video, tracks, video_output_dir / "local_tracks.json", error=str(exc))
            print(f"Video analysis failed: {exc}")
            continue
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
        grouped, elapsed = _timed(
            timers,
            "clustering",
            lambda: group_and_export(
                output_root, records, embeddings, tracks_by_id, config.clustering, config.export,
                video_metadata={video.video_id: video for video in videos},
            ),
        )
        print(f"Clustering and album export time: {format_duration(elapsed)}")
        player_groups, reports = grouped
        logs["video_sorting"] = reports["video_sorting"]
        conflict_resolution = reports["cluster_conflict_resolution"]
        identity_resolution = reports["identity_cluster_resolution"]
        if conflict_resolution["split_clusters"]:
            logs["cluster_conflict_resolution"] = conflict_resolution
            print(
                "Same-video conflict resolution: "
                f"split {len(conflict_resolution['split_clusters'])} cluster(s)"
            )
        if identity_resolution.get("split_clusters"):
            logs["identity_cluster_resolution"] = identity_resolution
            print(
                "Identity consistency resolution: "
                f"split {len(identity_resolution['split_clusters'])} cluster(s)"
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
    signature_unchanged = (not logs.get("inference_signature")
                           or model_signature(config) == logs["inference_signature"])
    if not signature_unchanged:
        errors.append({"stage": "provenance", "error": "Model files changed during processing"})
        logs.pop("inference_signature", None)
    state = {
        "version": 1, "video_fingerprints": video_fingerprints,
        "inference_signature": logs.get("inference_signature"),
        "legacy_provenance_unverified": not bool(logs.get("inference_signature")),
        "embedding_status": track_states(tracks_by_id, records, embedding_output.track_statuses),
        "next_player_number": max((int(g["player_id"][1:]) for g in player_groups), default=0) + 1,
        "operations": [],
    }
    state = write_cache_outputs(output_root, CollectionCache(videos, tracks_by_id, records, embeddings,
                                {v.video_id for v in videos}), state)
    write_json(global_dir / "collection_state.json", state)
    write_json(output_root / "logs.json", logs)
    total_time = sum(event.elapsed_sec for event in timers.events if event.name != "scan_videos")
    print(f"Total folder time: {format_duration(total_time)}")
    print(f"Output: {output_root}")
    return 2 if errors else 0


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
