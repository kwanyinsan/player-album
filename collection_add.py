"""Add new video analysis to a collection, then rebuild its global albums."""
from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from album_pipeline import group_and_export
from collection_transaction import collection_lock, publish, recover, prune_backups
from collection_cache import CollectionCache, cache_statuses, write_cache_outputs
from inference_provenance import model_signature
from config import (CropConfig, DetectorConfig, FilterConfig, PipelineConfig,
                    ReidConfig, parse_bool, validate_config)
from gallery_export import export_gallery_html, export_standalone_gallery
from json_export import build_video_index, write_global_indexes
from reid_embedding import extract_track_embeddings
from saved_run import SavedRun, checksum, construct, load_saved_run, read_json, saved_settings, write_registry
from utils import file_sha1, stable_video_id, write_json
from video_io import read_video_metadata
from video_processing import analyze_video
from yoloe_player_detection import YOLOEPlayerDetector


def inference_config(run: SavedRun, input_dir: Path, args: argparse.Namespace) -> PipelineConfig:
    logs = run.logs
    required = ("yoloe_model", "reid_model", "reid_model_path", "crop_config", "filter_config")
    missing = [key for key in required if key not in logs]
    if missing:
        raise ValueError(f"Missing saved inference settings: {missing}; cannot safely infer them")
    detector = {f.name: logs[f.name] for f in fields(DetectorConfig) if f.name in logs}
    detector["yoloe_model"] = Path(args.yoloe_model or logs["yoloe_model"])
    if args.device is not None:
        detector["device"] = args.device
    checkpoint = args.reid_model_path or logs["reid_model_path"]
    if not checkpoint:
        raise ValueError("Add requires an explicit ReID checkpoint, not automatic pretrained downloads")
    clustering, export = saved_settings(run)
    # Reclustering disables debug rendering in its export settings, not in new analysis.
    export = replace(export, save_debug_video=args.save_debug_video)
    return PipelineConfig(input_dir, run.root, tuple(logs.get("video_extensions", (".mp4", ".mov", ".avi", ".mkv"))),
                          args.max_videos, construct(DetectorConfig, detector),
                          construct(CropConfig, logs["crop_config"]), construct(FilterConfig, logs["filter_config"]),
                          ReidConfig(logs["reid_model"], Path(checkpoint), logs.get("reid_distance", "cosine")),
                          clustering, export)


def discover(run: SavedRun, input_dir: Path, extensions: tuple[str, ...], state: dict[str, Any],
             limit: int | None = None) -> tuple[list[tuple[Path, str, str]], list[str], dict[str, str], list[dict], list[str]]:
    fingerprints = dict(state.get("video_fingerprints", {}))
    cache = run.complete_cache
    for video in cache.videos:
        if video.video_id in fingerprints:
            continue
        if not video.path.is_file():
            raise ValueError(f"Original video unavailable for fingerprint migration: {video.path}. "
                             "Restore its recorded path before the first Add.")
        # Legacy IDs contain a SHA1 prefix. Do not attach a changed file to old analysis.
        if run.logs.get("video_id_strategy") == "sha1_content":
            prefix = video.video_id[2:].split("_")[0].lower()
            if not file_sha1(video.path).startswith(prefix):
                raise ValueError(f"Original video content changed: {video.path}")
        fingerprints[video.video_id] = checksum(video.path)
    known = set(fingerprints.values())
    used = {v.video_id for v in cache.videos}
    used.update(p.name for p in (run.root / "videos").iterdir() if p.is_dir())
    new, duplicates, errors = [], [], []
    excluded = []
    excluded_digests = {digest for video_id, digest in fingerprints.items() if video_id not in cache.active_video_ids}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            digest = checksum(path)
            if digest in known:
                duplicates.append(str(path))
                if digest in excluded_digests:
                    excluded.append(str(path))
                    print(f"Excluded video remains inactive: {path}", flush=True)
                continue
            if limit is not None and len(new) >= limit:
                continue
            new.append((path, stable_video_id(path, used), digest))
            known.add(digest)
        except OSError as exc:
            errors.append({"stage": "scan", "path": str(path), "error": str(exc)})
    return new, duplicates, fingerprints, errors, excluded


def execute_add(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root, input_dir = args.run_dir.resolve(), args.input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input folder not found: {input_dir}")
    if input_dir.is_relative_to(root) or root.is_relative_to(input_dir):
        raise ValueError("Input and collection folders must not overlap")
    run = load_saved_run(root, progress=print)
    state_path = root / "global/collection_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    config = inference_config(run, input_dir, args)
    cache = run.complete_cache
    statuses = cache_statuses(cache, state.get("embedding_status"))
    retry_tracks = [run.tracks[key] for key, entry in statuses.items()
                    if key in run.tracks and entry["status"] in {"pending", "retryable_error"}]
    blocked = [key for key, entry in statuses.items() if key in run.tracks and entry["status"] == "blocked_error"]
    new, duplicates, fingerprints, errors, excluded = discover(run, input_dir, config.video_extensions, state, args.max_videos)
    print(f"Existing: {len(run.videos)}; new: {len(new)}; duplicates skipped: {len(duplicates)}; "
          f"tracks to retry: {len(retry_tracks)}", flush=True)
    for path, _, _ in new:
        print(f"  New: {path}", flush=True)
    if args.dry_run or (not new and not retry_tracks):
        print("Dry run; no inference or album changes." if args.dry_run else "Nothing to add; no inference or album changes.")
        for error in errors:
            print(error)
        if blocked:
            print(f"Blocked tracks require their recorded error to be resolved: {blocked}")
        return 2 if errors or blocked else 0
    validate_config(config)
    signature = model_signature(config)
    previous_signature = state.get("inference_signature") or run.logs.get("inference_signature")
    if previous_signature and previous_signature != signature:
        raise ValueError("Inference checkpoint or embedding recipe differs from the collection; rebuild separately")
    if not previous_signature:
        # A different path is not evidence that a legacy checkpoint is equivalent.
        for override, saved in ((args.yoloe_model, run.logs["yoloe_model"]),
                                (args.reid_model_path, run.logs["reid_model_path"])):
            if override and Path(override).resolve() != Path(saved).resolve():
                raise ValueError("Cannot verify a relocated checkpoint for a legacy collection")
        print("Warning: original checkpoint provenance is unverified; retaining that warning.", flush=True)
    operation = root / ".updates" / uuid.uuid4().hex
    if not operation.resolve().is_relative_to(root):
        raise ValueError("Update workspace must stay inside the collection")
    stage = operation / "stage"
    (stage / "global").mkdir(parents=True)
    print(f"Update workspace: {operation}", flush=True)
    group_digest = checksum(root / "global/player_groups.json")
    detector = None
    if new:
        detector = YOLOEPlayerDetector(config.detector)
        detector.load()
    videos, tracks = list(cache.videos), dict(cache.tracks)
    new_tracks, added, timings = [], [], {}
    for path, video_id, digest in new:
        try:
            print(f"Processing new video: {path.name}", flush=True)
            video = read_video_metadata(path, video_id)
            local, elapsed = analyze_video(video, detector, config, stage)
            if checksum(path) != digest:
                raise ValueError("Input video changed during analysis")
            videos.append(video)
            tracks.update({t.local_player_id: t for t in local})
            new_tracks.extend(local)
            added.append(video_id)
            fingerprints[video_id] = digest
            timings[video_id] = elapsed
            print(f"Video timings: {elapsed}", flush=True)
        except Exception as exc:
            errors.append({"stage": "video", "path": str(path), "error": str(exc)})
            print(f"Video failed, continuing: {exc}", flush=True)
    if not added and not retry_tracks:
        write_json(operation / "failure_report.json", {"errors": errors})
        print("No successful additions; published collection unchanged.")
        return 2
    if detector:
        detector.reset_model()
    reid_started = time.perf_counter()
    extraction_tracks = retry_tracks + new_tracks
    extracted = extract_track_embeddings(extraction_tracks, stage, operation / "new_embeddings", config.reid,
                                         config.detector.device, source_root=root,
                                         expected_dimension=cache.embeddings.shape[1] if len(cache.embeddings) else None)
    errors.extend({"stage": "reid_crop", **e} for e in extracted.errors)
    timings["reid_sec"] = time.perf_counter() - reid_started
    records = list(cache.records) + [replace(r, embedding_row=len(cache.records) + i) for i, r in enumerate(extracted.records)]
    for track in extraction_tracks:
        key = track.local_player_id
        previous_attempts = statuses.get(key, {}).get("attempt_count", 0)
        entry = extracted.track_statuses.get(key, {})
        statuses[key] = {**entry, "attempt_count": previous_attempts + int(bool(track.crop_paths)),
                         "model_signature": signature}
        if "status" not in entry:
            statuses[key]["status"] = "retryable_error" if track.crop_paths else "no_usable_crops"
    arrays = [a for a in (cache.embeddings, extracted.embeddings) if len(a)]
    if len(arrays) == 2 and arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError("New embedding dimension differs from existing embeddings")
    embeddings = np.concatenate(arrays) if arrays else np.zeros((0, 0), dtype=np.float32)
    cache = CollectionCache(videos, tracks, records, embeddings, cache.active_video_ids | set(added))
    active = cache.active_view()
    statuses = cache_statuses(cache, statuses)
    unfinished = [key for key, entry in statuses.items() if key in active.tracks
                  and entry["status"] in {"pending", "retryable_error", "blocked_error"}]
    if unfinished and not extracted.errors:
        errors.append({"stage": "reid_track", "error": f"{len(unfinished)} track(s) have no valid embedding"})
    old_groups = read_json(root / "global/player_groups.json")
    cluster_started = time.perf_counter()
    groups, reports = group_and_export(stage, active.records, active.embeddings, active.tracks, config.clustering, config.export,
                                      old_groups, state.get("next_player_number", 1), source_root=root,
                                      video_metadata={v.video_id: v for v in active.videos})
    timings["clustering_export_sec"] = time.perf_counter() - cluster_started
    write_global_indexes(stage / "global", groups, build_video_index(active.videos, groups, active.tracks))
    if config.export.save_gallery:
        export_gallery_html(stage, groups, active.tracks, config.export.gallery_include_review_rejected)
        export_standalone_gallery(stage, source_root=root)
    from main import _base_logs
    state = {**state, "version": 1, "video_fingerprints": fingerprints,
             "inference_signature": signature, "embedding_status": statuses,
             "legacy_provenance_unverified": state.get("legacy_provenance_unverified", not bool(run.logs.get("inference_signature"))),
             "next_player_number": reports["player_identity_continuity"]["next_player_number"],
             "operations": [*state.get("operations", []), {"id": operation.name, "added_video_ids": added,
                 "retried_tracks": [t.local_player_id for t in retry_tracks],
                 "lineage": reports["player_identity_continuity"]["lineage"]}]}
    state = write_cache_outputs(stage, cache, state)
    write_json(stage / "global/collection_state.json", state)
    timings["before_publication_sec"] = time.perf_counter() - started
    logs = {**run.logs, **_base_logs(config), **reports, "operation": "add",
            "source_errors": run.logs.get("errors", []), "errors": errors,
            "warnings": [*run.warnings, *(detector.runtime_warnings if detector else [])],
            "number_of_videos": len(active.videos), "number_of_local_tracks": len(active.tracks),
            "number_of_registered_videos": len(cache.videos), "number_of_excluded_videos": len(cache.videos) - len(active.videos),
            "number_of_cached_embeddings": len(cache.records), "number_of_active_embeddings": len(active.records),
            "number_of_global_players": len(groups), "processing_times": timings,
            "inference_executed": True, "inference_video_ids": added,
            "retried_tracks": [t.local_player_id for t in retry_tracks], "unfinished_embedding_tracks": unfinished,
            "duplicate_inputs_skipped": len(duplicates), "excluded_inputs_skipped": len(excluded),
            "existing_videos_reused": len(run.videos)}
    for status in ("accepted", "review", "rejected"):
        logs[f"number_of_{status}_players"] = sum(g["export_status"] == status for g in groups)
    write_json(stage / "logs.json", logs)
    validated = load_saved_run(stage, source_root=root)
    write_registry(stage, validated, source_root=root)
    load_saved_run(stage, source_root=root)
    for relative, digest in run.artifacts.items():
        if checksum(root / relative) != digest:
            raise ValueError(f"Existing analysis changed during update: {relative}")
    if checksum(root / "global/player_groups.json") != group_digest:
        raise ValueError("Existing player groups changed during update")
    if model_signature(config) != signature:
        raise ValueError("Model weights changed during update")
    publish(root, operation, added)
    try:
        prune_backups(root, args.keep_backups)
    except OSError as exc:
        print(f"Update committed, but backup cleanup failed: {exc}", flush=True)
    print(f"Added {len(added)} videos; {len(active.records)} active embedded tracks; "
          f"{logs['number_of_accepted_players']} accepted players. Total: {time.perf_counter() - started:.1f}s", flush=True)
    return 2 if errors else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Add new videos or recover an interrupted collection update.")
    parser.add_argument("command", choices=("add", "recover"))
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_videos", type=int, help="Maximum new videos, after duplicate filtering")
    parser.add_argument("--device")
    parser.add_argument("--yoloe_model", type=Path, help="Relocate a fingerprinted checkpoint without changing it")
    parser.add_argument("--reid_model_path", type=Path)
    parser.add_argument("--save_debug_video", type=parse_bool, default=True)
    parser.add_argument("--keep_backups", type=int, default=1, help="Previous committed export backups to retain")
    args = parser.parse_args(argv)
    if args.command == "add" and args.input_dir is None:
        parser.error("add requires --input_dir")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("max_videos must be positive")
    if args.keep_backups < 1:
        parser.error("keep_backups must be at least 1")
    try:
        if not args.run_dir.is_dir():
            raise ValueError("Existing collection folder not found")
        with collection_lock(args.run_dir):
            if args.command == "recover":
                print("Recovered update." if recover(args.run_dir) else "No interrupted publication.")
                return 0
            if (args.run_dir / ".collection-update.json").exists():
                raise ValueError("Run the recover command before adding more videos")
            return execute_add(args)
    except Exception as exc:
        print(f"Collection operation failed: {exc}", flush=True)
        return 2
