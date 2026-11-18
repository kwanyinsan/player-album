"""Exclude videos from clustering while preserving their complete analysis cache."""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from album_pipeline import group_and_export
from collection_cache import CollectionCache, write_cache_outputs
from collection_transaction import collection_lock, publish, prune_backups
from gallery_export import export_gallery_html, export_standalone_gallery
from json_export import build_video_index, write_global_indexes
from saved_run import checksum, load_saved_run, read_json, saved_settings, write_registry
from utils import write_json


def execute_delete(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root = args.run_dir.resolve()
    run = load_saved_run(root, progress=print)
    cache = run.complete_cache
    requested = set(args.video_ids)
    known = {v.video_id for v in cache.videos}
    if requested - known:
        raise ValueError(f"Unknown video IDs: {sorted(requested - known)}")
    selected = requested & cache.active_video_ids
    affected_tracks = {key for key, t in cache.tracks.items() if t.video_id in selected}
    groups_path = root / "global/player_groups.json"
    previous_groups = read_json(groups_path)
    group_digest = checksum(groups_path)
    affected_players = [g["player_id"] for g in previous_groups if set(g["local_tracks"]) & affected_tracks]
    print(f"Requested: {len(requested)}; newly excluded: {len(selected)}; "
          f"affected tracks: {len(affected_tracks)}; affected albums: {affected_players}", flush=True)
    for video in cache.videos:
        if video.video_id in selected:
            print(f"  Exclude {video.video_id}: {video.filename}", flush=True)
    if args.dry_run or not selected:
        print("Dry run; collection unchanged." if args.dry_run else "Videos already excluded; nothing to update.")
        return 0
    operation = root / ".updates" / uuid.uuid4().hex
    if not operation.resolve().is_relative_to(root):
        raise ValueError("Update workspace escapes collection")
    stage = operation / "stage"
    (stage / "global").mkdir(parents=True)
    print(f"Update workspace: {operation}", flush=True)
    state_path = root / "global/collection_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    cache = CollectionCache(cache.videos, cache.tracks, cache.records, cache.embeddings,
                            cache.active_video_ids - selected)
    active = cache.active_view()
    clustering, export = saved_settings(run)
    timings = {"validation_selection_sec": time.perf_counter() - started}
    cluster_started = time.perf_counter()
    groups, reports = group_and_export(stage, active.records, active.embeddings, active.tracks, clustering, export,
                                      previous_groups, state.get("next_player_number", 1), source_root=root,
                                      video_metadata={v.video_id: v for v in active.videos})
    timings["clustering_export_sec"] = time.perf_counter() - cluster_started
    write_global_indexes(stage / "global", groups, build_video_index(active.videos, groups, active.tracks))
    if export.save_gallery:
        export_gallery_html(stage, groups, active.tracks, export.gallery_include_review_rejected)
        export_standalone_gallery(stage, source_root=root)
    retired = sorted({g["player_id"] for g in previous_groups} - {g["player_id"] for g in groups})
    state.update(next_player_number=reports["player_identity_continuity"]["next_player_number"])
    state.setdefault("operations", []).append({"id": operation.name, "operation": "delete",
        "requested_video_ids": sorted(requested), "excluded_video_ids": sorted(selected),
        "retired_player_ids": retired, "lineage": reports["player_identity_continuity"]["lineage"],
        "recorded_at": time.time()})
    state = write_cache_outputs(stage, cache, state)
    write_json(stage / "global/collection_state.json", state)
    timings["before_publication_sec"] = time.perf_counter() - started
    logs = {**run.logs, **reports, "operation": "delete", "output_dir": str(root),
            "number_of_videos": len(active.videos), "number_of_local_tracks": len(active.tracks),
            "number_of_registered_videos": len(cache.videos),
            "number_of_excluded_videos": len(cache.videos) - len(active.videos),
            "number_of_cached_embeddings": len(cache.records), "number_of_active_embeddings": len(active.records),
            "number_of_global_players": len(groups), "requested_video_ids": sorted(requested),
            "excluded_video_ids": sorted(selected), "retired_player_ids": retired,
            "active_videos_before": len(run.videos), "active_videos_after": len(active.videos),
            "inference_executed": False, "inference_video_ids": [], "retried_tracks": [],
            "unfinished_embedding_tracks": [key for key, entry in state["embedding_status"].items()
                if key in active.tracks and entry["status"] in {"pending", "retryable_error", "blocked_error"}],
            "duplicate_inputs_skipped": 0, "excluded_inputs_skipped": 0,
            "source_errors": run.logs.get("errors", []), "errors": [], "warnings": run.warnings,
            "processing_times": timings}
    for status in ("accepted", "review", "rejected"):
        logs[f"number_of_{status}_players"] = sum(g["export_status"] == status for g in groups)
    write_json(stage / "logs.json", logs)
    validated = load_saved_run(stage, source_root=root)
    write_registry(stage, validated, source_root=root)
    load_saved_run(stage, source_root=root)
    for relative, digest in run.artifacts.items():
        if checksum(root / relative) != digest:
            raise ValueError(f"Existing artifact changed during Delete: {relative}")
    if checksum(groups_path) != group_digest:
        raise ValueError("Player groups changed during Delete")
    publish(root, operation, [])
    try:
        prune_backups(root, args.keep_backups)
    except OSError as exc:
        print(f"Delete committed, but backup cleanup failed: {exc}", flush=True)
    print(f"Excluded {len(selected)} videos; {len(active.videos)} active videos; "
          f"{len(active.records)} active embeddings; {len(cache.records)} cached embeddings. "
          f"Total: {time.perf_counter() - started:.1f}s; no model inference.", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Exclude videos and rebuild albums using saved embeddings.")
    parser.add_argument("command", choices=("delete",))
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--video_ids", nargs="+", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_backups", type=int, default=1)
    args = parser.parse_args(argv)
    if args.keep_backups < 1:
        parser.error("keep_backups must be at least 1")
    try:
        if not args.run_dir.is_dir():
            raise ValueError("Existing collection folder not found")
        with collection_lock(args.run_dir):
            return execute_delete(args)
    except Exception as exc:
        print(f"Delete failed: {exc}", flush=True)
        return 2
