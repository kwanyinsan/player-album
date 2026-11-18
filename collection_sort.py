"""Refresh player video rankings and galleries without changing identity decisions."""
from __future__ import annotations

import argparse
import re
import shutil
import time
import uuid
from pathlib import Path

from artifact_paths import artifact_path
from collection_transaction import collection_lock, publish, prune_backups
from gallery_export import export_gallery_html, export_standalone_gallery
from player_video_sorting import ranking_fields, SORTING_VERSION
from saved_run import SavedRun, checksum, load_saved_run, read_json, saved_settings, write_registry
from utils import write_json


def validate_groups(groups: list[dict], run: SavedRun) -> None:
    assigned, players = set(), set()
    for group in groups:
        player_id = group["player_id"]
        ids = group["local_tracks"]
        if (not re.fullmatch(r"P[0-9]+", player_id) or player_id in players
                or len(ids) != len(set(ids)) or assigned & set(ids)):
            raise ValueError(f"Invalid player ID or duplicate track assignment: {player_id}")
        players.add(player_id)
        assigned.update(ids)
    if assigned != {r.local_player_id for r in run.records}:
        raise ValueError("Player groups must contain each active embedded track exactly once")


def copy_exports(root: Path, stage: Path) -> dict[str, str]:
    digests = {}
    for name in ("global", "players", "review", "rejected"):
        directory = root / name
        if not directory.exists():
            continue
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Managed export directory escapes collection")
        (stage / name).mkdir(parents=True, exist_ok=True)
        for path in directory.rglob("*"):
            if path.is_symlink() and path.is_dir():
                raise ValueError("Managed export directories must not contain directory links")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            source = artifact_path(root, relative)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = checksum(source)
            shutil.copy2(source, target)
            if checksum(target) != digest:
                raise ValueError(f"Export changed during copy: {relative}")
            digests[relative] = digest
    (stage / "players").mkdir(parents=True, exist_ok=True)
    return digests


def execute_sort(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    root = args.run_dir.resolve()
    run = load_saved_run(root, progress=print)
    groups = read_json(root / "global/player_groups.json")
    groups_digest = checksum(root / "global/player_groups.json")
    validate_groups(groups, run)
    videos = {v.video_id: v for v in run.videos}
    ranked_groups = [{**group, **ranking_fields(group["local_tracks"], run.tracks, videos)} for group in groups]
    unavailable = [{"player_id": g["player_id"], "video_id": r["video_id"], "reason": r["coverage_status"]}
                   for g in ranked_groups for r in g["video_rankings"] if r["coverage_status"] != "valid"]
    total = sum(len(g["video_rankings"]) for g in ranked_groups)
    print(f"Player groups: {len(groups)}; ranked video entries: {total}; unavailable coverage: {len(unavailable)}", flush=True)
    if args.dry_run:
        print("Dry run; exports unchanged.")
        return 0
    operation = root / ".updates" / uuid.uuid4().hex
    if not operation.resolve().is_relative_to(root):
        raise ValueError("Update workspace escapes collection")
    stage = operation / "stage"
    stage.mkdir(parents=True)
    print(f"Update workspace: {operation}", flush=True)
    copied = copy_exports(root, stage)
    if copied.get("global/player_groups.json") != groups_digest:
        raise ValueError("Player groups changed during Sort")
    for group in ranked_groups:
        player_id = group["player_id"]
        expected_album = stage / "players" / player_id / "metadata.json"
        if group.get("export_status") == "accepted" and not expected_album.is_file():
            raise ValueError(f"Accepted album metadata missing: {player_id}")
        for category in ("players", "review", "rejected"):
            metadata_path = stage / category / player_id / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            if (metadata.get("player_id") != player_id or metadata.get("local_tracks") != group["local_tracks"]
                    or metadata.get("export_status") != group.get("export_status")):
                raise ValueError(f"Album metadata disagrees with global groups: {player_id}")
            fields = {key: group[key] for key in ("video_rankings", "videos", "video_ids")}
            write_json(metadata_path, {**metadata, **fields})
    write_json(stage / "global/player_groups.json", ranked_groups)
    _, export = saved_settings(run)
    if export.save_gallery:
        export_gallery_html(stage, ranked_groups, run.tracks, export.gallery_include_review_rejected)
        export_standalone_gallery(stage, source_root=root)
    state_path = stage / "global/collection_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    state["video_sorting_version"] = SORTING_VERSION
    state.setdefault("operations", []).append({"id": operation.name, "operation": "sort", "recorded_at": time.time()})
    write_json(state_path, state)
    report = {"version": SORTING_VERSION, "ranked_video_entries": total, "unavailable_coverage": unavailable}
    logs = {**run.logs, "operation": "sort", "source_errors": run.logs.get("errors", []), "errors": [],
            "inference_executed": False, "inference_video_ids": [], "retried_tracks": [],
            "video_sorting": report, "processing_times": {"sort_before_publication_sec": time.perf_counter() - started}}
    write_json(stage / "logs.json", logs)
    validated = load_saved_run(stage, source_root=root)
    write_registry(stage, validated, source_root=root)
    load_saved_run(stage, source_root=root)
    for relative, digest in {**run.artifacts, **copied}.items():
        if checksum(root / relative) != digest:
            raise ValueError(f"Collection changed during Sort: {relative}")
    publish(root, operation, [])
    try:
        prune_backups(root, args.keep_backups)
    except OSError as exc:
        print(f"Sort committed, but backup cleanup failed: {exc}", flush=True)
    print(f"Sorted {total} video entries across {len(groups)} players in {time.perf_counter() - started:.1f}s; "
          "no inference or reclustering.", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Rank each player's videos by tracked frame coverage.")
    parser.add_argument("command", choices=("sort",))
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_backups", type=int, default=1)
    args = parser.parse_args(argv)
    if args.keep_backups < 1:
        parser.error("keep_backups must be at least 1")
    try:
        if not args.run_dir.is_dir():
            raise ValueError("Existing collection folder not found")
        with collection_lock(args.run_dir):
            return execute_sort(args)
    except Exception as exc:
        print(f"Sort failed: {exc}", flush=True)
        return 2
