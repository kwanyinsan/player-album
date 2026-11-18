"""Commands that work exclusively with previously saved analysis."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from album_pipeline import group_and_export
from collection_cache import write_cache_outputs
from gallery_export import export_gallery_html
from json_export import build_video_index, write_global_indexes
from saved_run import checksum, load_saved_run, read_json, saved_settings, write_registry
from utils import write_json


def main(argv: list[str]) -> int:
    if argv and argv[0] == "sort":
        from collection_sort import main as sort_main
        return sort_main(argv)
    if argv and argv[0] == "delete":
        from collection_delete import main as delete_main
        return delete_main(argv)
    if argv and argv[0] in {"add", "recover"}:
        from collection_add import main as add_main
        return add_main(argv)
    parser = argparse.ArgumentParser(description="Inspect or recluster saved analysis without model inference.")
    parser.add_argument("command", choices=("inspect", "recluster"))
    parser.add_argument("--run_dir", type=Path, required=True, help="Existing saved run.")
    parser.add_argument("--output_dir", type=Path, help="New destination; must not already exist.")
    parser.add_argument("--distance_threshold", type=float, help="Override saved clustering threshold.")
    args = parser.parse_args(argv)
    if args.command == "inspect" and (args.output_dir is not None or args.distance_threshold is not None):
        parser.error("inspect is read-only and does not accept output or clustering overrides")
    if args.command == "recluster" and args.output_dir is None:
        parser.error("recluster requires --output_dir (a new folder)")
    if args.distance_threshold is not None and not 0 <= args.distance_threshold <= 2:
        parser.error("distance_threshold must be in [0, 2]")
    started = time.perf_counter()
    staging = None
    try:
        source = args.run_dir.resolve()
        if args.output_dir is not None:
            destination = args.output_dir.resolve()
            if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError("Output must be a new folder outside the source run")
        print(f"Validating saved analysis: {source}", flush=True)
        run = load_saved_run(source, progress=lambda message: print(message, flush=True))
        print(json.dumps(run.report(), indent=2), flush=True)
        if args.command == "inspect":
            return 0
        clustering, export = saved_settings(run)
        if args.distance_threshold is not None:
            clustering = replace(clustering, distance_threshold=args.distance_threshold)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-pending-", dir=destination.parent))
        print("Copying validated analysis to a separate collection (debug videos are not duplicated).", flush=True)
        for relative, digest in run.artifacts.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
            if checksum(target) != digest:
                raise ValueError(f"Source artifact changed during import: {relative}")
        # Canonicalize the index order without changing embedding row associations.
        write_json(staging / "global/local_track_index.json", [r.to_dict() for r in run.records])
        print("Clustering saved embeddings and exporting albums...", flush=True)
        state_path = source / "global/collection_state.json"
        state = read_json(state_path) if state_path.exists() else {}
        groups_path = source / "global/player_groups.json"
        previous_groups = read_json(groups_path) if groups_path.exists() else None
        groups, reports = group_and_export(staging, run.records, run.embeddings, run.tracks, clustering, export,
                                          previous_groups, state.get("next_player_number", 1),
                                          video_metadata={v.video_id: v for v in run.videos})
        continuity = reports["player_identity_continuity"]
        state["next_player_number"] = continuity.get("next_player_number", state.get("next_player_number", len(groups) + 1))
        state.setdefault("operations", []).append({"operation": "recluster_saved", "lineage": continuity.get("lineage", [])})
        state = write_cache_outputs(staging, run.complete_cache, state)
        write_json(staging / "global/collection_state.json", state)
        write_global_indexes(staging / "global", groups, build_video_index(run.videos, groups, run.tracks))
        if export.save_gallery:
            export_gallery_html(staging, groups, run.tracks, export.gallery_include_review_rejected)
        logs = {
            **run.logs, **reports,
            "operation": "recluster_saved", "source_run": str(source), "output_dir": str(destination),
            "clustering_config": asdict(clustering), "export_config": asdict(export),
            "clustering_method": clustering.cluster_method, "distance_threshold": clustering.distance_threshold,
            "number_of_global_players": len(groups),
            "number_of_registered_videos": len(run.complete_cache.videos),
            "number_of_excluded_videos": len(run.complete_cache.videos) - len(run.videos),
            "number_of_accepted_players": sum(g["export_status"] == "accepted" for g in groups),
            "number_of_review_players": sum(g["export_status"] == "review" for g in groups),
            "number_of_rejected_players": sum(g["export_status"] == "rejected" for g in groups),
            "source_errors": run.logs.get("errors", []), "errors": [], "warnings": run.warnings,
            "inference_executed": False, "inference_cache_provenance": run.report()["inference_cache_provenance"],
            "processing_times": {"recluster_saved_sec": time.perf_counter() - started},
        }
        write_json(staging / "logs.json", logs)
        write_json(staging / "global/reuse_report.json", run.report())
        write_registry(staging, load_saved_run(staging))
        load_saved_run(staging)
        # Verify source immutability before publishing a fully built destination.
        for relative, digest in run.artifacts.items():
            if checksum(source / relative) != digest:
                raise ValueError(f"Source artifact changed during reclustering: {relative}")
        staging.rename(destination)
        print(f"Saved {len(groups)} groups ({logs['number_of_accepted_players']} accepted) to {destination}")
        print(f"Total: {time.perf_counter() - started:.1f}s; no video inference executed.")
        return 0
    except Exception as exc:
        print(f"Saved-run operation failed: {exc}")
        if staging is not None and staging.exists():
            print(f"Unpublished diagnostic files retained at: {staging}")
        return 2
