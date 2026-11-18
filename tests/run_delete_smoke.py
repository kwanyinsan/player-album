"""Test Delete against a separate copy of a real saved collection, without inference."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from album_pipeline import group_and_export
from collection_cli import main as collection_main
from json_export import build_video_index, write_global_indexes
from main import main as pipeline_main
from saved_run import checksum, load_saved_run, read_json, saved_settings
from utils import write_json


def file_snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): checksum(p) for p in root.rglob("*") if p.is_file()}


def memberships(groups: list[dict]) -> list[tuple]:
    return sorted((tuple(sorted(g["local_tracks"])), g["export_status"]) for g in groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--delete_videos", type=int, default=3)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError("Test destination must not exist")
    source = load_saved_run(args.source_run.resolve(), progress=print)
    embedded_videos = {r.video_id for r in source.records}
    selected = [v.video_id for v in source.videos if v.video_id in embedded_videos][:args.delete_videos]
    if args.delete_videos < 1 or len(selected) != args.delete_videos or len(selected) >= len(source.videos):
        raise ValueError("Choose fewer positive embedded videos than the active collection size")
    root = destination / "collection"
    shutil.copytree(source.root, root, ignore=shutil.ignore_patterns(".updates", ".collection.lock"))
    before = load_saved_run(root)
    old_cache = before.complete_cache
    video_digests = file_snapshot(root / "videos")
    original_digests = {str(v.path): checksum(v.path) for v in old_cache.videos if v.path.is_file()}
    model_digests = {str(p): checksum(Path(p)) for p in (source.logs.get("yoloe_model"), source.logs.get("reid_model_path"))
                     if p and Path(p).is_file()}
    command = ["delete", "--run_dir", str(root), "--video_ids", *selected]
    started = time.perf_counter()
    with patch("reid_embedding._load_extractor", side_effect=AssertionError("ReID inference called")), \
         patch("yoloe_player_detection.YOLOEPlayerDetector.load", side_effect=AssertionError("YOLOE inference called")):
        assert pipeline_main(command) == 0
        delete_seconds = time.perf_counter() - started
        updated = load_saved_run(root)
        assert len(updated.videos) == len(before.videos) - len(selected)
        assert not set(selected) & updated.complete_cache.active_video_ids
        np.testing.assert_array_equal(updated.complete_cache.embeddings, old_cache.embeddings)
        assert updated.complete_cache.records == old_cache.records
        rows = [r.embedding_row for r in old_cache.records if r.video_id not in selected]
        np.testing.assert_array_equal(updated.embeddings, old_cache.embeddings[rows])
        assert video_digests == file_snapshot(root / "videos")
        snapshot = file_snapshot(root)
        assert pipeline_main(command) == 0
        assert snapshot == file_snapshot(root)
        inputs = destination / "inputs"
        inputs.mkdir()
        excluded_source = next(v.path for v in old_cache.videos if v.video_id == selected[0])
        shutil.copy2(excluded_source, inputs / "renamed_excluded_video.mp4")
        assert pipeline_main(["add", "--run_dir", str(root), "--input_dir", str(inputs)]) == 0
        assert snapshot == file_snapshot(root)
        reference = destination / "recluster_reference"
        assert collection_main(["recluster", "--run_dir", str(root), "--output_dir", str(reference)]) == 0
        reloaded = load_saved_run(reference)
        assert reloaded.complete_cache.active_video_ids == updated.complete_cache.active_video_ids
        np.testing.assert_array_equal(reloaded.complete_cache.embeddings, old_cache.embeddings)
        actual_groups = read_json(root / "global/player_groups.json")
        assert memberships(actual_groups) == memberships(read_json(reference / "global/player_groups.json"))
        direct = destination / "filtered_reference"
        (direct / "global").mkdir(parents=True)
        records = [replace(r, embedding_row=i) for i, r in enumerate(old_cache.records[j] for j in rows)]
        tracks = {key: t for key, t in old_cache.tracks.items() if t.video_id not in selected}
        videos = [v for v in before.videos if v.video_id not in selected]
        clustering, export = saved_settings(before)
        expected_groups, _ = group_and_export(direct, records, old_cache.embeddings[rows], tracks,
                                              clustering, export, source_root=root,
                                              video_metadata={v.video_id: v for v in videos})
        write_global_indexes(direct / "global", expected_groups, build_video_index(videos, expected_groups, tracks))
        assert memberships(actual_groups) == memberships(expected_groups)
    assert original_digests == {p: checksum(Path(p)) for p in original_digests}
    assert model_digests == {p: checksum(Path(p)) for p in model_digests}
    assert source.artifacts == load_saved_run(source.root).artifacts
    with closing(sqlite3.connect((root / "project.db").as_uri() + "?mode=ro", uri=True)) as db:
        registry_version = db.execute("SELECT schema_version FROM project").fetchone()[0]
        active_rows, cache_rows = db.execute("SELECT count(embedding_row),count(cache_row) FROM tracks").fetchone()
    report = {"active_videos_before": len(before.videos), "active_videos_after": len(updated.videos),
              "registered_videos": len(updated.complete_cache.videos), "excluded_video_ids": selected,
              "cached_embeddings": len(old_cache.records), "active_embeddings": len(updated.records),
              "registry_schema_version": registry_version, "active_registry_rows": active_rows,
              "cached_registry_rows": cache_rows, "delete_seconds": delete_seconds,
              "global_groups": len(actual_groups),
              "accepted_players": updated.logs["number_of_accepted_players"],
              "video_analysis_unchanged": True, "cached_embedding_values_and_rows_unchanged": True,
              "original_videos_and_models_unchanged": True, "source_collection_unchanged": True,
              "repeated_delete_noop": True, "excluded_add_noop": True,
              "recluster_preserves_exclusions": True, "filtered_reference_matches": True,
              "model_inference_executed": False}
    write_json(destination / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
