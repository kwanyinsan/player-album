"""Verify real-result sorting against raw frame sets, without inference or reclustering."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main as pipeline_main
from saved_run import checksum, load_saved_run, read_json, saved_settings
from utils import write_json


def snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): checksum(p) for p in root.rglob("*") if p.is_file()}


def protected_files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): checksum(p) for p in root.rglob("*") if p.is_file()
            and ".updates" not in p.relative_to(root).parts
            and (p.relative_to(root).parts[0] == "videos" or p.suffix in {".jpg", ".npy", ".csv"}
                 or p.name in {"local_track_index.json", "track_cache_index.json", "video_index.json"})}


def identity_fields(group: dict) -> dict:
    return {key: value for key, value in group.items() if key not in {"videos", "video_ids", "video_rankings"}}


class GalleryOrder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.player = None
        self.stack = []
        self.orders = {}

    def handle_starttag(self, tag, attrs):
        if tag == "section":
            attrs = dict(attrs)
            self.stack.append(self.player)
            if "player" in attrs.get("class", "").split():
                self.player = attrs["data-player-id"]
                self.orders[self.player] = []
            if attrs.get("data-video-id") and self.player:
                self.orders[self.player].append(attrs["data-video-id"])

    def handle_endtag(self, tag):
        if tag == "section":
            self.player = self.stack.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError("Test destination must not exist")
    source = args.source_run.resolve()
    source_digests = snapshot(source)
    root = destination / "collection"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns(".updates", ".collection.lock"))
    run = load_saved_run(root)
    before = read_json(root / "global/player_groups.json")
    protected = protected_files(root)
    state_before = read_json(root / "global/collection_state.json")
    wanted = {key for g in before for key in g["local_tracks"]}
    frame_sets = {}
    for video in run.videos:
        payload = read_json(root / "videos" / video.video_id / "local_tracks.json")
        for local in payload["tracks"]:
            if local["local_player_id"] in wanted:
                frame_sets[local["local_player_id"]] = {f["frame_index"] for f in local["frames"]}
    command = ["sort", "--run_dir", str(root)]
    with patch("reid_embedding._load_extractor", side_effect=AssertionError("ReID inference called")), \
         patch("yoloe_player_detection.YOLOEPlayerDetector.load", side_effect=AssertionError("YOLOE called")), \
         patch("clustering.cluster_embeddings", side_effect=AssertionError("Reclustered")), \
         patch("video_io.iter_video_frames", side_effect=AssertionError("Decoded source video")):
        started = time.perf_counter()
        assert pipeline_main(command) == 0
        elapsed = time.perf_counter() - started
        after = read_json(root / "global/player_groups.json")
        assert [identity_fields(g) for g in before] == [identity_fields(g) for g in after]
        assert protected_files(root) == protected
        videos = {v.video_id: v for v in run.videos}
        checked, unavailable = 0, 0
        for group in after:
            independent = []
            for entry in group["video_rankings"]:
                indices = set().union(*(frame_sets[key] for key in entry["local_tracks"]))
                video = videos[entry["video_id"]]
                total = video.frame_count
                valid = total > 0
                if video.processed_frame_count is not None:
                    total = video.processed_frame_count
                    valid = video.analysis_complete is True and total > 0 and (
                        video.frame_count <= 0 or total == video.frame_count)
                valid = valid and (not indices or min(indices) >= 0 and max(indices) < total)
                assert entry["visible_frames"] == len(indices)
                if valid:
                    assert math.isclose(entry["appearance_percentage"], 100 * len(indices) / total, abs_tol=1e-10)
                else:
                    assert entry["appearance_percentage"] is None
                    unavailable += 1
                independent.append((not valid, -Fraction(len(indices), total) if valid else Fraction(0),
                                    -len(indices) if valid else 0, entry["video_id"]))
                checked += 1
            assert group["video_ids"] == [entry[3] for entry in sorted(independent)]
            assert group["videos"] == [entry["original_filename"] for entry in group["video_rankings"]]
            assert [entry["rank"] for entry in group["video_rankings"]] == list(range(1, len(independent) + 1))
            if group["export_status"] == "accepted":
                metadata = read_json(root / "players" / group["player_id"] / "metadata.json")
                assert metadata["video_rankings"] == group["video_rankings"]
        _, export = saved_settings(run)
        expected = {g["player_id"]: g["video_ids"] for g in after
                    if export.gallery_include_review_rejected or g["export_status"] == "accepted"}
        for name in ("gallery.html", "gallery_standalone.html"):
            parsed = GalleryOrder()
            parsed.feed((root / "global" / name).read_text())
            assert parsed.orders == expected
        assert pipeline_main(command) == 0
        assert read_json(root / "global/player_groups.json") == after
        assert protected_files(root) == protected
    updated = load_saved_run(root)
    assert updated.complete_cache.active_video_ids == run.complete_cache.active_video_ids
    state_after = read_json(root / "global/collection_state.json")
    for key in ("video_membership", "embedding_status", "next_player_number", "inference_signature"):
        assert state_after.get(key) == state_before.get(key)
    assert source_digests == snapshot(source)
    report = {"active_videos": len(updated.videos), "registered_videos": len(updated.complete_cache.videos),
              "active_embeddings": len(updated.records), "cached_embeddings": len(updated.complete_cache.records),
              "player_groups": len(after), "accepted_players": sum(g["export_status"] == "accepted" for g in after),
              "ranked_video_entries": checked, "unavailable_coverage_entries": unavailable,
              "sort_seconds": elapsed, "raw_frame_set_reference_matches": True,
              "galleries_match_ranked_metadata": True, "repeated_sort_is_deterministic": True,
              "analysis_images_embeddings_indexes_and_similarity_unchanged": True,
              "player_assignments_and_quality_unchanged": True, "exclusions_and_processing_states_unchanged": True,
              "source_collection_unchanged": True, "inference_executed": False, "reclustering_executed": False}
    write_json(destination / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
