"""Exercise fresh inference, Add, Delete, Sort, and crash recovery on existing videos."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main as pipeline_main
from run_sort_smoke import GalleryOrder, identity_fields, protected_files
from saved_run import checksum, load_saved_run, read_json
from utils import write_json
from video_io import read_video_metadata

REPOSITORY = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {"sha256": checksum(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def video_snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): checksum(p) for p in root.rglob("*") if p.is_file()}


def published_snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): checksum(p) for p in root.rglob("*") if p.is_file()
            and ".updates" not in p.relative_to(root).parts and p.name != ".collection.lock"}


def check_rankings(run) -> int:
    groups = read_json(run.root / "global/player_groups.json")
    wanted = {key for group in groups for key in group["local_tracks"]}
    observed = {}
    videos = {v.video_id: v for v in run.videos}
    for video in run.videos:
        payload = read_json(run.root / "videos" / video.video_id / "local_tracks.json")
        for track in payload["tracks"]:
            if track["local_player_id"] in wanted:
                observed[track["local_player_id"]] = {f["frame_index"] for f in track["frames"]}
    unavailable = 0
    for group in groups:
        ordering = []
        assert not group.get("conflicts"), "A final group violates a same-video conflict rule"
        for entry in group["video_rankings"]:
            frames = set().union(*(observed[key] for key in entry["local_tracks"]))
            video = videos[entry["video_id"]]
            total = video.processed_frame_count if video.processed_frame_count is not None else video.frame_count
            valid = total > 0 and (not frames or min(frames) >= 0 and max(frames) < total)
            if video.processed_frame_count is not None:
                valid = valid and video.analysis_complete is True and (
                    video.frame_count <= 0 or total == video.frame_count)
            assert entry["visible_frames"] == len(frames)
            if valid:
                assert math.isclose(entry["appearance_percentage"], 100 * len(frames) / total, abs_tol=1e-10)
            else:
                assert entry["appearance_percentage"] is None
                unavailable += 1
            ordering.append((not valid, -Fraction(len(frames), total) if valid else Fraction(0),
                             -len(frames) if valid else 0, entry["video_id"]))
        assert group["video_ids"] == [item[3] for item in sorted(ordering)]
        if group["export_status"] == "accepted":
            metadata = read_json(run.root / "players" / group["player_id"] / "metadata.json")
            assert metadata["video_rankings"] == group["video_rankings"]
    expected = {g["player_id"]: g["video_ids"] for g in groups if g["export_status"] == "accepted"}
    for name in ("gallery.html", "gallery_standalone.html"):
        path = run.root / "global" / name
        if path.exists():
            parser = GalleryOrder()
            parser.feed(path.read_text(encoding="utf-8"))
            assert parser.orders == expected
    return unavailable


def phase_summary(run) -> dict:
    groups = read_json(run.root / "global/player_groups.json")
    state = read_json(run.root / "global/collection_state.json")
    assert not run.logs.get("errors"), f"Processing errors: {run.logs.get('errors')}"
    assert run.provenance_verified, "A fresh workflow must have verified model provenance"
    assert all(v.analysis_complete is True and v.processed_frame_count > 0 for v in run.complete_cache.videos)
    return {"active_videos": len(run.videos), "registered_videos": len(run.complete_cache.videos),
            "active_tracks": len(run.tracks), "active_embeddings": len(run.records),
            "cached_embeddings": len(run.complete_cache.records), "global_groups": len(groups),
            "accepted_players": sum(g["export_status"] == "accepted" for g in groups),
            "unavailable_coverage_entries": check_rankings(run), "errors": run.logs.get("errors", []),
            "next_player_number": state["next_player_number"]}


def crash_delete(root: Path, video_id: str) -> None:
    original_rename = Path.rename
    def exit_during_publication(path, target):
        if path.name == "logs.json" and path.parent.name == "stage":
            os._exit(73)
        return original_rename(path, target)
    with patch.object(Path, "rename", exit_during_publication):
        pipeline_main(["delete", "--run_dir", str(root), "--video_ids", video_id])
    raise AssertionError("Crash injection was not reached")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=REPOSITORY / "highlights")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--initial_videos", type=int, default=20)
    parser.add_argument("--add_videos", type=int, default=5)
    parser.add_argument("--delete_videos", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--crash_run_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--crash_video_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.crash_run_dir:
        crash_delete(args.crash_run_dir, args.crash_video_id)
        return
    if args.output_dir is None:
        parser.error("--output_dir is required")
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir() or not source.is_relative_to(REPOSITORY):
        raise ValueError("Dataset must be an existing directory in this project")
    if destination.exists() or not destination.is_relative_to(REPOSITORY / "runs") or destination.is_relative_to(source):
        raise ValueError("Output must be a new folder under the project's runs directory, outside the dataset")
    if min(args.initial_videos, args.add_videos, args.delete_videos) <= 0:
        raise ValueError("Video counts must be positive")
    destination.mkdir(parents=True)
    root = destination / "collection"
    initial_dir, addition_dir = destination / "dataset/initial", destination / "dataset/additional"
    logs_dir = destination / "stage_logs"
    logs_dir.mkdir()
    report = {"status": "running", "source_dataset": str(source), "output": str(destination),
              "seed": args.seed, "stages": [], "phases": {}, "checks": {}}
    report_path = destination / "verification.json"
    write_json(report_path, report)
    started = time.perf_counter()

    def execute(name: str, arguments: list[str], expected_exit: int = 0, script: Path | None = None):
        command = [sys.executable, "-u", str(script or REPOSITORY / "main.py"), *arguments]
        log_path = logs_dir / (name + ".log")
        entry = {"name": name, "command": command, "expected_exit": expected_exit, "log": str(log_path)}
        report["stages"].append(entry)
        report["progress"] = {"stage": name, "videos_started": 0, "videos_completed": 0}
        write_json(report_path, report)
        print(f"\nStage: {name}", flush=True)
        phase_started = time.perf_counter()
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=REPOSITORY, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith(("Processing video:", "Processing new video:")):
                        report["progress"]["videos_started"] += 1
                        report["progress"]["current_video"] = line.strip().split(":", 1)[1].strip()
                        write_json(report_path, report)
                    if line.startswith(("Total video time:", "Video timings:")):
                        report["progress"]["videos_completed"] += 1
                        write_json(report_path, report)
                entry["exit_code"] = process.wait()
        entry["seconds"] = time.perf_counter() - phase_started
        write_json(report_path, report)
        assert entry["exit_code"] == expected_exit, f"{name} exited {entry['exit_code']}; see {log_path}"

    try:
        originals = {p.relative_to(source).as_posix(): file_record(p) for p in sorted(source.rglob("*"))
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS}
        models = {str(p): file_record(p) for p in (REPOSITORY / "models").rglob("*") if p.is_file()}
        if (REPOSITORY / "mobileclip2_b.ts").is_file():
            models[str(REPOSITORY / "mobileclip2_b.ts")] = file_record(REPOSITORY / "mobileclip2_b.ts")
        unique = {}
        for relative, record in originals.items():
            unique.setdefault(record["sha256"], relative)
        choices = sorted(unique.values())
        random.Random(args.seed).shuffle(choices)
        count = args.initial_videos + args.add_videos
        if len(choices) < count or args.delete_videos >= count:
            raise ValueError("Insufficient distinct videos for the requested workflow")
        manifest = []
        for index, relative in enumerate(choices[:count]):
            original = source / relative
            group = "initial" if index < args.initial_videos else "additional"
            target = (initial_dir if group == "initial" else addition_dir) / relative
            metadata = read_video_metadata(original, "selection")
            assert metadata.frame_count > 0 and metadata.fps > 0, f"Invalid video metadata: {original}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            assert checksum(target) == originals[relative]["sha256"]
            manifest.append({"source": str(original), "copy": str(target), "phase": group,
                             **originals[relative], "duration_sec": metadata.duration_sec,
                             "frame_count": metadata.frame_count, "fps": metadata.fps})
        report.update(source_video_count=len(originals), source_unique_video_count=len(unique),
                      selected_video_count=count, selected_minutes=sum(v["duration_sec"] for v in manifest) / 60)
        write_json(destination / "dataset_manifest.json", manifest)
        write_json(report_path, report)
        print(f"Selected {count} existing videos ({report['selected_minutes']:.2f} minutes): "
              f"{args.initial_videos} initial, {args.add_videos} additions.", flush=True)

        execute("fresh", ["--input_dir", str(initial_dir), "--output_dir", str(root)])
        initial = load_saved_run(root)
        assert len(initial.videos) == args.initial_videos and len(initial.records) > 0
        report["phases"]["fresh"] = phase_summary(initial)
        initial_images = video_snapshot(root / "videos")
        initial_records = list(initial.complete_cache.records)
        initial_vectors = initial.complete_cache.embeddings.copy()
        initial_ids = {v.video_id for v in initial.videos}

        execute("add", ["add", "--run_dir", str(root), "--input_dir", str(addition_dir)])
        added = load_saved_run(root)
        assert len(added.videos) == count and len(added.records) > len(initial.records)
        np.testing.assert_array_equal(added.complete_cache.embeddings[:len(initial_vectors)], initial_vectors)
        assert added.complete_cache.records[:len(initial_records)] == initial_records
        assert all(checksum(root / "videos" / relative) == digest for relative, digest in initial_images.items())
        report["checks"]["add_reused_existing_analysis_and_embeddings"] = True
        report["phases"]["add"] = phase_summary(added)
        snapshot = published_snapshot(root)
        execute("duplicate_add", ["add", "--run_dir", str(root), "--input_dir", str(addition_dir)])
        assert snapshot == published_snapshot(root)
        report["checks"]["duplicate_add_noop"] = True

        added_ids = {v.video_id for v in added.videos} - initial_ids
        embedded_ids = {r.video_id for r in added.records}
        candidates = sorted(added_ids & embedded_ids)[:1] + sorted(initial_ids & embedded_ids)
        candidates += sorted(embedded_ids - set(candidates))
        excluded = candidates[:args.delete_videos]
        assert len(excluded) == args.delete_videos
        retained_images = video_snapshot(root / "videos")
        retained_vectors = added.complete_cache.embeddings.copy()
        execute("delete", ["delete", "--run_dir", str(root), "--video_ids", *excluded])
        deleted = load_saved_run(root)
        assert len(deleted.videos) == count - len(excluded)
        assert not set(excluded) & deleted.complete_cache.active_video_ids
        np.testing.assert_array_equal(deleted.complete_cache.embeddings, retained_vectors)
        assert retained_images == video_snapshot(root / "videos")
        report["excluded_video_ids"] = excluded
        report["checks"]["delete_preserved_complete_cache_and_video_analysis"] = True
        report["phases"]["delete"] = phase_summary(deleted)
        snapshot = published_snapshot(root)
        execute("duplicate_delete", ["delete", "--run_dir", str(root), "--video_ids", *excluded])
        assert snapshot == published_snapshot(root)
        execute("add_after_delete", ["add", "--run_dir", str(root), "--input_dir", str(addition_dir)])
        assert snapshot == published_snapshot(root)
        report["checks"]["duplicate_delete_and_excluded_add_noop"] = True

        identity_before = [identity_fields(g) for g in read_json(root / "global/player_groups.json")]
        protected = protected_files(root)
        execute("sort", ["sort", "--run_dir", str(root)])
        sorted_run = load_saved_run(root)
        assert protected == protected_files(root)
        assert identity_before == [identity_fields(g) for g in read_json(root / "global/player_groups.json")]
        assert sorted_run.complete_cache.active_video_ids == deleted.complete_cache.active_video_ids
        report["phases"]["sort"] = phase_summary(sorted_run)
        report["checks"]["sort_preserved_identity_quality_cache_and_exclusions"] = True

        snapshot = published_snapshot(root)
        crash_id = sorted(sorted_run.complete_cache.active_video_ids)[0]
        execute("interrupted_update", ["--crash_run_dir", str(root), "--crash_video_id", crash_id],
                expected_exit=73, script=Path(__file__))
        assert (root / ".collection-update.json").is_file()
        execute("inspection_blocked_until_recovery", ["inspect", "--run_dir", str(root)], expected_exit=2)
        execute("recover", ["recover", "--run_dir", str(root)])
        assert snapshot == published_snapshot(root)
        recovered = load_saved_run(root)
        report["phases"]["recovered"] = phase_summary(recovered)
        report["checks"]["real_process_interruption_recovered_exactly"] = True

        import cv2
        debug = []
        for video in recovered.complete_cache.videos:
            path = root / "videos" / video.video_id / "tracking_debug.mp4"
            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
                assert ok and frame.shape[:2] == (video.height, video.width), f"Invalid debug video: {path}"
                expected_frames = video.processed_frame_count
                assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames, f"Incomplete debug video: {path}"
                capture.set(cv2.CAP_PROP_POS_FRAMES, expected_frames - 1)
                ok, last_frame = capture.read()
                assert ok and last_frame.shape[:2] == (video.height, video.width), f"Invalid final debug frame: {path}"
                debug.append(video.video_id)
            finally:
                capture.release()
        assert (root / "global/gallery_standalone.html").is_file()
        report["checks"]["all_debug_videos_decodable"] = True
        report["checks"]["standalone_gallery_present"] = True
        report["checks"]["all_original_videos_unchanged"] = originals == {
            relative: file_record(source / relative) for relative in originals}
        report["checks"]["all_model_files_unchanged"] = models == {path: file_record(Path(path)) for path in models}
        report["checks"]["dataset_copies_unchanged"] = all(checksum(Path(v["copy"])) == v["sha256"] for v in manifest)
        assert all(report["checks"].values())
        report.update(status="passed", total_seconds=time.perf_counter() - started,
                      debug_videos_checked=len(debug), final_gallery=str(root / "global/gallery_standalone.html"))
        report.pop("progress", None)
        write_json(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc(),
                      total_seconds=time.perf_counter() - started)
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
