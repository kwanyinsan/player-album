"""Opt-in real-model Add test using a small copy of an existing saved run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from album_pipeline import group_and_export
from collection_add import main as add_main
from json_export import build_video_index, write_global_indexes
from saved_run import checksum, load_saved_run, saved_settings, write_registry
from utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--baseline_videos", type=int, default=20)
    parser.add_argument("--new_videos", type=int, default=5)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError("Smoke test destination must not exist")
    source = load_saved_run(args.source_run.resolve(), progress=print)
    embedded = {r.video_id for r in source.records}
    count = args.baseline_videos + args.new_videos
    if args.baseline_videos < 1 or args.new_videos < 1:
        raise ValueError("Video counts must be positive")
    choices = sorted((v for v in source.videos if v.video_id in embedded and v.path.is_file()),
                     key=lambda v: (v.duration_sec, v.video_id))[:count]
    if len(choices) < count:
        raise ValueError(f"{count} original videos with saved embeddings are needed")
    baseline, holdout = choices[:args.baseline_videos], choices[args.baseline_videos:]
    baseline_ids = {v.video_id for v in baseline}
    root, inputs = destination / "collection", destination / "inputs"
    (root / "global").mkdir(parents=True)
    inputs.mkdir()
    for relative, digest in source.artifacts.items():
        if relative.startswith("videos/") and relative.split("/")[1] in baseline_ids:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.root / relative, target)
            assert checksum(target) == digest
    selected = [r for r in source.records if r.video_id in baseline_ids]
    embeddings = source.embeddings[[r.embedding_row for r in selected]].copy()
    records = [replace(r, embedding_row=i) for i, r in enumerate(selected)]
    tracks = {k: t for k, t in source.tracks.items() if t.video_id in baseline_ids}
    clustering, export = saved_settings(source)
    groups, reports = group_and_export(root, records, embeddings, tracks, clustering, export,
                                      video_metadata={v.video_id: v for v in baseline})
    np.save(root / "global/local_track_embeddings.npy", embeddings)
    write_json(root / "global/local_track_index.json", [r.to_dict() for r in records])
    write_global_indexes(root / "global", groups, build_video_index(baseline, groups, tracks))
    write_json(root / "logs.json", {**source.logs, **reports, "number_of_videos": len(baseline),
                                  "number_of_local_tracks": len(tracks), "number_of_global_players": len(groups)})
    write_registry(root, load_saved_run(root))
    original_digests = {str(v.path): checksum(v.path) for v in choices}
    preserved = {p: checksum(p) for p in (root / "videos").rglob("*") if p.is_file()}
    for video in holdout:
        shutil.copy2(video.path, inputs / video.filename)
    model_digests = {p: checksum(Path(p)) for p in (source.logs["yoloe_model"], source.logs["reid_model_path"])}
    command = ["add", "--run_dir", str(root), "--input_dir", str(inputs)]
    if add_main(command) != 0:
        raise RuntimeError("Real inference Add did not complete without errors")
    updated = load_saved_run(root)
    assert len(updated.videos) == count
    assert len(updated.records) > len(records), "Holdout must produce valid embeddings"
    np.testing.assert_array_equal(updated.embeddings[:len(records)], embeddings)
    assert preserved == {p: checksum(p) for p in preserved}
    assert original_digests == {p: checksum(Path(p)) for p in original_digests}
    assert model_digests == {p: checksum(Path(p)) for p in model_digests}
    snapshot = {p: checksum(p) for p in root.rglob("*") if p.is_file()}
    assert add_main(command) == 0
    assert snapshot == {p: checksum(p) for p in snapshot}
    # Independently recluster the combined cache into a separate output.
    from collection_cli import main as saved_main
    reference = destination / "recluster_reference"
    assert saved_main(["recluster", "--run_dir", str(root), "--output_dir", str(reference)]) == 0
    def memberships(path):
        payload = json.loads((path / "global/player_groups.json").read_text())
        return sorted((tuple(sorted(g["local_tracks"])), g["export_status"]) for g in payload)
    assert memberships(root) == memberships(reference)
    report = {"baseline_videos": [v.filename for v in baseline], "added_videos": [v.filename for v in holdout],
              "baseline_video_count": len(baseline), "added_video_count": len(holdout),
              "existing_embeddings": len(records), "combined_embeddings": len(updated.records),
              "old_analysis_unchanged": True, "input_videos_unchanged": True,
              "duplicate_add_noop": True, "combined_cache_recluster_matches": True,
              "model_weights_unchanged": True,
              "old_analysis_copied_to_update_workspace": any(
                  (p / "stage" / relative).exists() for p in (root / ".updates").iterdir()
                  for relative in source.artifacts if relative.startswith("videos/") and relative.split("/")[1] in baseline_ids),
              "timings": updated.logs["processing_times"]}
    write_json(destination / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
