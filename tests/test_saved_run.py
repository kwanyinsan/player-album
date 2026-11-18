import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from collection_cli import main
from config import ClusteringConfig, ExportConfig
from player_tracking import LocalTrack, TrackFrame
from reid_embedding import EmbeddingRecord
from saved_run import checksum, load_saved_run, load_video_tracks
from utils import write_json


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "output"
        self.records = []
        videos = []
        for row, video_id in enumerate(("V_A", "V_B")):
            track_id = video_id + "_track_0001"
            crop = f"videos/{video_id}/crops/{track_id}/crop_0001.jpg"
            target = self.root / crop
            target.parent.mkdir(parents=True)
            target.write_bytes(b"fixture-crop")
            frame = TrackFrame(0, track_id, 1, [0, 0, 20, 40], 0.9, "person")
            track = LocalTrack(track_id, video_id, 1, video_id + ".mp4", frames=[frame],
                               frame_count=1, crop_paths=[crop], representative_crop=crop)
            video = dict(video_id=video_id, filename=video_id + ".mp4", path="missing/input.mp4",
                         width=100, height=100, fps=30, frame_count=1, duration_sec=1/30)
            write_json(self.root / f"videos/{video_id}/local_tracks.json",
                       {"video": video, "tracks": [track.to_dict()]})
            videos.append({**video, "original_filename": video["filename"]})
            self.records.append(EmbeddingRecord(track_id, video_id, video["filename"], 0, 0, 1,
                                                [crop], crop, row))
        write_json(self.root / "global/video_index.json", videos)
        write_json(self.root / "global/local_track_index.json", [r.to_dict() for r in reversed(self.records)])
        np.save(self.root / "global/local_track_embeddings.npy", np.eye(2, dtype=np.float32))
        write_json(self.root / "logs.json", {
            "number_of_videos": 2,
            "clustering_config": asdict(ClusteringConfig(identity_split_enabled=False)),
            "export_config": asdict(ExportConfig(quality_filter_export=False)),
        })

    def call_recluster(self):
        return main(["recluster", "--run_dir", str(self.root), "--output_dir", str(self.output)])

    def mutate_index(self, callback):
        path = self.root / "global/local_track_index.json"
        records = json.loads(path.read_text())
        callback(records)
        write_json(path, records)

    def test_reordered_index_preserves_embedding_associations(self):
        run = load_saved_run(self.root)
        self.assertEqual(run.records[0].local_player_id, self.records[0].local_player_id)
        np.testing.assert_array_equal(run.embeddings, np.eye(2))
        self.assertFalse(run.report()["inference_cache_reuse_verified"])
        video, tracks = load_video_tracks(self.root / "videos/V_A/local_tracks.json", include_frames=True)
        self.assertEqual(tracks[0].frames[0].frame_index, 0)

    def test_duplicate_rows_rejected(self):
        self.mutate_index(lambda records: records[0].update(embedding_row=0))
        with self.assertRaisesRegex(ValueError, "unique complete permutation"):
            load_saved_run(self.root)

    def test_bad_track_mapping_rejected(self):
        self.mutate_index(lambda records: records[0].update(video_id="V_A"))
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            load_saved_run(self.root)

    def test_missing_crop_rejected(self):
        (self.root / self.records[0].crop_paths[0]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing or external"):
            load_saved_run(self.root)

    def test_nonfinite_embeddings_rejected(self):
        np.save(self.root / "global/local_track_embeddings.npy", np.full((2, 2), np.nan))
        with self.assertRaisesRegex(ValueError, "Invalid embedding"):
            load_saved_run(self.root)

    def test_zero_embedding_rejected(self):
        np.save(self.root / "global/local_track_embeddings.npy", np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Invalid embedding"):
            load_saved_run(self.root)

    def test_bad_frame_summary_rejected(self):
        path = self.root / "videos/V_A/local_tracks.json"
        payload = json.loads(path.read_text())
        payload["tracks"][0]["frame_count"] = 2
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "frame or video mapping"):
            load_saved_run(self.root)

    def test_empty_collection_supported(self):
        write_json(self.root / "global/video_index.json", [])
        write_json(self.root / "global/local_track_index.json", [])
        write_json(self.root / "logs.json", {"number_of_videos": 0})
        np.save(self.root / "global/local_track_embeddings.npy", np.zeros((0, 0), dtype=np.float32))
        self.assertEqual(self.call_recluster(), 0)
        self.assertEqual(load_saved_run(self.output).records, [])

    def test_main_routes_to_read_only_inspection(self):
        from main import main as pipeline_main
        with patch("main.parse_config", side_effect=AssertionError("batch config requested")):
            self.assertEqual(pipeline_main(["inspect", "--run_dir", str(self.root)]), 0)

    def test_override_saved_threshold(self):
        self.assertEqual(main(["recluster", "--run_dir", str(self.root), "--output_dir", str(self.output),
                               "--distance_threshold", "0.25"]), 0)
        logs = json.loads((self.output / "logs.json").read_text())
        self.assertEqual(logs["clustering_config"]["distance_threshold"], 0.25)

    def test_escaping_artifact_rejected(self):
        path = self.root / "videos/V_A/local_tracks.json"
        payload = json.loads(path.read_text())
        payload["tracks"][0]["crop_paths"] = ["../outside.jpg"]
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "relative path"):
            load_saved_run(self.root)

    def test_recluster_without_models_and_source_unchanged(self):
        before = {str(p): checksum(p) for p in self.root.rglob("*") if p.is_file()}
        with patch("reid_embedding._load_extractor", side_effect=AssertionError("inference called")), \
             patch("yoloe_player_detection.YOLOEPlayerDetector.load", side_effect=AssertionError("inference called")):
            self.assertEqual(self.call_recluster(), 0)
        self.assertEqual(before, {str(p): checksum(p) for p in self.root.rglob("*") if p.is_file()})
        self.assertTrue((self.output / "global/gallery.html").is_file())
        self.assertTrue((self.output / "players/P0001/representative.jpg").is_file())
        self.assertEqual(len(load_saved_run(self.output).records), 2)
        with closing(sqlite3.connect(self.output / "project.db")) as db:
            self.assertEqual(db.execute("SELECT sum(active) FROM videos").fetchone()[0], 2)

    def test_registered_tampering_rejected(self):
        self.assertEqual(self.call_recluster(), 0)
        (self.output / self.records[0].crop_paths[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Registered artifacts have changed"):
            load_saved_run(self.output)

    def test_existing_destination_rejected(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        self.assertEqual(self.call_recluster(), 2)
        self.assertEqual(marker.read_text(), "keep")

    def test_failed_export_never_publishes(self):
        with patch("collection_cli.group_and_export", side_effect=RuntimeError("export failed")):
            self.assertEqual(self.call_recluster(), 2)
        self.assertFalse(self.output.exists())

    def test_inspect_is_read_only(self):
        before = set(self.root.rglob("*"))
        self.assertEqual(main(["inspect", "--run_dir", str(self.root)]), 0)
        self.assertEqual(before, set(self.root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
