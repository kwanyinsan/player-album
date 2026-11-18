import argparse
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from collection_add import main
from collection_transaction import atomic_json, collection_lock, publish, recover
from config import CropConfig, FilterConfig
from json_export import write_local_tracks_json
from player_identity import reconcile_player_ids
from player_tracking import LocalTrack, TrackFrame
from reid_embedding import EmbeddingOutput, EmbeddingRecord, extract_track_embeddings
from saved_run import checksum, load_saved_run, read_json
import test_saved_run
from utils import write_json
from video_io import VideoInfo


class AddTests(unittest.TestCase):
    def setUp(self):
        test_saved_run.SavedRunTests.setUp(self)
        self.inputs = Path(self.temp.name) / "inputs"
        self.inputs.mkdir()
        logs = read_json(self.root / "logs.json")
        detector = Path(self.temp.name) / "detector.pt"
        detector.write_bytes(b"detector")
        reid = Path(self.temp.name) / "reid.pth"
        reid.write_bytes(b"reid")
        logs.update(yoloe_model=str(detector), reid_model="osnet_ain_x1_0", reid_model_path=str(reid),
                    crop_config=asdict(CropConfig()), filter_config=asdict(FilterConfig()))
        write_json(self.root / "logs.json", logs)
        for video_id in ("V_A", "V_B"):
            path = Path(self.temp.name) / (video_id + ".mp4")
            path.write_bytes(video_id.encode())
            trackfile = self.root / f"videos/{video_id}/local_tracks.json"
            payload = read_json(trackfile)
            payload["video"]["path"] = str(path)
            write_json(trackfile, payload)
        write_json(self.root / "global/player_groups.json", [
            {"player_id": f"P{i+1:04d}", "local_tracks": [r.local_player_id]} for i, r in enumerate(self.records)
        ])
        self.signature = {"reid": "fixture-v1"}

    def add(self, *args):
        return main(["add", "--run_dir", str(self.root), "--input_dir", str(self.inputs), *args])

    def fake_analysis(self, video, detector, config, root):
        track_id = video.video_id + "_track_0001"
        crop = f"videos/{video.video_id}/crops/{track_id}/crop_0001.jpg"
        (root / crop).parent.mkdir(parents=True)
        (root / crop).write_bytes(b"fixture-crop")
        track = LocalTrack(track_id, video.video_id, 1, video.filename,
                           frames=[TrackFrame(0, track_id, 1, [0, 0, 20, 40], .9, "person")],
                           frame_count=1, crop_paths=[crop], representative_crop=crop)
        write_local_tracks_json(video, [track], root / f"videos/{video.video_id}/local_tracks.json")
        return [track], {"tracking_sec": 0}

    def fake_embeddings(self, tracks, *args, **kwargs):
        records = [EmbeddingRecord(t.local_player_id, t.video_id, t.source_video_name, 0, 0, 1,
                                   t.crop_paths, t.representative_crop, i) for i, t in enumerate(tracks)]
        embeddings = np.array([[1., 0.]] * len(tracks), dtype=np.float32).reshape(-1, 2)
        global_dir = args[1]
        global_dir.mkdir(parents=True, exist_ok=True)
        np.save(global_dir / "local_track_embeddings.npy", embeddings)
        write_json(global_dir / "local_track_index.json", [r.to_dict() for r in records])
        return EmbeddingOutput(embeddings, records, [], "test")

    def mocked_add(self, *args, analysis=None):
        with patch("collection_add.model_signature", return_value=self.signature), \
             patch("collection_add.YOLOEPlayerDetector") as detector, \
             patch("collection_add.read_video_metadata", side_effect=lambda p, v: VideoInfo(v, p.name, p, 100, 100, 30, 1, 1/30)), \
             patch("collection_add.analyze_video", side_effect=analysis or self.fake_analysis), \
             patch("collection_add.extract_track_embeddings", side_effect=self.fake_embeddings) as reid:
            result = self.add(*args)
            self.last_detector = detector
            return result, reid.call_count

    def test_add_preserves_existing_analysis_and_duplicate_is_noop(self):
        (self.inputs / "new.mp4").write_bytes(b"new-video")
        before = {p: checksum(p) for p in (self.root / "videos").rglob("*") if p.is_file()}
        self.assertEqual(self.mocked_add()[0], 0)
        run = load_saved_run(self.root)
        self.assertEqual(len(run.videos), 3)
        np.testing.assert_array_equal(run.embeddings[:2], np.eye(2))
        self.assertEqual(before, {p: checksum(p) for p in before})
        groups = read_json(self.root / "global/player_groups.json")
        self.assertEqual(next(g for g in groups if self.records[0].local_player_id in g["local_tracks"])["player_id"], "P0001")
        self.assertIn("data:image/jpeg;base64,", (self.root / "global/gallery_standalone.html").read_text())
        published = {p: checksum(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.mocked_add(), (0, 0))
        self.assertEqual(published, {p: checksum(p) for p in published})

    def test_renamed_duplicate_skipped_without_model(self):
        (self.inputs / "renamed.mp4").write_bytes(b"V_A")
        with patch("collection_add.model_signature", side_effect=AssertionError("model checked")):
            self.assertEqual(self.add(), 0)

    def test_dry_run_no_inference(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        with patch("collection_add.YOLOEPlayerDetector", side_effect=AssertionError("model loaded")):
            self.assertEqual(self.add("--dry_run"), 0)
        self.assertFalse((self.root / ".updates").exists())

    def test_same_filename_different_contents_is_new(self):
        (self.inputs / "V_A.mp4").write_bytes(b"different")
        self.assertEqual(self.mocked_add()[0], 0)
        self.assertEqual(len(load_saved_run(self.root).videos), 3)

    def test_all_video_failures_preserve_published_run(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        before = checksum(self.root / "logs.json")
        self.assertEqual(self.mocked_add(analysis=lambda *args: (_ for _ in ()).throw(RuntimeError("bad video")))[0], 2)
        self.assertEqual(checksum(self.root / "logs.json"), before)
        self.assertEqual(len(load_saved_run(self.root).videos), 2)

    def test_export_failure_does_not_publish(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        with patch("collection_add.group_and_export", side_effect=RuntimeError("failed export")):
            self.assertEqual(self.mocked_add()[0], 2)
        self.assertEqual(len(load_saved_run(self.root).videos), 2)

    def test_known_checkpoint_mismatch_rejected(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        (self.inputs / "next.mp4").write_bytes(b"next")
        self.signature = {"reid": "different"}
        self.assertEqual(self.mocked_add(), (2, 0))

    def test_empty_tracks_video_is_registered(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        def empty(video, detector, config, root):
            write_local_tracks_json(video, [], root / f"videos/{video.video_id}/local_tracks.json")
            return [], {}
        self.assertEqual(self.mocked_add(analysis=empty)[0], 0)
        self.assertEqual(len(load_saved_run(self.root).videos), 3)

    def test_interrupted_publication_can_rollback_repeatedly(self):
        operation = self.root / ".updates" / "fixture"
        (operation / "stage").mkdir(parents=True)
        (operation / "backup").mkdir()
        (operation / "stage/logs.json").write_text("new")
        original = checksum(self.root / "logs.json")
        payload = {"operation": "fixture", "status": "publishing",
                   "entries": [{"path": "logs.json", "install": True}]}
        atomic_json(self.root / ".collection-update.json", payload)
        (self.root / "logs.json").rename(operation / "backup/logs.json")
        (operation / "stage/logs.json").rename(self.root / "logs.json")
        self.assertTrue(recover(self.root))
        self.assertEqual(checksum(self.root / "logs.json"), original)
        self.assertFalse(recover(self.root))

    def test_split_merge_ids_are_deterministic_and_not_recycled(self):
        previous = [{"player_id": "P0007", "local_tracks": ["a", "b"]}]
        records = [argparse.Namespace(local_player_id=t) for t in ("a", "b", "c")]
        ids, report = reconcile_player_ids([0, 1, 2], records, previous, 20)
        self.assertEqual(ids, {0: "P0007", 1: "P0020", 2: "P0021"})
        self.assertEqual(report["next_player_number"], 22)

    def test_partial_failure_publishes_success_and_returns_failure_status(self):
        (self.inputs / "bad.mp4").write_bytes(b"bad")
        (self.inputs / "good.mp4").write_bytes(b"good")
        def analyze(video, *args):
            if video.filename == "bad.mp4":
                raise RuntimeError("unreadable video")
            return self.fake_analysis(video, *args)
        self.assertEqual(self.mocked_add(analysis=analyze)[0], 2)
        run = load_saved_run(self.root)
        self.assertEqual(len(run.videos), 3)
        self.assertEqual(run.videos[-1].filename, "good.mp4")
        self.assertEqual(len(run.logs["errors"]), 1)

    def test_limit_applies_after_duplicate_filter(self):
        (self.inputs / "a_duplicate.mp4").write_bytes(b"V_A")
        (self.inputs / "b_new.mp4").write_bytes(b"new")
        (self.inputs / "c_new.mp4").write_bytes(b"next")
        self.assertEqual(self.mocked_add("--max_videos", "1")[0], 0)
        run = load_saved_run(self.root)
        self.assertEqual(len(run.videos), 3)
        self.assertEqual(run.videos[-1].filename, "b_new.mp4")

    def test_multiple_new_copies_processed_once(self):
        (self.inputs / "a.mp4").write_bytes(b"new")
        (self.inputs / "b.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        self.assertEqual(len(load_saved_run(self.root).videos), 3)

    def test_missing_legacy_original_is_not_guessed(self):
        (Path(self.temp.name) / "V_A.mp4").unlink()
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add(), (2, 0))

    def test_dimension_mismatch_does_not_publish(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        normal = self.fake_embeddings
        def wrong(tracks, *args, **kwargs):
            result = normal(tracks, *args, **kwargs)
            result.embeddings = np.ones((len(tracks), 3), dtype=np.float32)
            return result
        self.fake_embeddings = wrong
        self.assertEqual(self.mocked_add()[0], 2)
        self.assertEqual(len(load_saved_run(self.root).videos), 2)

    def test_lock_blocks_second_writer(self):
        with collection_lock(self.root):
            self.assertEqual(self.add("--dry_run"), 2)

    def test_publication_failure_rolls_back_all_managed_outputs(self):
        operation = self.root / ".updates" / "rollback"
        stage = operation / "stage"
        (stage / "global").mkdir(parents=True)
        (stage / "global/marker").write_text("new")
        (stage / "logs.json").write_text("new logs")
        before = checksum(self.root / "logs.json")
        original_rename = Path.rename
        def fail_once(path, target):
            if path == stage / "logs.json":
                raise OSError("injected publication failure")
            return original_rename(path, target)
        with patch.object(Path, "rename", fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                publish(self.root, operation, [])
        self.assertEqual(checksum(self.root / "logs.json"), before)
        self.assertFalse((self.root / ".collection-update.json").exists())
        self.assertEqual(len(load_saved_run(self.root).videos), 2)

    def test_merge_keeps_one_prior_id_and_records_both(self):
        previous = [{"player_id": "P0001", "local_tracks": ["a"]},
                    {"player_id": "P0002", "local_tracks": ["b"]}]
        records = [argparse.Namespace(local_player_id=t) for t in ("a", "b")]
        ids, report = reconcile_player_ids([0, 0], records, previous)
        self.assertEqual(ids[0], "P0001")
        self.assertEqual(report["lineage"][0]["previous_player_ids"], ["P0001", "P0002"])

    def test_failed_embeddings_retry_from_saved_crops_without_detector(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.fake_embeddings = extract_track_embeddings
        before = {g["player_id"]: g["local_tracks"] for g in read_json(self.root / "global/player_groups.json")}
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: (_ for _ in ()).throw(RuntimeError("temporary failure")), "test")):
            self.assertEqual(self.mocked_add()[0], 2)
        failed = load_saved_run(self.root)
        self.assertEqual(len(failed.videos), 3)
        self.assertEqual(len(failed.records), 2)
        self.assertEqual({g["player_id"]: g["local_tracks"] for g in read_json(self.root / "global/player_groups.json")}, before)
        state = read_json(self.root / "global/collection_state.json")
        pending = [k for k, e in state["embedding_status"].items() if e["status"] == "retryable_error"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(state["embedding_status"][pending[0]]["attempt_count"], 1)
        preserved = {p: checksum(p) for p in (self.root / "videos").rglob("*") if p.is_file()}
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: np.array([[1., 0.]]), "test")) as loader:
            self.assertEqual(self.mocked_add(analysis=lambda *args: self.fail("Video analysis repeated"))[0], 0)
            self.assertEqual(loader.call_count, 1)
        self.last_detector.assert_not_called()
        recovered = load_saved_run(self.root)
        self.assertEqual(len(recovered.records), 3)
        np.testing.assert_array_equal(recovered.embeddings[:2], np.eye(2))
        self.assertEqual(preserved, {p: checksum(p) for p in preserved})
        status = read_json(self.root / "global/collection_state.json")["embedding_status"][pending[0]]
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["attempt_count"], 2)
        self.assertEqual(status["embedding_row"], 2)

    def test_recluster_preserves_player_ids_and_state(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        from collection_cli import main as saved_main
        previous = read_json(self.root / "global/player_groups.json")
        self.assertEqual(saved_main(["recluster", "--run_dir", str(self.root), "--output_dir", str(self.output)]), 0)
        current = read_json(self.output / "global/player_groups.json")
        self.assertEqual({g["player_id"]: g["local_tracks"] for g in previous},
                         {g["player_id"]: g["local_tracks"] for g in current})
        load_saved_run(self.output)

    def test_no_old_analysis_copied_and_only_one_backup_retained(self):
        for name in ("one", "two", "three"):
            (self.inputs / (name + ".mp4")).write_bytes(name.encode())
            self.assertEqual(self.mocked_add("--max_videos", "1")[0], 0)
        updates = list((self.root / ".updates").iterdir())
        self.assertEqual(len(updates), 3)
        self.assertEqual(sum((p / "backup/global").exists() for p in updates), 1)
        self.assertFalse(any(list((p / "stage/videos").rglob("local_tracks.json")) for p in updates))
        self.assertEqual(len(load_saved_run(self.root).videos), 5)

    def test_inconsistent_complete_status_is_rejected(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        state = read_json(self.root / "global/collection_state.json")
        state["embedding_status"][self.records[0].local_player_id]["embedding_row"] = 999
        write_json(self.root / "global/collection_state.json", state)
        with self.assertRaisesRegex(ValueError, "Invalid embedding status"):
            load_saved_run(self.root)

    def test_failed_retry_publication_does_not_claim_complete(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.fake_embeddings = extract_track_embeddings
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: np.zeros((1, 2)), "test")):
            self.assertEqual(self.mocked_add()[0], 2)
        state_before = checksum(self.root / "global/collection_state.json")
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: np.array([[1., 0.]]), "test")), \
             patch("collection_add.publish", side_effect=RuntimeError("interrupted before publication")):
            self.assertEqual(self.mocked_add()[0], 2)
        self.assertEqual(checksum(self.root / "global/collection_state.json"), state_before)
        self.assertEqual(len(load_saved_run(self.root).records), 2)

    def test_fresh_batch_records_provenance_fingerprints_and_states(self):
        from config import parse_config
        from main import main as batch_main
        logs = read_json(self.root / "logs.json")
        video_path = self.inputs / "new.mp4"
        video_path.write_bytes(b"new")
        config = parse_config(["--input_dir", str(self.inputs), "--output_dir", str(self.output),
                               "--yoloe_model", logs["yoloe_model"], "--reid_model_path", logs["reid_model_path"]])
        video = VideoInfo("V_C", video_path.name, video_path, 100, 100, 30, 1, 1/30)
        with patch("main.parse_config", return_value=config), \
             patch("main.model_signature", return_value=self.signature), \
             patch("main.scan_videos", return_value=[video]), \
             patch("main.YOLOEPlayerDetector") as detector, \
             patch("main.analyze_video", side_effect=lambda v, d, c, r, **kw: self.fake_analysis(v, d, c, r)), \
             patch("main.extract_track_embeddings", side_effect=self.fake_embeddings):
            detector.return_value.prompt_warning = None
            detector.return_value.runtime_warnings = []
            self.assertEqual(batch_main([]), 0)
        saved = load_saved_run(self.output)
        self.assertTrue(saved.provenance_verified)
        state = read_json(self.output / "global/collection_state.json")
        self.assertEqual(state["video_fingerprints"]["V_C"], checksum(video_path))
        self.assertEqual(state["embedding_status"]["V_C_track_0001"]["status"], "complete")
        self.assertEqual(state["inference_signature"], self.signature)

    def test_blocked_tracks_report_failure_without_repeated_inference(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.fake_embeddings = extract_track_embeddings
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: (_ for _ in ()).throw(OSError("corrupt crop")), "test")):
            self.assertEqual(self.mocked_add()[0], 2)
        state = read_json(self.root / "global/collection_state.json")
        self.assertEqual(sum(e["status"] == "blocked_error" for e in state["embedding_status"].values()), 1)
        self.assertEqual(self.mocked_add(), (2, 0))
        self.last_detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
