import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_collection_add
from collection_cli import main as collection_main
from collection_transaction import collection_lock
from main import main as pipeline_main
from reid_embedding import extract_track_embeddings
from saved_run import checksum, load_saved_run, read_json, write_registry


class DeleteTests(unittest.TestCase):
    def setUp(self):
        test_collection_add.AddTests.setUp(self)

    add = test_collection_add.AddTests.add
    mocked_add = test_collection_add.AddTests.mocked_add
    fake_analysis = test_collection_add.AddTests.fake_analysis
    fake_embeddings = test_collection_add.AddTests.fake_embeddings

    def delete(self, *ids, dry_run=False):
        args = ["delete", "--run_dir", str(self.root), "--video_ids", *ids]
        if dry_run:
            args.append("--dry_run")
        with patch("reid_embedding._load_extractor", side_effect=AssertionError("ReID inference called")), \
             patch("yoloe_player_detection.YOLOEPlayerDetector.load", side_effect=AssertionError("YOLOE called")):
            return pipeline_main(args)

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): checksum(p) for p in self.root.rglob("*") if p.is_file()
                and ".updates" not in p.relative_to(self.root).parts and p.name != ".collection.lock"}

    def test_delete_preserves_cache_and_analysis_and_reindexes_active_rows(self):
        before = {p: checksum(p) for p in (self.root / "videos").rglob("*") if p.is_file()}
        original_inputs = {p: checksum(p) for p in Path(self.temp.name).glob("*.mp4")}
        self.assertEqual(self.delete("V_A"), 0)
        run = load_saved_run(self.root)
        self.assertEqual([v.video_id for v in run.videos], ["V_B"])
        self.assertEqual(len(run.complete_cache.videos), 2)
        np.testing.assert_array_equal(run.complete_cache.embeddings, np.eye(2))
        np.testing.assert_array_equal(run.embeddings, [[0., 1.]])
        self.assertEqual(run.records[0].embedding_row, 0)
        self.assertEqual(before, {p: checksum(p) for p in before})
        self.assertEqual(original_inputs, {p: checksum(p) for p in original_inputs})
        with closing(sqlite3.connect(self.root / "project.db")) as db:
            self.assertEqual(db.execute("SELECT schema_version FROM project").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT active FROM videos WHERE video_id='V_A'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT embedding_row,cache_row FROM tracks WHERE video_id='V_A'").fetchone(), (None, 0))
            self.assertEqual(db.execute("SELECT embedding_row,cache_row FROM tracks WHERE video_id='V_B'").fetchone(), (0, 1))
        groups = read_json(self.root / "global/player_groups.json")
        self.assertEqual([g["player_id"] for g in groups], ["P0002"])
        self.assertNotIn("V_A_track_0001", (self.root / "global/gallery.html").read_text())
        state = read_json(self.root / "global/collection_state.json")
        self.assertEqual(state["embedding_status"]["V_A_track_0001"]["status"], "complete")
        self.assertIsNone(state["embedding_status"]["V_A_track_0001"]["embedding_row"])

    def test_dry_run_is_read_only(self):
        before = self.snapshot()
        self.assertEqual(self.delete("V_A", dry_run=True), 0)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / ".updates").exists())

    def test_unknown_id_rejects_entire_request(self):
        before = self.snapshot()
        self.assertEqual(self.delete("V_A", "V_UNKNOWN"), 2)
        self.assertEqual(self.snapshot(), before)

    def test_repeated_delete_is_noop(self):
        self.assertEqual(self.delete("V_A"), 0)
        before = self.snapshot()
        with patch("collection_delete.group_and_export", side_effect=AssertionError("Reclustered")):
            self.assertEqual(self.delete("V_A", "V_A"), 0)
        self.assertEqual(self.snapshot(), before)

    def test_delete_all_keeps_complete_cache_and_empty_valid_exports(self):
        self.assertEqual(self.delete("V_A", "V_B"), 0)
        run = load_saved_run(self.root)
        self.assertEqual(run.videos, [])
        self.assertEqual(run.records, [])
        self.assertEqual(run.embeddings.shape, (0, 2))
        self.assertEqual(run.complete_cache.embeddings.shape, (2, 2))
        self.assertEqual(read_json(self.root / "global/player_groups.json"), [])
        self.assertEqual(list((self.root / "players").iterdir()), [])
        self.assertTrue((self.root / "global/gallery_standalone.html").is_file())

    def test_add_excluded_duplicates_does_not_reactivate(self):
        self.assertEqual(self.delete("V_A"), 0)
        (self.inputs / "renamed.mp4").write_bytes(b"V_A")
        before = self.snapshot()
        self.assertEqual(self.mocked_add(), (0, 0))
        self.last_detector.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(load_saved_run(self.root).complete_cache.active_video_ids, {"V_B"})

    def test_add_new_video_preserves_inactive_embeddings(self):
        self.assertEqual(self.delete("V_A"), 0)
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        run = load_saved_run(self.root)
        self.assertEqual(len(run.videos), 2)
        self.assertEqual(len(run.complete_cache.videos), 3)
        np.testing.assert_array_equal(run.complete_cache.embeddings[:2], np.eye(2))
        np.testing.assert_array_equal(run.embeddings, [[0., 1.], [1., 0.]])
        self.assertNotIn("V_A", run.complete_cache.active_video_ids)

    def test_recluster_preserves_exclusions_and_full_cache(self):
        self.assertEqual(self.delete("V_A"), 0)
        self.assertEqual(collection_main(["recluster", "--run_dir", str(self.root), "--output_dir", str(self.output)]), 0)
        run = load_saved_run(self.output)
        self.assertEqual(run.complete_cache.active_video_ids, {"V_B"})
        np.testing.assert_array_equal(run.complete_cache.embeddings, np.eye(2))
        self.assertTrue((self.output / "videos/V_A/local_tracks.json").is_file())

    def test_delete_does_not_require_original_videos_or_model_files(self):
        logs = read_json(self.root / "logs.json")
        for path in [Path(logs["yoloe_model"]), Path(logs["reid_model_path"]),
                     *Path(self.temp.name).glob("*.mp4")]:
            path.unlink()
        self.assertEqual(self.delete("V_A"), 0)

    def test_inactive_pending_tracks_are_not_retried(self):
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.fake_embeddings = extract_track_embeddings
        with patch("reid_embedding._load_extractor", return_value=(lambda paths: np.zeros((1, 2)), "test")):
            self.assertEqual(self.mocked_add()[0], 2)
        video_id = load_saved_run(self.root).videos[-1].video_id
        self.assertEqual(self.delete(video_id), 0)
        self.assertEqual(self.mocked_add(), (0, 0))
        self.last_detector.assert_not_called()

    def test_active_vector_tampering_is_detected(self):
        self.assertEqual(self.delete("V_A"), 0)
        np.save(self.root / "global/local_track_embeddings.npy", np.array([[1., 1.]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "differ from the complete cache"):
            load_saved_run(self.root)

    def test_inactive_cache_tampering_is_detected(self):
        self.assertEqual(self.delete("V_A"), 0)
        values = np.load(self.root / "global/track_cache_embeddings.npy")
        values[0] = [1., 1.]
        np.save(self.root / "global/track_cache_embeddings.npy", values)
        with self.assertRaisesRegex(ValueError, "Registered artifacts have changed"):
            load_saved_run(self.root)

    def test_registry_membership_tampering_is_detected(self):
        self.assertEqual(self.delete("V_A"), 0)
        with closing(sqlite3.connect(self.root / "project.db")) as db, db:
            db.execute("UPDATE videos SET active=1 WHERE video_id='V_A'")
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            load_saved_run(self.root)

    def test_publication_failure_restores_previous_collection(self):
        before = self.snapshot()
        original_rename = Path.rename
        def fail(path, target):
            if path.name == "logs.json" and path.parent.name == "stage":
                raise OSError("injected failure")
            return original_rename(path, target)
        with patch.object(Path, "rename", fail):
            self.assertEqual(self.delete("V_A"), 2)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(load_saved_run(self.root).videos), 2)
        self.assertFalse((self.root / ".collection-update.json").exists())

    def test_export_failure_does_not_publish_migration(self):
        before = self.snapshot()
        with patch("collection_delete.group_and_export", side_effect=RuntimeError("failed export")):
            self.assertEqual(self.delete("V_A"), 2)
        self.assertEqual(self.snapshot(), before)

    def test_schema_one_registry_migrates_with_delete(self):
        write_registry(self.root, load_saved_run(self.root))
        with closing(sqlite3.connect(self.root / "project.db")) as db, db:
            db.executescript("""
                UPDATE project SET schema_version=1;
                CREATE TABLE tracks_v1 (local_player_id TEXT PRIMARY KEY, video_id TEXT, embedding_row INTEGER UNIQUE);
                INSERT INTO tracks_v1 SELECT local_player_id,video_id,embedding_row FROM tracks;
                DROP TABLE tracks;
                ALTER TABLE tracks_v1 RENAME TO tracks;
            """)
        self.assertEqual(len(load_saved_run(self.root).videos), 2)
        self.assertEqual(self.delete("V_A"), 0)
        with closing(sqlite3.connect(self.root / "project.db")) as db:
            self.assertEqual(db.execute("SELECT schema_version FROM project").fetchone()[0], 2)
        self.assertEqual(len(load_saved_run(self.root).complete_cache.records), 2)

    def test_delete_respects_collection_lock(self):
        with collection_lock(self.root):
            self.assertEqual(self.delete("V_A"), 2)

    def test_player_ids_are_not_recycled_after_deleting_every_video(self):
        self.assertEqual(self.delete("V_A", "V_B"), 0)
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        groups = read_json(self.root / "global/player_groups.json")
        self.assertEqual([g["player_id"] for g in groups], ["P0003"])


if __name__ == "__main__":
    unittest.main()
