import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import test_collection_add
from collection_transaction import collection_lock
from main import main as pipeline_main
from saved_run import checksum, load_saved_run, read_json
from utils import write_json


class VideoOrder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.stack = []
        self.orders = {}

    def handle_starttag(self, tag, attrs):
        if tag != "section":
            return
        attrs = dict(attrs)
        self.stack.append(self.current)
        classes = attrs.get("class", "").split()
        if "player" in classes:
            self.current = attrs["data-player-id"]
            self.orders[self.current] = []
        if "video" in classes and self.current and attrs.get("data-video-id"):
            self.orders[self.current].append(attrs["data-video-id"])

    def handle_endtag(self, tag):
        if tag == "section":
            self.current = self.stack.pop()


def identity_fields(group):
    return {key: value for key, value in group.items() if key not in {"videos", "video_ids", "video_rankings"}}


class SortTests(unittest.TestCase):
    add = test_collection_add.AddTests.add
    mocked_add = test_collection_add.AddTests.mocked_add
    fake_analysis = test_collection_add.AddTests.fake_analysis
    fake_embeddings = test_collection_add.AddTests.fake_embeddings

    def setUp(self):
        test_collection_add.AddTests.setUp(self)
        index = read_json(self.root / "global/video_index.json")
        for entry in index:
            total = 100 if entry["video_id"] == "V_A" else 10
            path = self.root / "videos" / entry["video_id"] / "local_tracks.json"
            payload = read_json(path)
            payload["video"].update(frame_count=total, duration_sec=total / 30)
            entry.update(frame_count=total, duration_sec=total / 30)
            write_json(path, payload)
        write_json(self.root / "global/video_index.json", index)
        (self.inputs / "new.mp4").write_bytes(b"new")
        self.assertEqual(self.mocked_add()[0], 0)
        groups = read_json(self.root / "global/player_groups.json")
        for group in groups:
            group.pop("video_rankings")
            group["video_ids"].sort()
            group["videos"].sort()
            path = self.root / "players" / group["player_id"] / "metadata.json"
            payload = read_json(path)
            payload.pop("video_rankings")
            payload["video_ids"].sort()
            payload["videos"].sort()
            write_json(path, payload)
        write_json(self.root / "global/player_groups.json", groups)

    def sort(self, *args):
        with patch("reid_embedding._load_extractor", side_effect=AssertionError("ReID inference called")), \
             patch("yoloe_player_detection.YOLOEPlayerDetector.load", side_effect=AssertionError("YOLOE called")), \
             patch("clustering.cluster_embeddings", side_effect=AssertionError("Reclustered")), \
             patch("video_io.iter_video_frames", side_effect=AssertionError("Decoded input video")):
            return pipeline_main(["sort", "--run_dir", str(self.root), *args])

    def protected(self):
        return {p.relative_to(self.root).as_posix(): checksum(p) for p in self.root.rglob("*") if p.is_file()
                and ".updates" not in p.relative_to(self.root).parts
                and (p.relative_to(self.root).parts[0] == "videos" or p.suffix == ".jpg"
                     or p.suffix in {".npy", ".csv"} or p.name in {"local_track_index.json", "track_cache_index.json"})}

    def test_sort_preserves_identity_quality_vectors_and_images(self):
        before = read_json(self.root / "global/player_groups.json")
        protected = self.protected()
        self.assertEqual(self.sort(), 0)
        after = read_json(self.root / "global/player_groups.json")
        self.assertEqual([identity_fields(g) for g in before], [identity_fields(g) for g in after])
        self.assertEqual(protected, self.protected())
        group = next(g for g in after if "V_A_track_0001" in g["local_tracks"])
        self.assertEqual([r["appearance_percentage"] for r in group["video_rankings"]], [100., 1.])
        self.assertEqual(group["video_ids"][-1], "V_A")
        for group in after:
            metadata = read_json(self.root / "players" / group["player_id"] / "metadata.json")
            self.assertEqual(metadata["video_rankings"], group["video_rankings"])
        load_saved_run(self.root)

    def test_galleries_follow_ranked_video_order(self):
        self.assertEqual(self.sort(), 0)
        groups = read_json(self.root / "global/player_groups.json")
        expected = {g["player_id"]: g["video_ids"] for g in groups}
        for name in ("gallery.html", "gallery_standalone.html"):
            parser = VideoOrder()
            content = (self.root / "global" / name).read_text()
            parser.feed(content)
            self.assertEqual(parser.orders, expected)
            self.assertIn("100.0% (1 of 1 frames)", content)
        self.assertIn("data:image/jpeg;base64,", (self.root / "global/gallery_standalone.html").read_text())

    def test_dry_run_does_not_publish(self):
        before = {p: checksum(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.sort("--dry_run"), 0)
        self.assertEqual(before, {p: checksum(p) for p in before})

    def test_repeated_sort_is_deterministic(self):
        self.assertEqual(self.sort(), 0)
        groups = read_json(self.root / "global/player_groups.json")
        self.assertEqual(self.sort(), 0)
        self.assertEqual(read_json(self.root / "global/player_groups.json"), groups)

    def test_sort_after_delete_preserves_exclusions(self):
        from collection_delete import main as delete_main
        video_id = load_saved_run(self.root).videos[-1].video_id
        self.assertEqual(delete_main(["delete", "--run_dir", str(self.root), "--video_ids", video_id]), 0)
        updated = read_json(self.root / "global/player_groups.json")
        player = next(g for g in updated if "V_A_track_0001" in g["local_tracks"])
        self.assertEqual([r["appearance_percentage"] for r in player["video_rankings"]], [1.])
        before = self.protected()
        self.assertEqual(self.sort(), 0)
        self.assertEqual(before, self.protected())
        run = load_saved_run(self.root)
        self.assertNotIn(video_id, run.complete_cache.active_video_ids)
        self.assertTrue(all(video_id not in g["video_ids"] for g in read_json(self.root / "global/player_groups.json")))

    def test_sort_empty_active_collection(self):
        from collection_delete import main as delete_main
        ids = [v.video_id for v in load_saved_run(self.root).videos]
        self.assertEqual(delete_main(["delete", "--run_dir", str(self.root), "--video_ids", *ids]), 0)
        self.assertEqual(self.sort(), 0)
        self.assertEqual(load_saved_run(self.root).records, [])

    def test_publication_failure_rolls_back(self):
        before = {p.relative_to(self.root).as_posix(): checksum(p) for p in self.root.rglob("*")
                  if p.is_file() and ".updates" not in p.relative_to(self.root).parts}
        original_rename = Path.rename
        def fail(path, target):
            if path.name == "logs.json" and path.parent.name == "stage":
                raise OSError("injected failure")
            return original_rename(path, target)
        with patch.object(Path, "rename", fail):
            self.assertEqual(self.sort(), 2)
        self.assertEqual(before, {relative: checksum(self.root / relative) for relative in before})
        load_saved_run(self.root)

    def test_missing_album_metadata_blocks_publication(self):
        path = next((self.root / "players").glob("*/metadata.json"))
        path.unlink()
        groups = checksum(self.root / "global/player_groups.json")
        self.assertEqual(self.sort(), 2)
        self.assertEqual(checksum(self.root / "global/player_groups.json"), groups)

    def test_bad_group_membership_is_rejected(self):
        groups = read_json(self.root / "global/player_groups.json")
        groups[0]["local_tracks"].append("missing_track")
        write_json(self.root / "global/player_groups.json", groups)
        self.assertEqual(self.sort(), 2)

    def test_no_source_videos_or_weights_required(self):
        logs = read_json(self.root / "logs.json")
        for path in [Path(logs["yoloe_model"]), Path(logs["reid_model_path"]), *self.inputs.glob("*.mp4"),
                     *Path(self.temp.name).glob("*.mp4")]:
            path.unlink()
        self.assertEqual(self.sort(), 0)

    def test_sort_respects_lock(self):
        with collection_lock(self.root):
            self.assertEqual(self.sort(), 2)

    def test_add_recomputes_rankings(self):
        self.assertEqual(self.sort(), 0)
        (self.inputs / "second.mp4").write_bytes(b"second")
        self.assertEqual(self.mocked_add()[0], 0)
        player = next(g for g in read_json(self.root / "global/player_groups.json") if "V_A_track_0001" in g["local_tracks"])
        self.assertEqual([r["appearance_percentage"] for r in player["video_rankings"]], [100., 100., 1.])

    def test_recluster_produces_valid_rankings(self):
        from collection_cli import main as collection_main
        self.assertEqual(collection_main(["recluster", "--run_dir", str(self.root), "--output_dir", str(self.output)]), 0)
        groups = read_json(self.output / "global/player_groups.json")
        self.assertTrue(all(r["coverage_status"] == "valid" for g in groups for r in g["video_rankings"]))


if __name__ == "__main__":
    unittest.main()
