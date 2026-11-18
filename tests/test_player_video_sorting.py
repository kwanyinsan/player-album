import unittest
from pathlib import Path
from types import SimpleNamespace

from config import FilterConfig
from player_tracking import LocalTrack, _process_video_tracking_once
from player_video_sorting import compact_frame_ranges, merge_frame_ranges, rank_player_videos
from video_io import VideoInfo


def video(key, total, filename=None, **kwargs):
    return VideoInfo(key, filename or key + ".mp4", Path(key + ".mp4"), 640, 480, 30,
                     total, total / 30, **kwargs)


def track(key, video_id, ranges):
    merged = merge_frame_ranges(ranges)
    return LocalTrack(key, video_id, 1, video_id + ".mp4",
                      frame_count=sum(b - a + 1 for a, b in merged), observed_frame_ranges=ranges)


class RankingTests(unittest.TestCase):
    def rank(self, tracks, videos):
        return rank_player_videos([t.local_player_id for t in tracks],
                                  {t.local_player_id: t for t in tracks}, {v.video_id: v for v in videos})

    def test_compaction_and_overlap_merge(self):
        self.assertEqual(compact_frame_ranges([3, 1, 2, 2, 8]), [[1, 3], [8, 8]])
        self.assertEqual(merge_frame_ranges([[10, 20], [0, 12], [21, 22], [30, 30]]), [[0, 22], [30, 30]])
        with self.assertRaises(ValueError):
            compact_frame_ranges([1, True])

    def test_overlap_is_counted_once(self):
        rows = self.rank([track("a", "V_A", [[0, 99]]), track("b", "V_A", [[80, 179]])], [video("V_A", 300)])
        self.assertEqual(rows[0]["visible_frames"], 180)
        self.assertEqual(rows[0]["appearance_percentage"], 60.)

    def test_gaps_are_not_counted(self):
        row = self.rank([track("a", "V_A", [[0, 9], [20, 29]])], [video("V_A", 100)])[0]
        self.assertEqual(row["visible_frames"], 20)

    def test_percentage_overrides_raw_count(self):
        rows = self.rank([track("a", "V_A", [[0, 299]]), track("b", "V_B", [[0, 5]])],
                         [video("V_A", 1000), video("V_B", 10)])
        self.assertEqual([r["video_id"] for r in rows], ["V_B", "V_A"])

    def test_equal_ratios_use_count_then_video_id(self):
        rows = self.rank([track("a", "V_A", [[0, 1]]), track("b", "V_B", [[0, 2]]), track("c", "V_C", [[0, 1]])],
                         [video("V_A", 4), video("V_B", 6), video("V_C", 4)])
        self.assertEqual([r["video_id"] for r in rows], ["V_B", "V_A", "V_C"])

    def test_ranking_uses_unrounded_ratio(self):
        rows = self.rank([track("a", "V_A", [[0, 4998]]), track("z", "V_Z", [[0, 4999]])],
                         [video("V_A", 10000), video("V_Z", 10001)])
        self.assertEqual([r["video_id"] for r in rows], ["V_Z", "V_A"])
        self.assertEqual(f"{rows[0]['appearance_percentage']:.1f}", f"{rows[1]['appearance_percentage']:.1f}")

    def test_duplicate_filenames_remain_separate_entries(self):
        rows = self.rank([track("a", "V_A", [[0, 0]]), track("b", "V_B", [[0, 0]])],
                         [video("V_A", 10, "same.mp4"), video("V_B", 20, "same.mp4")])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["original_filename"] for r in rows], ["same.mp4", "same.mp4"])

    def test_invalid_counts_sort_last(self):
        rows = self.rank([track("a", "V_A", [[0, 0]]), track("b", "V_B", [])],
                         [video("V_A", 0), video("V_B", 100)])
        self.assertEqual(rows[0]["appearance_percentage"], 0)
        self.assertIsNone(rows[1]["appearance_percentage"])

    def test_out_of_range_frames_are_unavailable(self):
        row = self.rank([track("a", "V_A", [[0, 100]])], [video("V_A", 100)])[0]
        self.assertIsNone(row["appearance_percentage"])
        self.assertEqual(row["coverage_status"], "out_of_range_frames")

    def test_processed_count_is_recorded_and_cross_checked(self):
        row = self.rank([track("a", "V_A", [[0, 9]])],
                        [video("V_A", 100, processed_frame_count=100, analysis_complete=True)])[0]
        self.assertEqual(row["frame_count_source"], "processed")
        self.assertEqual(row["appearance_percentage"], 10)
        row = self.rank([track("a", "V_A", [[0, 9]])],
                        [video("V_A", 100, processed_frame_count=90, analysis_complete=True)])[0]
        self.assertEqual(row["coverage_status"], "frame_count_mismatch")
        self.assertIsNone(row["appearance_percentage"])

    def test_processed_count_can_supply_missing_metadata(self):
        row = self.rank([track("a", "V_A", [[0, 9]])],
                        [video("V_A", 0, processed_frame_count=100, analysis_complete=True)])[0]
        self.assertEqual(row["appearance_percentage"], 10)

    def test_incomplete_analysis_is_unavailable(self):
        row = self.rank([track("a", "V_A", [[0, 9]])],
                        [video("V_A", 100, processed_frame_count=10, analysis_complete=False)])[0]
        self.assertEqual(row["coverage_status"], "incomplete_analysis")

    def test_missing_observations_do_not_use_track_span(self):
        local = LocalTrack("a", "V_A", 1, "a.mp4", start_frame=0, end_frame=99, frame_count=20)
        row = self.rank([local], [video("V_A", 100)])[0]
        self.assertEqual(row["coverage_status"], "missing_frame_observations")
        self.assertIsNone(row["visible_frames"])

    def test_invalid_or_inconsistent_ranges_are_unavailable(self):
        for ranges in ([[2, 1]], [[-1, 2]], [None]):
            local = LocalTrack("a", "V_A", 1, "a.mp4", frame_count=1, observed_frame_ranges=ranges)
            self.assertEqual(self.rank([local], [video("V_A", 100)])[0]["coverage_status"], "invalid_frame_observations")
        local = LocalTrack("a", "V_A", 1, "a.mp4", frame_count=1, observed_frame_ranges=[[0, 9]])
        self.assertEqual(self.rank([local], [video("V_A", 100)])[0]["coverage_status"], "inconsistent_frame_observations")

    def test_missing_metadata_and_empty_groups(self):
        row = self.rank([track("a", "V_A", [[0, 0]])], [])[0]
        self.assertEqual(row["coverage_status"], "missing_video_metadata")
        self.assertEqual(self.rank([], []), [])

    def test_tracking_counts_frames_without_detections(self):
        info = video("V_A", 3)
        detector = SimpleNamespace(config=SimpleNamespace(prompt="person"),
                                   track_video=lambda path: iter([SimpleNamespace(boxes=None)] * 3))
        self.assertEqual(_process_video_tracking_once(info, detector, FilterConfig(), 96), [])
        self.assertEqual(info.processed_frame_count, 3)
        self.assertTrue(info.analysis_complete)


if __name__ == "__main__":
    unittest.main()
