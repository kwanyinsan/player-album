import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import ReidConfig
from player_tracking import LocalTrack
from reid_embedding import EmbeddingDimensionError, extract_track_embeddings


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index in range(3):
            (self.root / f"crop{index}.jpg").write_bytes(b"fixture")
        self.track = LocalTrack("V_A_track_0001", "V_A", 1, "a.mp4",
                                crop_paths=["crop0.jpg", "crop1.jpg", "crop2.jpg"])

    def extract(self, outputs, dimension=2):
        iterator = iter(outputs)
        def extractor(paths):
            output = next(iterator)
            if isinstance(output, Exception):
                raise output
            return output
        with patch("reid_embedding._load_extractor", return_value=(extractor, "test")):
            return extract_track_embeddings([self.track], self.root, self.root / "global", ReidConfig(), "cpu",
                                            expected_dimension=dimension)

    def test_one_crop_fails_average_uses_only_successful_vectors(self):
        output = self.extract([RuntimeError("temporary"), [[1., 0.]], [[0., 1.]]])
        np.testing.assert_allclose(output.embeddings[0], [2**-.5, 2**-.5], atol=1e-6)
        self.assertEqual(output.records[0].crop_embedding_count, 2)
        self.assertEqual(len(output.errors), 1)
        self.assertEqual(output.track_statuses[self.track.local_player_id]["status"], "complete")

    def test_nonfinite_and_zero_vectors_are_never_saved(self):
        for invalid in (np.nan, np.inf, 0.):
            with self.subTest(invalid=invalid):
                output = self.extract([np.full((1, 2), invalid)] * 3)
                self.assertEqual(len(output.records), 0)
                self.assertEqual(output.track_statuses[self.track.local_player_id]["status"], "retryable_error")

    def test_dimension_mismatch_is_fatal(self):
        with self.assertRaises(EmbeddingDimensionError):
            self.extract([np.ones((1, 3))])

    def test_invalid_output_shape_is_rejected(self):
        output = self.extract([np.ones((2, 2)), np.array(1.), np.ones((1, 1, 2))])
        self.assertEqual(len(output.records), 0)

    def test_cancelled_mean_is_rejected(self):
        self.track.crop_paths = self.track.crop_paths[:2]
        output = self.extract([[[1., 0.]], [[-1., 0.]]])
        self.assertEqual(len(output.records), 0)
        self.assertEqual(output.track_statuses[self.track.local_player_id]["failed_stage"], "reid_average")

    def test_no_crops_does_not_load_model(self):
        self.track.crop_paths = []
        with patch("reid_embedding._load_extractor", side_effect=AssertionError("Model loaded")):
            result = extract_track_embeddings([self.track], self.root, self.root / "global", ReidConfig(), "cpu")
        self.assertEqual(result.track_statuses[self.track.local_player_id]["status"], "no_usable_crops")

    def test_missing_crops_are_blocked(self):
        self.track.crop_paths = ["missing.jpg"]
        output = self.extract([])
        self.assertEqual(output.track_statuses[self.track.local_player_id]["status"], "blocked_error")

    def test_initialization_out_of_memory_leaves_retryable_tracks(self):
        with patch("reid_embedding._load_extractor", side_effect=RuntimeError("CUDA out of memory")):
            output = extract_track_embeddings([self.track], self.root, self.root / "global", ReidConfig(), "0")
        self.assertEqual(len(output.records), 0)
        self.assertEqual(output.track_statuses[self.track.local_player_id]["failed_stage"], "reid_initialization")

    def test_invalid_checkpoint_initialization_is_fatal(self):
        with patch("reid_embedding._load_extractor", side_effect=RuntimeError("Invalid checkpoint")):
            with self.assertRaisesRegex(RuntimeError, "Invalid checkpoint"):
                extract_track_embeddings([self.track], self.root, self.root / "global", ReidConfig(), "cpu")


if __name__ == "__main__":
    unittest.main()
