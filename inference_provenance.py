"""Fingerprint the weights and feature extraction implementation used for inference."""
import inspect
from importlib.metadata import version
from pathlib import Path

from config import PipelineConfig
from saved_run import checksum


def model_signature(config: PipelineConfig) -> dict[str, str]:
    import torchreid
    from torchreid.utils import FeatureExtractor

    return {"reid_model": config.reid.reid_model,
            "reid_sha256": checksum(config.reid.reid_model_path),
            "detector_sha256": checksum(config.detector.yoloe_model),
            "embedding_recipe": "feature-extractor-defaults-l2-mean-l2-v1",
            "torchreid_version": getattr(torchreid, "__version__", "unknown"),
            "extractor_sha256": checksum(Path(inspect.getfile(FeatureExtractor))),
            "torchvision_version": version("torchvision")}
