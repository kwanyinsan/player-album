from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import ReidConfig
from player_tracking import LocalTrack
from utils import ensure_dir, normalize_vector, write_json


@dataclass
class EmbeddingRecord:
    local_player_id: str
    video_id: str
    source_video_name: str
    start_frame: int
    end_frame: int
    frame_count: int
    crop_paths: list[str]
    representative_crop: str | None
    embedding_row: int
    crop_embedding_count: int = 0
    embedding_consistency_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_player_id": self.local_player_id,
            "video_id": self.video_id,
            "source_video_name": self.source_video_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "crop_paths": self.crop_paths,
            "representative_crop": self.representative_crop,
            "embedding_row": self.embedding_row,
            "crop_embedding_count": self.crop_embedding_count,
            "embedding_consistency_score": self.embedding_consistency_score,
        }


@dataclass
class EmbeddingOutput:
    embeddings: np.ndarray
    records: list[EmbeddingRecord]
    errors: list[dict[str, str]]
    device_used: str


def extract_track_embeddings(
    tracks: list[LocalTrack],
    output_root: Path,
    global_dir: Path,
    reid_config: ReidConfig,
    requested_device: str,
) -> EmbeddingOutput:
    ensure_dir(global_dir)
    eligible_tracks = [track for track in tracks if track.crop_paths]
    errors: list[dict[str, str]] = []
    if not eligible_tracks:
        embeddings = np.zeros((0, 0), dtype=np.float32)
        records: list[EmbeddingRecord] = []
        _save_embedding_outputs(global_dir, embeddings, records)
        return EmbeddingOutput(embeddings=embeddings, records=records, errors=errors, device_used="none")

    extractor, device_used = _load_extractor(reid_config, requested_device)
    rows: list[np.ndarray] = []
    records = []

    for track in eligible_tracks:
        crop_embeddings: list[np.ndarray] = []
        crop_abs_paths = [str(output_root / p) for p in track.crop_paths]
        
        if crop_abs_paths:
            try:
                # Batch extract all crops for this track
                features = extractor(crop_abs_paths)
                
                # Convert the features tensor/array to numpy
                if hasattr(features, "detach"):
                    features = features.detach()
                if hasattr(features, "cpu"):
                    features = features.cpu()
                if hasattr(features, "numpy"):
                    features_array = features.numpy()
                else:
                    features_array = np.asarray(features)
                    
                features_array = np.asarray(features_array, dtype=np.float32)
                
                # If only 1 crop, it might return 1D or 2D (1xN). Ensure 2D.
                if features_array.ndim == 1:
                    features_array = features_array.reshape(1, -1)
                elif features_array.ndim > 2:
                    features_array = features_array.reshape(features_array.shape[0], -1)
                    
                for i in range(features_array.shape[0]):
                    crop_embeddings.append(normalize_vector(features_array[i]))
            except Exception as exc:
                errors.append(
                    {
                        "local_player_id": track.local_player_id,
                        "crop_path": "batch_extraction",
                        "error": str(exc),
                    }
                )

        if not crop_embeddings:
            errors.append(
                {
                    "local_player_id": track.local_player_id,
                    "crop_path": "",
                    "error": "No valid crop embeddings produced for this track.",
                }
            )
            continue

        crop_stack = np.vstack(crop_embeddings)
        track_embedding = normalize_vector(np.mean(crop_stack, axis=0))
        consistency_score = _embedding_consistency(crop_stack)
        row_index = len(rows)
        rows.append(track_embedding)
        records.append(
            EmbeddingRecord(
                local_player_id=track.local_player_id,
                video_id=track.video_id,
                source_video_name=track.source_video_name,
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                frame_count=track.frame_count,
                crop_paths=list(track.crop_paths),
                representative_crop=track.representative_crop,
                embedding_row=row_index,
                crop_embedding_count=len(crop_embeddings),
                embedding_consistency_score=consistency_score,
            )
        )

    embeddings = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    _save_embedding_outputs(global_dir, embeddings, records)
    return EmbeddingOutput(embeddings=embeddings, records=records, errors=errors, device_used=device_used)


def _load_extractor(reid_config: ReidConfig, requested_device: str):
    try:
        import torch
        from torchreid.utils import FeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "Torchreid and PyTorch are required for ReID embedding extraction. "
            "Install torchreid as a dependency; do not copy its source into this project."
        ) from exc

    device = _resolve_torch_device(torch, requested_device)
    extractor = FeatureExtractor(
        model_name=reid_config.reid_model,
        model_path=str(reid_config.reid_model_path or ""),
        device=device,
    )
    return extractor, device


def _resolve_torch_device(torch_module: object, requested_device: str) -> str:
    requested = str(requested_device).strip().lower()
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    if requested == "cpu" or not cuda_available:
        return "cpu"
    if requested.startswith("cuda"):
        return requested
    if requested.isdigit():
        return f"cuda:{requested}"
    return "cuda"


def _feature_to_vector(features: object) -> np.ndarray:
    if isinstance(features, (list, tuple)):
        features = features[0]
    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "cpu"):
        features = features.cpu()
    if hasattr(features, "numpy"):
        array = features.numpy()
    else:
        array = np.asarray(features)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("ReID extractor returned a scalar feature.")
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1)[0]


def _embedding_consistency(crop_embeddings: np.ndarray) -> float:
    if len(crop_embeddings) <= 1:
        return 1.0
    similarities: list[float] = []
    for left_index, left in enumerate(crop_embeddings):
        for right in crop_embeddings[left_index + 1 :]:
            similarities.append(float(np.dot(left, right)))
    if not similarities:
        return 1.0
    mean_similarity = float(np.mean(similarities))
    return float(np.clip((mean_similarity - 0.55) / 0.40, 0.0, 1.0))


def _save_embedding_outputs(global_dir: Path, embeddings: np.ndarray, records: list[EmbeddingRecord]) -> None:
    ensure_dir(global_dir)
    np.save(global_dir / "local_track_embeddings.npy", embeddings)
    write_json(global_dir / "local_track_index.json", [record.to_dict() for record in records])
