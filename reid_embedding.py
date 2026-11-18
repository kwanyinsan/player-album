from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import ReidConfig
from artifact_paths import artifact_path
from player_tracking import LocalTrack
from utils import ensure_dir, write_json


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
    track_statuses: dict[str, dict[str, Any]] = field(default_factory=dict)


class EmbeddingDimensionError(ValueError):
    """An extractor returned vectors incompatible with the collection."""


def validated_vector(vector: np.ndarray, expected_dimension: int | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not vector.size:
        raise ValueError("Expected a nonempty one-dimensional embedding")
    if expected_dimension is not None and vector.size != expected_dimension:
        raise EmbeddingDimensionError(f"Embedding dimension {vector.size} differs from expected {expected_dimension}")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding contains nonfinite values or has zero norm")
    return vector / norm


def extract_track_embeddings(
    tracks: list[LocalTrack],
    output_root: Path,
    global_dir: Path,
    reid_config: ReidConfig,
    requested_device: str,
    source_root: Path | None = None,
    expected_dimension: int | None = None,
) -> EmbeddingOutput:
    ensure_dir(global_dir)
    eligible_tracks = [track for track in tracks if track.crop_paths]
    errors: list[dict[str, str]] = []
    statuses = {track.local_player_id: {"status": "no_usable_crops", "attempt_count": 0,
                "embedding_row": None, "failed_stage": None, "last_error": None} for track in tracks}
    if not eligible_tracks:
        embeddings = np.zeros((0, 0), dtype=np.float32)
        records: list[EmbeddingRecord] = []
        _save_embedding_outputs(global_dir, embeddings, records)
        return EmbeddingOutput(embeddings, records, errors, "none", statuses)

    try:
        extractor, device_used = _load_extractor(reid_config, requested_device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        for track in eligible_tracks:
            error = {"local_player_id": track.local_player_id, "crop_path": "", "error": str(exc)}
            errors.append(error)
            statuses[track.local_player_id] = {"status": "retryable_error", "attempt_count": 1,
                "embedding_row": None, "failed_stage": "reid_initialization", "last_error": str(exc)}
        embeddings = np.zeros((0, 0), dtype=np.float32)
        _save_embedding_outputs(global_dir, embeddings, [])
        return EmbeddingOutput(embeddings, [], errors, "unavailable", statuses)
    rows: list[np.ndarray] = []
    records = []

    for track in eligible_tracks:
        crop_embeddings: list[np.ndarray] = []
        track_errors = []
        for crop_rel_path in track.crop_paths:
            try:
                crop_path = artifact_path(output_root, crop_rel_path, source_root)
                features = extractor([str(crop_path)])
                vector = validated_vector(_feature_to_vector(features), expected_dimension)
                if expected_dimension is None:
                    expected_dimension = vector.size
                crop_embeddings.append(vector)
            except EmbeddingDimensionError:
                raise
            except Exception as exc:
                error = (
                    {
                        "local_player_id": track.local_player_id,
                        "crop_path": crop_rel_path,
                        "error": str(exc),
                        "status": "blocked_error" if isinstance(exc, OSError) or "Missing or external artifact" in str(exc)
                                  else "retryable_error",
                    }
                )
                errors.append(error)
                track_errors.append(error)

        status = {"status": "retryable_error", "attempt_count": 1, "embedding_row": None,
                  "failed_stage": "reid_crop", "last_error": track_errors[-1]["error"] if track_errors else None,
                  "crop_errors": track_errors}
        statuses[track.local_player_id] = status

        if not crop_embeddings:
            if track_errors and all(e["status"] == "blocked_error" for e in track_errors):
                status["status"] = "blocked_error"
            errors.append(
                {
                    "local_player_id": track.local_player_id,
                    "crop_path": "",
                    "error": "No valid crop embeddings produced for this track.",
                }
            )
            continue

        crop_stack = np.vstack(crop_embeddings)
        try:
            track_embedding = validated_vector(np.mean(crop_stack, axis=0), expected_dimension)
        except ValueError as exc:
            status.update(failed_stage="reid_average", last_error=str(exc))
            errors.append({"local_player_id": track.local_player_id, "crop_path": "", "error": str(exc)})
            continue
        consistency_score = _embedding_consistency(crop_stack)
        row_index = len(rows)
        status.update(status="complete", embedding_row=row_index, failed_stage=None, last_error=None)
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
    return EmbeddingOutput(embeddings, records, errors, device_used, statuses)


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
    if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[0] != 1):
        raise ValueError("ReID extractor must return one vector for one crop")
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
