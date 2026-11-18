"""Read-only validation of saved analysis and a portable collection registry.

Frame arrays remain on disk: a full run can contain gigabytes of polygons.
Call load_video_tracks with include_frames=True when frame data is needed.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import ClusteringConfig, ExportConfig
from artifact_paths import artifact_path
from embedding_state import validate_track_states
from collection_cache import AnalysisView, CollectionCache
from player_tracking import LocalTrack, TrackFrame
from reid_embedding import EmbeddingRecord
from video_io import VideoInfo
from player_video_sorting import compact_frame_ranges


SCHEMA_VERSION = 2


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def construct(cls: type, payload: dict[str, Any]) -> Any:
    return cls(**{f.name: payload[f.name] for f in fields(cls) if f.name in payload})


def load_video_tracks(path: Path, include_frames: bool = False) -> tuple[VideoInfo, list[LocalTrack]]:
    payload = read_json(path)
    video = construct(VideoInfo, {**payload["video"], "path": Path(payload["video"]["path"])})
    if video.video_id != path.parent.name:
        raise ValueError(f"Video ID does not match its folder: {path}")
    if payload.get("error"):
        raise ValueError(f"Saved video analysis failed: {path}: {payload['error']}")
    tracks = []
    for raw in payload["tracks"]:
        raw_frames = raw["frames"]
        track = construct(LocalTrack, {**raw, "frames": []})
        indices = [frame["frame_index"] for frame in raw_frames]
        if (track.video_id != video.video_id or track.source_video_name != video.filename
                or track.frame_count != len(indices) or indices != sorted(set(indices))
                or any(index < 0 for index in indices)
                or (indices and (track.start_frame != indices[0] or track.end_frame != indices[-1]))
                or any(frame["local_track_id"] != track.local_player_id for frame in raw_frames)):
            raise ValueError(f"Inconsistent frame or video mapping: {track.local_player_id}")
        observed = compact_frame_ranges(indices)
        if track.observed_frame_ranges is not None and track.observed_frame_ranges != observed:
            raise ValueError(f"Stored frame ranges disagree with observations: {track.local_player_id}")
        track.observed_frame_ranges = observed
        if include_frames:
            track.frames = [construct(TrackFrame, frame) for frame in raw_frames]
        tracks.append(track)
    return video, tracks


@dataclass
class SavedRun:
    root: Path
    videos: list[VideoInfo]
    tracks: dict[str, LocalTrack]
    records: list[EmbeddingRecord]
    embeddings: np.ndarray
    logs: dict[str, Any]
    artifacts: dict[str, str]
    warnings: list[str]
    provenance_verified: bool = False
    cache: CollectionCache | None = None

    @property
    def complete_cache(self) -> CollectionCache:
        return self.cache or CollectionCache(self.videos, self.tracks, self.records, self.embeddings,
                                              {v.video_id for v in self.videos})

    def report(self) -> dict[str, Any]:
        return {
            "videos": len(self.videos), "local_tracks": len(self.tracks),
            "embedded_tracks": len(self.records), "embedding_shape": list(self.embeddings.shape),
            "registered_videos": len(self.complete_cache.videos),
            "excluded_videos": len(self.complete_cache.videos) - len(self.videos),
            "cached_embeddings": len(self.complete_cache.records),
            "validated_artifacts": len(self.artifacts), "warnings": self.warnings,
            "can_recluster": True,
            "inference_cache_provenance": "verified" if self.provenance_verified else "unverified",
            "inference_cache_reuse_verified": self.provenance_verified,
        }


def load_saved_run(root: Path, progress: Callable[[str], None] | None = None,
                   source_root: Path | None = None) -> SavedRun:
    root = root.resolve()
    if (root / ".collection-update.json").exists():
        raise ValueError("Interrupted update: run python main.py recover --run_dir <collection> first")
    artifacts: dict[str, str] = {}

    def register(relative: str) -> Path:
        path = artifact_path(root, relative, source_root)
        normalized = Path(relative.replace("\\", "/")).as_posix()
        if normalized not in artifacts:
            artifacts[normalized] = checksum(path)
        return path

    logs = read_json(register("logs.json"))
    state = {}
    if (root / "global/collection_state.json").exists():
        state = read_json(register("global/collection_state.json"))
    video_index = read_json(register("global/video_index.json"))
    if len({v["video_id"] for v in video_index}) != len(video_index):
        raise ValueError("Duplicate video IDs in video_index.json")
    entries = {entry["video_id"]: entry for entry in video_index}
    has_cache = "cache_format_version" in state
    membership = state.get("video_membership") if has_cache else {key: True for key in entries}
    if (has_cache and state["cache_format_version"] != 1) or not isinstance(membership, dict):
        raise ValueError("Unsupported complete cache or missing video membership")
    if any(type(active) is not bool for active in membership.values()):
        raise ValueError("Video membership must contain boolean active flags")
    if {key for key, active in membership.items() if active} != set(entries):
        raise ValueError("Active membership differs from video_index.json")
    videos, tracks = [], {}
    for position, video_id in enumerate(membership, start=1):
        video, local_tracks = load_video_tracks(register(f"videos/{video_id}/local_tracks.json"))
        entry = entries.get(video_id)
        if entry and (entry["original_filename"] != video.filename
                or any(entry.get(key) != getattr(video, key) for key in
                       ("width", "height", "fps", "frame_count", "duration_sec", "processed_frame_count", "analysis_complete"))):
            raise ValueError(f"Video metadata mismatch: {video_id}")
        videos.append(video)
        for track in local_tracks:
            if track.local_player_id in tracks:
                raise ValueError(f"Duplicate local track: {track.local_player_id}")
            tracks[track.local_player_id] = track
            for relative in [*track.crop_paths, track.representative_crop]:
                if relative:
                    register(relative)
            if track.crop_paths:
                metadata = f"videos/{video_id}/crops/{track.local_player_id}/metadata.json"
                if (root / metadata).is_file() or (source_root and (source_root / metadata).is_file()):
                    crop_metadata = read_json(register(metadata))
                    for crop in crop_metadata.get("crops", []):
                        for key in ("path", "masked_path"):
                            if crop.get(key):
                                register(crop[key])
        if progress and (position % 25 == 0 or position == len(membership)):
            progress(f"Validated {position}/{len(membership)} registered videos")

    records, embeddings = _load_embeddings(register, "local_track", "embedding_row", tracks)
    if has_cache:
        cache_records, cache_embeddings = _load_embeddings(register, "track_cache", "cache_row", tracks)
    else:
        cache_records, cache_embeddings = records, embeddings
    cache = CollectionCache(videos, tracks, cache_records, cache_embeddings,
                            {key for key, active in membership.items() if active})
    active = cache.active_view()
    if (active.records != records or active.embeddings.dtype != embeddings.dtype
            or not np.array_equal(active.embeddings, embeddings)):
        raise ValueError("Active embedding rows or values differ from the complete cache")
    if "embedding_status" in state:
        validate_track_states(state["embedding_status"], tracks, records, cache_records if has_cache else None)

    if (root / "project.db").exists():
        validate_registry(root, artifacts, active, cache)
    provenance_verified = bool(state.get("inference_signature") or logs.get("inference_signature")) and not state.get("legacy_provenance_unverified", True)
    warnings = [] if provenance_verified else [
        "Saved embeddings can be reclustered, but original model checksums and stage provenance "
        "are not verified. Do not use this import to claim compatibility with new inference settings.",
    ]
    if logs.get("errors"):
        warnings.append(f"Source logs contain {len(logs['errors'])} error(s); inspect logs before relying on completeness.")
    if "clustering_config" not in logs:
        warnings.append("Legacy logs omit some clustering settings; missing values use current defaults. "
                        "The resolved settings will be recorded in the new logs.json.")
    for key, count in (("number_of_videos", len(active.videos)), ("number_of_local_tracks", len(active.tracks)),
                       ("number_of_registered_videos", len(videos))):
        if logs.get(key) is not None and logs[key] != count:
            raise ValueError(f"{key} differs from logs.json")
    embedded_ids = {r.local_player_id for r in records}
    missing = sum(bool(t.crop_paths) and key not in embedded_ids for key, t in active.tracks.items())
    if missing:
        warnings.append(f"{missing} active track(s) have crops but no embedding; excluded from clustering.")
    return SavedRun(root, active.videos, active.tracks, records, embeddings, logs, artifacts, warnings,
                    provenance_verified, cache)


def _load_embeddings(register, prefix: str, row_key: str, tracks: dict) -> tuple[list[EmbeddingRecord], np.ndarray]:
    raw_records = read_json(register(f"global/{prefix}_index.json"))
    embeddings = np.load(register(f"global/{prefix}_embeddings.npy"), allow_pickle=False)
    if (embeddings.ndim != 2 or embeddings.dtype.kind != "f"
            or len(embeddings) != len(raw_records) or not np.isfinite(embeddings).all()
            or (len(embeddings) and np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12))):
        raise ValueError("Invalid embedding array: expected finite, nonzero vectors matching the index")
    rows = [record[row_key] for record in raw_records]
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(embeddings))):
        raise ValueError("Embedding rows must be a unique complete permutation of array rows")
    records = [construct(EmbeddingRecord, {**raw, "embedding_row": raw[row_key]})
               for raw in sorted(raw_records, key=lambda r: r[row_key])]
    if len({r.local_player_id for r in records}) != len(records):
        raise ValueError("Duplicate embedding track IDs")
    for record in records:
        track = tracks.get(record.local_player_id)
        if track is None or any(getattr(track, key) != getattr(record, key) for key in (
            "video_id", "source_video_name", "start_frame", "end_frame", "frame_count",
            "crop_paths", "representative_crop",
        )) or not record.crop_paths:
            raise ValueError(f"Embedding-to-track mapping mismatch: {record.local_player_id}")

    return records, embeddings


def validate_registry(
    root: Path, artifacts: dict[str, str], active: AnalysisView, cache: CollectionCache,
) -> None:
    connection = sqlite3.connect((root / "project.db").as_uri() + "?mode=ro", uri=True)
    try:
        versions = connection.execute("SELECT schema_version FROM project").fetchall()
        if versions not in ([(1,)], [(SCHEMA_VERSION,)]):
            raise ValueError("Unsupported collection registry schema")
        membership = connection.execute("SELECT video_id, active FROM videos").fetchall()
        expected_membership = {(v.video_id, int(v.video_id in cache.active_video_ids)) for v in cache.videos}
        if set(membership) != expected_membership:
            raise ValueError("Registry video membership mismatch")
        rows = {r.local_player_id: r.embedding_row for r in active.records}
        cached_rows = {r.local_player_id: r.embedding_row for r in cache.records}
        expected_tracks = {(t.local_player_id, t.video_id, rows.get(t.local_player_id)) for t in cache.tracks.values()}
        query = "SELECT local_player_id, video_id, embedding_row FROM tracks"
        if versions == [(SCHEMA_VERSION,)]:
            expected_tracks = {(*entry, cached_rows.get(entry[0])) for entry in expected_tracks}
            query = "SELECT local_player_id, video_id, embedding_row, cache_row FROM tracks"
        elif cache.active_video_ids != {v.video_id for v in cache.videos}:
            raise ValueError("Schema 1 cannot represent excluded video membership")
        if set(connection.execute(query)) != expected_tracks:
            raise ValueError("Registry track or embedding membership mismatch")
        registered = dict(connection.execute("SELECT path, sha256 FROM artifacts"))
        if registered != artifacts:
            changed = sorted(key for key in registered.keys() | artifacts.keys()
                             if registered.get(key) != artifacts.get(key))
            raise ValueError(f"Registered artifacts have changed: {changed[:5]}")
    finally:
        connection.close()


def saved_settings(run: SavedRun) -> tuple[ClusteringConfig, ExportConfig]:
    clustering = dict(run.logs.get("clustering_config") or {})
    clustering.setdefault("cluster_method", run.logs.get("clustering_method", "agglomerative"))
    clustering.setdefault("distance_threshold", run.logs.get("distance_threshold", 0.18))
    legacy = run.logs.get("identity_cluster_resolution") or {}
    for old, new in (("enabled", "identity_split_enabled"), ("color_weight", "identity_color_weight"),
                     ("split_distance_threshold", "identity_split_distance_threshold"),
                     ("split_min_cluster_size", "identity_split_min_cluster_size")):
        if old in legacy:
            clustering.setdefault(new, legacy[old])
    export = {**(run.logs.get("export_config") or {}), "save_debug_video": False}
    return construct(ClusteringConfig, clustering), construct(ExportConfig, export)


def write_registry(root: Path, run: SavedRun, source_root: Path | None = None) -> None:
    """Create a staged registry, retaining inactive membership and complete cache rows."""
    path = root / "project.db"
    if path.exists():
        raise ValueError(f"Refusing to replace an existing registry: {path}")
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.executescript("""
                CREATE TABLE project (schema_version INTEGER, provenance TEXT, source_config TEXT);
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, active INTEGER NOT NULL CHECK(active IN (0,1)),
                    source_path TEXT, tracks_path TEXT NOT NULL, status TEXT NOT NULL);
                CREATE TABLE tracks (local_player_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
                    embedding_row INTEGER UNIQUE, cache_row INTEGER UNIQUE,
                    FOREIGN KEY(video_id) REFERENCES videos(video_id));
                CREATE TABLE artifacts (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            """)
            connection.execute("INSERT INTO project VALUES (?, ?, ?)",
                               (SCHEMA_VERSION, "verified" if run.provenance_verified else "unverified", json.dumps(run.logs)))
            cache = run.complete_cache
            for video in cache.videos:
                connection.execute("INSERT INTO videos VALUES (?, ?, ?, ?, 'analysed')", (
                    video.video_id, int(video.video_id in cache.active_video_ids),
                    video.path.as_posix(), f"videos/{video.video_id}/local_tracks.json",
                ))
            row_by_track = {r.local_player_id: r.embedding_row for r in run.records}
            cache_rows = {r.local_player_id: r.embedding_row for r in cache.records}
            connection.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?)", [
                (t.local_player_id, t.video_id, row_by_track.get(t.local_player_id), cache_rows.get(t.local_player_id))
                for t in cache.tracks.values()
            ])
            connection.executemany("INSERT INTO artifacts VALUES (?, ?)", [
                (relative, checksum(artifact_path(root, relative, source_root))) for relative in run.artifacts
            ])
    finally:
        connection.close()
