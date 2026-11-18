"""Complete analysis cache and explicit active views for collection operations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from embedding_state import track_states
from player_tracking import LocalTrack
from reid_embedding import EmbeddingRecord
from utils import write_json
from video_io import VideoInfo


@dataclass
class AnalysisView:
    videos: list[VideoInfo]
    tracks: dict[str, LocalTrack]
    records: list[EmbeddingRecord]
    embeddings: np.ndarray


@dataclass
class CollectionCache(AnalysisView):
    active_video_ids: set[str]

    def active_view(self) -> AnalysisView:
        selected = [r for r in self.records if r.video_id in self.active_video_ids]
        rows = [r.embedding_row for r in selected]
        return AnalysisView(
            [v for v in self.videos if v.video_id in self.active_video_ids],
            {key: t for key, t in self.tracks.items() if t.video_id in self.active_video_ids},
            [replace(r, embedding_row=i) for i, r in enumerate(selected)],
            self.embeddings[rows].copy(),
        )


def cache_statuses(cache: CollectionCache, previous: dict | None = None) -> dict:
    states = track_states(cache.tracks, cache.records, previous)
    active_rows = {r.local_player_id: r.embedding_row for r in cache.active_view().records}
    cache_rows = {r.local_player_id: r.embedding_row for r in cache.records}
    for key, entry in states.items():
        entry["cache_row"] = cache_rows.get(key)
        entry["embedding_row"] = active_rows.get(key)
    return states


def write_cache_outputs(root: Path, cache: CollectionCache, state: dict[str, Any]) -> dict[str, Any]:
    """Write small embedding arrays and manifests; video analysis stays in place."""
    directory = root / "global"
    directory.mkdir(parents=True, exist_ok=True)
    active = cache.active_view()
    np.save(directory / "local_track_embeddings.npy", active.embeddings)
    write_json(directory / "local_track_index.json", [r.to_dict() for r in active.records])
    np.save(directory / "track_cache_embeddings.npy", cache.embeddings)
    cached_records = []
    for record in cache.records:
        raw = record.to_dict()
        raw["cache_row"] = raw.pop("embedding_row")
        cached_records.append(raw)
    write_json(directory / "track_cache_index.json", cached_records)
    return {**state, "cache_format_version": 1,
            "video_membership": {v.video_id: v.video_id in cache.active_video_ids for v in cache.videos},
            "embedding_status": cache_statuses(cache, state.get("embedding_status"))}
