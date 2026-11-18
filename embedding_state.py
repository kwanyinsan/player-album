"""Persist and validate the relationship between track jobs and embedding rows."""
from copy import deepcopy

STATES = {"complete", "pending", "retryable_error", "no_usable_crops", "blocked_error"}


def track_states(tracks, records, previous=None):
    rows = {r.local_player_id: r.embedding_row for r in records}
    states = {}
    for key, track in tracks.items():
        entry = deepcopy((previous or {}).get(key, {}))
        entry.setdefault("attempt_count", 0)
        entry.setdefault("failed_stage", None)
        entry.setdefault("last_error", None)
        if key in rows:
            entry.update(status="complete", embedding_row=rows[key], failed_stage=None, last_error=None)
        elif not track.crop_paths:
            entry.update(status="no_usable_crops", embedding_row=None)
        else:
            entry.setdefault("status", "pending")
            entry["embedding_row"] = None
        states[key] = entry
    return states


def validate_track_states(states, tracks, records, cache_records=None):
    rows = {r.local_player_id: r.embedding_row for r in records}
    cached_rows = {r.local_player_id: r.embedding_row for r in (records if cache_records is None else cache_records)}
    if set(states) != set(tracks):
        raise ValueError("Embedding status membership differs from saved tracks")
    for key, entry in states.items():
        status = entry.get("status")
        if (status not in STATES or type(entry.get("attempt_count")) is not int or entry["attempt_count"] < 0
                or (status == "complete") != (key in cached_rows)
                or entry.get("embedding_row") != rows.get(key)
                or (cache_records is not None and entry.get("cache_row") != cached_rows.get(key))
                or (status == "no_usable_crops" and tracks[key].crop_paths)
                or (status in {"pending", "retryable_error"} and not tracks[key].crop_paths)):
            raise ValueError(f"Invalid embedding status or row mapping: {key}")
