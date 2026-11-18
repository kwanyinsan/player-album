"""Rank a player's videos by the union of observed tracking frames."""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from player_tracking import LocalTrack
    from video_io import VideoInfo

SORTING_VERSION = 1


def compact_frame_ranges(indices: Iterable[int]) -> list[list[int]]:
    values = list(indices)
    if any(type(index) is not int or index < 0 for index in values):
        raise ValueError("Frame indices must be nonnegative integers")
    values = sorted(set(values))
    ranges: list[list[int]] = []
    for index in values:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ranges


def merge_frame_ranges(ranges: Iterable[list[int]]) -> list[list[int]]:
    values = list(ranges)
    if any(not isinstance(r, (tuple, list)) or len(r) != 2 or any(type(n) is not int for n in r)
           or r[0] < 0 or r[1] < r[0] for r in values):
        raise ValueError("Invalid inclusive frame range")
    merged: list[list[int]] = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _frame_total(video: VideoInfo | None) -> tuple[int | None, str, str]:
    if video is None:
        return None, "metadata", "missing_video_metadata"
    reported, processed = video.frame_count, video.processed_frame_count
    if processed is not None:
        if type(processed) is not int or processed <= 0 or video.analysis_complete is not True:
            return None, "processed", "incomplete_analysis"
        if type(reported) is int and reported > 0 and reported != processed:
            return processed, "processed", "frame_count_mismatch"
        return processed, "processed", "valid"
    if video.analysis_complete is False:
        return None, "metadata", "incomplete_analysis"
    if type(reported) is not int or reported <= 0:
        return None, "metadata", "invalid_total_frames"
    return reported, "metadata", "valid"


def rank_player_videos(local_track_ids: list[str], tracks: dict[str, LocalTrack],
                      videos: dict[str, VideoInfo]) -> list[dict[str, Any]]:
    by_video: dict[str, list[LocalTrack]] = {}
    for key in dict.fromkeys(local_track_ids):
        if key not in tracks:
            raise ValueError(f"Player group references a missing or inactive track: {key}")
        track = tracks[key]
        by_video.setdefault(track.video_id, []).append(track)
    rankings = []
    for video_id, local in by_video.items():
        video = videos.get(video_id)
        total, source, status = _frame_total(video)
        observed = []
        valid_observations = True
        for track in local:
            try:
                ranges = track.observed_frame_ranges
                if ranges is None:
                    if track.frames or track.frame_count == 0:
                        ranges = compact_frame_ranges(f.frame_index for f in track.frames)
                    else:
                        valid_observations = False
                        status = "missing_frame_observations"
                        break
                ranges = merge_frame_ranges(ranges)
            except (ValueError, TypeError):
                valid_observations = False
                status = "invalid_frame_observations"
                break
            if sum(end - start + 1 for start, end in ranges) != track.frame_count:
                valid_observations = False
                status = "inconsistent_frame_observations"
                break
            observed.extend(ranges)
        merged = merge_frame_ranges(observed) if valid_observations else []
        visible = sum(end - start + 1 for start, end in merged) if valid_observations else None
        if status == "valid" and merged and merged[-1][1] >= total:
            status = "out_of_range_frames"
        percentage = 100.0 * visible / total if status == "valid" else None
        rankings.append({"video_id": video_id,
            "original_filename": video.filename if video else local[0].source_video_name,
            "source_path": video.path.as_posix() if video else None,
            "visible_frames": visible, "total_frames": total, "appearance_percentage": percentage,
            "frame_count_source": source, "coverage_status": status,
            "metadata_frame_count": video.frame_count if video else None,
            "processed_frame_count": video.processed_frame_count if video else None,
            "local_tracks": sorted(t.local_player_id for t in local)})
    def order(entry):
        valid = entry["coverage_status"] == "valid"
        ratio = Fraction(entry["visible_frames"], entry["total_frames"]) if valid else Fraction(0)
        return (not valid, -ratio, -entry["visible_frames"] if valid else 0, entry["video_id"])
    rankings.sort(key=order)
    return [{"rank": index, **entry} for index, entry in enumerate(rankings, start=1)]


def ranking_fields(local_track_ids: list[str], tracks: dict[str, LocalTrack],
                   videos: dict[str, VideoInfo]) -> dict[str, Any]:
    rankings = rank_player_videos(local_track_ids, tracks, videos)
    return {"video_rankings": rankings, "videos": [r["original_filename"] for r in rankings],
            "video_ids": [r["video_id"] for r in rankings]}
