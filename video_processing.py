"""Per-video analysis shared by batch and incremental collection processing."""
from time import perf_counter
from pathlib import Path
from typing import Callable

from config import PipelineConfig
from crop_extraction import extract_crops_for_video
from json_export import write_local_tracks_json
from player_tracking import LocalTrack, process_video_tracking, render_tracking_debug_video
from video_io import VideoInfo
from yoloe_player_detection import YOLOEPlayerDetector


def analyze_video(
    video: VideoInfo, detector: YOLOEPlayerDetector, config: PipelineConfig, output_root: Path,
    on_optional_error: Callable[[str, Exception], None] | None = None,
) -> tuple[list[LocalTrack], dict[str, float]]:
    directory = output_root / "videos" / video.video_id
    directory.mkdir(parents=True, exist_ok=True)
    timings = {}
    started = perf_counter()
    tracks = process_video_tracking(video, detector, config.filtering, config.crops.min_crop_height)
    timings["tracking_sec"] = perf_counter() - started
    started = perf_counter()
    try:
        extract_crops_for_video(video, tracks, directory, output_root, config.crops)
    except Exception as exc:
        if on_optional_error is None:
            raise
        on_optional_error("crop_extraction", exc)
    timings["crops_sec"] = perf_counter() - started
    started = perf_counter()
    if config.export.save_debug_video:
        try:
            render_tracking_debug_video(video, tracks, directory / "tracking_debug.mp4", config.detector.prompt)
        except Exception as exc:
            if on_optional_error is None:
                raise
            on_optional_error("debug_video_rendering", exc)
    timings["debug_sec"] = perf_counter() - started
    write_local_tracks_json(video, tracks, directory / "local_tracks.json")
    return tracks, timings
