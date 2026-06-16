from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from utils import stable_video_id


@dataclass
class VideoInfo:
    video_id: str
    filename: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "filename": self.filename,
            "path": self.path.as_posix(),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
        }


def read_video_metadata(path: Path, video_id: str) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(
            video_id=video_id,
            filename=path.name,
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
        )
    finally:
        cap.release()


def scan_videos(input_dir: Path, extensions: tuple[str, ...], max_videos: int | None = None) -> list[VideoInfo]:
    used_ids: set[str] = set()
    extension_set = {ext.lower() for ext in extensions}
    paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extension_set
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().lower(),
    )
    if max_videos is not None:
        paths = paths[:max_videos]
    videos: list[VideoInfo] = []
    for path in paths:
        video_id = stable_video_id(path, used_ids)
        videos.append(read_video_metadata(path, video_id))
    return videos


def iter_video_frames(path: Path) -> Iterator[tuple[int, object]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()


def create_video_writer(path: Path, fps: float, width: int, height: int):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer
