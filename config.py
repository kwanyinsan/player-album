from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_VIDEO_EXTENSIONS = ".mp4,.mov,.avi,.mkv"
DEFAULT_REID_MODEL_PATH = (
    "models/"
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_video_extensions(value: str) -> tuple[str, ...]:
    extensions: list[str] = []
    for raw in value.split(","):
        ext = raw.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.append(ext)
    if not extensions:
        raise argparse.ArgumentTypeError("At least one video extension is required.")
    return tuple(dict.fromkeys(extensions))


@dataclass(frozen=True)
class DetectorConfig:
    yoloe_model: Path
    prompt: str = "person"
    tracker: str = "botsort.yaml"
    imgsz: int = 960
    conf: float = 0.40
    iou: float = 0.70
    device: str = "0"
    half: bool = True


@dataclass(frozen=True)
class CropConfig:
    max_crops_per_track: int = 16
    min_crop_height: int = 150
    crop_padding: float = 0.08
    blur_threshold: float = 110.0
    mask_crop_mode: str = "bbox"
    min_mask_area_ratio: float = 0.02
    min_person_shape_score: float = 0.35
    max_edge_touch_ratio: float = 0.75


@dataclass(frozen=True)
class FilterConfig:
    active_player_filter: bool = True
    min_track_frames: int = 25
    min_movement_score: float = 0.04
    min_track_duration_sec: float = 0.8


@dataclass(frozen=True)
class ReidConfig:
    reid_model: str = "osnet_ain_x1_0"
    reid_model_path: Path | None = Path(DEFAULT_REID_MODEL_PATH)
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class ClusteringConfig:
    cluster_method: str = "agglomerative"
    distance_threshold: float = 0.18
    identity_split_enabled: bool = True
    identity_split_min_cluster_size: int = 8
    identity_split_distance_threshold: float = 0.14
    identity_color_weight: float = 0.25
    identity_min_consistency: float = 0.74
    identity_min_color_consistency: float = 0.35
    identity_split_mean_threshold: float = 0.78
    identity_split_color_threshold: float = 0.52
    identity_split_p10_threshold: float = 0.62
    identity_split_std_threshold: float = 0.07
    identity_split_large_min_similarity: float = 0.58
    identity_album_min_consistency: float = 0.70
    identity_album_large_min_consistency: float = 0.78
    identity_album_min_color_consistency: float = 0.50


@dataclass(frozen=True)
class ExportConfig:
    save_debug_video: bool = True
    save_gallery: bool = True
    quality_filter_export: bool = True
    album_accept_threshold: float = 0.78
    album_review_threshold: float = 0.55
    export_review_rejected_folders: bool = False
    gallery_include_review_rejected: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    input_dirs: tuple[Path, ...]
    output_dir: Path
    video_extensions: tuple[str, ...]
    max_videos: Optional[int]
    detector: DetectorConfig
    crops: CropConfig
    filtering: FilterConfig
    reid: ReidConfig
    clustering: ClusteringConfig
    export: ExportConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build player and person albums from tracked highlight videos."
    )

    parser.add_argument("--input_dirs", nargs="+", help="Folders of input videos.")
    parser.add_argument("--input_dir", default="highlights", help="Folder of input videos (deprecated, use --input_dirs).")
    parser.add_argument("--output_dir", default="runs/player_album_best", help="Output run folder.")
    parser.add_argument("--yoloe_model", default="models/yoloe-26x-seg.pt", help="YOLOE segmentation weights.")
    parser.add_argument("--prompt", default="person", help="YOLOE text prompt, for example 'person' or 'player'.")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker YAML, for example botsort.yaml.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLOE inference image size.")
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Torch and Ultralytics device, for example 0 or cpu.")
    parser.add_argument("--half", type=parse_bool, default=True, help="Use FP16 inference when supported.")
    parser.add_argument("--reid_model", default="osnet_ain_x1_0", help="Torchreid model name.")
    parser.add_argument("--reid_model_path", default=DEFAULT_REID_MODEL_PATH, help="Optional Torchreid checkpoint path.")
    parser.add_argument("--max_crops_per_track", type=int, default=16, help="Best crops to keep per local track.")
    parser.add_argument("--min_track_frames", type=int, default=25, help="Minimum frames for a local track.")
    parser.add_argument("--min_crop_height", type=int, default=150, help="Reject crops shorter than this many pixels.")
    parser.add_argument("--cluster_method", default="agglomerative", help="Clustering backend.")
    parser.add_argument("--distance_threshold", type=float, default=0.18, help="Cosine distance threshold.")
    parser.add_argument("--identity_split_enabled", type=parse_bool, default=True, help="Split visually inconsistent clusters after ReID clustering.")
    parser.add_argument("--identity_split_min_cluster_size", type=int, default=8, help="Only run identity splitting on clusters at least this large.")
    parser.add_argument("--identity_split_distance_threshold", type=float, default=0.14, help="Second-pass combined distance threshold for identity splitting.")
    parser.add_argument("--identity_color_weight", type=float, default=0.25, help="Weight for crop color fingerprint similarity during identity splitting.")
    parser.add_argument("--identity_min_consistency", type=float, default=0.74, help="Minimum combined identity consistency before a cluster is marked weak.")
    parser.add_argument("--identity_min_color_consistency", type=float, default=0.35, help="Minimum color consistency before a cluster is marked outfit-diverse.")
    parser.add_argument("--identity_split_mean_threshold", type=float, default=0.78, help="Split a cluster when its combined identity mean falls below this value.")
    parser.add_argument("--identity_split_color_threshold", type=float, default=0.52, help="Split a cluster when its outfit and color consistency falls below this value.")
    parser.add_argument("--identity_split_p10_threshold", type=float, default=0.62, help="Low-tail combined similarity threshold for identity splitting.")
    parser.add_argument("--identity_split_std_threshold", type=float, default=0.07, help="Combined similarity standard-deviation threshold for identity splitting.")
    parser.add_argument("--identity_split_large_min_similarity", type=float, default=0.58, help="Minimum pair similarity tolerated for large identity clusters.")
    parser.add_argument("--identity_album_min_consistency", type=float, default=0.70, help="Album scoring penalty threshold for identity consistency.")
    parser.add_argument("--identity_album_large_min_consistency", type=float, default=0.78, help="Album scoring penalty threshold for large identity clusters.")
    parser.add_argument("--identity_album_min_color_consistency", type=float, default=0.50, help="Album scoring penalty threshold for outfit and color consistency.")
    parser.add_argument("--save_debug_video", type=parse_bool, default=True, help="Render per-video tracking debug MP4.")
    parser.add_argument("--save_gallery", type=parse_bool, default=True, help="Render global verification gallery HTML.")

    parser.add_argument("--max_videos", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument(
        "--video_extensions",
        type=parse_video_extensions,
        default=parse_video_extensions(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated extensions. Default: .mp4,.mov,.avi,.mkv",
    )
    parser.add_argument("--crop_padding", type=float, default=0.08, help="BBox padding ratio for crop extraction.")
    parser.add_argument("--blur_threshold", type=float, default=110.0, help="Minimum Laplacian blur score.")
    parser.add_argument(
        "--mask_crop_mode",
        choices=("bbox", "masked", "both"),
        default="bbox",
        help="Use rectangular crops, masked crops, or save both.",
    )
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.02, help="Minimum mask area divided by bbox area.")
    parser.add_argument("--min_person_shape_score", type=float, default=0.35, help="Minimum person-like crop shape score.")
    parser.add_argument("--max_edge_touch_ratio", type=float, default=0.75, help="Reject crops touching too many frame edges.")
    parser.add_argument("--active_player_filter", type=parse_bool, default=True, help="Enable simple active-track filter.")
    parser.add_argument("--min_movement_score", type=float, default=0.04, help="Minimum normalized movement score.")
    parser.add_argument("--min_track_duration_sec", type=float, default=0.8, help="Minimum visible track duration.")
    parser.add_argument("--quality_filter_export", type=parse_bool, default=True, help="Split albums into accepted, review, and rejected groups.")
    parser.add_argument("--album_accept_threshold", type=float, default=0.78, help="Minimum album quality score for the players folder.")
    parser.add_argument("--album_review_threshold", type=float, default=0.55, help="Minimum album quality score for review status.")
    parser.add_argument("--export_review_rejected_folders", type=parse_bool, default=False, help="Copy review and rejected albums into output folders.")
    parser.add_argument("--gallery_include_review_rejected", type=parse_bool, default=False, help="Show review and rejected groups in gallery.html.")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    input_dirs = tuple(Path(p) for p in args.input_dirs) if getattr(args, "input_dirs", None) else (Path(args.input_dir),)
    return PipelineConfig(
        input_dirs=input_dirs,
        output_dir=Path(args.output_dir),
        video_extensions=tuple(args.video_extensions),
        max_videos=args.max_videos,
        detector=DetectorConfig(
            yoloe_model=Path(args.yoloe_model),
            prompt=args.prompt,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=str(args.device),
            half=args.half,
        ),
        crops=CropConfig(
            max_crops_per_track=args.max_crops_per_track,
            min_crop_height=args.min_crop_height,
            crop_padding=args.crop_padding,
            blur_threshold=args.blur_threshold,
            mask_crop_mode=args.mask_crop_mode,
            min_mask_area_ratio=args.min_mask_area_ratio,
            min_person_shape_score=args.min_person_shape_score,
            max_edge_touch_ratio=args.max_edge_touch_ratio,
        ),
        filtering=FilterConfig(
            active_player_filter=args.active_player_filter,
            min_track_frames=args.min_track_frames,
            min_movement_score=args.min_movement_score,
            min_track_duration_sec=args.min_track_duration_sec,
        ),
        reid=ReidConfig(
            reid_model=args.reid_model,
            reid_model_path=Path(args.reid_model_path) if args.reid_model_path else None,
        ),
        clustering=ClusteringConfig(
            cluster_method=args.cluster_method,
            distance_threshold=args.distance_threshold,
            identity_split_enabled=args.identity_split_enabled,
            identity_split_min_cluster_size=args.identity_split_min_cluster_size,
            identity_split_distance_threshold=args.identity_split_distance_threshold,
            identity_color_weight=args.identity_color_weight,
            identity_min_consistency=args.identity_min_consistency,
            identity_min_color_consistency=args.identity_min_color_consistency,
            identity_split_mean_threshold=args.identity_split_mean_threshold,
            identity_split_color_threshold=args.identity_split_color_threshold,
            identity_split_p10_threshold=args.identity_split_p10_threshold,
            identity_split_std_threshold=args.identity_split_std_threshold,
            identity_split_large_min_similarity=args.identity_split_large_min_similarity,
            identity_album_min_consistency=args.identity_album_min_consistency,
            identity_album_large_min_consistency=args.identity_album_large_min_consistency,
            identity_album_min_color_consistency=args.identity_album_min_color_consistency,
        ),
        export=ExportConfig(
            save_debug_video=args.save_debug_video,
            save_gallery=args.save_gallery,
            quality_filter_export=args.quality_filter_export,
            album_accept_threshold=args.album_accept_threshold,
            album_review_threshold=args.album_review_threshold,
            export_review_rejected_folders=args.export_review_rejected_folders,
            gallery_include_review_rejected=args.gallery_include_review_rejected,
        ),
    )


def parse_config(argv: Optional[list[str]] = None) -> PipelineConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return config_from_args(args)


def validate_config(config: PipelineConfig) -> None:
    if not config.input_dirs:
        raise ValueError("At least one input directory must be provided.")
    for input_dir in config.input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if not config.detector.yoloe_model.exists():
        raise FileNotFoundError(f"YOLOE model file does not exist: {config.detector.yoloe_model}")
    if config.reid.reid_model_path is not None and not config.reid.reid_model_path.exists():
        raise FileNotFoundError(f"ReID model checkpoint does not exist: {config.reid.reid_model_path}")
    if config.crops.max_crops_per_track <= 0:
        raise ValueError("--max_crops_per_track must be greater than 0.")
    if config.filtering.min_track_frames < 0:
        raise ValueError("--min_track_frames must be non-negative.")
    if not (0.0 <= config.clustering.distance_threshold <= 2.0):
        raise ValueError("--distance_threshold must be in the cosine distance range [0, 2].")
    if config.clustering.identity_split_min_cluster_size < 2:
        raise ValueError("--identity_split_min_cluster_size must be at least 2.")
    if not (0.0 <= config.clustering.identity_split_distance_threshold <= 2.0):
        raise ValueError("--identity_split_distance_threshold must be in the distance range [0, 2].")
    if not (0.0 <= config.clustering.identity_color_weight <= 1.0):
        raise ValueError("--identity_color_weight must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_min_consistency <= 1.0):
        raise ValueError("--identity_min_consistency must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_min_color_consistency <= 1.0):
        raise ValueError("--identity_min_color_consistency must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_split_mean_threshold <= 1.0):
        raise ValueError("--identity_split_mean_threshold must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_split_color_threshold <= 1.0):
        raise ValueError("--identity_split_color_threshold must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_split_p10_threshold <= 1.0):
        raise ValueError("--identity_split_p10_threshold must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_split_std_threshold <= 1.0):
        raise ValueError("--identity_split_std_threshold must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_split_large_min_similarity <= 1.0):
        raise ValueError("--identity_split_large_min_similarity must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_album_min_consistency <= 1.0):
        raise ValueError("--identity_album_min_consistency must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_album_large_min_consistency <= 1.0):
        raise ValueError("--identity_album_large_min_consistency must be in [0, 1].")
    if not (0.0 <= config.clustering.identity_album_min_color_consistency <= 1.0):
        raise ValueError("--identity_album_min_color_consistency must be in [0, 1].")
    if not (0.0 <= config.crops.min_person_shape_score <= 1.0):
        raise ValueError("--min_person_shape_score must be in [0, 1].")
    if not (0.0 <= config.crops.max_edge_touch_ratio <= 1.0):
        raise ValueError("--max_edge_touch_ratio must be in [0, 1].")
    if not (0.0 <= config.export.album_review_threshold <= config.export.album_accept_threshold <= 1.0):
        raise ValueError("--album_review_threshold must be <= --album_accept_threshold, both in [0, 1].")
