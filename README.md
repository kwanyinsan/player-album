# Player Album

Player Album is a player/person album generation pipeline for highlight videos. It uses YOLOE segmentation and Ultralytics tracking to create local player tracks, saves high-quality player crops, extracts Torchreid ReID embeddings, clusters matching identities across videos, and exports player album folders, JSON indexes, debug videos, and verification galleries.

Third-party credit: this project uses [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) as an environment-level installed dependency for person ReID feature extraction, and uses OSNet/OSNet-AIN model checkpoints from the Torchreid ecosystem. Torchreid is MIT licensed. For research or academic use, credit the Torchreid project and the OSNet/OSNet-AIN papers listed by the upstream repository, including "Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch", "Omni-Scale Feature Learning for Person Re-Identification", and "Learning Generalisable Omni-Scale Representations for Person Re-Identification".

## Pipeline

```text
input highlight videos
-> YOLOE segmentation prompt detection
-> Ultralytics BoT-SORT or ByteTrack tracking
-> local player tracks per video
-> best crop extraction and crop quality filtering
-> Torchreid OSNet-AIN embedding extraction
-> global player clustering
-> same-video conflict and identity consistency checks
-> accepted player albums, JSON indexes, debug videos, gallery.html
```

## Scope

This project is only for player/person album grouping across highlight videos.

```text
In scope:
  YOLOE person/player segmentation
  BoT-SORT or ByteTrack local tracking
  best crop extraction
  Torchreid ReID embedding extraction
  global player clustering
  player album export
  verification gallery and debug videos

Out of scope:
  ball detection
  sport-specific object detection
  video auto-framing
  16:9 or 9:16 cropped video generation
  SAM3
  court-line model training
```

The current defaults are tuned for quality-first player album processing:

```text
YOLOE model: models/yoloe-26x-seg.pt
YOLOE prompt: person
tracker: botsort.yaml
imgsz: 960
conf: 0.40
device: 0
half: true
ReID model: osnet_ain_x1_0
ReID checkpoint: models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
cluster method: agglomerative
cluster distance threshold: 0.18
debug video: true
gallery: true
```

## Project Structure

```text
main.py
  Owns the end-to-end pipeline. It parses config, scans videos, runs YOLOE tracking,
  extracts crops, runs Torchreid embeddings, clusters players, exports JSON/gallery,
  records timing, and continues past per-video failures.

config.py
  Defines CLI arguments, default values, dataclass configs, boolean parsing,
  extension parsing, and config validation.

video_io.py
  Scans input folders recursively for supported videos, assigns stable content-hash
  video IDs, and reads video metadata such as width, height, FPS, frame count,
  and duration.

yoloe_player_detection.py
  Loads the Ultralytics YOLOE segmentation model, applies the text prompt with
  model.set_classes when supported, and exposes the model/tracker backend.

player_tracking.py
  Runs Ultralytics model.track with BoT-SORT or ByteTrack, converts raw detections
  into local player tracks, estimates active-player scores, stores per-frame boxes
  and masks, and renders tracking_debug.mp4.

crop_extraction.py
  Reopens the video frames, extracts player crops for every local track, applies
  crop quality filters, saves the best crops, and writes crop metadata.

crop_quality.py
  Scores crop sharpness, crop size, person-like shape, edge contact, mask area,
  detection confidence, and near-duplicate crop spacing.

reid_embedding.py
  Calls torchreid.utils.FeatureExtractor, extracts one embedding per crop, normalizes
  crop embeddings, averages them into one embedding per local track, and writes
  local_track_embeddings.npy plus local_track_index.json.

clustering.py
  Builds cosine similarity matrices, runs agglomerative clustering, converts cluster
  labels into global player IDs, and writes similarity_matrix.csv.

conflict_rules.py
  Enforces the same-video cannot-link rule. Two local tracks visible at the same
  time in the same video cannot be the same global player.

identity_validation.py
  Computes ReID plus outfit/color consistency, splits visually weak clusters, and
  reports identity validation stats for each final player group.

album_quality.py
  Computes album_quality_score from track quality, crop quality, motion, activity
  zone, cluster consistency, identity consistency, color consistency, and conflicts.

gallery_export.py
  Builds global/gallery.html for visual verification. Accepted albums are shown
  by default, with full crop images using object-fit contain.

json_export.py
  Copies accepted player crops into players/, writes per-player metadata, builds
  player_groups.json and video_index.json, and optionally exports review/rejected
  folders when enabled.

timing_utils.py
  Provides stage timers and duration formatting for terminal output and logs.json.

utils.py
  Shared helpers for directory creation, JSON writing, content-hash video IDs,
  path formatting, bbox math, clamping, and vector normalization.
```

## Install

This is the same install order used in the clean conda environment test. The test environment was created fresh as `player-album-readme-test`, then the project was run successfully on one video.

Clone the repo:

```powershell
git clone https://github.com/kwanyinsan/player-album.git
cd player-album
```

Prerequisites:

```text
Conda
Git on PATH
NVIDIA driver compatible with CUDA 12.1 wheels
Input videos in a local folder such as highlights/ or test/
Model weights placed in models/
```

Create and activate the environment:

```powershell
conda create -n player-album python=3.10 -y
conda activate player-album
```

Install ffmpeg before Python packages:

```powershell
conda install -c conda-forge ffmpeg -y
```

`ffmpeg` is installed through conda because it is a system video tool, not a normal Python package. It gives OpenCV and the project a reliable video decode/encode backend for reading highlight clips and writing `tracking_debug.mp4`.

Upgrade pip tooling:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch CUDA wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Torchreid as an environment-level source checkout:

```powershell
New-Item -ItemType Directory -Force "$env:CONDA_PREFIX\src" | Out-Null
git clone https://github.com/KaiyangZhou/deep-person-reid.git "$env:CONDA_PREFIX\src\deep-person-reid"
python -c "import os, pathlib, site; p = pathlib.Path(site.getsitepackages()[-1]) / 'deep_person_reid_src.pth'; p.write_text(str(pathlib.Path(os.environ['CONDA_PREFIX']) / 'src' / 'deep-person-reid') + '\n', encoding='utf-8'); print(p)"
```

This source checkout is inside the conda environment, not inside this project. It avoids building Torchreid's optional Cython ranking extension on Windows, which otherwise requires Microsoft C++ Build Tools. The project still imports Torchreid as a dependency through Python's environment path.

Check the environment:

```powershell
python -c "import torch; import ultralytics; import cv2; import torchreid; from torchreid.utils import FeatureExtractor; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__); print('torchreid', getattr(torchreid, '__version__', 'unknown')); print('FeatureExtractor OK')"
ffmpeg -version
python main.py --help
```

## GPU

The recommended first setup is for an NVIDIA RTX A2000 8GB laptop GPU:

```text
CUDA wheel: cu121
device: 0
half: true
YOLOE model: yoloe-26x-seg.pt
imgsz: 960
conf: 0.40
tracker: botsort.yaml
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If VRAM is tight, reduce quality in this order:

```text
1. Use --imgsz 640
2. Use models/yoloe-26m-seg.pt instead of yoloe-26x-seg.pt
3. Use --save_debug_video false for faster tests
4. Use --device cpu only for debugging, not for full runs
```

YOLOE may disable half precision internally if the model path produces a dtype mismatch. That warning is acceptable if the run continues.

## Models

Put model files in `models/`. The folder is ignored by git because weights are large local artifacts.

```text
models/
  yoloe-26x-seg.pt
  osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

The default command expects those exact paths. You can override them with `--yoloe_model` and `--reid_model_path`.

Torchreid checkpoint used by the default config:

```text
Model: osnet_ain_x1_0
Training source: MSMT17 combineall=True
Input: 256x128
Distance: cosine
Torchreid model-zoo result: 70.1 rank-1 / 43.3 mAP on MSMT17 -> Market1501,
and 71.1 rank-1 / 52.7 mAP on MSMT17 -> DukeMTMC-reID
Download: https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing
Local path expected by default:
models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

Download it with `gdown` after `requirements.txt` is installed:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
gdown --fuzzy "https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing" -O "models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
```

The YOLOE segmentation model must also be placed in `models/`. The default is:

```text
models/yoloe-26x-seg.pt
```

## Running

Minimal quality-first run:

```powershell
python main.py --input_dir highlights --output_dir runs/player_album_best
```

This short command works because model paths and quality defaults are already set in `config.py`.

Full default-equivalent command:

```powershell
python main.py --input_dir highlights --output_dir runs/player_album_best --yoloe_model models/yoloe-26x-seg.pt --prompt person --tracker botsort.yaml --imgsz 960 --conf 0.40 --iou 0.70 --device 0 --half true --reid_model osnet_ain_x1_0 --reid_model_path models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth --max_crops_per_track 16 --min_track_frames 25 --min_crop_height 150 --cluster_method agglomerative --distance_threshold 0.18 --save_debug_video true --save_gallery true
```

Small test run:

```powershell
python main.py --input_dir test --output_dir runs/player_album_test --max_videos 1 --save_debug_video false --save_gallery false
```

Prompt or confidence test:

```powershell
python main.py --input_dir highlights --output_dir runs/player_album_person_conf045 --prompt person --conf 0.45
```

CPU debug run:

```powershell
python main.py --input_dir test --output_dir runs/player_album_cpu_debug --max_videos 1 --device cpu --half false --save_debug_video false
```

## Arguments

All arguments are configurable from the CLI. The important defaults are already in `config.py`, so the normal run command can stay short.

```text
--input_dir
  Default: highlights
  Folder containing input videos. Videos are scanned recursively.

--output_dir
  Default: runs/player_album_best
  Run output folder.

--yoloe_model
  Default: models/yoloe-26x-seg.pt
  YOLOE segmentation model weights.

--prompt
  Default: person
  YOLOE text prompt. Good first choices are person, player, or athlete.

--tracker
  Default: botsort.yaml
  Ultralytics tracker config. Use botsort.yaml first; bytetrack.yaml is the backup.

--imgsz
  Default: 960
  YOLOE inference image size.

--conf
  Default: 0.40
  Detection confidence threshold. Higher values produce fewer but cleaner tracks.

--iou
  Default: 0.70
  Detection NMS IoU threshold.

--device
  Default: 0
  Torch/Ultralytics device. Use 0 for first GPU, cpu for CPU debug.

--half
  Default: true
  Use FP16 where supported.

--reid_model
  Default: osnet_ain_x1_0
  Torchreid model name used by FeatureExtractor.

--reid_model_path
  Default: models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
  Local Torchreid checkpoint path.

--max_crops_per_track
  Default: 16
  Maximum best crops saved per local track.

--min_track_frames
  Default: 25
  Reject local tracks shorter than this many frames when active filtering is enabled.

--min_crop_height
  Default: 150
  Reject crops shorter than this many pixels.

--cluster_method
  Default: agglomerative
  Global clustering backend.

--distance_threshold
  Default: 0.18
  Agglomerative clustering cosine distance threshold.

--identity_split_enabled
  Default: true
  Enable second-pass splitting for visually inconsistent clusters.

--identity_split_min_cluster_size
  Default: 8
  Only run identity splitting on clusters at least this large.

--identity_split_distance_threshold
  Default: 0.14
  Distance threshold used for second-pass identity splitting.

--identity_color_weight
  Default: 0.25
  Weight of outfit/color fingerprint similarity in identity validation.

--identity_min_consistency
  Default: 0.74
  Combined identity consistency threshold before a cluster is marked weak.

--identity_min_color_consistency
  Default: 0.35
  Color consistency threshold before a cluster is marked outfit-diverse.

--identity_split_mean_threshold
  Default: 0.78
  Split a cluster when average combined identity similarity falls below this.

--identity_split_color_threshold
  Default: 0.52
  Split a cluster when outfit/color consistency falls below this.

--identity_split_p10_threshold
  Default: 0.62
  Low-tail combined similarity threshold for identity splitting.

--identity_split_std_threshold
  Default: 0.07
  Combined similarity standard deviation threshold for identity splitting.

--identity_split_large_min_similarity
  Default: 0.58
  Minimum pair similarity tolerated inside large identity clusters.

--identity_album_min_consistency
  Default: 0.70
  Album scoring penalty threshold for identity consistency.

--identity_album_large_min_consistency
  Default: 0.78
  Album scoring penalty threshold for large identity clusters.

--identity_album_min_color_consistency
  Default: 0.50
  Album scoring penalty threshold for outfit/color consistency.

--save_debug_video
  Default: true
  Render per-video tracking_debug.mp4.

--save_gallery
  Default: true
  Render global/gallery.html.

--max_videos
  Default: none
  Optional cap for quick tests.

--video_extensions
  Default: .mp4,.mov,.avi,.mkv
  Comma-separated supported extensions.

--crop_padding
  Default: 0.08
  BBox padding ratio before crop extraction.

--blur_threshold
  Default: 110.0
  Minimum Laplacian sharpness score.

--mask_crop_mode
  Default: bbox
  Crop mode: bbox, masked, or both.

--min_mask_area_ratio
  Default: 0.02
  Minimum mask area divided by bbox area.

--min_person_shape_score
  Default: 0.35
  Minimum person-like crop shape score.

--max_edge_touch_ratio
  Default: 0.75
  Reject crops touching too much of the frame edge.

--active_player_filter
  Default: true
  Enable track filtering based on duration, movement, crop height, size, and activity zone.

--min_movement_score
  Default: 0.04
  Minimum normalized movement score for active-player filtering.

--min_track_duration_sec
  Default: 0.8
  Minimum visible duration for active-player filtering.

--quality_filter_export
  Default: true
  Use album_quality_score to separate accepted, review, and rejected groups.

--album_accept_threshold
  Default: 0.78
  Minimum album_quality_score copied into players/.

--album_review_threshold
  Default: 0.55
  Minimum album_quality_score considered review instead of rejected.

--export_review_rejected_folders
  Default: false
  If true, copy review/ and rejected/ folders. Default keeps only players/.

--gallery_include_review_rejected
  Default: false
  If true, show review/rejected groups in gallery.html.
```

## Input And Output

Input videos stay untouched. The code does not delete, move, rename, modify, or overwrite input videos or model weights.

Supported video extensions:

```text
.mp4, .mov, .avi, .mkv
```

Example output:

```text
runs/player_album_best/
  videos/
    V_ABC123/
      tracking_debug.mp4
      local_tracks.json
      crops/
        V_ABC123_track_0001/
          crop_0001.jpg
          crop_0002.jpg
          metadata.json

  global/
    local_track_embeddings.npy
    local_track_index.json
    similarity_matrix.csv
    player_groups.json
    video_index.json
    gallery.html

  players/
    P0001/
      representative.jpg
      V_ABC123_track_0001_best.jpg
      metadata.json

  logs.json
```

Output file meanings:

```text
videos/<video_id>/tracking_debug.mp4
  Optional debug video showing YOLOE/tracker inference boxes, masks, IDs, confidence,
  prompt label, and active/ignored status.

videos/<video_id>/local_tracks.json
  Per-video track data: source metadata, local track IDs, frame indexes, bboxes,
  confidence, active status, crop paths, and track scores.

videos/<video_id>/crops/<local_track_id>/crop_####.jpg
  Saved best crops for a local player track.

videos/<video_id>/crops/<local_track_id>/metadata.json
  Crop-level metadata and quality scores for that local track.

global/local_track_embeddings.npy
  One normalized ReID embedding per local player track.

global/local_track_index.json
  JSON index mapping each embedding row to local_player_id, video_id, source video,
  crop paths, representative crop, and embedding quality information.

global/similarity_matrix.csv
  Pairwise cosine similarity matrix between local track embeddings.

global/player_groups.json
  Global player groups with player_id, assigned local tracks, videos, representative
  crop, album_quality_score, export status, conflict info, and identity validation.

global/video_index.json
  Video-level index showing which accepted global players appear in each video.

global/gallery.html
  Visual verification gallery. By default it shows accepted player groups only.
  It references crop images with relative paths, so send the run folder or a
  packaged standalone export if sharing with another person.

players/P####/
  Accepted player album folder. Contains representative.jpg, one best crop per
  assigned local track, and metadata.json.

logs.json
  Full run config, model paths, prompt, tracker, timing, warnings, errors, number
  of videos, number of local tracks, and number of global/accepted/review/rejected players.
```

Only accepted player albums are copied into `players/` by default. Review and rejected groups remain in JSON metadata unless `--export_review_rejected_folders true` is enabled.

## Tested Setup

The clean environment test used the same install order above. The final smoke test command was:

```powershell
python main.py --input_dir test --output_dir runs/readme_env_smoke_clean --max_videos 1 --save_debug_video false --save_gallery false
```

Result:

```text
Videos processed: 1
Local tracks: 29
Global player groups: 5
Accepted players: 3
Review players: 2
Errors: 0
Total folder time: 63.7s
```

The only expected warnings were Torchreid's optional Cython ranking warning and a PyTorch checkpoint warning from upstream Torchreid.

## Reproducibility

Internal video IDs use a content hash of the video bytes. The same video file should receive the same `video_id` even if it is moved into a different folder, copied to another machine, or mounted through Docker.

The most important run settings are saved into `logs.json`, including model paths, prompt, tracker, confidence threshold, ReID model, clustering config, export config, timing, warnings, and errors.
