# Player Album

Player Album is a player and person album generation pipeline for highlight videos.
It uses YOLOE segmentation and Ultralytics tracking to create local player tracks,
saves high-quality player crops, extracts Torchreid ReID embeddings, clusters
matching identities across videos, and exports player album folders, JSON indexes,
debug videos, and verification galleries.

Third-party credit: this project uses [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
as an environment-level installed dependency for person ReID feature extraction.
It also uses OSNet and OSNet-AIN model checkpoints from the Torchreid ecosystem.
Torchreid is MIT licensed. For research or academic use, credit the Torchreid
project and the OSNet and OSNet-AIN papers listed by the upstream repository,
including "Torchreid: A Library for Deep Learning Person Re-Identification in
Pytorch", "Omni-Scale Feature Learning for Person Re-Identification", and
"Learning Generalisable Omni-Scale Representations for Person Re-Identification".

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

This project processes a folder of highlight videos and groups repeated players
or people into identity albums. It detects player candidates with YOLOE, tracks
each local player within each video, extracts representative crops, computes ReID
embeddings with Torchreid, clusters likely matching identities across videos, and
exports albums plus verification artifacts.

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
cluster method: agglomerative
cluster distance threshold: 0.18
debug video: true
gallery: true
```

## Project Structure

**`main.py`**

Owns the end-to-end pipeline. It parses config, scans videos, runs YOLOE
tracking, extracts crops, runs Torchreid embeddings, clusters players, exports
JSON files and the gallery, records timing, and continues past per-video failures.

**`config.py`**

Defines CLI arguments, default values, dataclass configs, boolean parsing,
extension parsing, and config validation.

**`video_io.py`**

Scans input folders recursively for supported videos, assigns stable content-hash
video IDs, and reads video metadata such as width, height, FPS, frame count, and
duration.

**`yoloe_player_detection.py`**

Loads the Ultralytics YOLOE segmentation model, applies the text prompt with
`model.set_classes` when supported, and exposes the model and tracker backend.

**`player_tracking.py`**

Runs Ultralytics `model.track` with BoT-SORT or ByteTrack, converts raw detections
into local player tracks, estimates active-player scores, stores per-frame boxes
and masks, and renders `tracking_debug.mp4`.

**`crop_extraction.py`**

Reopens video frames, extracts player crops for every local track, applies crop
quality filters, saves the best crops, and writes crop metadata.

**`crop_quality.py`**

Scores crop sharpness, crop size, person-like shape, edge contact, mask area,
detection confidence, and near-duplicate crop spacing.

**`reid_embedding.py`**

Calls `torchreid.utils.FeatureExtractor`, extracts one embedding per crop,
normalizes crop embeddings, averages them into one embedding per local track, and
writes `local_track_embeddings.npy` plus `local_track_index.json`.

**`clustering.py`**

Builds cosine similarity matrices, runs agglomerative clustering, converts cluster
labels into global player IDs, and writes `similarity_matrix.csv`.

**`conflict_rules.py`**

Enforces the same-video cannot-link rule. Two local tracks visible at the same
time in the same video cannot be the same global player.

**`identity_validation.py`**

Computes ReID plus outfit and color consistency, splits visually weak clusters,
and reports identity validation stats for each final player group.

**`album_quality.py`**

Computes `album_quality_score` from track quality, crop quality, motion, activity
zone, cluster consistency, identity consistency, color consistency, and conflicts.

**`gallery_export.py`**

Builds the gallery HTML file under the global output folder for visual
verification. Accepted albums are shown by default, with full crop images using
`object-fit: contain`.

**`json_export.py`**

Copies accepted player crops into the `players` folder, writes per-player metadata,
builds `player_groups.json` and `video_index.json`, and optionally exports review
and rejected folders when enabled.

**`timing_utils.py`**

Provides stage timers and duration formatting for terminal output and `logs.json`.

**`utils.py`**

Provides shared helpers for directory creation, JSON writing, content-hash video
IDs, path formatting, bbox math, clamping, and vector normalization.

**`saved_run.py`**

Validates saved video analysis and embedding associations, loads track summaries
or full frame records, restores saved settings, and manages the initial SQLite
collection registry with artifact checksums and explicit provenance limitations.

**`collection_cli.py`**

Handles read-only inspection and reclustering into a new collection. It copies
validated artifacts, verifies source integrity, and publishes completed output.

**`album_pipeline.py`**

Shares clustering, conflict resolution, identity checks, and album export between
new video processing and saved-run processing so both workflows use the same rules.

**`tests/test_saved_run.py`**

Exercises saved-run validation, embedding mappings, source preservation, registry
integrity, and failure handling without detector or ReID inference.

## Install

This is the same install order used in the clean conda environment test.

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
Input videos in a local folder such as highlights or test
Model weights placed in the models folder
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

Install `ffmpeg` through Conda as a system video tool. The pipeline reads source
frames and writes `tracking_debug.mp4` through OpenCV's video backend.

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
$torchreidSrc = "$env:CONDA_PREFIX\src\deep-person-reid"
git clone https://github.com/KaiyangZhou/deep-person-reid.git $torchreidSrc
$sitePackages = python -c "import site; print(site.getsitepackages()[-1])"
$pthFile = Join-Path $sitePackages "deep_person_reid_src.pth"
Set-Content -Path $pthFile -Value $torchreidSrc
```

This source checkout is inside the conda environment, not inside this project.
It avoids building Torchreid's optional Cython ranking extension on Windows,
which otherwise requires Microsoft C++ Build Tools. The project still imports
Torchreid as a dependency through Python's environment path.

Check the environment:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "import ultralytics, cv2; print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__)"
python -c "import torchreid; print('torchreid', getattr(torchreid, '__version__', 'unknown'))"
python -c "from torchreid.utils import FeatureExtractor; print('FeatureExtractor OK')"
ffmpeg -version
python main.py --help
```

## GPU

The clean setup was tested on an NVIDIA RTX A2000 laptop GPU with 8GB VRAM. This
is the minimal verified GPU setup used during testing:

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

For lower VRAM, use smaller test settings first:

```text
Use --imgsz 640
Use models/yoloe-26m-seg.pt instead of yoloe-26x-seg.pt
Use --save_debug_video false for faster tests
Use --device cpu only for debugging
```

YOLOE may disable half precision internally if the model path produces a dtype
mismatch. That warning is acceptable if the run continues.

## Models

Put model files in the `models` folder. The folder is ignored by git because
weights are large local artifacts.

```text
models/
  yoloe-26x-seg.pt
  osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

The default command expects those exact paths. You can override them with
`--yoloe_model` and `--reid_model_path`.

Torchreid checkpoint used by the default config:

```text
Model: osnet_ain_x1_0
Training source: MSMT17 combineall=True
Input: 256x128
Distance: cosine
Torchreid model-zoo result:
  70.1 rank-1 and 43.3 mAP on MSMT17 to Market1501
  71.1 rank-1 and 52.7 mAP on MSMT17 to DukeMTMC-reID
Download:
  https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing
Local path expected by default:
  models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

Download it with `gdown` after `requirements.txt` is installed:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
$reidUrl = "https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing"
$reidPath = "models/" + `
  "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
gdown --fuzzy $reidUrl -O $reidPath
```

The YOLOE segmentation model must also be placed in the `models` folder. The
default is `models/yoloe-26x-seg.pt`.

## Running

### Reuse An Existing Run

Saved analysis can be validated and reclustered without opening input videos or
loading YOLOE or Torchreid. These commands need NumPy; reclustering also uses
scikit-learn and OpenCV for the existing identity checks. GPU access is not needed.

Inspect a run without modifying it:

```powershell
python main.py inspect `
  --run_dir runs/player_album_final
```

Create a separate collection using its saved tracks, crops, and embeddings:

```powershell
python main.py recluster `
  --run_dir runs/player_album_final `
  --output_dir runs/player_album_reloaded
```

The destination must not exist and must be outside the source run. Settings are
loaded from the saved logs. Add `--distance_threshold 0.20` to override the
clustering threshold. Legacy logs that omit settings produce a warning; missing
values use the current defaults, so historical groups are not guaranteed to match.

The importer checks video and track associations, frame summaries, embedding row
mapping, finite nonzero vectors, and referenced crop files. It copies validated
analysis and crops, exports fresh albums and a gallery, and creates `project.db`
with video membership, track-to-embedding rows, source settings, and artifact
SHA-256 checksums. Large per-frame records are read one video at a time. They
remain available on disk, with an optional typed loader for later frame operations.

Original input videos and weights are not required for these saved-run commands.
Debug videos and standalone galleries are not duplicated. The new regular gallery
uses the copied crops and can be shared together with its output folder.

Existing model checksums and inference provenance cannot be reconstructed reliably
from legacy logs. Such imports are explicitly marked unverified for future
inference-cache compatibility, even when their saved embeddings are valid for
reclustering. The resolved settings and validation warnings are saved in
`logs.json` and `global/reuse_report.json`. A subsequent load verifies registered
artifact checksums. The separate Add command below extends this foundation.
Delete and tracked-appearance video sorting are available as described below.

Failed operations return a nonzero status and leave any staging files unpublished
for diagnosis. A complete collection is published only after export and source
checksum verification. The source run is never updated by these commands.

### Add Videos To A Collection

Add processes only videos whose contents are not already registered. It reuses
existing local tracks, crops, and embeddings, then reclusters the combined set.
Existing per-video files and debug videos are not rewritten. Global player
memberships and acceptance decisions can change when new evidence is added.

```powershell
python main.py add `
  --run_dir runs/step1_test `
  --input_dir new_highlights
```

The input folder is scanned recursively and must be outside the collection.
Renamed copies are skipped using full SHA-256 fingerprints. Different contents
with the same filename are new inputs. The first Add to a legacy collection
needs the original videos at their recorded paths to establish fingerprints.
Run from the project directory when saved paths are relative. Missing originals
stop the migration instead of relying on filenames to guess duplicate identity.

**`--dry_run`**

Lists new inputs and duplicate counts without inference or album publication.
A lock file may be created, but saved analysis remains unchanged.

**`--max_videos`**

Limits the number of new videos after duplicate filtering, useful for testing.

**`--device`**

Overrides the processing device. Other inference and quality settings are loaded
from the saved collection rather than taken from changing command defaults.

**`--save_debug_video`**

Defaults to true for new videos, including collections created by reclustering.
Pass false to disable new debug exports. Existing debug videos remain untouched.

**`--yoloe_model` and `--reid_model_path`**

Allow relocating weights once their fingerprints are recorded. Different
checkpoint contents or a changed embedding recipe are rejected. Legacy runs
without historical fingerprints retain an explicit provenance warning; the
first Add records the current models but cannot prove which original weights
produced the old embeddings. Legacy checkpoint path overrides are rejected.

Unchanged player groups retain their IDs. Splits and merges use deterministic
shared-track overlap matching, and new IDs are allocated without recycling
retired numbers. Add and standalone reclustering use the same continuity rules.
Fresh batch runs record model fingerprints and video fingerprints from the start.

**`global/collection_state.json`**

Records video fingerprints, current inference fingerprints, the next player ID,
track embedding states, and operation history including identity lineage. Its checksum is registered in
the existing SQLite registry; no external database service is required.

Each local track has a state of complete, pending, retryable_error,
no_usable_crops, or blocked_error. The state stores its embedding row, attempt
count, failed stage, and last error. Rows marked complete must agree with the
saved embedding index or collection validation fails.

Add automatically retries pending and retryable tracks using their saved crops,
including when every supplied video is already registered. Existing complete
embeddings are preserved. Crops rejected by quality filtering have no_usable_crops
status and are not retried. A track with some successful crops has a complete
embedding and retains its crop warnings; it is not recomputed automatically.

Embedding vectors must be finite, nonzero, and of the expected dimension. Crop
and final averaged vectors are normalised before saving. Invalid feature values
are excluded. A dimension or checkpoint mismatch blocks publication. GPU memory
failure during ReID model initialisation leaves the analysed tracks retryable.
Other model initialisation errors stop the operation for diagnosis.

**`global/gallery_standalone.html`**

Add exports a self-contained gallery with embedded images alongside the normal
gallery. The standalone file can be sent by itself.

**`.updates` and `.collection-update.json`**

Each operation stages its outputs separately. Publication replaces managed global
exports and appends new video folders while keeping previous exports in a backup.
Old analysis is read directly from the collection during staging, avoiding a full
copy on every Add. By default only the latest previous committed export backup
is retained. Use --keep_backups with a positive integer to retain more. Cleanup
only touches older committed global and player export backups. Existing video
analysis, model files, and source videos are preserved. Failed workspaces and
backups from older versions without a commit marker remain available for diagnosis.
Do not modify an update workspace while processing or recovery is active.
There is a brief publication interval during which the gallery should not be
opened. This is recoverable publication, not a single atomic database transaction.

If interrupted during publication, restore the previous coherent collection:

```powershell
python main.py recover `
  --run_dir runs/step1_test
```

Concurrent Add operations are blocked by an operating-system lock that releases
when the process exits. Successful videos can be published despite individual
video failures; the command returns status 2 for partial failures and records
errors in the logs. Failed videos remain retryable. Successful video analysis can
be registered even if every eligible track needs an embedding retry, preserving
the crops for the next attempt. Unfinished tracks are excluded from clustering.
A duplicate-only operation with no pending tracks performs no inference or export.
Run the same Add command again to retry pending tracks. A pure retry performs no
YOLOE detection, video tracking, or crop extraction.

**`collection_add.py`**

Implements discovery, inherited inference settings, compatibility checks, new
video processing, combined embeddings, validation, and updated exports.

**`collection_transaction.py`**

Implements collection locking, journalled publication, and rollback recovery.

**`video_processing.py`**

Shares per-video tracking, crop extraction, and debug rendering between ordinary
batch processing and Add.

**`player_identity.py`**

Matches new groups to previous player IDs and records split and merge lineage.

**`embedding_state.py`**

Builds and validates persistent track processing states against saved embeddings.

**`artifact_paths.py`**

Resolves new staged files and existing collection artifacts with path checks.

**`inference_provenance.py`**

Fingerprints model weights and the embedding implementation for compatibility checks.

**`tests/test_reid_embedding.py`**

Tests invalid vectors, partial crop failures, failed averages, incompatible
dimensions, and memory errors during extractor initialisation.

**`tests/test_collection_add.py`**

Checks duplicate handling, preserved analysis, identity continuity, compatibility
failures, partial success, locking, and publication recovery with isolated fixtures.

**`tests/run_add_smoke.py`**

Runs an opt-in real-model test in a new output folder. By default it reuses 20 saved videos,
analyses five held-out videos, checks source and cache preservation, repeats Add to
verify duplicate skipping, and compares grouping against reclustering the combined
cache. It is not an identity-accuracy benchmark or a fresh inference comparison
of every source video.

```powershell
python tests/run_add_smoke.py `
  --source_run runs/step1_test `
  --output_dir runs/new_add_smoke `
  --baseline_videos 20 `
  --new_videos 5
```

Run the focused regression tests:

```powershell
python -m unittest discover -s tests -v
```

### Exclude Videos From A Collection

Delete marks selected videos inactive and rebuilds albums from the remaining
saved embeddings. Source videos, model files, local tracks, crops, and debug
videos remain available. This operation uses CPU clustering and crop appearance
checks; YOLOE and Torchreid inference are not run. Original input videos and model
weights do not need to be available for Delete.

Find internal video IDs and filenames in global/video_index.json, then preview:

```powershell
python main.py delete `
  --run_dir runs/step1_test `
  --video_ids V_653D16 V_2FBFA8 `
  --dry_run
```

Run the update by omitting --dry_run:

```powershell
python main.py delete `
  --run_dir runs/step1_test `
  --video_ids V_653D16 V_2FBFA8
```

**`--video_ids`**

Accepts one or more internal video IDs. Every ID is validated before publication.
An unknown ID rejects the entire request. Duplicate IDs are deduplicated, and
already excluded IDs are ignored. Repeating a completed Delete makes no changes.

**`--dry_run`**

Shows newly excluded videos, affected local-track count, and affected player IDs
without inference, migration, or export. The collection lock file may be created.

**`--keep_backups`**

Uses the same committed-export backup retention as Add, defaulting to one previous
backup. Delete stages its database and global exports and uses the existing
publication journal. The recover command handles interrupted publication.

**Complete cache and active view**

The complete cache contains all registered video metadata and every successful
track embedding, including excluded videos. The active view contains only videos
participating in clustering. Existing local_track files in the global output
represent the active view. Clustering, album export, and video_index.json use
active data exclusively. An empty active collection is supported.

**`global/track_cache_embeddings.npy`**

Stores all successful cached embeddings. Excluding videos preserves these vectors
and their cache rows. Add appends newly extracted embeddings to this complete cache.

**`global/track_cache_index.json`**

Maps cache_row to local track, source video, crops, and embedding metadata. Active
embedding_row values are separate and can change when active arrays are rebuilt.

**`global/collection_state.json`**

Stores video_membership flags, cache format version, per-track processing states,
and operation history. A complete track from an excluded video retains its
cache_row and complete status, while its active embedding_row is null. Exclusion
is a video membership decision, not an extraction failure.

**`project.db` schema version 2**

Records every registered video with its active flag, and every track with separate
cache_row and embedding_row mappings. The loader supports version 1 registries.
Successful collection updates stage a version 2 registry and bootstrap complete
cache files from saved embeddings, without repeating inference. Validation checks
active vectors against cached vectors and verifies artifacts for excluded videos.

**Add and recluster after Delete**

Add recognises excluded video fingerprints and reports them as inactive duplicates.
It does not reactivate them, and it retries unfinished embeddings only for active
tracks. Standalone reclustering preserves exclusions and copies the complete cache
to its new destination. Explicit restoration can be added later using the cache.

Remaining groups use the existing player-ID continuity rules. Groups without
remaining tracks disappear from current albums; their IDs are not recycled.
Clustering and quality decisions may change after evidence is excluded.

**Logs and inspection**

number_of_videos and number_of_local_tracks describe active data.
number_of_registered_videos and number_of_excluded_videos describe membership.
Inspection also reports cached embeddings. Delete logs requested and newly
excluded IDs, active counts before and after, retired player IDs, stage timing,
and inference_executed set to false.

**`collection_cache.py`**

Defines the complete analysis cache, creates active views, writes both embedding
datasets, and maps cached and active rows into persistent processing states.

**`collection_delete.py`**

Implements exclusion selection, staged reclustering, export, validation, and
recoverable publication.

**`tests/test_collection_delete.py`**

Tests exclusion, reindexing, complete-cache preservation, empty collections,
idempotence, migration, transaction recovery, and Add and recluster integration.

**`tests/run_delete_smoke.py`**

Creates a separate copy of a real result, excludes selected videos, verifies all
retained video-analysis files and embedding values, checks duplicate Add and
reclustering behaviour, and compares grouping with independently filtered cached
embeddings. Inference functions are patched to fail if invoked.

```powershell
python tests/run_delete_smoke.py `
  --source_run runs/add_reliability_test/collection `
  --output_dir runs/new_delete_test `
  --delete_videos 3
```

### Sort A Player's Videos By Appearance

Player metadata and both galleries rank each player's active videos by the
percentage of frames containing that player's assigned local tracks. The
calculation merges observed frame ranges across tracks from the same video.
Overlaps count once and tracking gaps do not count. For example, tracks covering
frames 0 through 99 and 80 through 179 contribute 180 unique frames; in a
300-frame video, appearance coverage is 60 percent.

Fresh runs, Add, Delete, and recluster generate these rankings automatically.
Refresh an existing result with:

```powershell
python main.py sort `
  --run_dir runs/step1_test
```

Sort reads saved frame observations and refreshes metadata and galleries without
YOLOE, Torchreid, video decoding, or reclustering. Existing player IDs, local-track
assignments, quality scores, acceptance decisions, embeddings, and exclusions are
preserved. Source videos and model weights do not need to be available. Use
--dry_run to preview the number of rankings and unavailable measurements. The
--keep_backups option uses the same retention policy as Add and Delete.

**`video_rankings`**

Each player entry in global/player_groups.json and each exported album's
metadata.json contains a ranked list. Its fields include rank, video_id,
original_filename, source_path, visible_frames, total_frames,
appearance_percentage, frame_count_source, coverage_status, and local_tracks.
The existing videos and video_ids string arrays follow the same order. Videos
with identical filenames remain separate entries identified by their video IDs.

**Coverage and ordering**

The denominator is the saved video frame count for legacy analysis, labelled
metadata. Fresh tracking also records processed_frame_count and analysis_complete.
Completed processing counts are cross-checked against positive metadata counts;
a disagreement makes coverage unavailable. A completed processing count can supply
the denominator when metadata has no positive frame count. The recorded source
is processed for these new measurements.

Missing totals, missing observations, incomplete processing, and out-of-range
frame indices produce a null percentage with an explicit coverage_status reason.
Unknown measurements sort last and do not change album acceptance. Valid entries
sort by exact coverage ratio descending, then visible-frame count descending,
then video ID. Display rounding never controls the order.

The percentage measures recognised tracking coverage. Detection misses and
identity mistakes can affect it; it is not a ground-truth visibility measurement.
The number of saved crop images does not affect coverage.

**Gallery**

Each player has ranked video sections showing source filename, percentage, and
observed frame count. Local tracks from that video appear together. The normal
and standalone galleries use the same ordering and preserve complete crop images.

**`observed_frame_ranges`**

Track summaries retain compact inclusive ranges instead of mask arrays. The saved
loader derives these ranges from existing frame records without rewriting old
tracking JSON. Fresh analysis saves the ranges and processed frame counts.

**`player_video_sorting.py`**

Compacts and merges observed ranges, validates coverage measurements, and generates
deterministic per-player video rankings.

**`collection_sort.py`**

Implements the standalone Sort command. It validates player assignments, stages
small global and album exports, and uses the collection publication journal.
Existing large video-analysis files are read in place. The recover command handles
an interrupted publication. Logs include video_sorting with unavailable-coverage
reasons and stage timing.

**`tests/test_player_video_sorting.py` and `tests/test_collection_sort.py`**

Test overlap, gaps, exact ratios, invalid counts, metadata disagreements,
preserved identity decisions, gallery ordering, Add and Delete integration,
standalone reclustering, and rollback.

**`tests/run_sort_smoke.py`**

Creates a separate copy of a real result and independently computes frame unions
using sets of raw frame indices. It checks every ranking, both galleries,
preserved embeddings and images, unchanged identity decisions and exclusions,
and deterministic repeated Sort. Inference and clustering functions are patched
to fail if invoked.

```powershell
python tests/run_sort_smoke.py `
  --source_run runs/delete_feature_test/collection `
  --output_dir runs/new_sort_test
```

**`tests/run_workflow_smoke.py`**

Runs the complete workflow in one new collection using videos already in the
project. By default it selects a deterministic sample of 25 distinct videos,
copies 20 into an initial dataset and five into an additional dataset, and runs
fresh inference, Add, Delete, and Sort with the configured default models.
It checks duplicate operations, retained embeddings and video analysis, video
rankings against raw frame sets, and a real interrupted publication followed by
recovery. Original dataset files and model files are checked for changes.

```powershell
python tests/run_workflow_smoke.py `
  --input_dir highlights `
  --output_dir runs/new_complete_workflow_test
```

The destination must be a new folder under runs. It contains dataset copies,
dataset_manifest.json, the collection, stage_logs, and verification.json. The
report records each command, return code, timing, and validation outcome. Expected
nonzero exits during the deliberate interruption test are recorded separately.
Use --initial_videos, --add_videos, --delete_videos, and --seed to change the test
selection. This is a functional workflow test, not a labelled identity benchmark.

### Process Input Videos

Minimal quality-first run:

```powershell
python main.py `
  --input_dir highlights `
  --output_dir runs/player_album_best
```

This short command works because model paths and quality defaults are already set
in `config.py`.

Full default-equivalent command:

```powershell
$reidPath = "models/" + `
  "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
python main.py `
  --input_dir highlights `
  --output_dir runs/player_album_best `
  --yoloe_model models/yoloe-26x-seg.pt `
  --prompt person `
  --tracker botsort.yaml `
  --imgsz 960 `
  --conf 0.40 `
  --iou 0.70 `
  --device 0 `
  --half true `
  --reid_model osnet_ain_x1_0 `
  --reid_model_path $reidPath `
  --max_crops_per_track 16 `
  --min_track_frames 25 `
  --min_crop_height 150 `
  --cluster_method agglomerative `
  --distance_threshold 0.18 `
  --save_debug_video true `
  --save_gallery true
```

Small test run:

```powershell
python main.py `
  --input_dir test `
  --output_dir runs/player_album_test `
  --max_videos 1 `
  --save_debug_video false `
  --save_gallery false
```

Prompt or confidence test:

```powershell
python main.py `
  --input_dir highlights `
  --output_dir runs/player_album_person_conf045 `
  --prompt person `
  --conf 0.45
```

CPU debug run:

```powershell
python main.py `
  --input_dir test `
  --output_dir runs/player_album_cpu_debug `
  --max_videos 1 `
  --device cpu `
  --half false `
  --save_debug_video false
```

## Arguments

All arguments are configurable from the CLI. The important defaults are already
in `config.py`, so the normal run command can stay short.

**`--input_dir`**

Default: `highlights`. Pass one parent folder. The scanner includes supported
videos in every nested folder under that parent.

**`--output_dir`**

Default: `runs/player_album_best`. Run output folder.

**`--yoloe_model`**

Default: `models/yoloe-26x-seg.pt`. YOLOE segmentation model weights.

**`--prompt`**

Default: `person`. YOLOE text prompt. Good first choices are `person`, `player`,
or `athlete`.

**`--tracker`**

Default: `botsort.yaml`. Ultralytics tracker config. Use `botsort.yaml` first.
Use `bytetrack.yaml` as the backup tracker.

**`--imgsz`**

Default: `960`. YOLOE inference image size.

**`--conf`**

Default: `0.40`. Detection confidence threshold. Higher values produce fewer but
cleaner tracks.

**`--iou`**

Default: `0.70`. Detection NMS IoU threshold.

**`--device`**

Default: `0`. Torch and Ultralytics device. Use `0` for the first GPU or `cpu`
for CPU debug.

**`--half`**

Default: `true`. Use FP16 where supported.

**`--reid_model`**

Default: `osnet_ain_x1_0`. Torchreid model name used by `FeatureExtractor`.

**`--reid_model_path`**

Default: the OSNet-AIN MSMT17 checkpoint in the `models` folder. This is the
local Torchreid checkpoint path.

**`--max_crops_per_track`**

Default: `16`. Maximum best crops saved per local track.

**`--min_track_frames`**

Default: `25`. Reject local tracks shorter than this many frames when active
filtering is enabled.

**`--min_crop_height`**

Default: `150`. Reject crops shorter than this many pixels.

**`--cluster_method`**

Default: `agglomerative`. Global clustering backend.

**`--distance_threshold`**

Default: `0.18`. Agglomerative clustering cosine distance threshold.

**`--identity_split_enabled`**

Default: `true`. Enable second-pass splitting for visually inconsistent clusters.

**`--identity_split_min_cluster_size`**

Default: `8`. Only run identity splitting on clusters at least this large.

**`--identity_split_distance_threshold`**

Default: `0.14`. Distance threshold used for second-pass identity splitting.

**`--identity_color_weight`**

Default: `0.25`. Weight of outfit and color fingerprint similarity in identity
validation.

**`--identity_min_consistency`**

Default: `0.74`. Combined identity consistency threshold before a cluster is
marked weak.

**`--identity_min_color_consistency`**

Default: `0.35`. Color consistency threshold before a cluster is marked
outfit-diverse.

**`--identity_split_mean_threshold`**

Default: `0.78`. Split a cluster when average combined identity similarity falls
below this.

**`--identity_split_color_threshold`**

Default: `0.52`. Split a cluster when outfit and color consistency falls below
this.

**`--identity_split_p10_threshold`**

Default: `0.62`. Low-tail combined similarity threshold for identity splitting.

**`--identity_split_std_threshold`**

Default: `0.07`. Combined similarity standard deviation threshold for identity
splitting.

**`--identity_split_large_min_similarity`**

Default: `0.58`. Minimum pair similarity tolerated inside large identity clusters.

**`--identity_album_min_consistency`**

Default: `0.70`. Album scoring penalty threshold for identity consistency.

**`--identity_album_large_min_consistency`**

Default: `0.78`. Album scoring penalty threshold for large identity clusters.

**`--identity_album_min_color_consistency`**

Default: `0.50`. Album scoring penalty threshold for outfit and color consistency.

**`--save_debug_video`**

Default: `true`. Render per-video `tracking_debug.mp4`.

**`--save_gallery`**

Default: `true`. Render `global/gallery.html`.

**`--max_videos`**

Default: none. Optional cap for quick tests.

**`--video_extensions`**

Default: `.mp4,.mov,.avi,.mkv`. Comma-separated supported extensions.

**`--crop_padding`**

Default: `0.08`. BBox padding ratio before crop extraction.

**`--blur_threshold`**

Default: `110.0`. Minimum Laplacian sharpness score.

**`--mask_crop_mode`**

Default: `bbox`. Crop mode: `bbox`, `masked`, or `both`.

**`--min_mask_area_ratio`**

Default: `0.02`. Minimum mask area divided by bbox area.

**`--min_person_shape_score`**

Default: `0.35`. Minimum person-like crop shape score.

**`--max_edge_touch_ratio`**

Default: `0.75`. Reject crops touching too much of the frame edge.

**`--active_player_filter`**

Default: `true`. Enable track filtering based on duration, movement, crop height,
size, and activity zone.

**`--min_movement_score`**

Default: `0.04`. Minimum normalized movement score for active-player filtering.

**`--min_track_duration_sec`**

Default: `0.8`. Minimum visible duration for active-player filtering.

**`--quality_filter_export`**

Default: `true`. Use `album_quality_score` to separate accepted, review, and
rejected groups.

**`--album_accept_threshold`**

Default: `0.78`. Minimum `album_quality_score` copied into the `players` folder.

**`--album_review_threshold`**

Default: `0.55`. Minimum `album_quality_score` considered review instead of
rejected.

**`--export_review_rejected_folders`**

Default: `false`. If true, copy review and rejected folders. The default keeps
only the `players` folder.

**`--gallery_include_review_rejected`**

Default: `false`. If true, show review and rejected groups in `gallery.html`.

## Input And Output

Input videos stay untouched. The code does not delete, move, rename, modify, or
overwrite input videos or model weights.

A fresh run and Add each take one `--input_dir` per command. For example,
`highlights` can contain separate `session_a`, `session_b`, and `session_c`
folders, and one run will scan all three. For folders with separate roots, start
with one folder and use Add for each additional root against the same collection.

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

**`videos/<video_id>/tracking_debug.mp4`**

Optional debug video showing YOLOE and tracker inference boxes, masks, IDs,
confidence, prompt label, and active or ignored status.

**`videos/<video_id>/local_tracks.json`**

Per-video track data: source metadata, local track IDs, frame indexes, bboxes,
confidence, active status, crop paths, and track scores.

**`videos/<video_id>/crops/<local_track_id>/crop_####.jpg`**

Saved best crops for a local player track.

**`videos/<video_id>/crops/<local_track_id>/metadata.json`**

Crop-level metadata and quality scores for that local track.

**`global/local_track_embeddings.npy`**

One normalized ReID embedding per local player track.

**`global/local_track_index.json`**

JSON index mapping each embedding row to local player ID, video ID, source video,
crop paths, representative crop, and embedding quality information.

**`global/similarity_matrix.csv`**

Pairwise cosine similarity matrix between local track embeddings.

**`global/player_groups.json`**

Global player groups with player ID, assigned local tracks, videos, representative
crop, `album_quality_score`, export status, conflict info, and identity validation.

**`global/video_index.json`**

Video-level index showing which accepted global players appear in each video.

**`global/gallery.html`**

Visual verification gallery. By default it shows accepted player groups only. It
references crop images with relative paths, so send the run folder or a packaged
standalone export if sharing with another person.

**`players/P####/`**

Accepted player album folder. It contains `representative.jpg`, one best crop per
assigned local track, and `metadata.json`.

**`logs.json`**

Full run config, model paths, prompt, tracker, timing, warnings, errors, number
of videos, number of local tracks, and number of global, accepted, review, and
rejected players.

Only accepted player albums are copied into the `players` folder by default.
Review and rejected groups remain in JSON metadata unless
`--export_review_rejected_folders true` is enabled.

## Video ID Reproducibility

Internal video IDs use a content hash of the video bytes. The same video file
should receive the same `video_id` even if it is moved into a different folder,
copied to another machine, or mounted through Docker.

This matters because output folders, local track IDs, `video_index.json`, and
`player_groups.json` all use the internal video ID. Content-based IDs make old
and new runs easier to compare when the file paths change.

The most important run settings are saved into `logs.json`, including model
paths, prompt, tracker, confidence threshold, ReID model, clustering config,
export config, timing, warnings, and errors.
