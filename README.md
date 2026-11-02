# Visual Localization against Mesh-Rendered Databases

The reference database of each scene is **rendered from a colored 3D mesh**, so every reference
image comes with an exact depth map. A query image is localized in four steps:

```
query image
   |  1. global retrieval      retriever.embed() + FAISS cosine search -> top-k rendered references
   |  2. 2D-2D matching        matcher.match(query, reference) for each of the k candidates
   |  3. geometric rerank      the candidate with the most fundamental-matrix RANSAC inliers wins
   |  4. absolute PnP          reference keypoints lifted to 3D with the rendered depth,
   v                           then PnP-RANSAC -> 6-DoF camera pose of the query
estimated pose (cam2world)
```

Retrievers and matchers are interchangeable components. The experiments evaluate every
combination of 6 retrievers and 7 matchers on 10 scenes.

| Retrievers | Matchers |
|---|---|
| `salad` (DINOv2-SALAD), `megaloc` (MegaLoc), `anyloc` (AnyLoc VLAD), `netvlad` (NetVLAD, Pitts30k), `dinov2` (DINOv2 CLS token), `dinov3` (DINOv3 CLS token) | `lightglue` (SuperPoint + LightGlue), `superglue` (SuperPoint + SuperGlue), `roma_precise` / `roma_turbo` (RoMa v2, two presets), `omniglue` (OmniGlue), `dinov2_matcher` (SuperPoint keypoints + DINOv2 features, mutual NN), `salad_matcher` (SuperPoint keypoints + SALAD cluster probabilities, mutual NN) |

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Data preparation](#data-preparation)
- [Running locally](#running-locally)
- [Running on an HPC cluster with SLURM (CINECA Leonardo)](#running-on-an-hpc-cluster-with-slurm-cineca-leonardo)
- [Outputs](#outputs)
- [Timing and memory measurement](#timing-and-memory-measurement)
- [Evaluation and metrics](#evaluation-and-metrics)
- [Reference results](#reference-results)
- [Configuration](#configuration)
- [Adding a new retriever or matcher](#adding-a-new-retriever-or-matcher)

## Repository structure

```
loc/                        localization pipeline (Python package)
  localize.py               command-line entry point: runs one retriever/matcher pair on one scene
  database.py               FAISS index of the reference descriptors, with a shared cache
  pose.py                   matching of the k candidates, geometric rerank and PnP
  geometry.py               depth back-projection, pose composition, pose error
  datasets.py               registry of the scenes and loaders of references and queries
  capture_io.py             readers/writers of the Capture text format (LaMAR)
  evaluate.py               recall and median errors of an estimated trajectory
  vram.py                   GPU memory counters (torch and NVML)
  paths.py                  default locations of data, outputs and third-party code
  utils.py                  JSON helpers, logging, parsing of --*-opt options
  retrievers/               one file per retriever + registry (retrievers/__init__.py)
  matchers/                 one file per matcher + registry (matchers/__init__.py)
hpc/                        execution on a SLURM cluster
  submit.py                 writes the plan of the experiments and submits a job array
  localize.sbatch           one array task = one (scene, retriever, matcher) combination
  env.sh                    environment shared by the login node and the jobs
  prefetch_models.py        downloads every model; checks that they load offline
  offline_hub.py            lets torch.hub load cached repositories without internet
scripts/
  run_matrix_local.py       runs several combinations sequentially on a local GPU
  prepare/                  conversion of the public datasets and rendering of the databases
tests/                      unit and end-to-end tests (CPU only, no data needed)
```

Folders created at run time, ignored by git: `data/` (datasets), `output/` (results and caches),
`third_party/` (cloned repositories and OmniGlue weights), `cache/` (model weights on a cluster),
`envs/` (virtual environments on a cluster), `jobs/` and `logs/` (SLURM plans and logs).

## Installation

The code was run with Python 3.12 on Linux with NVIDIA A100 GPUs (CUDA 12.1). A GPU is needed to run
the experiments in a reasonable time; the tests run on the CPU.

```bash
git clone https://github.com/RiccardoN001/visual-localization-benchmark.git
cd visual-localization-benchmark
```

**Main environment** (every retriever, every matcher except OmniGlue):

```bash
python -m venv envs/loc
source envs/loc/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install romav2 --no-deps    # its metadata declares an unsatisfiable dependency
```

**Model weights.** DINOv3 (used by RoMa v2 and by the `dinov3` retriever) is gated on
HuggingFace: request access on the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and export a token. Then download everything once:

```bash
export HF_TOKEN=hf_...
python -m hpc.prefetch_models
```

`prefetch_models.py` builds every registered component once on the CPU, which downloads its
weights into the torch.hub and HuggingFace caches. It also clones AnyLoc, SuperGlue and OmniGlue
into `third_party/` and downloads the OmniGlue weights. It is idempotent; `--only a,b` and
`--skip a,b` restrict it to some components.

**OmniGlue environment.** OmniGlue uses TensorFlow 2.15, which needs cuDNN 8 and therefore an
older torch, so it lives in its own environment:

```bash
python -m venv envs/omniglue
source envs/omniglue/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-omniglue.txt
python -m hpc.prefetch_models --only omniglue
```

**Data preparation environment.** The scripts in `scripts/prepare/` use Open3D and NumPy 2, so
they get a separate environment (no GPU needed):

```bash
python -m venv envs/prep
source envs/prep/bin/activate
pip install -r requirements-prep.txt
```

**Tests:**

```bash
python -m pytest tests
```

## Datasets

No data is included in this repository. The experiments use the following scenes:

| Scene (`--dataset`) | Source | Queries | Reference database |
|---|---|---|---|
| `hge`, `cab`, `lin` | [LaMAR](https://lamar.ethz.ch) ([data format and download](https://github.com/microsoft/lamar-benchmark)) | images of the NavVis scanner session (`raw_data/images_undistr_center`) | rendered from the NavVis mesh `proc/meshes/mesh_simplified.ply` |
| `hgephone`, `linphone` | LaMAR, `sessions/query_val_phone` | official phone validation queries (`queries.txt`), taken at a different time than the scan | the database of `hge` / `lin` |
| `bodleian`, `christchurch`, `keble`, `observatory` | [Oxford Spires](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/) | `cam0` fisheye images, undistorted | rendered from a colored mesh reconstructed from the laser scans |
| `sanandrea` | San Andrea, available from the authors on request | three cameras with different intrinsics | rendered with Blender from a textured mesh |

The NavVis sessions used for LaMAR are `HGE/navvis_2022-02-06_12.55.11`,
`CAB/navvis_2022-06-25_11.14.36` and `LIN/navvis_2022-07-03_08.30.21`. The Oxford Spires meshes
and the rendered databases are not part of the public datasets and are available from the
authors on request.

LaMAR is distributed under CC BY-SA 4.0; please follow the license terms of each dataset.

## Data preparation

Every scene ends up in the same layout, read by `loc/datasets.py`:

```
data/<SCENE>/
  capture/                  queries
    sensors.txt             PINHOLE intrinsics of each query camera
    images.txt              timestamp, sensor_id, image path relative to raw_data/
    trajectories.txt        ground-truth cam2world poses (queries without a pose are allowed)
    raw_data/...            the images
  db/                       rendered reference database
    db_sensors.txt          intrinsics of the renders
    db_trajectories.txt     cam2world pose of each render
    images/<stem>.jpg       rendered RGB images
    depth_png/<stem>.png    rendered depth, uint16 PNG, depth_m = value / depth_scale
    depth_scale.json        {"depth_scale": 250.0}
```

`<SCENE>` is the upper-case dataset name (`HGE`, `KEBLE`, ...). The phone scenes only have a
`capture/` folder: their database is the one of the base scene (`data/HGE/db`, `data/LIN/db`).
All files use the [Capture format](https://github.com/microsoft/lamar-benchmark/blob/main/CAPTURE.md):
poses are cam2world, quaternions are written as `qw, qx, qy, qz`, and depth is the Z coordinate
along the optical axis.

**Database poses.** The rendered views are generated with `scripts/prepare/make_db_poses.py` from
positions sampled on the walkable part of each mesh (a JSON list of `"position": [x, y, z]`
entries): 4 headings 90 degrees apart (`--yaws`), no pitch (`--pitch`), and only positions within
0.25 m of the height range of the query trajectory (`--z-margin`):

```bash
python scripts/prepare/make_db_poses.py --points <sampled_positions.json> \
    --reference-trajectories data/HGE/capture/trajectories.txt --out data/HGE/db_poses.txt
```

Renders that see the mesh in less than half of the image are skipped, so a database can have
slightly fewer images than poses.

The commands below run in the data preparation environment (`source envs/prep/bin/activate`).

### LaMAR NavVis scenes (`hge`, `cab`, `lin`)

The queries are the NavVis session itself, and mesh, database poses and ground truth are all in
the frame of the session.

```bash
S=<lamar>/HGE/navvis_2022-02-06_12.55.11
mkdir -p data/HGE/capture/raw_data
cp $S/sensors.txt $S/images.txt $S/trajectories.txt data/HGE/capture/
ln -s $S/raw_data/images_undistr_center data/HGE/capture/raw_data/images_undistr_center

python scripts/prepare/render_db.py --root data/HGE/capture \
    --mesh-path $S/proc/meshes/mesh_simplified.ply \
    --poses data/HGE/db_poses.txt --out data/HGE/db
```

Repeat for `CAB` and `LIN` with their NavVis sessions and pose files.

### LaMAR phone scenes (`hgephone`, `linphone`)

The phone ground truth is in the global LaMAR frame. It is moved to the NavVis frame of the
database with the inverse of the session alignment:

```bash
python scripts/prepare/phone_prepare.py \
    --session <lamar>/HGE/sessions/query_val_phone \
    --alignment <lamar>/HGE/navvis_2022-02-06_12.55.11/proc/alignment_global.txt \
    --out data/HGEPHONE/capture
```

The same for `LIN` -> `data/LINPHONE/capture`. The database of `data/HGE` and `data/LIN` is reused.

### Oxford Spires scenes (`bodleian`, `christchurch`, `keble`, `observatory`)

The meshes are assumed to be in the frame of the ground truth.

```bash
# 1. queries: undistort the fisheye images and convert the TUM ground truth (body frame) to
#    optical camera poses
python scripts/prepare/oxford_prepare.py --images <sequence>/cam0 --calib <oxford>/cam0.yaml \
    --gt <sequence>/gt-tum.txt --out data/KEBLE/capture

# 2. only for meshes larger than the available RAM: out-of-core simplification,
#    written to <mesh>_simplified.ply
python scripts/prepare/simplify_mesh.py <mesh>.ply --voxel 0.03

# 3. database, rendered with the undistorted intrinsics of step 1
python scripts/prepare/render_db.py --root data/KEBLE/capture --mesh-path <mesh>_simplified.ply \
    --poses data/KEBLE/db_poses.txt --out data/KEBLE/db
```

### San Andrea (`sanandrea`)

```bash
python scripts/prepare/sanandrea_prepare.py --scene <sanandrea_raw> --out data/SANANDREA
```

### Expected sizes

| Scene | Database poses | Database images | Queries |
|---|---|---|---|
| HGE | 1392 | 1361 | 2408 |
| CAB | 3000 | 2998 | 3696 |
| LIN | 2504 | 2443 | 3972 |
| HGEPHONE | (HGE) | (HGE) | 476 |
| LINPHONE | (LIN) | (LIN) | 605 |
| BODLEIAN | 1368 | 1368 | 4966 |
| CHRISTCHURCH | 1192 | 1154 | 11775 |
| KEBLE | 568 | 565 | 5682 |
| OBSERVATORY | 576 | 576 | 5746 |
| SANANDREA | 126 | 126 | 519 |

The database sizes are those of the experiments: a database rendered from other poses has a
different size. `db/images` and `db/depth_png` must contain the same number of files.

## Running locally

One retriever/matcher pair on one scene:

```bash
python -m loc.localize --dataset keble --retriever salad --matcher lightglue
python -m loc.evaluate --dataset keble --results output/localization/keble_salad_lightglue_k3
```

Useful options: `--limit 20` (first 20 queries only, for a quick check), `--query <qid>` (one
query), `--resume` (keep the queries that already have a result, e.g. after an interruption),
`--k 1` (no rerank). Every option is listed by `python -m loc.localize --help` and in
[Configuration](#configuration).

Several combinations, one after the other on the same GPU:

```bash
python scripts/run_matrix_local.py --dataset keble --all \
    --omniglue-python envs/omniglue/bin/python
python scripts/run_matrix_local.py --dataset keble,observatory --retriever salad,megaloc \
    --matcher lightglue,roma_turbo -- --limit 50
```

OmniGlue combinations are run with the interpreter of the OmniGlue environment, and skipped when
`--omniglue-python` is not given. Options after `--` are forwarded to `loc.localize`.

The first time a retriever is used on a scene, the descriptors of all references are computed and
cached in `output/retrieval_cache/<retriever>/<fingerprint>/`. The fingerprint depends only on the
database content, so every matcher (and every later run) reuses the cache.

## Running on an HPC cluster with SLURM (CINECA Leonardo)

The full experiment matrix (10 scenes x 6 retrievers x 7 matchers) was run on the Booster
partition of Leonardo at CINECA ([user documentation](https://docs.hpc.cineca.it/)), where each
combination is an independent task of a SLURM job array. This section describes that setup;
the same scripts work on other SLURM clusters after the changes listed at the end.

### How the cluster is used

- **Login nodes** have internet access but must not run computations (CPU time is limited). They
  are used to create the environments, download the models and submit jobs.
- **Compute nodes** have GPUs but **no internet access**. Every model must therefore be
  downloaded in advance from the login node; the jobs run with `HF_HUB_OFFLINE=1` and with
  `hpc/offline_hub.py`, which lets `torch.hub` load already cached repositories.
- **Storage.** `$HOME` has a small quota (50 GB) and must not host virtual environments;
  `$WORK` (project space, 1 TB) holds the repository, the environments, the model caches, the data
  and the results. `$SCRATCH` deletes files older than 40 days and is not suited to results.
- **Jobs.** A Booster node has 4 A100 GPUs and 32 cores. Each task asks for 1 GPU and 8 cores, a
  quarter of a node, so SLURM packs four tasks per node and accounts 1 GPU-hour per hour.
  `boost_usr_prod` is the GPU partition; the QoS sets the limits: `normal` (24 h walltime, the
  default), `boost_qos_dbg` (30 min, at most 2 jobs, higher priority: for quick tests),
  `boost_qos_lprod` (up to 4 days).

### 1. Access

Leonardo uses short-lived SSH certificates (12 h) issued by the CINECA certificate authority with
the `step` client, plus two-factor authentication. Follow the access instructions of the
[CINECA documentation](https://docs.hpc.cineca.it/), then:

```bash
ssh <username>@login.leonardo.cineca.it
saldo -b                                   # budget of your projects: the account name is here
echo 'export LOC_SLURM_ACCOUNT=<account>' >> ~/.bashrc
```

### 2. Code, environments and models (login node)

```bash
cd $WORK
git clone https://github.com/RiccardoN001/visual-localization-benchmark.git
cd visual-localization-benchmark
source hpc/env.sh       # loads the modules (profile/deeplrn, cuda/12.1) and sets the model caches
```

Create `envs/loc` and `envs/omniglue` as in [Installation](#installation), with `python -m venv`
or `uv venv`. Then download the models with `HF_TOKEN` set:

```bash
source envs/loc/bin/activate && python -m hpc.prefetch_models
source envs/omniglue/bin/activate && python -m hpc.prefetch_models --only omniglue
```

`hpc/env.sh` puts the torch.hub and HuggingFace caches inside the repository (`cache/`), so the jobs
read exactly what was downloaded. The download is I/O bound and fits in the CPU-time limit of the
login node; if it gets killed, run it on the `lrd_all_serial` partition.

Check, **on a compute node**, that everything loads without network access:

```bash
srun -A $LOC_SLURM_ACCOUNT -p boost_usr_prod -q boost_qos_dbg --gres=gpu:1 \
     --cpus-per-task=8 --time=00:15:00 --pty bash
source hpc/env.sh && source envs/loc/bin/activate
python -m hpc.prefetch_models --check-only
exit
```

### 3. Data

Prepare the scenes on a workstation (see [Data preparation](#data-preparation)) and copy
`data/<SCENE>/{capture,db}` to `$WORK/visual-localization-benchmark/data/`, for example with
`rsync -a data/KEBLE <username>@login.leonardo.cineca.it:'$WORK/visual-localization-benchmark/data/'`.
Check the counts against [Expected sizes](#expected-sizes).

### 4. Submitting experiments

`hpc/submit.py` expands the requested scenes x retrievers x matchers, writes a **plan file**
(`jobs/<timestamp>_<name>.tsv`, one line per combination) and submits **one job array** whose task
*i* reads line *i* of the plan and runs `loc.localize` for that combination
(`hpc/localize.sbatch`). OmniGlue tasks automatically use `envs/omniglue`.

A first test with one combination and a few queries, on the debug QoS:

```bash
python -m hpc.submit --dataset keble --retriever salad --matcher lightglue \
    --limit 20 --qos boost_qos_dbg --time 00:25:00
```

The full experiment matrix:

```bash
python -m hpc.submit --dataset hge,cab,lin,hgephone,linphone,bodleian,christchurch,keble,observatory,sanandrea \
    --all --warmup --dry-run      # prints the plan and the sbatch commands
python -m hpc.submit --dataset hge,cab,lin,hgephone,linphone,bodleian,christchurch,keble,observatory,sanandrea \
    --all --warmup
```

- `--all` uses every registered retriever and matcher for the fields not given explicitly.
- `--warmup` first submits a small array with one task per (scene, retriever), which builds the
  retrieval cache on a single query. The main array starts only after all of them succeed
  (`--dependency=afterok`), so the matchers of the same pair do not all compute the same reference
  descriptors at the same time.
- `--throttle N` limits the number of tasks running at the same time.
- `--time` sets the walltime of each task (default 6 h; a shorter walltime starts sooner).
- `--resume` resubmits the same combinations and skips the queries that already have a result:
  use it, with the same arguments, after a task hit its walltime.

### 5. Monitoring

```bash
python -m hpc.submit --status                     # = squeue --me
python -m hpc.submit --progress                   # state of the last plan, per scene
python -m hpc.submit --progress --detail          # also one line per combination
python -m hpc.submit --progress --only partial    # only the interrupted combinations
tail -f logs/<job-name>_<jobid>_<task>.out        # log of one task
ls output/localization/<run>/results | wc -l      # queries done in a run
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode   # finished tasks
scancel <jobid>                                    # the whole array (or <jobid>_<task>)
saldo -b                                           # remaining budget
```

`--progress` combines the state in `squeue`, the result files and the last line of each log, so it
also works after the jobs have left the queue: `partial` marks a combination that stopped before
the end, to run again with `--resume`. `--plan <file.tsv>` selects a plan other than the last one,
and `--only` also accepts `running`, `pending`, `done`, `ended`, `idle`, `todo` or part of a run
name.

A task that ends early usually shows the reason at the end of its log: missing data
(`database not found`), a model that was not downloaded (network error with `HF_HUB_OFFLINE`), or
a module name that changed. `PEND:DependencyNeverSatisfied` means that a warm-up task failed and
the dependent array will never start: cancel it, fix the problem and submit again.

### Other SLURM clusters

- `hpc/localize.sbatch`: change the `#SBATCH --partition` and the resources (or pass
  `--partition` to `submit.py`).
- `hpc/env.sh`: set `LOC_MODULES` to the modules of the cluster (or to an empty string).
- `submit.py --qos`: use a QoS that exists on the cluster.

## Outputs

Each run writes `output/localization/<dataset>_<retriever>_<matcher>_k<k>/`:

| File | Content |
|---|---|
| `estimated_trajectories.txt` | estimated cam2world pose of every localized query (Capture format) |
| `batch_timing.json` | summary of the run: configuration, timing statistics, recall, median errors, GPU memory |
| `results/<qid>_result.json` | per query: success, estimated pose, chosen candidate and inliers, ground-truth errors, timing, memory peaks |
| `retrieval/<qid>_retrieval.json` | top-k candidates with their similarity |
| `matches/<qid>_matches.json` | matched keypoints of every candidate (disable with `--no-save-matches`) |

## Timing and memory measurement

Everything is measured in `localize_query()` in `loc/localize.py`, with `time.perf_counter()`.
GPU work is asynchronous, so the device is synchronized before each timer is read. Each
`results/<qid>_result.json` has the same `timing` block for every combination:

| Field | What it includes |
|---|---|
| `retrieval_sec` | image loading and preprocessing + forward pass of the retriever (`embed_sec`) + FAISS search on the CPU (`faiss_sec`) |
| `match_sec` | the matcher on all k candidates (image loading included; the query features are extracted once and cached) + the fundamental-matrix RANSAC rerank |
| `pnp_sec` | reading the depth map (`depth_load_sec`) + back-projection (`backproject_sec`) + `solvePnPRansac` (`solve_sec`) |
| `estimate_sec` | `retrieval_sec + match_sec + pnp_sec` |
| `total_sec` | wall-clock time of the query, including the JSON files written for it |
| `detail` | internal timers of the components, e.g. keypoint extraction vs matching, or the four stages of RoMa v2 |

The first query also pays for CUDA initialization; no query is excluded, and the summary in
`batch_timing.json` reports both means and medians (`timing_mean_sec`, `timing_median_sec`).
`setup_time_sec` is the time to load the models and build or load the retrieval index.

GPU memory (`loc/vram.py`) is measured in MB at well-defined points:

| Field of `batch_timing.json` (`vram_mb`) | Meaning |
|---|---|
| `retriever_weights`, `matcher_weights` | increase of allocated memory when the model is loaded (models stay resident for the whole run) |
| `db_build_peak` | peak while building or loading the retrieval index |
| `retrieval_peak_median`, `match_peak_median` | median over queries of the peak allocated memory of each stage; peak counters are reset before every stage |
| `*_peak_reserved_median` | the same with reserved memory (blocks kept by the PyTorch caching allocator) |
| `max_stage_peak` | maximum of all stage peaks of the run |
| `run_reserved_peak` | memory reserved by PyTorch at the end of the run, its high-water mark |
| `nvml_process_at_end` | memory of the process according to the driver (NVML), including the CUDA context and non-PyTorch allocations |

The PyTorch counters do not see TensorFlow: for OmniGlue only `nvml_process_at_end` is
meaningful. NVML needs `nvidia-ml-py` and assumes a single visible GPU, as in the SLURM jobs.
Memory statistics are computed only over the queries processed in the current invocation, not
over the queries reloaded by `--resume`.

## Evaluation and metrics

For each query the position error is the distance between the estimated and the ground-truth
camera centers, and the rotation error is the angle of `R_est^T R_gt`. A query is localized at a
threshold `(m, deg)` when both errors are below it. The default thresholds are

```
(0.10 m, 1 deg)   (0.25 m, 2 deg)   (0.50 m, 3 deg)   (1.00 m, 5 deg)
```

defined once in `THRESHOLDS` in `loc/evaluate.py`. Recall is computed over all queries with
ground truth: a query whose PnP fails counts as a miss. `localize.py` writes these numbers into
`batch_timing.json`; `loc.evaluate` recomputes them from `estimated_trajectories.txt`, with any
thresholds, without running the experiment again:

```bash
python -m loc.evaluate --dataset keble --results output/localization/keble_salad_lightglue_k3 \
    --thresholds 0.25:2,0.5:5,5:10 --denominator all
```

`--denominator all` also counts the queries without ground truth as misses (on ChristChurch, 623
queries were recorded outside the ground-truth time window). The report is written to
`evaluation.json` next to the trajectory.

## Reference results

Recall (%) of four combinations with the default options (k = 3), at (0.25 m, 2 deg) /
(1 m, 5 deg), as computed by `loc.evaluate`. They can be used to check a run; small differences
are expected with other GPUs and library versions.

| Scene | MegaLoc + RoMa v2 precise | MegaLoc + RoMa v2 turbo | SALAD + LightGlue | NetVLAD + LightGlue |
|---|---|---|---|---|
| `keble` | 88.6 / 94.1 | 76.5 / 88.6 | 64.7 / 81.7 | 48.0 / 60.6 |
| `bodleian` | 90.2 / 92.1 | 82.6 / 88.4 | 80.3 / 91.1 | 68.8 / 80.2 |
| `christchurch` | 77.3 / 94.4 | 59.7 / 88.4 | 55.4 / 84.9 | 39.3 / 63.2 |
| `observatory` | 21.6 / 98.5 | 25.7 / 95.6 | 26.7 / 94.9 | 25.2 / 90.9 |
| `hge` | 89.7 / 91.6 | 86.7 / 89.7 | 73.0 / 76.0 | 63.1 / 65.0 |
| `cab` | 73.3 / 79.0 | 65.6 / 74.7 | 58.8 / 65.1 | 50.4 / 55.5 |
| `lin` | 89.8 / 92.4 | 86.4 / 90.4 | 81.1 / 85.2 | 74.0 / 76.8 |
| `hgephone` | 76.5 / 87.8 | 60.9 / 77.1 | 66.8 / 75.8 | 47.9 / 59.0 |
| `linphone` | 83.6 / 92.1 | 66.8 / 80.3 | 70.6 / 85.0 | 45.0 / 56.2 |
| `sanandrea` | 58.4 / 89.8 | 50.1 / 85.5 | 51.4 / 76.5 | 49.7 / 70.7 |

Latency per query and peak GPU memory of each component, measured separately on an NVIDIA A100
(64 GB):

| Retriever | Latency (ms) | VRAM (MB) | | Matcher | Latency (ms) | VRAM (MB) |
|---|---|---|---|---|---|---|
| `netvlad` | 40 | 1465 | | `superglue` | 99 | 2988 |
| `dinov2` | 76 | 1785 | | `lightglue` | 170 | 1558 |
| `dinov3` | 43 | 1756 | | `omniglue` | 550 | 3481 |
| `anyloc` | 106 | 1795 | | `roma_turbo` | 65 | 3469 |
| `salad` | 48 | 992 | | `roma_precise` | 264 | 6807 |
| `megaloc` | 47 | 1500 | | `dinov2_matcher` | 277 | 2679 |
| | | | | `salad_matcher` | 239 | 1851 |

## Configuration

### `loc.localize`

| Option | Default | Meaning |
|---|---|---|
| `--dataset`, `--retriever`, `--matcher` | required | the scene and the two components |
| `--k` | 3 | number of retrieved candidates matched and reranked |
| `--retriever-opt KEY=VALUE`, `--matcher-opt KEY=VALUE` | | override a constructor argument of the component (repeatable), see below |
| `--pnp-reprojection-error` | 8.0 | RANSAC threshold of PnP, in pixels |
| `--pnp-iterations` | 100 | RANSAC iterations of PnP |
| `--pnp-confidence` | 0.99 | RANSAC confidence of PnP |
| `--rerank-threshold` | 3.0 | RANSAC threshold of the fundamental-matrix rerank, in pixels |
| `--limit`, `--query`, `--query-stride` | | subset of the queries (first N, one query, one capture stop every N) |
| `--resume` | off | reuse existing per-query results |
| `--no-save-matches` | off | do not write `matches/` |
| `--name` | `<dataset>_<retriever>_<matcher>_k<k>` | name of the run folder |
| `--data-root`, `--output-root` | `data/`, `output/` | locations of the datasets and of the results |

The configuration actually used (including the component options) is stored in
`batch_timing.json`. Changing options does not change the run name: use `--name` to keep
different settings apart.

### Component options

Every argument of a component constructor can be set with `--retriever-opt` / `--matcher-opt`
(values are parsed as Python literals), for example
`--matcher-opt max_kpts=4096 --matcher-opt resize=1024`. Defaults are the values used in the experiments.

| Component | Options (default) |
|---|---|
| `lightglue` | `max_kpts` (2048), `resize` (None = full resolution) |
| `superglue` | `weights` ("outdoor"), `max_kpts` (2048), `kp_threshold` (0.005), `match_threshold` (0.2), `sinkhorn_iterations` (20) |
| `roma_precise`, `roma_turbo` | `setting` ("precise" / "turbo"), `certainty` (0.5), `max_kp` (2000), `n_sample` (5000) |
| `omniglue` | `min_conf` (0.02) |
| `dinov2_matcher` | `max_kpts` (2048), `min_score` (None), `ratio` (None), `height` (336), `width` (448), `hub_model` ("dinov2_vitl14") |
| `salad_matcher` | `max_kpts` (2048), `min_score` (None), `ratio` (None) |
| `salad`, `megaloc` | `height` (322), `width` (322) |
| `anyloc` | `model` ("dinov2_vitl14"), `layer` (18), `facet` ("value"), `num_clusters` (32), `seed` (0), `height` (336), `width` (448) |
| `netvlad` | `height` (480), `width` (640), `whiten` (True) |
| `dinov2` | `height` (336), `width` (448), `hub_model` ("dinov2_vitl14") |
| `dinov3` | `height`, `width` (None = 224x224 of the checkpoint), `hf_model` |

With `hpc/submit.py`, the same `--retriever-opt` / `--matcher-opt` / `--limit` / `--k` options are
forwarded to every task (values cannot contain spaces).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LOC_DATA_ROOT` | `<repo>/data` | datasets |
| `LOC_OUTPUT_ROOT` | `<repo>/output` | results and retrieval cache |
| `LOC_THIRD_PARTY` | `<repo>/third_party` | cloned AnyLoc, SuperGlue and OmniGlue |
| `TORCH_HOME`, `HF_HOME` | user cache (`<repo>/cache` with `hpc/env.sh`) | model weights |
| `HF_TOKEN` | | HuggingFace token, needed only to download DINOv3 |
| `LOC_OFFLINE_HUB` | unset | `1` on nodes without internet access (set by the job) |
| `LOC_OFFLINE_HUB_VERBOSE` | unset | `1` prints the repositories found in the torch.hub cache |
| `LOC_SLURM_ACCOUNT` | | SLURM account for `hpc/submit.py` |
| `LOC_MODULES` | `profile/deeplrn cuda/12.1` | modules loaded by `hpc/env.sh` |
| `LOC_VENV`, `LOC_VENV_OMNIGLUE` | `envs/loc`, `envs/omniglue` | environments used by the jobs |

## Adding a new retriever or matcher

A component is a class in its own file plus one line in the registry. Nothing else in the
pipeline, the local runner or the SLURM launcher has to change: `--retriever` / `--matcher`
choices, `--all`, `prefetch_models.py` and the run names are all derived from the registries.

**Matcher.** Create `loc/matchers/my_matcher.py`:

```python
import numpy as np
import torch

from . import Matcher


class MyMatcher(Matcher):
    name = "my_matcher"

    def __init__(self, device, max_kpts=2048, **cfg):
        super().__init__(device, max_kpts=max_kpts, **cfg)   # cfg is saved with the results
        self.model = ...                                      # load the weights here

    @torch.inference_mode()
    def match(self, query_path, ref_path):
        ...
        # two (N, 2) float32 arrays: matched pixels in the query and in the reference,
        # in full-resolution coordinates of the original images
        return pts_q.astype(np.float32), pts_r.astype(np.float32)
```

and register it in `MATCHER_SPECS` in `loc/matchers/__init__.py`:

```python
"my_matcher": (".my_matcher", "MyMatcher", {}),
```

The third element holds preset options: two entries can share a class with different presets, as
`roma_precise` and `roma_turbo` do. Coordinates must be at full resolution because the reference
points are looked up in the rendered depth map. `match()` is called once per candidate with the
same query, so caching the query features (as `lightglue.py` does) avoids extracting them k times.
Optional methods: `timing_detail()` / `reset_timers()` to report internal timers, and
`config_summary()` to change what is stored as configuration.

**Retriever.** Create `loc/retrievers/my_retriever.py` with a subclass of `Retriever` whose
`embed(img_path)` returns a `(1, D)` float32 **L2-normalized** descriptor (the index uses inner
products as cosine similarity), and add it to `RETRIEVER_SPECS` in `loc/retrievers/__init__.py`.
If the descriptor depends on the database (like the VLAD vocabulary of AnyLoc), implement
`prepare(db_image_paths, cache_dir)`: it is called before any `embed()`, both when the index is
built and when it is loaded from the cache.

**Then:**

```bash
python -m hpc.prefetch_models --only my_matcher       # downloads the weights (built on the CPU)
python -m loc.localize --dataset keble --retriever salad --matcher my_matcher --limit 20
```

If the code of the method must be cloned instead of installed, add it to `CLONES` in
`hpc/prefetch_models.py`. If it needs its own environment, select it in `hpc/localize.sbatch` as
done for OmniGlue, and pass its interpreter to `scripts/run_matrix_local.py`.
