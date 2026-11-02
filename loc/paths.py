import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every location can be redirected with an environment variable, so the same code runs on a
# workstation and on a cluster where data, outputs and model caches live on different filesystems.
DATA_ROOT = Path(os.environ.get("LOC_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("LOC_OUTPUT_ROOT", REPO_ROOT / "output"))
THIRD_PARTY_ROOT = Path(os.environ.get("LOC_THIRD_PARTY", REPO_ROOT / "third_party"))


def experiment_name(dataset, retriever, matcher, k):
    return f"{dataset}_{retriever}_{matcher.replace('_', '-')}_k{k}"


def run_dirs(output_root, name):
    root = Path(output_root) / "localization" / name
    dirs = {
        "root": root,
        "retrieval": root / "retrieval",
        "matches": root / "matches",
        "results": root / "results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def db_fingerprint(db_trajectories_path, n_images):
    """Identity of a reference database: content of db_trajectories.txt plus the image count.

    Re-rendering the database changes the fingerprint, which invalidates the retrieval cache.
    """
    h = hashlib.sha256(Path(db_trajectories_path).read_bytes())
    h.update(str(n_images).encode())
    return h.hexdigest()[:12]


def retrieval_cache_dir(output_root, retriever_name, fingerprint):
    # The cache depends only on the retriever and the database, so every matcher reuses it.
    d = Path(output_root) / "retrieval_cache" / retriever_name / fingerprint
    d.mkdir(parents=True, exist_ok=True)
    return d
