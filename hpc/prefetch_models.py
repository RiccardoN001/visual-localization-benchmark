"""Download every model needed by the experiments, or verify that they work offline.

Run it from a node with internet access (the login node on Leonardo), after `source hpc/env.sh`:

    export HF_TOKEN=hf_...        # DINOv3 (used by RoMa v2 and by the dinov3 retriever) is gated
    python -m hpc.prefetch_models
    python -m hpc.prefetch_models --only salad,lightglue

Then check, from a compute node, that everything loads without network access:

    python -m hpc.prefetch_models --check-only
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tarfile
import traceback
import zipfile
from pathlib import Path

# Components whose code is cloned instead of installed, and the files OmniGlue needs.
CLONES = {
    "anyloc": ("AnyLoc", "https://github.com/AnyLoc/AnyLoc.git"),
    "superglue": ("SuperGluePretrainedNetwork",
                  "https://github.com/magicleap/SuperGluePretrainedNetwork.git"),
    "omniglue": ("omniglue", "https://github.com/google-research/omniglue.git"),
}
OMNIGLUE_WEIGHTS = {
    "sp_v6": "https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz",
    "dinov2_vitb14_pretrain.pth":
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "og_export": "https://storage.googleapis.com/omniglue/og_export.zip",
}


def clone(name, check_only):
    from loc.paths import THIRD_PARTY_ROOT

    folder, url = CLONES[name]
    target = THIRD_PARTY_ROOT / folder
    if (target / ".git").exists():
        print(f"  already present: {target}")
        return target
    if check_only:
        raise SystemExit(f"{target} not found")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    return target


def download_omniglue_weights(repo_dir):
    import torch.hub

    models = Path(repo_dir) / "models"
    models.mkdir(exist_ok=True)
    for name, url in OMNIGLUE_WEIGHTS.items():
        if (models / name).exists():
            continue
        archive = models / Path(url).name
        torch.hub.download_url_to_file(url, str(archive))
        if archive.suffix == ".tgz":
            with tarfile.open(archive) as tar:
                tar.extractall(models)
            archive.unlink()
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(models)
            archive.unlink()


def prefetch(kind, name, check_only):
    """Construct the component once on the CPU, which downloads (or reads) all its weights."""
    from loc.matchers import get_matcher
    from loc.retrievers import get_retriever

    if name in CLONES:
        repo = clone(name, check_only)
        if name == "omniglue":
            if not check_only:
                download_omniglue_weights(repo)
            missing = [w for w in OMNIGLUE_WEIGHTS if not (repo / "models" / w).exists()]
            if missing:
                raise SystemExit(f"OmniGlue weights not found in {repo / 'models'}: {missing}")
            if importlib.util.find_spec("tensorflow") is None:
                print("  weights present; loading skipped (run again from envs/omniglue)")
                return
    build = get_retriever if kind == "retriever" else get_matcher
    build(name, "cpu")


def folder_size_gb(path):
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def report_caches():
    """Size of every cache, to see what was downloaded and how much of the quota it takes."""
    import torch.hub

    from loc.paths import THIRD_PARTY_ROOT

    hub = Path(torch.hub.get_dir())
    targets = [("third_party", THIRD_PARTY_ROOT), ("torch hub", hub),
               ("  checkpoints", hub / "checkpoints")]
    if os.environ.get("HF_HOME"):
        targets.append(("HF cache", Path(os.environ["HF_HOME"])))
    print("caches:")
    for label, path in targets:
        size = f"{folder_size_gb(path):8.2f} GB" if path.exists() else "missing"
        print(f"  {label:<14} {size}   {path}")
    for repo in sorted(hub.glob("*_*")) if hub.is_dir() else []:
        if repo.is_dir():
            print(f"    hub repository: {repo.name}")


def main():
    from loc.matchers import MATCHER_NAMES
    from loc.paths import REPO_ROOT
    from loc.retrievers import RETRIEVER_NAMES
    from loc.utils import setup_logging

    setup_logging()
    components =[("retriever", n) for n in RETRIEVER_NAMES] + [("matcher", n) for n in MATCHER_NAMES]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated component names (default: all)")
    p.add_argument("--skip", default="", help="comma-separated component names to skip")
    p.add_argument("--check-only", action="store_true",
                   help="do not download: load every component with network access disabled")
    args = p.parse_args()

    only = {n for n in args.only.split(",") if n}
    skip = {n for n in args.skip.split(",") if n}
    unknown = (only | skip) - {n for _, n in components}
    if unknown:
        raise SystemExit(f"unknown components {sorted(unknown)}")

    print(f"{'offline check' if args.check_only else 'download'} | repository: {REPO_ROOT}")
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME')}  HF_HOME={os.environ.get('HF_HOME')}")
    if args.check_only:
        print("HF_HUB_OFFLINE=1, offline torch.hub, HF_TOKEN removed")
    failed, done = [], 0
    for kind, name in components:
        if (only and name not in only) or name in skip:
            continue
        print(f"[{kind}] {name}")
        try:
            prefetch(kind, name, args.check_only)
            print("  ok")
            done += 1
        except BaseException as exc:  # noqa: BLE001  SystemExit included: report and continue
            traceback.print_exc()
            failed.append(name)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    report_caches()
    if failed:
        print(f"failed: {failed}")
        return 1
    print(f"{done} components ok")
    return 0


if __name__ == "__main__":
    if "--check-only" in sys.argv:
        # Must happen before huggingface_hub and torch.hub are imported.
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            os.environ.pop(var, None)
        from hpc import offline_hub  # noqa: F401
    sys.exit(main())
