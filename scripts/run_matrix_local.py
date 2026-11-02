"""Run a set of retriever x matcher combinations sequentially on the local GPU.

    python scripts/run_matrix_local.py --dataset keble --retriever salad,netvlad --matcher lightglue
    python scripts/run_matrix_local.py --dataset keble --all --omniglue-python envs/omniglue/bin/python

Any option after `--` is forwarded to loc.localize (for example `-- --limit 20 --resume`).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc.matchers import MATCHER_NAMES  # noqa: E402
from loc.paths import REPO_ROOT  # noqa: E402
from loc.retrievers import RETRIEVER_NAMES  # noqa: E402


def names(value, allowed, use_all):
    if value:
        return value.split(",")
    if use_all:
        return list(allowed)
    raise SystemExit("specify the names or use --all")


def main():
    argv = sys.argv[1:]
    forwarded = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="comma-separated dataset names")
    p.add_argument("--retriever", default=None)
    p.add_argument("--matcher", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--omniglue-python", default=None,
                   help="Python interpreter of the OmniGlue environment (OmniGlue is skipped without it)")
    args = p.parse_args(argv)

    failed = []
    for dataset in args.dataset.split(","):
        for retriever in names(args.retriever, RETRIEVER_NAMES, args.all):
            for matcher in names(args.matcher, MATCHER_NAMES, args.all):
                python, env = sys.executable, dict(os.environ)
                if matcher == "omniglue":
                    if not args.omniglue_python:
                        print(f"skipping {dataset} {retriever} omniglue: --omniglue-python not given")
                        continue
                    # Without memory growth TensorFlow preallocates the whole GPU.
                    python, env["TF_FORCE_GPU_ALLOW_GROWTH"] = args.omniglue_python, "true"
                cmd = [python, "-m", "loc.localize", "--dataset", dataset, "--retriever", retriever,
                       "--matcher", matcher, "--k", str(args.k), *forwarded]
                print("$ " + " ".join(cmd), flush=True)
                if subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode != 0:
                    failed.append((dataset, retriever, matcher))
    if failed:
        print(f"failed combinations: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
