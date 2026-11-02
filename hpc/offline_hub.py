# torch.hub.load("owner/repo", ...) asks GitHub for the default branch before looking at the local
# cache, which fails on compute nodes without internet access. AnyLoc hardcodes such a call in its
# own code. This patch resolves a missing branch from the repository already present in the cache
# (<hub_dir>/<owner>_<repo>_<branch>), and falls back to the original behaviour otherwise, so it is
# a no-op when the network is available. It is enabled by setting LOC_OFFLINE_HUB=1.
import os
from pathlib import Path

import torch.hub

_original_parse_repo_info = torch.hub._parse_repo_info


def _cached_ref(owner, name):
    prefix = f"{owner}_{name}_"
    refs = [d.name[len(prefix):] for d in Path(torch.hub.get_dir()).glob(f"{prefix}*") if d.is_dir()]
    for preferred in ("main", "master"):
        if preferred in refs:
            return preferred
    return sorted(refs)[0] if refs else None


def _parse_repo_info_offline(github):
    if ":" not in github and "/" in github:
        owner, _, name = github.partition("/")
        ref = _cached_ref(owner, name)
        if ref is not None:
            return owner, name, ref
    return _original_parse_repo_info(github)


torch.hub._parse_repo_info = _parse_repo_info_offline

# Diagnostics for a job that cannot find a model: shows what is actually in the cache.
if os.environ.get("LOC_OFFLINE_HUB_VERBOSE"):
    hub_dir = Path(torch.hub.get_dir())
    print(f"offline_hub active, hub_dir={hub_dir}")
    for repo in sorted(hub_dir.glob("*_*")) if hub_dir.is_dir() else []:
        print(f"  cached repository: {repo.name}")
