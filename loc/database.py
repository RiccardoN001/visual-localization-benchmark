import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np

from . import paths
from .utils import load_json, save_json

log = logging.getLogger(__name__)

METADATA_KEYS = ("pose_id", "image_path", "depth_path", "sensor_id", "K_ref", "R_c2w", "t_c2w")


def build_or_load_index(retriever, references, db_root, output_root):
    """FAISS index of the reference descriptors, cached per (retriever, database content)."""
    valid = [e for e in references if Path(e["image_path"]).exists()]
    if len(valid) != len(references):
        log.warning("%d references have no image on disk and are skipped",
                    len(references) - len(valid))

    fingerprint = paths.db_fingerprint(Path(db_root) / "db_trajectories.txt", len(valid))
    cache = paths.retrieval_cache_dir(output_root, retriever.name, fingerprint)
    index_path, meta_path = cache / "index.faiss", cache / "database_meta.json"

    retriever.prepare([e["image_path"] for e in valid], cache_dir=cache)

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        if index.ntotal == len(valid):
            log.info("Loaded cached %s index (dim=%d, N=%d) from %s", retriever.name, index.d,
                     index.ntotal, cache)
            return index, load_json(meta_path)
        log.warning("Cached index has %d entries instead of %d: rebuilding", index.ntotal, len(valid))

    log.info("Building %s index on %d references", retriever.name, len(valid))
    t0 = time.perf_counter()
    embeddings, metadata = [], []
    for i, e in enumerate(valid):
        embeddings.append(retriever.embed(e["image_path"])[0])
        metadata.append({"faiss_idx": i, **{k: e[k] for k in METADATA_KEYS}})
        if (i + 1) % 250 == 0:
            log.info("  %d/%d", i + 1, len(valid))

    embeddings = np.vstack(embeddings).astype("float32")
    # Inner product on L2-normalized descriptors is cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Atomic write: another job using the same cache must never read a partial index.
    tmp = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, index_path)
    save_json(metadata, meta_path)
    log.info("Index ready (dim=%d, N=%d) in %.1fs", embeddings.shape[1], index.ntotal,
             time.perf_counter() - t0)
    return index, metadata


def search(index, metadata, embedding, k):
    """Top-k references as [{rank, similarity, metadata}], best first."""
    sims, idxs = index.search(embedding, k)
    return [{"rank": r + 1, "similarity": float(sims[0][r]), "metadata": metadata[int(idxs[0][r])]}
            for r in range(idxs.shape[1]) if 0 <= int(idxs[0][r]) < len(metadata)]
