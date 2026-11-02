import torch


def similarity_matrix(D0, D1, metric):
    """Full [N0, N1] similarity. 'bhattacharyya' expects rows that are probability distributions."""
    if metric == "cosine":
        p = torch.nn.functional.normalize(D0, dim=-1)
        q = torch.nn.functional.normalize(D1, dim=-1)
        return p @ q.T
    if metric == "bhattacharyya":
        return torch.sqrt(D0.clamp_min(0)) @ torch.sqrt(D1.clamp_min(0)).T
    raise ValueError(f"unknown metric {metric!r}: expected 'cosine' or 'bhattacharyya'")


def mutual_nn_matches(D0, D1, metric="cosine", min_score=None, ratio=None):
    """Mutual nearest-neighbour matches between two descriptor sets -> (idx0, idx1, scores).

    min_score: minimum similarity of a match.
    ratio: Lowe ratio test on similarities, keeps a match if second_best / best <= ratio.
    """
    empty = torch.zeros(0, dtype=torch.long, device=D0.device)
    if D0.numel() == 0 or D1.numel() == 0:
        return empty, empty.clone(), torch.zeros(0, dtype=D0.dtype, device=D0.device)

    sim = similarity_matrix(D0, D1, metric)
    best1 = sim.argmax(dim=1)
    best0 = sim.argmax(dim=0)
    idx0 = torch.arange(sim.shape[0], device=sim.device)
    keep = best0[best1] == idx0

    if ratio is not None and sim.shape[1] >= 2:
        top2 = sim.topk(2, dim=1).values
        keep &= (top2[:, 1] / top2[:, 0].clamp_min(1e-9)) <= ratio

    scores = sim[idx0, best1]
    if min_score is not None:
        keep &= scores >= min_score
    return idx0[keep], best1[keep], scores[keep]
