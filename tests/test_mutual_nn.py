import torch

from loc.matchers._mutual_nn import mutual_nn_matches, similarity_matrix


def pairs(idx0, idx1):
    return {(int(a), int(b)) for a, b in zip(idx0, idx1)}


def test_recovers_a_known_permutation():
    torch.manual_seed(0)
    D0 = torch.randn(12, 32)
    perm = torch.randperm(12)
    idx0, idx1, scores = mutual_nn_matches(D0, D0[perm].clone(), "cosine")
    assert pairs(idx0, idx1) == {(int(perm[k]), k) for k in range(12)}
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-5)


def test_is_symmetric():
    torch.manual_seed(1)
    D0, D1 = torch.randn(9, 16), torch.randn(11, 16)
    a0, a1, _ = mutual_nn_matches(D0, D1, "cosine")
    b1, b0, _ = mutual_nn_matches(D1, D0, "cosine")
    assert pairs(a0, a1) == pairs(b0, b1)


def test_a_reference_is_matched_at_most_once():
    D0 = torch.tensor([[1.0, 0.0], [0.94, 0.34], [0.0, 1.0]])
    D1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    idx0, idx1, _ = mutual_nn_matches(D0, D1, "cosine")
    assert len(idx1) == len(set(idx1.tolist()))
    assert (0, 0) in pairs(idx0, idx1)


def test_empty_inputs():
    empty = torch.zeros(0, 8)
    for A, B in ((empty, torch.randn(5, 8)), (torch.randn(5, 8), empty), (empty, empty)):
        idx0, idx1, scores = mutual_nn_matches(A, B, "cosine")
        assert len(idx0) == len(idx1) == len(scores) == 0
    assert len(mutual_nn_matches(torch.randn(1, 8), torch.randn(1, 8), "cosine")[0]) == 1


def test_bhattacharyya_on_distributions():
    P = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    Q = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    sim = similarity_matrix(P, Q, "bhattacharyya")
    assert torch.allclose(torch.diagonal(sim.flip(1)), torch.ones(2), atol=1e-6)
    assert bool((sim <= 1.0 + 1e-6).all())
    assert pairs(*mutual_nn_matches(P, Q, "bhattacharyya")[:2]) == {(0, 1), (1, 0)}


def test_filters_only_remove_matches():
    torch.manual_seed(2)
    D0, D1 = torch.randn(20, 24), torch.randn(20, 24)
    base = len(mutual_nn_matches(D0, D1, "cosine")[0])
    assert len(mutual_nn_matches(D0, D1, "cosine", min_score=0.9)[0]) <= base
    assert len(mutual_nn_matches(D0, D1, "cosine", ratio=0.8)[0]) <= base
    scores = mutual_nn_matches(D0, D1, "cosine", min_score=0.5)[2]
    assert bool((scores >= 0.5).all())
