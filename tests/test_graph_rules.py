# tests/test_graph_rules.py
import torch
from son_goku.approx.graph_build import (
    adjacency_from_cos, edge_density, calibrate_tau_for_density, knn_k_for_density
)

def _mk_C():
    # K=4, hand-picked cosines
    # upper tri: [-0.8, 0.3, -0.1, -0.2, 0.05, 0.7]
    C = torch.tensor([
        [ 1.00, -0.80,  0.30, -0.10],
        [-0.80,  1.00, -0.20,  0.05],
        [ 0.30, -0.20,  1.00,  0.70],
        [-0.10,  0.05,  0.70,  1.00],
    ], dtype=torch.float32)
    return C

def test_threshold_rule_symmetry_and_diag():
    C = _mk_C()
    A = adjacency_from_cos(C, mode="threshold", tau=0.0)
    # Expect edges where C_ij < 0: (0,1), (0,3), (1,2)
    expect = torch.tensor([
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
    ], dtype=torch.bool)
    assert torch.equal(A, expect)
    assert torch.all(A == A.t())
    assert not torch.any(torch.diag(A))

def test_signed_equals_threshold_tau0():
    C = _mk_C()
    A1 = adjacency_from_cos(C, mode="signed")
    A2 = adjacency_from_cos(C, mode="threshold", tau=0.0)
    assert torch.equal(A1, A2)

def test_quantile_p50_matches_expected_edges():
    C = _mk_C()
    # 50th percentile is between -0.1 and 0.05 -> expect edges at vals < ~ -0.025:
    # (0,1), (1,2), (0,3)
    A = adjacency_from_cos(C, mode="quantile", quantile_p=0.5)
    expect = torch.tensor([
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
    ], dtype=torch.bool)
    assert torch.equal(A, expect)

def test_knn_k1_is_union_of_per_node_most_conflicting():
    C = _mk_C()
    A = adjacency_from_cos(C, mode="knn", knn_k=1)
    # For each row, smallest cosine neighbor:
    # 0->1, 1->0, 2->1, 3->0 => union same as above
    expect = torch.tensor([
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
    ], dtype=torch.bool)
    assert torch.equal(A, expect)

def test_calibrate_tau_hits_target_density():
    C = _mk_C()
    target = 0.5
    tau = calibrate_tau_for_density(C, target_density=target)
    A = adjacency_from_cos(C, mode="threshold", tau=tau)
    dens = edge_density(A)
    # K=4 => possible edges=6; density quantization step=1/6 ≈ 0.1667
    assert abs(dens - target) <= (1.0/6.0 + 1e-6)

def test_knn_k_for_density_reasonable():
    for K in [3, 4, 5]:
        for δ in [0.1, 0.3, 0.5, 0.8]:
            k = knn_k_for_density(K, δ)
            assert 1 <= k <= (K-1)
            # expected density approx 2k/(K-1)
            approx = (2.0 * k) / (K - 1)
            assert 0.0 <= approx <= 1.0