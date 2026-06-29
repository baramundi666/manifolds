"""
operator_ga.py
--------------
Simulated Binary Crossover (SBX) + Polynomial Mutation (PM) for real-valued
variables, adapted for points living on a Riemannian manifold.

Original MATLAB source: OperatorGA.m (PlatEMO / BIMK Group, 2023).
References
----------
[1] Deb et al., "Self-adaptive SBX for real-parameter optimization", GECCO 2007.
[2] Deb & Goyal, "GeneAS for engineering design", CS & Informatics, 1996.
"""

from __future__ import annotations
import numpy as np
from typing import List


def operator_ga(
    manifold,
    parents: List[np.ndarray],
    pro_c: float = 1.0,
    dis_c: float = 20.0,
    pro_m: float = 1.0,
    dis_m: float = 20.0,
) -> List[np.ndarray]:
    """
    Apply SBX crossover and polynomial mutation to a list of parent points.

    The parents list is split in half: parents[0..n//2-1] are paired with
    parents[n//2..n-1].  Each pair produces two offspring, so the output
    list has the same length as the input.
    """
    half     = len(parents) // 2
    parents1 = parents[:half]
    parents2 = parents[half: half * 2]
    offspring: List[np.ndarray] = [None] * len(parents)

    for pop_idx in range(len(parents1)):
        p1, p2   = parents1[pop_idx], parents2[pop_idx]
        children = _ga_real(p1, p2, pro_c, dis_c, pro_m, dis_m)
        n_rows   = children.shape[0]
        offspring[2 * pop_idx]     = children[: n_rows // 2].reshape(p1.shape)
        offspring[2 * pop_idx + 1] = children[n_rows // 2 :].reshape(p1.shape)
    return offspring


def _ga_real(parent1, parent2, pro_c, dis_c, pro_m, dis_m):
    """SBX + PM on flat real arrays; bounds fixed at [-1, 1]."""
    orig_shape = parent1.shape
    p1 = parent1.reshape(orig_shape[0], -1) if parent1.ndim >= 2 else parent1.reshape(1, -1)
    p2 = parent2.reshape(orig_shape[0], -1) if parent2.ndim >= 2 else parent2.reshape(1, -1)
    N, D   = p1.shape
    lower  = np.full((1, D), -1.0)
    upper  = np.full((1, D),  1.0)

    # Simulated Binary Crossover
    beta   = np.zeros((N, D))
    mu     = np.random.rand(N, D)
    mask_lo = mu <= 0.5
    beta[mask_lo]  = (2.0 * mu[mask_lo])           ** (1.0 / (dis_c + 1))
    beta[~mask_lo] = (2.0 - 2.0 * mu[~mask_lo])    ** (-1.0 / (dis_c + 1))
    beta *= (-1.0) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1.0
    no_cross = (np.random.rand(N, 1) > pro_c).repeat(D, axis=1)
    beta[no_cross] = 1.0
    offspring = np.vstack([beta * (p1 - p2) / 2.0, -beta * (p1 - p2) / 2.0])

    # Polynomial Mutation
    Lower    = np.tile(lower, (2 * N, 1))
    Upper    = np.tile(upper, (2 * N, 1))
    offspring = np.clip(offspring, Lower, Upper)
    site     = np.random.rand(2 * N, D) < (pro_m / D)
    mu2      = np.random.rand(2 * N, D)

    mask = site & (mu2 <= 0.5)
    dlo  = (offspring[mask] - Lower[mask]) / (Upper[mask] - Lower[mask])
    offspring[mask] += (Upper[mask] - Lower[mask]) * (
        (2.0 * mu2[mask] + (1.0 - 2.0 * mu2[mask]) * (1.0 - dlo) ** (dis_m + 1))
        ** (1.0 / (dis_m + 1)) - 1.0
    )
    mask = site & (mu2 > 0.5)
    dhi  = (Upper[mask] - offspring[mask]) / (Upper[mask] - Lower[mask])
    offspring[mask] += (Upper[mask] - Lower[mask]) * (
        1.0 - (2.0 * (1.0 - mu2[mask]) + 2.0 * (mu2[mask] - 0.5) * (1.0 - dhi) ** (dis_m + 1))
        ** (1.0 / (dis_m + 1))
    )
    return offspring