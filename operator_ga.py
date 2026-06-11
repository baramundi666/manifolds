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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    manifold :
        A pymanopt manifold object (only ``dim`` is used here for the
        mutation bound calculation).
    parents : list of np.ndarray
        Population points (numpy arrays). Must have even length.
    pro_c : float
        Crossover probability (applied per pair).
    dis_c : float
        Distribution index for SBX (higher → offspring closer to parents).
    pro_m : float
        Expected number of mutations per individual (used as pro_m/D per gene).
    dis_m : float
        Distribution index for polynomial mutation.

    Returns
    -------
    offspring : list of np.ndarray
        Same length as ``parents``, same array shape per element.

    Notes
    -----
    The MATLAB version handled three cases based on the type of manifold point
    (vector, matrix, struct).  In Python with pymanopt all points are numpy
    arrays (vectors or matrices), so only those two cases are needed.
    The vector case uses bounds [-1, 1] per dimension (matching the MATLAB
    ``lower/upper`` in ``GAreal``).  The matrix case applies the same bounds
    element-wise after flattening.
    """
    half = len(parents) // 2
    parents1 = parents[:half]   # first  set of parents
    parents2 = parents[half: half * 2]  # second set of parents

    offspring: List[np.ndarray] = [None] * len(parents)

    for pop_idx in range(len(parents1)):
        p1 = parents1[pop_idx]
        p2 = parents2[pop_idx]

        # Apply SBX + PM; result contains two children stacked along axis 0.
        children = _ga_real(p1, p2, pro_c, dis_c, pro_m, dis_m)

        # children has shape (2*H, W, ...) where p1 has shape (H, W, ...).
        n_rows = children.shape[0]
        child_a = children[: n_rows // 2]   # first  offspring
        child_b = children[n_rows // 2 :]   # second offspring

        # Reshape back to the original parent shape
        offspring[2 * pop_idx]     = child_a.reshape(p1.shape)
        offspring[2 * pop_idx + 1] = child_b.reshape(p1.shape)

    return offspring


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ga_real(
    parent1: np.ndarray,
    parent2: np.ndarray,
    pro_c: float,
    dis_c: float,
    pro_m: float,
    dis_m: float,
) -> np.ndarray:
    """
    SBX crossover followed by polynomial mutation for two real-valued parents.

    Both parents are flattened to 1-D (or 2-D for matrix points), the
    operators are applied, and the result is returned with shape
    ``(2 * N, D)`` where N × D = parent1.size.

    The variable bounds are fixed at [-1, +1] for each dimension, mirroring
    the MATLAB code (``lower = zeros(1,D)-1; upper = zeros(1,D)+1``).
    """
    orig_shape = parent1.shape

    # ---- reshape to 2-D: (N_rows, D_cols) ----------------------------------
    # For a vector parent this gives (1, D); for a matrix (N, D).
    p1 = parent1.reshape(orig_shape[0], -1) if parent1.ndim >= 2 else parent1.reshape(1, -1)
    p2 = parent2.reshape(orig_shape[0], -1) if parent2.ndim >= 2 else parent2.reshape(1, -1)
    N, D = p1.shape

    lower = np.full((1, D), -1.0)
    upper = np.full((1, D),  1.0)

    # ---- Simulated Binary Crossover (SBX) -----------------------------------
    beta = np.zeros((N, D))
    mu   = np.random.rand(N, D)

    # Spread factor β:  β = (2μ)^(1/(η+1))  if μ ≤ 0.5
    #                   β = (2 - 2μ)^(-1/(η+1)) otherwise
    mask_lo = mu <= 0.5
    beta[mask_lo]  = (2.0 * mu[mask_lo])              ** (1.0 / (dis_c + 1))
    beta[~mask_lo] = (2.0 - 2.0 * mu[~mask_lo])       ** (-1.0 / (dis_c + 1))

    # Randomly flip sign of β
    beta *= (-1.0) ** np.random.randint(0, 2, size=(N, D))

    # With 50 % probability keep β = 1 for each gene (no crossover effect)
    beta[np.random.rand(N, D) < 0.5] = 1.0

    # If the pair's crossover probability check fails, set whole row to β = 1.
    # rand(N,1) produces a column vector; broadcasting sets entire rows to 1.
    no_cross = (np.random.rand(N, 1) > pro_c).repeat(D, axis=1)  # (N, D) bool mask
    beta[no_cross] = 1.0

    # Offspring = ±β * (p1 − p2) / 2   (centred difference, no mean term)
    # This matches the *commented-out* MATLAB line that keeps only the
    # deviation from the midpoint, not the midpoint itself.
    offspring = np.vstack([
         beta * (p1 - p2) / 2.0,   # child A
        -beta * (p1 - p2) / 2.0,   # child B
    ])  # shape (2*N, D)

    # ---- Polynomial Mutation (PM) -------------------------------------------
    Lower = np.tile(lower, (2 * N, 1))  # (2N, D)
    Upper = np.tile(upper, (2 * N, 1))  # (2N, D)

    # Clamp offspring to [Lower, Upper] before mutation
    offspring = np.clip(offspring, Lower, Upper)

    # Each gene is mutated independently with probability pro_m / D
    site = np.random.rand(2 * N, D) < (pro_m / D)
    mu2  = np.random.rand(2 * N, D)

    # Lower-tail mutation  (μ ≤ 0.5)
    mask = site & (mu2 <= 0.5)
    delta_lo = (offspring[mask] - Lower[mask]) / (Upper[mask] - Lower[mask])
    offspring[mask] += (Upper[mask] - Lower[mask]) * (
        (2.0 * mu2[mask] + (1.0 - 2.0 * mu2[mask]) * (1.0 - delta_lo) ** (dis_m + 1))
        ** (1.0 / (dis_m + 1)) - 1.0
    )

    # Upper-tail mutation  (μ > 0.5)
    mask = site & (mu2 > 0.5)
    delta_hi = (Upper[mask] - offspring[mask]) / (Upper[mask] - Lower[mask])
    offspring[mask] += (Upper[mask] - Lower[mask]) * (
        1.0 - (2.0 * (1.0 - mu2[mask]) + 2.0 * (mu2[mask] - 0.5) * (1.0 - delta_hi) ** (dis_m + 1))
        ** (1.0 / (dis_m + 1))
    )

    return offspring  # shape (2*N, D)
