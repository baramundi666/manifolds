"""
mdtma.py
--------
mDTMA — manifold Differential-evolution / Tournament-selection /
Memetic / Approximation optimiser.

A population-based, derivative-free minimiser for functions defined on
Riemannian manifolds.  The update rule fuses a GA-style crossover (SBX +
polynomial mutation) with a PSO-style retraction step so that every candidate
stays on the manifold throughout the optimisation.

Original MATLAB source: mDTMA.m  (Lingping Kong, 2023)
Based on pymanopt for manifold operations: https://www.pymanopt.org/
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from operator_ga import operator_ga
from tournament_selection import tournament_selection


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    """Snapshot of the optimiser state after one iteration."""
    iteration:  int
    cost:       float          # best cost so far
    cost_evals: int            # cumulative function evaluations
    time:       float          # cumulative wall-clock seconds
    population: List[np.ndarray]
    best_point: np.ndarray


@dataclass
class OptimiseResult:
    """Full result returned by :func:`mdtma`."""
    x:          np.ndarray          # best point found
    cost:       float               # cost at best point
    history:    List[IterationStats]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def mdtma(
    manifold,
    cost_fn:       Callable[[np.ndarray], float],
    *,
    population_size: int = 20,
    max_iter:        int = 100,
    w0:              float = 0.5,
    x0:              Optional[List[np.ndarray]] = None,
    verbosity:       int = 1,
    # GA operator hyper-parameters (rarely need changing)
    pro_c: float = 1.0,
    dis_c: float = 20.0,
    pro_m: float = 1.0,
    dis_m: float = 20.0,
) -> OptimiseResult:
    """
    Minimise *cost_fn* on *manifold* using the mDTMA algorithm.

    Parameters
    ----------
    manifold :
        A pymanopt manifold instance (e.g. ``pymanopt.manifolds.Sphere(n)``).
        Must expose ``dim``, ``random_point()``, ``retraction(x, v)``,
        and ``projection(x, v)`` (to project a Euclidean vector onto the
        tangent space at x).
    cost_fn : callable
        ``cost_fn(x) -> float`` — the scalar objective to minimise.
    population_size : int
        Number of particles.  Capped internally at ``4 * manifold.dim``.
    max_iter : int
        Maximum number of iterations.  At least ``10 * manifold.dim``,
        capped at 800.
    w0 : float
        Base inertia weight.  The effective weight decays slightly each
        iteration: ``w = w0 + 0.1 * (1 - iter / max_iter)``.
    x0 : list of np.ndarray, optional
        Initial population.  If *None* a random population is generated.
    verbosity : int
        0 = silent, 1 = summary line each iteration, 2 = detailed.
    pro_c, dis_c, pro_m, dis_m : float
        GA operator parameters forwarded to :func:`operator_ga`.

    Returns
    -------
    OptimiseResult
        ``.x``       – best point found
        ``.cost``    – its cost value
        ``.history`` – list of :class:`IterationStats`, one per iteration
    """
    dim = manifold.dim

    # ---- Resolve effective hyperparameters ----------------------------------
    # Match MATLAB capping logic exactly.
    max_iter        = min(800, max(max_iter, 10 * dim))
    population_size = min(population_size, 4 * dim)

    # ---- Initialise population ----------------------------------------------
    if x0 is None:
        population = [manifold.random_point() for _ in range(population_size)]
    else:
        if not isinstance(x0, list):
            raise TypeError("x0 must be a list of numpy arrays (one per particle).")
        population = list(x0)
        population_size = len(population)  # honour user-supplied size

    # Velocities (tangent vectors at each particle's current position).
    # Initialised to zero; updated during the first iteration.
    velocities: List[Optional[np.ndarray]] = [None] * population_size

    # ---- Evaluate initial costs ---------------------------------------------
    costs      = np.array([cost_fn(x) for x in population], dtype=float)
    cost_evals = population_size

    # ---- Identify initial best ----------------------------------------------
    best_idx  = int(np.argmin(costs))
    best_cost = float(costs[best_idx])
    best_x    = population[best_idx].copy()

    # ---- History bookkeeping ------------------------------------------------
    history: List[IterationStats] = []
    t_start = time.perf_counter()

    if verbosity >= 1:
        print(f"{'Iter':>6}  {'Cost evals':>10}  {'Best cost':>18}")
        print("-" * 40)

    # =========================================================================
    # Main optimisation loop
    # =========================================================================
    for iteration in range(1, max_iter + 1):

        # ---- Decaying inertia weight ----------------------------------------
        # Starts just above w0 and decreases to w0 as iterations proceed.
        w = w0 + 0.1 * (1.0 - iteration / max_iter)

        # ---- Tournament selection → mating pool ----------------------------
        # Returns population_size indices (with repetition allowed).
        mating_pool = tournament_selection(2, population_size, costs)

        # ---- GA crossover + mutation to generate velocity candidates --------
        # operator_ga expects a list of parent arrays selected by mating_pool.
        parents_selected = [population[i] for i in mating_pool]
        velocity_spring  = operator_ga(
            manifold, parents_selected,
            pro_c=pro_c, dis_c=dis_c, pro_m=pro_m, dis_m=dis_m,
        )

        # ---- Update positions using the GA-generated velocity ---------------
        for i in range(population_size):
            pool_idx = mating_pool[i]
            xi       = population[pool_idx]        # current position
            vi_raw   = velocity_spring[i]          # raw GA output (Euclidean)

            # -- Project vi_raw onto the tangent space at xi ------------------
            # The tangent-space projection removes the component of vi_raw
            # that is normal to the manifold at xi.
            vi_tangent = manifold.projection(xi, vi_raw)

            # -- "firstgo": displacement toward the new position --------------
            # vtemp is the component that moves xi *along* the manifold;
            # scaled by the inertia weight w.
            vtemp    = vi_tangent - xi              # deviation from xi
            firstgo  = w * vtemp                   # scaled displacement

            # Retract to stay on manifold
            population[pool_idx] = manifold.retraction(xi, firstgo)

            # -- Residual tangent velocity (orthogonal component) -------------
            # After retraction to the new position, keep the part of vi_raw
            # that was already tangential (perpendicular to xi's normal).
            vi_orth            = vi_raw - vi_tangent   # tangential residual
            velocities[pool_idx] = w * vi_orth         # store scaled velocity

        # ---- Evaluate offspring: retract each particle by its velocity ------
        offspring  = [None] * population_size
        off_costs  = np.zeros(population_size)

        for i in range(population_size):
            pool_idx   = mating_pool[i]
            xi_new     = population[pool_idx]
            vi_new     = velocities[pool_idx]

            if vi_new is None:
                # Safety fallback for the very first call before velocity is set
                vi_new = manifold.zero_vector(xi_new)

            # Retract along the stored velocity to generate the offspring
            offspring[i] = manifold.retraction(xi_new, vi_new)
            off_costs[i] = cost_fn(offspring[i])
            cost_evals  += 1

        # ---- Survival selection (elitist μ + λ) ----------------------------
        # Merge current population and offspring; keep the best population_size.
        combined_pop   = population + offspring
        combined_costs = np.concatenate([costs, off_costs])

        sorted_indices = np.argsort(combined_costs)
        population = [combined_pop[j]   for j in sorted_indices[:population_size]]
        costs      =  combined_costs[sorted_indices[:population_size]]

        # ---- Update global best --------------------------------------------
        if costs[0] < best_cost:
            best_cost = float(costs[0])
            best_x    = population[0].copy()

        # ---- Record stats --------------------------------------------------
        elapsed = time.perf_counter() - t_start
        history.append(IterationStats(
            iteration  = iteration,
            cost       = best_cost,
            cost_evals = cost_evals,
            time       = elapsed,
            population = [p.copy() for p in population],
            best_point = best_x.copy(),
        ))

        if verbosity >= 1:
            print(f"{iteration:>6}  {cost_evals:>10}  {best_cost:>+18.8e}")

    # =========================================================================
    return OptimiseResult(x=best_x, cost=best_cost, history=history)
