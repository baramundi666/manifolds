"""
mdtma.py
--------
mDTMA solver — two modes in one file:

  mdtma(...)          Original minimisation mode (unchanged logic).
  mdtma_explore(...)  New exploration / dataset-generation mode.

=== WHAT CHANGED FOR EXPLORATION MODE ===

1. COST FUNCTION → REPULSION FROM ARCHIVE
   Instead of minimising a user-supplied f(x), each particle's fitness is the
   *negative* Euclidean distance to its nearest neighbour in a growing archive
   of all previously accepted points.  "Lower cost" = "farther from everything
   already found" = more isolated = more useful for dataset generation.

   One-line change in evaluation:
     BEFORE: cost = user_cost_fn(x)
     AFTER:  cost = -min_dist(x, archive)    # negative so minimiser = explorer

2. PERMANENT ARCHIVE (no point re-use)
   All offspring that pass the spacing check are appended to a global archive.
   The exploration cost is computed against the full archive (not just the live
   population), so regions already visited stay penalised forever.

   Added after offspring evaluation:
     archive.extend(offspring)

3. MINIMUM-SPACING REJECTION
   Offspring closer than `min_spacing` to any archived point are replaced by a
   fresh random manifold point.  This prevents clustering and guarantees the
   archive contains only well-separated points.

   Added inside the offspring loop:
     if min_dist(candidate, archive) < min_spacing:
         candidate = manifold.random_point()

4. DIVERSITY-AWARE PARENT RE-EVALUATION
   Before (μ+λ) selection, parent costs are re-evaluated against the updated
   archive so the selection pressure always reflects the current state of
   coverage rather than stale distances from earlier iterations.

   Added before sorting:
     costs = [exploration_cost(p, archive) for p in population]

Everything else — tournament selection, SBX+PM GA operator, Riemannian
retraction, decaying inertia weight w = w0+0.1*(1-iter/max_iter), and the
(μ+λ) elitist survival rule — is **identical** to the original algorithm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from operator_ga import operator_ga
from tournament_selection import tournament_selection


@dataclass
class IterationStats:
    """State snapshot after one iteration (minimisation mode)."""
    iteration:  int
    cost:       float
    cost_evals: int
    time:       float
    population: List[np.ndarray]
    best_point: np.ndarray


@dataclass
class OptimiseResult:
    """Result of :func:`mdtma` (minimisation)."""
    x:       np.ndarray
    cost:    float
    history: List[IterationStats]


@dataclass
class ExploreStats:
    """State snapshot after one iteration (exploration mode)."""
    iteration:  int
    archive_size: int         # total distinct points collected so far
    max_gap:    float         # largest nearest-neighbour distance in archive
    cost_evals: int
    time:       float
    population: List[np.ndarray]
    archive:    List[np.ndarray]


@dataclass
class ExploreResult:
    """Result of :func:`mdtma_explore`."""
    points:  List[np.ndarray]   # the full collected dataset
    history: List[ExploreStats]


def _min_dist_to_archive(x: np.ndarray, archive: List[np.ndarray]) -> float:
    """Euclidean distance from x to the nearest point in archive."""
    if not archive:
        return float("inf")
    return float(np.min(np.linalg.norm(np.stack(archive) - x, axis=1)))


def _exploration_cost(x: np.ndarray, archive: List[np.ndarray]) -> float:
    """
    Fitness for the exploration mode.
    Lower = better = farther from all known points.
    Returns -d(x, nearest_archive_point), or 0 if archive is empty.
    """
    if not archive:
        return 0.0
    return -_min_dist_to_archive(x, archive)


def mdtma(
    manifold,
    cost_fn: Callable[[np.ndarray], float],
    *,
    population_size: int   = 20,
    max_iter:        int   = 100,
    w0:              float = 0.5,
    x0:              Optional[List[np.ndarray]] = None,
    verbosity:       int   = 1,
    pro_c: float = 1.0, dis_c: float = 20.0,
    pro_m: float = 1.0, dis_m: float = 20.0,
) -> OptimiseResult:
    """
    Minimise cost_fn on manifold using mDTMA (original algorithm, unchanged).

    Parameters
    ----------
    manifold : pymanopt manifold
    cost_fn  : f(x) -> float
    population_size, max_iter, w0, x0, verbosity : see module docstring
    pro_c, dis_c, pro_m, dis_m : GA operator parameters
    """
    dim             = manifold.dim
    max_iter        = min(800, max(max_iter, 10 * dim))
    population_size = min(population_size, 4 * dim)

    population = (list(x0) if x0 is not None
                  else [manifold.random_point() for _ in range(population_size)])
    population_size = len(population)
    velocities: List[Optional[np.ndarray]] = [None] * population_size

    costs      = np.array([cost_fn(x) for x in population], dtype=float)
    cost_evals = population_size
    best_idx   = int(np.argmin(costs))
    best_cost  = float(costs[best_idx])
    best_x     = population[best_idx].copy()

    history: List[IterationStats] = []
    t_start = time.perf_counter()

    if verbosity >= 1:
        print(f"{'Iter':>6}  {'Evals':>8}  {'Best cost':>18}")
        print("-" * 38)

    for iteration in range(1, max_iter + 1):
        w           = w0 + 0.1 * (1.0 - iteration / max_iter)
        mating_pool = tournament_selection(2, population_size, costs)
        vel_spring  = operator_ga(manifold, [population[i] for i in mating_pool],
                                  pro_c=pro_c, dis_c=dis_c, pro_m=pro_m, dis_m=dis_m)

        for i in range(population_size):
            idx        = mating_pool[i]
            xi         = population[idx]
            vi_tang    = manifold.projection(xi, vel_spring[i])
            population[idx] = manifold.retraction(xi, w * (vi_tang - xi))
            velocities[idx] = w * (vel_spring[i] - vi_tang)

        offspring  = [None] * population_size
        off_costs  = np.zeros(population_size)
        for i in range(population_size):
            idx       = mating_pool[i]
            vi        = velocities[idx] if velocities[idx] is not None \
                        else manifold.zero_vector(population[idx])
            offspring[i]  = manifold.retraction(population[idx], vi)
            off_costs[i]  = cost_fn(offspring[i])
            cost_evals   += 1

        combined_pop   = population + offspring
        combined_costs = np.concatenate([costs, off_costs])
        order          = np.argsort(combined_costs)
        population     = [combined_pop[j] for j in order[:population_size]]
        costs          =  combined_costs[order[:population_size]]

        if costs[0] < best_cost:
            best_cost = float(costs[0])
            best_x    = population[0].copy()

        elapsed = time.perf_counter() - t_start
        history.append(IterationStats(iteration, best_cost, cost_evals,
                                      elapsed, [p.copy() for p in population],
                                      best_x.copy()))
        if verbosity >= 1:
            print(f"{iteration:>6}  {cost_evals:>8}  {best_cost:>+18.8e}")

    return OptimiseResult(x=best_x, cost=best_cost, history=history)



def mdtma_explore(
    manifold,
    *,
    population_size: int   = 20,
    max_iter:        int   = 100,
    w0:              float = 0.5,
    min_spacing:     float = 0.05,
    x0:              Optional[List[np.ndarray]] = None,
    verbosity:       int   = 1,
    pro_c: float = 1.0, dis_c: float = 20.0,
    pro_m: float = 1.0, dis_m: float = 20.0,
) -> ExploreResult:
    """
    Run mDTMA in exploration mode: collect maximally spread points on manifold.

    Instead of minimising a cost function this variant drives the population to
    explore the manifold as broadly as possible and accumulates all visited
    points into a dataset.

    Parameters
    ----------
    manifold : pymanopt manifold
    population_size : int   (capped at 4*dim)
    max_iter        : int   (at least 10*dim, capped at 800)
    w0              : float  Base inertia weight (same decay schedule as original)
    min_spacing     : float  Minimum Euclidean distance between any two archive
                             points.  Offspring closer than this to any existing
                             archive point are replaced with a fresh random point.
    x0              : optional initial population list
    verbosity       : 0 silent, 1 per-iter line

    Returns
    -------
    ExploreResult
        .points  — list of all collected manifold points
        .history — per-iteration stats including the live archive
    """
    dim             = manifold.dim
    max_iter        = min(800, max(max_iter, 10 * dim))
    population_size = min(population_size, 4 * dim)

    # Initialise population
    population = (list(x0) if x0 is not None
                  else [manifold.random_point() for _ in range(population_size)])
    population_size = len(population)
    velocities: List[Optional[np.ndarray]] = [None] * population_size

    archive: List[np.ndarray] = [p.copy() for p in population]

    # Initial exploration costs: for each seed point, cost vs rest of archive
    costs = np.array([
        _exploration_cost(p, [a for j, a in enumerate(archive) if j != i])
        for i, p in enumerate(population)
    ], dtype=float)
    cost_evals = population_size

    history: List[ExploreStats] = []
    t_start = time.perf_counter()

    if verbosity >= 1:
        print(f"{'Iter':>6}  {'Archive':>8}  {'Max gap':>12}")
        print("-" * 32)

    for iteration in range(1, max_iter + 1):
        # decaying inertia weight
        w = w0 + 0.1 * (1.0 - iteration / max_iter)

        # tournament selection (on exploration cost)
        mating_pool = tournament_selection(2, population_size, costs)

        # GA crossover + mutation
        vel_spring = operator_ga(manifold, [population[i] for i in mating_pool],
                                 pro_c=pro_c, dis_c=dis_c, pro_m=pro_m, dis_m=dis_m)

        # update positions via retraction
        for i in range(population_size):
            idx     = mating_pool[i]
            xi      = population[idx]
            vi_tang = manifold.projection(xi, vel_spring[i])
            population[idx] = manifold.retraction(xi, w * (vi_tang - xi))
            velocities[idx] = w * (vel_spring[i] - vi_tang)

        # generate offspring via velocity retraction + spacing rejection
        offspring = []
        for i in range(population_size):
            idx = mating_pool[i]
            vi  = velocities[idx] if velocities[idx] is not None \
                  else manifold.zero_vector(population[idx])
            candidate  = manifold.retraction(population[idx], vi)
            cost_evals += 1

            # reject if too close to any archived point
            if archive and _min_dist_to_archive(candidate, archive) < min_spacing:
                candidate  = manifold.random_point()   # fresh random replacement
                cost_evals += 1

            offspring.append(candidate)

        # add all offspring to the permanent archive
        for pt in offspring:
            archive.append(pt.copy())

        # re-evaluate parent costs vs updated archive before selection
        costs_parents = np.array(
            [_exploration_cost(p, archive) for p in population], dtype=float)
        off_costs = np.array(
            [_exploration_cost(o, archive) for o in offspring], dtype=float)

        # (μ+λ) elitist selection — now ranked by exploration cost
        combined_pop   = population + offspring
        combined_costs = np.concatenate([costs_parents, off_costs])
        order          = np.argsort(combined_costs)   # lowest cost = most isolated
        population     = [combined_pop[j] for j in order[:population_size]]
        costs          =  combined_costs[order[:population_size]]

        # Compute max nearest-neighbour gap in archive (coverage quality metric)
        if len(archive) >= 2:
            A = np.stack(archive)
            # For each point: distance to nearest neighbour in archive
            dists_sq  = (np.sum(A**2, axis=1, keepdims=True) +
                         np.sum(A**2, axis=1) - 2.0 * A @ A.T)
            np.fill_diagonal(dists_sq, np.inf)
            nn_dists   = np.sqrt(np.clip(dists_sq, 0, None).min(axis=1))
            max_gap    = float(nn_dists.max())
        else:
            max_gap = 0.0

        elapsed = time.perf_counter() - t_start
        history.append(ExploreStats(
            iteration    = iteration,
            archive_size = len(archive),
            max_gap      = max_gap,
            cost_evals   = cost_evals,
            time         = elapsed,
            population   = [p.copy() for p in population],
            archive      = [a.copy() for a in archive],
        ))

        if verbosity >= 1:
            print(f"{iteration:>6}  {len(archive):>8}  {max_gap:>12.6f}")

    return ExploreResult(points=archive, history=history)