"""
example_run.py
--------------
Example driver for the mDTMA manifold optimiser.

Three cost-function presets are provided so you can see how to plug in your
own objective without touching the solver:

    • 'rayleigh'  – maximise the leading eigenvector of a symmetric matrix A
                    (minimise -x'Ax).  This matches the original MATLAB example.
    • 'ackley'    – a non-convex test function on the sphere (good stress test).
    • 'custom'    – a slot for your own callable.

After optimisation a visualisation window shows:
    1. Convergence curve (best cost vs iteration).
    2. For 2-D / 3-D spheres: the trajectory of the best particle projected
       onto the sphere surface.
    3. Population scatter at selected iterations (every visualise_every steps).

Usage
-----
    python example_run.py                         # default Rayleigh, n=10
    python example_run.py --n 4 --preset ackley
    python example_run.py --n 3 --preset rayleigh --visualise
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from pymanopt.manifolds import Sphere

from mdtma import mdtma, OptimiseResult


# ===========================================================================
# Cost function library
# ===========================================================================

def make_rayleigh_cost(A: np.ndarray) -> Callable[[np.ndarray], float]:
    """
    Returns cost(x) = -x' A x  (minimising this maximises the Rayleigh quotient).

    For a symmetric PSD matrix A the minimum cost equals -λ_max(A),
    achieved at the corresponding eigenvector.
    """
    def cost(x: np.ndarray) -> float:
        return float(-x @ (A @ x))
    return cost


def make_ackley_cost(a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> Callable[[np.ndarray], float]:
    """
    Ackley function adapted to unit-sphere inputs.

    The sphere constraint means ‖x‖ = 1 always, so the first exponential
    term is constant (-a * exp(-b)); we keep it for faithfulness to the
    original formula.  The second term still varies with x.
    """
    def cost(x: np.ndarray) -> float:
        n   = len(x)
        t1  = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n))
        t2  = -np.exp(np.sum(np.cos(c * x)) / n)
        return float(t1 + t2 + a + np.e)
    return cost


def make_custom_cost(fn: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """
    Wrap any user-supplied function so it is ready for the solver.

    Parameters
    ----------
    fn : callable
        Any function ``fn(x: np.ndarray) -> float``.

    Example
    -------
    >>> my_cost = make_custom_cost(lambda x: np.sum(x**2))
    >>> result  = run_optimisation(n=5, cost_fn=my_cost)
    """
    return fn


# ===========================================================================
# Solver wrapper
# ===========================================================================

def run_optimisation(
    n:               int   = 10,
    cost_fn:         Callable | None = None,
    population_size: int   = 20,
    max_iter:        int   = 100,
    w0:              float = 0.5,
    verbosity:       int   = 1,
    seed:            int | None = None,
) -> OptimiseResult:
    """
    Run mDTMA on the n-sphere with the given cost function.

    Parameters
    ----------
    n : int
        Ambient dimension of the sphere S^(n-1).
    cost_fn : callable, optional
        ``cost_fn(x) -> float``.  If *None*, the Rayleigh preset with a
        random symmetric matrix is used.
    population_size, max_iter, w0 :
        Forwarded to :func:`mdtma`.
    verbosity : int
        0 = silent, 1 = per-iteration line, 2 = verbose.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    OptimiseResult
    """
    if seed is not None:
        np.random.seed(seed)

    manifold = Sphere(n)

    if cost_fn is None:
        # Default: Rayleigh quotient on a random symmetric matrix
        B    = np.random.randn(n, n)
        A    = (B + B.T) / 2          # symmetric
        cost_fn = make_rayleigh_cost(A)
        print(f"Using Rayleigh cost on a {n}×{n} random symmetric matrix.")

    print(f"\nRunning mDTMA  |  sphere S^{n-1}  |  pop={population_size}  |  max_iter={max_iter}\n")
    t0     = time.perf_counter()
    result = mdtma(
        manifold,
        cost_fn,
        population_size = population_size,
        max_iter        = max_iter,
        w0              = w0,
        verbosity       = verbosity,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nFinished in {elapsed:.2f}s  |  best cost = {result.cost:+.8e}")
    return result


# ===========================================================================
# Visualisation
# ===========================================================================

def visualise(
    result:           OptimiseResult,
    manifold_dim:     int,
    visualise_every:  int = 1,
) -> None:
    """
    Produce diagnostic plots for an mDTMA run.

    Plot 1 – Convergence curve: best cost vs iteration number.
    Plot 2 – (2-D sphere only) Trajectory of the best point on the circle.
    Plot 3 – (3-D sphere only) Trajectory of the best point on the unit sphere.
    Plot 4 – Population diversity: population cost spread per iteration.

    Parameters
    ----------
    result : OptimiseResult
        Returned by :func:`run_optimisation`.
    manifold_dim : int
        Ambient dimension n (sphere lives in R^n).
    visualise_every : int
        Show population scatter only every this many iterations (avoids clutter).
    """
    history   = result.history
    iters     = [s.iteration for s in history]
    best_costs = [s.cost     for s in history]

    other = False

    n_plots   = (2 + (1 if manifold_dim in (2, 3) else 0)) if other else 1
    fig       = plt.figure(figsize=(6 * n_plots, 5))
    fig.suptitle("mDTMA Optimisation Results", fontsize=14, fontweight="bold")

    ax_idx = 1  # subplot counter
    if other:
        # ------------------------------------------------------------------
        # Plot 1 — Convergence curve
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(1, n_plots, ax_idx); ax_idx += 1
        ax1.plot(iters, best_costs, linewidth=2, color="steelblue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best cost")
        ax1.set_title("Convergence")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # ------------------------------------------------------------------
        # Plot 2 — Population cost spread (box plot at selected iterations)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(1, n_plots, ax_idx); ax_idx += 1

        # Collect all individual costs per iteration stored in history.
        # (Each particle's cost is re-evaluated lazily via the cost stored;
        #  here we use the per-particle cost from the population snapshot.)
        selected = [s for s in history if s.iteration % visualise_every == 0 or s.iteration == 1]
        box_data  = []
        box_labels = []
        for s in selected:
            # We don't re-evaluate; instead use a proxy: store spread via best_cost
            # and the iteration index. Full per-particle costs would need to be
            # logged separately — this plot shows diversity via trajectory spread.
            pop_norms = [float(np.linalg.norm(p - result.x)) for p in s.population]
            box_data.append(pop_norms)
            box_labels.append(str(s.iteration))

        ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="lightcoral", color="darkred"),
                    medianprops=dict(color="darkred", linewidth=2))
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Distance from best solution")
        ax2.set_title("Population diversity")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # ------------------------------------------------------------------
    # Plot 3 — Trajectory on the sphere (2-D or 3-D only)
    # ------------------------------------------------------------------
    if manifold_dim == 2:
        # Circle in R^2
        ax3 = fig.add_subplot(1, n_plots, ax_idx); ax_idx += 1
        theta = np.linspace(0, 2 * np.pi, 300)
        ax3.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.4)

        # Trajectory of best point
        traj = np.array([s.best_point for s in history])  # (T, 2)
        sc   = ax3.scatter(traj[:, 0], traj[:, 1],
                           c=np.arange(len(traj)), cmap="plasma",
                           s=30, zorder=3)
        ax3.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.5, color="gray")
        ax3.set_aspect("equal")
        plt.colorbar(sc, ax=ax3, label="Iteration")
        ax3.set_title("Best-point trajectory on S¹")

    elif manifold_dim == 3:
        # Sphere in R^3
        ax3 = fig.add_subplot(1, n_plots, ax_idx,
                               projection="3d"); ax_idx += 1
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax3.plot_surface(xs, ys, zs, alpha=0.08, color="lightblue")

        traj = np.array([s.best_point for s in history])  # (T, 3)
        sc   = ax3.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                           c=np.arange(len(traj)), cmap="plasma",
                           s=25, depthshade=True)
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                 linewidth=1.0, alpha=0.5, color="gray")
        fig.colorbar(sc, ax=ax3, shrink=0.6, label="Iteration")
        ax3.set_title("Best-point trajectory on S²")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    plt.tight_layout()
    plt.savefig("mdtma_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to mdtma_results.png")
    plt.show()


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mDTMA example runner")
    p.add_argument("--n",          type=int,   default=3,
                   help="Sphere dimension (ambient R^n, default 3)")
    p.add_argument("--preset",     type=str,   default="rayleigh",
                   choices=["rayleigh", "ackley", "custom"],
                   help="Cost function preset")
    p.add_argument("--pop",        type=int,   default=20,
                   help="Population size")
    p.add_argument("--iter",       type=int,   default=100,
                   help="Max iterations")
    p.add_argument("--w0",         type=float, default=0.5,
                   help="Base inertia weight")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed")
    p.add_argument("--visualise",  action="store_true",
                   help="Show result plots after optimisation")
    p.add_argument("--verbosity",  type=int,   default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Choose cost function based on preset --------------------------------
    if args.preset == "rayleigh":
        np.random.seed(args.seed)
        B       = np.random.randn(args.n, args.n)
        A       = (B + B.T) / 2
        cost_fn = make_rayleigh_cost(A)
        print(f"Preset: Rayleigh quotient  (minimise -x'Ax)")

    elif args.preset == "ackley":
        cost_fn = make_ackley_cost()
        print("Preset: Ackley function on the sphere")

    elif args.preset == "custom":
        # ----------------------------------------------------------------
        # *** PLUG YOUR OWN COST FUNCTION IN HERE ***
        #
        # Example: minimise the first coordinate of x.
        # Replace the lambda with any function f(x: np.ndarray) -> float.
        # ----------------------------------------------------------------
        cost_fn = make_custom_cost(lambda x: float(x[0]))
        print("Preset: custom  (minimise x[0])")

    # ---- Run -----------------------------------------------------------------
    result = run_optimisation(
        n               = args.n,
        cost_fn         = cost_fn,
        population_size = args.pop,
        max_iter        = args.iter,
        w0              = args.w0,
        verbosity       = args.verbosity,
        seed            = args.seed,
    )

    # ---- Visualise -----------------------------------------------------------
    if args.visualise or True:   # always visualise when running as script
        visualise(result, manifold_dim=args.n)


if __name__ == "__main__":
    main()
