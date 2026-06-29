"""
example_run.py
--------------
Example driver for mDTMA in both minimisation and exploration modes.

MINIMISATION EXAMPLES
---------------------
  python example_run.py --mode min --preset rayleigh --n 10
  python example_run.py --mode min --preset ackley   --n 5
  python example_run.py --mode min --preset custom   --n 4

EXPLORATION / DATASET GENERATION EXAMPLES
-----------------------------------------
The exploration mode collects maximally spread points on a manifold defined by
a zero-level set  h(x) = 0  (implicit surface), then visualises them.

Three implicit manifold examples with 2-D / 3-D zero sets are included:

  sphere    : ||x||² - 1 = 0  (unit sphere S², embedded in R³)
              Zero set is 2-D; visualised as a 3-D scatter on a sphere.

  torus     : (sqrt(x²+y²) - R)² + z² - r² = 0  (torus embedded in R³)
              Zero set is 2-D; visualised as a 3-D scatter on a torus surface.

  figure8   : x⁴ - x² + y² = 0  (lemniscate-of-Bernoulli-like curve in R²)
              Zero set is a figure-eight curve; visualised in 2-D.

  cylinder  : x² + y² - 1 = 0  (infinite cylinder, clipped to z in [-1,1])
              Zero set is 2-D; visualised as a 3-D scatter.

Usage examples:
  python example_run.py --mode explore --manifold sphere  --n 3
  python example_run.py --mode explore --manifold torus   --n 3
  python example_run.py --mode explore --manifold figure8 --n 2
  python example_run.py --mode explore --manifold cylinder --n 3
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymanopt.manifolds import Sphere

from mdtma import mdtma, mdtma_explore, OptimiseResult, ExploreResult
import os
import matplotlib

matplotlib.use("QtAgg", force=True)

from matplotlib.widgets import Button, TextBox
import queue


def demo(pts):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    state = {"idx": 0}
    q = queue.Queue()

    def redraw():
        ax.cla()

        vis = pts[:state["idx"]]

        ax.scatter(
            vis[:, 0],
            vis[:, 1],
            vis[:, 2],
            c=np.arange(len(vis)),
            cmap="plasma",
        )

        ax.set_title(f"Showing {state['idx']} points")
        fig.canvas.draw_idle()

    def next_step(event=None):
        state["idx"] = min(len(pts), state["idx"] + 1)
        redraw()

    def prev_step(event=None):
        state["idx"] = max(10, state["idx"] - 10)
        redraw()

    # Buttons
    ax_prev = plt.axes([0.72, 0.05, 0.1, 0.05])
    ax_next = plt.axes([0.84, 0.05, 0.1, 0.05])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")

    b_prev.on_clicked(prev_step)
    b_next.on_clicked(next_step)

    # Keyboard
    def on_key(event):
        if event.key == "right":
            next_step()
        elif event.key == "left":
            prev_step()

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()

    print("Interactive window should open now.")
    print("If it does not:")
    print("  Linux: sudo apt install python3-tk")
    print("  macOS: install python.org Python")
    print("  Windows: use normal Python, not headless environment")

    plt.show(block=True)

def visualise_exploration_interactive(
result: ExploreResult,
manifold_name: str,
outfile: str = "mdtma_explore.png",
):


    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np

    history = result.history
    total_iters = len(history)

    # Detect dimensionality
    sample_pts = np.array(history[-1].archive)
    is_3d = (sample_pts.shape[1] == 3)

    fig = plt.figure(figsize=(15, 6))

    fig.suptitle(
        f"mDTMA Exploration — {manifold_name}",
        fontsize=14,
        fontweight="bold"
    )

    ax1 = fig.add_subplot(1, 2, 1)

    iters = [s.iteration for s in history]
    arch_sizes = [s.archive_size for s in history]
    max_gaps = [s.max_gap for s in history]

    color1 = "steelblue"
    color2 = "tomato"

    ln1 = ax1.plot(
        iters,
        arch_sizes,
        color=color1,
        linewidth=2,
        label="Archive size"
    )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Points in archive", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()

    ln2 = ax2.plot(
        iters,
        max_gaps,
        color=color2,
        linewidth=2,
        linestyle="--",
        label="Max NN gap"
    )

    ax2.set_ylabel(
        "Max nearest-neighbour distance",
        color=color2
    )

    ax2.tick_params(axis="y", labelcolor=color2)

    lines = ln1 + ln2
    ax1.legend(
        lines,
        [l.get_label() for l in lines],
        loc="upper left"
    )

    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_title("Coverage over iterations")

    # Current iteration marker
    iter_marker = ax1.axvline(
        1,
        color="black",
        linewidth=2,
        alpha=0.8
    )

    if is_3d:
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax3 = fig.add_subplot(1, 2, 2)

    state = {"iter": 0}

    def draw_manifold_surface():
        if manifold_name == "sphere":
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.pi, 40)

            ax3.plot_surface(
                np.outer(np.cos(u), np.sin(v)),
                np.outer(np.sin(u), np.sin(v)),
                np.outer(np.ones_like(u), np.cos(v)),
                alpha=0.06,
                color="lightblue"
            )

        elif manifold_name == "torus":
            R, r = 1.0, 0.4

            u = np.linspace(0, 2 * np.pi, 80)
            v = np.linspace(0, 2 * np.pi, 40)

            U, V = np.meshgrid(u, v)

            Tx = (R + r * np.cos(V)) * np.cos(U)
            Ty = (R + r * np.cos(V)) * np.sin(U)
            Tz = r * np.sin(V)

            ax3.plot_surface(
                Tx,
                Ty,
                Tz,
                alpha=0.07,
                color="lightyellow"
            )

        elif manifold_name == "cylinder":
            theta = np.linspace(0, 2 * np.pi, 60)
            z = np.linspace(-1, 1, 20)

            T, Z = np.meshgrid(theta, z)

            ax3.plot_surface(
                np.cos(T),
                np.sin(T),
                Z,
                alpha=0.07,
                color="lightgreen"
            )

        elif manifold_name == "figure8":
            t = np.linspace(-1.1, 1.1, 2000)

            y_lo = np.sqrt(
                np.clip(t**2 - t**4, 0, None)
            )

            ax3.plot(
                t,
                y_lo,
                "k-",
                lw=0.8,
                alpha=0.3
            )

            ax3.plot(
                t,
                -y_lo,
                "k-",
                lw=0.8,
                alpha=0.3
            )

    def redraw():
        current_iter = state["iter"]

        hist = history[current_iter]
        points = np.array(hist.archive)

        elev = azim = None

        if is_3d:
            elev = ax3.elev
            azim = ax3.azim

        ax3.cla()

        draw_manifold_surface()

        order = np.arange(len(points))

        if is_3d:
            ax3.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=order,
                cmap="plasma",
                s=18,
                depthshade=True,
                alpha=0.85
            )

            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("z")

            if elev is not None:
                ax3.view_init(elev=elev, azim=azim)

        else:
            ax3.scatter(
                points[:, 0],
                points[:, 1],
                c=order,
                cmap="plasma",
                s=25,
                alpha=0.85
            )

            ax3.set_aspect("equal")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")

        ax3.set_title(
            f"{manifold_name} — iteration "
            f"{hist.iteration}/{total_iters} "
            f"({len(points)} points)"
        )

        ax3.grid(True, linestyle="--", alpha=0.3)

        iter_marker.set_xdata([hist.iteration, hist.iteration])

        fig.canvas.draw_idle()

    def next_iter(event=None):
        state["iter"] = min(
            total_iters - 1,
            state["iter"] + 1
        )
        redraw()

    def prev_iter(event=None):
        state["iter"] = max(
            0,
            state["iter"] - 1
        )
        redraw()

    # Buttons
    ax_prev = plt.axes([0.72, 0.02, 0.1, 0.05])
    ax_next = plt.axes([0.84, 0.02, 0.1, 0.05])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")

    b_prev.on_clicked(prev_iter)
    b_next.on_clicked(next_iter)

    # Keyboard
    def on_key(event):
        if event.key == "right":
            next_iter()

        elif event.key == "left":
            prev_iter()

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    def on_close(event):
        plt.savefig(
            outfile,
            dpi=150,
            bbox_inches="tight"
        )

        print(f"Saved → {outfile}")

    fig.canvas.mpl_connect(
        "close_event",
        on_close
    )

    redraw()

    plt.tight_layout()
    plt.show(block=True)



def make_rayleigh_cost(A: np.ndarray) -> Callable[[np.ndarray], float]:
    """Minimise -x'Ax  ≡  maximise the leading Rayleigh quotient."""
    def cost(x: np.ndarray) -> float:
        return float(-x @ (A @ x))
    return cost


def make_ackley_cost() -> Callable[[np.ndarray], float]:
    """Ackley function restricted to the unit sphere."""
    a, b, c = 20.0, 0.2, 2 * np.pi
    def cost(x: np.ndarray) -> float:
        n  = len(x)
        t1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
        t2 = -np.exp(np.sum(np.cos(c * x)) / n)
        return float(t1 + t2 + a + np.e)
    return cost


def make_custom_cost() -> Callable[[np.ndarray], float]:
    def cost(x: np.ndarray) -> float:
        return float(np.sum(x[::2] ** 2))
    return cost

class ImplicitManifold:
    """
    Thin wrapper so mdtma_explore can work with an implicit surface.

    The manifold is defined by  h(x) = 0  for x in R^n.  We approximate
    it with S^{n-1} for the purpose of retraction / random points, then
    project each generated point onto the true zero set by gradient descent
    on |h(x)|².

    This approach keeps the core algorithm completely unchanged: only the
    `random_point()` and `retraction()` methods are overridden to snap
    candidate points back onto h(x) = 0 after each move.
    """

    def __init__(self, n: int, h: Callable, grad_h: Callable,
                 name: str = "implicit", max_proj_steps: int = 60,
                 proj_lr: float = 0.02):
        """
        Parameters
        ----------
        n           : ambient dimension
        h           : scalar constraint h(x) -> float
        grad_h      : gradient ∇h(x) -> np.ndarray, shape (n,)
        name        : label for printing / plotting
        max_proj_steps : gradient steps used to snap a point onto h=0
        proj_lr     : step size for the projection gradient descent
        """
        self._sphere    = Sphere(n)
        self.dim        = n - 1          # manifold is (n-1)-dimensional
        self.n          = n
        self.h          = h
        self.grad_h     = grad_h
        self.name       = name
        self._steps     = max_proj_steps
        self._lr        = proj_lr

    def _project_to_zero_set(self, x: np.ndarray) -> np.ndarray:
        """
        Snap x onto h(x)=0 via gradient descent on (1/2)|h(x)|².

        ∇(½h²) = h(x) * ∇h(x)
        """
        pt = x.copy()
        for _ in range(self._steps):
            val = self.h(pt)
            if abs(val) < 1e-7:
                break
            g  = self.grad_h(pt)
            gn = np.dot(g, g) + 1e-12
            pt = pt - self._lr * val * g / gn
        return pt

    def random_point(self) -> np.ndarray:
        """Sample a random point from the sphere then project onto h=0."""
        x = self._sphere.random_point()
        return self._project_to_zero_set(x)

    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Retract along v then snap back onto h=0."""
        candidate = self._sphere.retraction(x, v)
        return self._project_to_zero_set(candidate)

    def projection(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project v onto the tangent plane of the sphere at x."""
        return self._sphere.projection(x, v)

    def zero_vector(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


def make_sphere_manifold(r: float = 1.0) -> ImplicitManifold:
    """Unit sphere: h(x,y,z) = x²+y²+z² - r²."""
    h      = lambda x: float(np.sum(x**2) - r**2)
    grad_h = lambda x: 2.0 * x
    return ImplicitManifold(3, h, grad_h, name=f"sphere(r={r})")


def make_torus_manifold(R: float = 1.0, r: float = 0.4) -> ImplicitManifold:
    """
    Torus: h(x,y,z) = (√(x²+y²) - R)² + z² - r²  embedded in R³.

    R = major radius (distance from torus centre to tube centre)
    r = minor radius (tube radius)
    """
    def h(x):
        rho = np.sqrt(x[0]**2 + x[1]**2 + 1e-12)
        return (rho - R)**2 + x[2]**2 - r**2

    def grad_h(x):
        rho = np.sqrt(x[0]**2 + x[1]**2 + 1e-12)
        g   = np.zeros(3)
        g[0] = 2.0 * (rho - R) * x[0] / rho
        g[1] = 2.0 * (rho - R) * x[1] / rho
        g[2] = 2.0 * x[2]
        return g

    return ImplicitManifold(3, h, grad_h, name=f"torus(R={R},r={r})",
                            max_proj_steps=120, proj_lr=0.03)


def make_figure8_manifold() -> ImplicitManifold:
    """
    Figure-eight (lemniscate-like) curve in R²:  h(x,y) = x⁴ - x² + y².

    The zero set is a figure-eight lying flat in the x-y plane centred at 0,
    with lobes at roughly x = ±0.7.
    """
    h      = lambda x: float(x[0]**4 - x[0]**2 + x[1]**2)
    grad_h = lambda x: np.array([4.0 * x[0]**3 - 2.0 * x[0], 2.0 * x[1]])

    # Use a tiny 2-sphere (circle S¹) as the ambient manifold trick
    m = ImplicitManifold(2, h, grad_h, name="figure8",
                         max_proj_steps=200, proj_lr=0.05)
    m.dim = 1   # curve is 1-D
    return m


def make_cylinder_manifold(height: float = 1.0) -> ImplicitManifold:
    """
    Unit cylinder capped at z = ±height:  h(x,y,z) = x² + y² - 1.

    Points are also clamped to |z| ≤ height after projection.
    """
    def h(x):
        return float(x[0]**2 + x[1]**2 - 1.0)

    def grad_h(x):
        return np.array([2.0 * x[0], 2.0 * x[1], 0.0])

    base = ImplicitManifold(3, h, grad_h, name="cylinder",
                            max_proj_steps=80, proj_lr=0.05)

    # Override random_point to ensure z stays in bounds
    _orig_rp = base.random_point
    def rp_bounded():
        pt    = _orig_rp()
        pt[2] = np.clip(pt[2], -height, height)
        return pt
    base.random_point = rp_bounded

    # Override retraction to clamp z after projection
    _orig_retr = base.retraction
    def retr_bounded(x, v):
        pt    = _orig_retr(x, v)
        pt[2] = np.clip(pt[2], -height, height)
        return pt
    base.retraction = retr_bounded

    return base


def run_minimisation(
    n:               int   = 10,
    preset:          str   = "rayleigh",
    population_size: int   = 20,
    max_iter:        int   = 100,
    w0:              float = 0.5,
    seed:            int   = 42,
    verbosity:       int   = 1,
) -> OptimiseResult:
    """Run mDTMA minimisation on S^{n-1} and return the result."""
    np.random.seed(seed)
    manifold = Sphere(n)

    if preset == "rayleigh":
        B  = np.random.randn(n, n); A = (B + B.T) / 2
        fn = make_rayleigh_cost(A)
        print(f"Preset: Rayleigh quotient on {n}×{n} symmetric matrix")
    elif preset == "ackley":
        fn = make_ackley_cost()
        print("Preset: Ackley function on the sphere")
    else:
        fn = make_custom_cost()
        print("Preset: custom (edit make_custom_cost() to change)")

    print(f"\nMinimisation | S^{n-1} | pop={population_size} | iter={max_iter}\n")
    return mdtma(manifold, fn,
                 population_size=population_size, max_iter=max_iter,
                 w0=w0, verbosity=verbosity)


def visualise_minimisation(result: OptimiseResult, n: int,
                           outfile: str = "mdtma_min.png") -> None:
    """Convergence + sphere trajectory plot for minimisation results."""
    history    = result.history
    iters      = [s.iteration  for s in history]
    best_costs = [s.cost       for s in history]

    n_cols = 2 if n in (2, 3) else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("mDTMA — Minimisation", fontsize=13, fontweight="bold")

    # Convergence
    ax = axes[0]
    ax.plot(iters, best_costs, linewidth=2, color="steelblue")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best cost")
    ax.set_title("Convergence"); ax.grid(True, linestyle="--", alpha=0.5)

    # Trajectory
    if n == 3:
        ax = fig.add_subplot(1, n_cols, 2, projection="3d")
        u  = np.linspace(0, 2 * np.pi, 50)
        v  = np.linspace(0, np.pi, 30)
        ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                        np.outer(np.sin(u), np.sin(v)),
                        np.outer(np.ones_like(u), np.cos(v)),
                        alpha=0.07, color="lightblue")
        traj = np.array([s.best_point for s in history])
        sc   = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                          c=iters, cmap="plasma", s=20)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.4, color="gray")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Iteration")
        ax.set_title("Best-point trajectory on S²")

    elif n == 2:
        ax    = axes[1]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, lw=0.8)
        traj = np.array([s.best_point for s in history])
        sc   = ax.scatter(traj[:, 0], traj[:, 1], c=iters, cmap="plasma", s=25)
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.4, color="gray")
        plt.colorbar(sc, ax=ax, label="Iteration")
        ax.set_aspect("equal"); ax.set_title("Best-point trajectory on S¹")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")


def run_exploration(
    manifold_name:   str   = "sphere",
    population_size: int   = 20,
    max_iter:        int   = 60,
    w0:              float = 0.5,
    min_spacing:     float = 0.05,
    seed:            int   = 42,
    verbosity:       int   = 1,
) -> ExploreResult:
    """Run mDTMA exploration on an implicit manifold and return the result."""
    np.random.seed(seed)

    manifold_map = {
        "sphere":   make_sphere_manifold,
        "torus":    make_torus_manifold,
        "figure8":  make_figure8_manifold,
        "cylinder": make_cylinder_manifold,
    }
    if manifold_name not in manifold_map:
        raise ValueError(f"Unknown manifold '{manifold_name}'. "
                         f"Choose from: {list(manifold_map)}")

    manifold = manifold_map[manifold_name]()
    print(f"\nExploration | {manifold.name} | pop={population_size} | iter={max_iter}")
    print(f"min_spacing={min_spacing}\n")

    return mdtma_explore(manifold,
                         population_size=population_size,
                         max_iter=max_iter, w0=w0,
                         min_spacing=min_spacing,
                         verbosity=verbosity)


def visualise_exploration(result: ExploreResult, manifold_name: str,
                          outfile: str = "mdtma_explore.png") -> None:
    """
    Plot the collected points iteratively on the manifold surface.

    Left panel  — archive size and max nearest-neighbour gap over iterations.
    Right panel — scatter of all found points on the manifold; colour = order
                  in which they were added (blue = early, yellow = late).
    """
    history    = result.history
    iters      = [s.iteration    for s in history]
    arch_sizes = [s.archive_size for s in history]
    max_gaps   = [s.max_gap      for s in history]
    points     = np.array(result.points)    # (N_total, ambient_dim)
    order      = np.arange(len(points))     # colour by discovery time

    is_3d = (points.shape[1] == 3)
    fig   = plt.figure(figsize=(14, 5))
    fig.suptitle(f"mDTMA Exploration — {manifold_name}  "
                 f"({len(points)} points collected)", fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 1)
    color1 = "steelblue"
    color2 = "tomato"
    ln1 = ax1.plot(iters, arch_sizes, color=color1, linewidth=2, label="Archive size")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Points in archive", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(iters, max_gaps, color=color2, linewidth=2, linestyle="--",
                   label="Max NN gap")
    ax2.set_ylabel("Max nearest-neighbour distance", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    lines = ln1 + ln2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax1.set_title("Coverage over iterations")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if is_3d:
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")

        # Draw a ghost of the manifold surface
        if manifold_name == "sphere":
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.pi,     40)
            ax3.plot_surface(np.outer(np.cos(u), np.sin(v)),
                             np.outer(np.sin(u), np.sin(v)),
                             np.outer(np.ones_like(u), np.cos(v)),
                             alpha=0.06, color="lightblue")

        elif manifold_name == "torus":
            R, r  = 1.0, 0.4
            u = np.linspace(0, 2 * np.pi, 80)
            v = np.linspace(0, 2 * np.pi, 40)
            U, V  = np.meshgrid(u, v)
            Tx = (R + r * np.cos(V)) * np.cos(U)
            Ty = (R + r * np.cos(V)) * np.sin(U)
            Tz = r * np.sin(V)
            ax3.plot_surface(Tx, Ty, Tz, alpha=0.07, color="lightyellow")

        elif manifold_name == "cylinder":
            theta = np.linspace(0, 2 * np.pi, 60)
            z     = np.linspace(-1, 1, 20)
            T, Z  = np.meshgrid(theta, z)
            ax3.plot_surface(np.cos(T), np.sin(T), Z,
                             alpha=0.07, color="lightgreen")

        # Scatter: colour = discovery order (blue → yellow = early → late)
        sc = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=order, cmap="plasma", s=18, depthshade=True,
                         alpha=0.85)
        fig.colorbar(sc, ax=ax3, shrink=0.55, label="Discovery order")
        ax3.set_title(f"Points on {manifold_name}")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    else:
        # 2-D manifold (figure-eight curve)
        ax3 = fig.add_subplot(1, 2, 2)
        # Draw the true curve densely
        t    = np.linspace(-1.1, 1.1, 2000)
        y_lo = np.sqrt(np.clip(t**2 - t**4, 0, None))
        ax3.plot(t,  y_lo, "k-", lw=0.8, alpha=0.3)
        ax3.plot(t, -y_lo, "k-", lw=0.8, alpha=0.3)
        sc = ax3.scatter(points[:, 0], points[:, 1],
                         c=order, cmap="plasma", s=25, alpha=0.85)
        plt.colorbar(sc, ax=ax3, label="Discovery order")
        ax3.set_aspect("equal")
        ax3.set_title(f"Points on {manifold_name}")
        ax3.set_xlabel("x"); ax3.set_ylabel("y")

    ax3.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mDTMA example runner")
    p.add_argument("--mode",     default="explore",
                   choices=["min", "explore"],
                   help="'min' = minimisation, 'explore' = dataset generation")
    # minimisation options
    p.add_argument("--n",        type=int,   default=3)
    p.add_argument("--preset",   default="rayleigh",
                   choices=["rayleigh", "ackley", "custom"])
    # exploration options
    p.add_argument("--manifold", default="sphere",
                   choices=["sphere", "torus", "figure8", "cylinder"])
    p.add_argument("--spacing",  type=float, default=0.08,
                   help="min_spacing for exploration (default 0.08)")
    # shared
    p.add_argument("--pop",      type=int,   default=20)
    p.add_argument("--iter",     type=int,   default=80)
    p.add_argument("--w0",       type=float, default=0.5)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--verbose",  type=int,   default=1)
    return p.parse_args()


def main() -> None:
    args = _parse()

    if args.mode == "min":
        result = run_minimisation(
            n=args.n, preset=args.preset,
            population_size=args.pop, max_iter=args.iter,
            w0=args.w0, seed=args.seed, verbosity=args.verbose,
        )
        print(f"\nBest cost = {result.cost:+.8e}")
        visualise_minimisation(result, n=args.n)

    else:  # explore
        result = run_exploration(
            manifold_name=args.manifold,
            population_size=args.pop, max_iter=args.iter,
            w0=args.w0, min_spacing=args.spacing,
            seed=args.seed, verbosity=args.verbose,
        )
        print(f"\nTotal points collected: {len(result.points)}")
        visualise_exploration_interactive(result, manifold_name=args.manifold)


if __name__ == "__main__":
    main()