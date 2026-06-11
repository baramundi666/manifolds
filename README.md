# mDTMA — Python Port

A clean Python translation of the **mDTMA** (manifold Differential-evolution /
Tournament-selection / Memetic / Approximation) optimiser from
[lingping-fuzzy/metaheuristic-manifold-optimization](https://github.com/lingping-fuzzy/metaheuristic-manifold-optimization).

The algorithm is a population-based, **derivative-free** minimiser for
functions defined on Riemannian manifolds.  It couples:

- **Tournament selection** to build a mating pool,
- **Simulated Binary Crossover (SBX) + Polynomial Mutation (PM)** to generate
  candidate moves (GA operator), and
- **Riemannian retraction** to keep every particle on the manifold.

---

## Files

| File | Description |
|---|---|
| `mdtma.py` | Core solver (`mdtma()` function) |
| `operator_ga.py` | SBX + PM GA operator |
| `tournament_selection.py` | K-tournament selection |
| `example_run.py` | Example driver with pluggable cost functions and visualisation |

---

## Requirements

```
pip install pymanopt numpy scipy matplotlib
```

---

## Quickstart

```python
import numpy as np
from pymanopt.manifolds import Sphere
from mdtma import mdtma

n   = 10
manifold = Sphere(n)

# Any function f(x: np.ndarray) -> float
B   = np.random.randn(n, n); A = (B + B.T) / 2
def cost(x):
    return float(-x @ (A @ x))      # maximise leading eigenvector

result = mdtma(manifold, cost, population_size=20, max_iter=100, w0=0.5)
print(result.x, result.cost)
```

---

## Changing the cost function

The solver is completely decoupled from the cost function — pass any callable
`f(x: np.ndarray) -> float`.  Three presets are provided in `example_run.py`:

### 1. Rayleigh quotient (original MATLAB example)
```python
from example_run import make_rayleigh_cost
A       = ...            # your symmetric matrix
cost_fn = make_rayleigh_cost(A)
```

### 2. Ackley (non-convex stress test)
```python
from example_run import make_ackley_cost
cost_fn = make_ackley_cost()
```

### 3. Fully custom
```python
# Any function that accepts a numpy array and returns a scalar:
cost_fn = lambda x: float(np.sin(5 * x[0]) + x[1] ** 2)

from mdtma import mdtma
from pymanopt.manifolds import Sphere
result = mdtma(Sphere(3), cost_fn, max_iter=100)
```

---

## Running from the command line

```bash
# Default: Rayleigh on S² (3-D sphere), with plots
python example_run.py --n 3 --preset rayleigh --visualise

# Ackley on S⁴
python example_run.py --n 5 --preset ackley --visualise

# Larger problem
python example_run.py --n 20 --pop 40 --iter 200 --w0 0.4
```

**CLI flags**

| Flag | Default | Description |
|---|---|---|
| `--n` | 3 | Ambient dimension (sphere S^{n-1} ⊂ R^n) |
| `--preset` | rayleigh | `rayleigh` / `ackley` / `custom` |
| `--pop` | 20 | Population size |
| `--iter` | 100 | Max iterations |
| `--w0` | 0.5 | Base inertia weight |
| `--seed` | 42 | RNG seed |
| `--verbosity` | 1 | 0 = silent, 1 = per-iter line |

---

## Visualisation

Call `visualise(result, manifold_dim=n)` after any run.  It produces:

1. **Convergence curve** — best cost vs. iteration.
2. **Population diversity** — box plot of per-particle distance from the best
   solution at selected iterations.
3. **Trajectory on S¹ or S²** — the path the best particle took across the
   sphere surface (only for n = 2 or n = 3, colour-coded by iteration).

The figure is also saved as `mdtma_results.png`.

---

## Key parameters

| Parameter | Effect | Suggested range |
|---|---|---|
| `population_size` | More particles → better exploration, slower | 10–50 |
| `max_iter` | More iterations → better convergence | 50–500 |
| `w0` | Base inertia: higher → larger steps | 0.3–0.9 |
| `dis_c` | SBX spread (higher → offspring closer to parents) | 5–30 |
| `dis_m` | PM spread (higher → smaller mutations) | 5–30 |

---

## Algorithm notes

The inertia weight decays slightly each iteration:

```
w = w0 + 0.1 * (1 - iter / max_iter)
```

so early iterations explore broadly and late iterations refine.

The GA offspring are projected onto the tangent space at each particle's
current position before retraction, ensuring the move direction is always
valid on the manifold.  Survival uses elitist (μ + λ) selection: the best
`population_size` individuals from the union of parents and offspring survive.
