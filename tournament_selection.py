"""
tournament_selection.py
-----------------------
K-tournament selection used to build the mating pool for the GA operator.

Original MATLAB source: TournamentSelection.m (PlatEMO, BIMK Group, 2023)
"""

import numpy as np


def tournament_selection(K: int, N: int, *fitness_arrays: np.ndarray) -> np.ndarray:
    """
    K-tournament selection that returns N winner indices.

    In every selection round K candidates are drawn at random from the
    population.  The candidate with the *smallest* combined fitness wins.
    Ties in the first fitness array are broken by the second, and so on
    (lexicographic ordering).

    Parameters
    ----------
    K : int
        Tournament size (number of candidates per round).
    N : int
        Number of winners (= output indices) to return.
    *fitness_arrays : 1-D array_like
        One or more fitness vectors of length equal to the population size.
        The first array is the primary criterion; subsequent arrays break ties.

    Returns
    -------
    index : np.ndarray, shape (N,)
        0-based indices of the N selected individuals.

    Notes
    -----
    The logic mirrors the MATLAB implementation exactly:
      1. Stack all fitness arrays column-wise and find unique combined rows.
      2. Sort those unique rows lexicographically to assign a scalar rank.
      3. Map every individual back to its rank.
      4. Draw K × N random candidates, pick the lowest-rank winner per column.
    """
    pop_size = len(fitness_arrays[0])

    # --- Step 1: build the combined fitness matrix (pop_size × n_criteria) ---
    # Each row is one individual's full fitness tuple.
    combined = np.column_stack([np.asarray(f).reshape(-1) for f in fitness_arrays])  # (pop_size, n_crit)

    # --- Step 2: assign a lexicographic rank to every unique fitness row ------
    # unique_rows  : (n_unique, n_crit) — the distinct fitness tuples
    # inverse      : (pop_size,)        — maps each individual → unique row idx
    unique_rows, inverse = np.unique(combined, axis=0, return_inverse=True)

    # Sort unique rows lexicographically; rank_of_unique[i] = position of
    # unique_rows[i] after sorting.
    lex_order = np.lexsort(unique_rows[:, ::-1].T)  # sort by last col first → first col last
    rank_of_unique = np.empty(len(unique_rows), dtype=int)
    rank_of_unique[lex_order] = np.arange(len(unique_rows))

    # Rank of every individual in the population
    individual_rank = rank_of_unique[inverse]  # shape (pop_size,)

    # --- Step 3: run K-tournament N times ------------------------------------
    # Draw K × N random candidate indices (each in [0, pop_size)).
    candidates = np.random.randint(0, pop_size, size=(K, N))  # shape (K, N)

    # Rank of each candidate
    candidate_ranks = individual_rank[candidates]             # shape (K, N)

    # For each of the N tournaments pick the row (candidate) with lowest rank
    best_in_tournament = np.argmin(candidate_ranks, axis=0)  # shape (N,)

    # Gather the actual population indices of the winners
    index = candidates[best_in_tournament, np.arange(N)]

    return index
