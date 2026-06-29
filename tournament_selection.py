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
    """
    pop_size = len(fitness_arrays[0])
    combined = np.column_stack([np.asarray(f).reshape(-1) for f in fitness_arrays])

    unique_rows, inverse = np.unique(combined, axis=0, return_inverse=True)
    lex_order = np.lexsort(unique_rows[:, ::-1].T)
    rank_of_unique = np.empty(len(unique_rows), dtype=int)
    rank_of_unique[lex_order] = np.arange(len(unique_rows))
    individual_rank = rank_of_unique[inverse]

    candidates = np.random.randint(0, pop_size, size=(K, N))
    candidate_ranks = individual_rank[candidates]
    best_in_tournament = np.argmin(candidate_ranks, axis=0)
    return candidates[best_in_tournament, np.arange(N)]