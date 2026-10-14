# -*- coding: utf-8 -*-

"""The numpy-only parts of query by committee: data folds and thresholds.

Adapted from code by Sina Mostafanejad, unchanged
except for being split off so that selection runs without torch or xnn. They are
re-exported by :mod:`active_learning_step.committee`, which keeps his API.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

__all__ = ["kfold_indices", "calibrate_threshold", "split_statistics", "select"]


def kfold_indices(n: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``k`` folds of a committee's training data.

    Member ``l`` trains on every fold but fold ``l`` and validates on fold
    ``l``, the 5-fold split of Smith *et al.* (``k = 5``). One permutation
    seeded by ``seed`` is cut into ``k`` nearly equal folds, so the folds are
    disjoint and cover the ``n`` indices exactly once.

    Parameters
    ----------
    n : int
        Number of training structures.
    k : int
        Number of folds, which is the number of committee members.
    seed : int, optional
        Seed of the permutation, by default 0. This is the data seed; the
        initialisation seed of each member is the trainer's ``cfg.seed``.

    Returns
    -------
    list of (ndarray, ndarray)
        For each member, ``(train_indices, held_indices)`` into the
        training set, sorted.

    Raises
    ------
    ValueError
        If ``k`` is smaller than 2 or larger than ``n``.
    """
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]; got k={k}, n={n}")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for held in folds:
        mask = np.ones(n, dtype=bool)
        mask[held] = False
        out.append((np.flatnonzero(mask), np.sort(held)))
    return out


def calibrate_threshold(
    rho: np.ndarray, eps: np.ndarray, fail_level: float, capture: float = 0.98
) -> float:
    """The inclusion threshold of Smith *et al.*, fig 1.

    The threshold is the largest value of ``rho`` such that at least
    ``capture`` of the structures whose error ``eps`` exceeds ``fail_level``
    have ``rho >= threshold``, so selecting by ``rho >= threshold`` catches
    that fraction of the failures.

    Parameters
    ----------
    rho : ndarray
        Committee spread per structure (one entry of :func:`disagreement`).
    eps : ndarray
        Worst error per structure (the matching entry of
        :func:`worst_error`), on the same structures.
    fail_level : float
        Error above which a structure counts as a failure (the paper: 1.5
        kcal/mol in its energy units).
    capture : float, optional
        Fraction of the failures the threshold must catch, by default 0.98.

    Returns
    -------
    float
        The threshold.

    Raises
    ------
    ValueError
        If no structure fails, so there is nothing to calibrate on.
    """
    rho = np.asarray(rho, dtype=float)
    eps = np.asarray(eps, dtype=float)
    failing = np.sort(rho[eps > fail_level])
    if failing.size == 0:
        raise ValueError(f"no structure has eps > {fail_level}; nothing to calibrate")
    # the k-th smallest rho of the failures, with k the number allowed to escape
    k = int(math.floor((1.0 - capture) * failing.size))
    return float(failing[k])


def split_statistics(
    rho: np.ndarray, err: np.ndarray, threshold: float
) -> Dict[str, float]:
    """What a threshold separates: the error above and below it.

    Parameters
    ----------
    rho : ndarray
        Committee spread per structure.
    err : ndarray
        An error per structure (the committee mean's, or the worst member's).
    threshold : float
        The threshold from :func:`calibrate_threshold`.

    Returns
    -------
    dict of str to float
        ``fraction_selected`` (``rho >= threshold``), ``n_above``,
        ``n_below``, and the RMS of ``err`` on each side (``rmse_above``,
        ``rmse_below``; ``nan`` for an empty side).
    """
    rho = np.asarray(rho, dtype=float)
    err = np.asarray(err, dtype=float)
    above = rho >= threshold

    def rms(x):
        return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")

    return {
        "fraction_selected": float(above.mean()),
        "n_above": int(above.sum()),
        "n_below": int((~above).sum()),
        "rmse_above": rms(err[above]),
        "rmse_below": rms(err[~above]),
    }


def select(rhos: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> np.ndarray:
    """Which structures any calibrated spread flags.

    Parameters
    ----------
    rhos : dict of str to ndarray
        Spreads per structure by name (``rho_energy``, ``rho_forces``, ...).
    thresholds : dict of str to float
        Threshold per name; names absent from ``rhos`` are ignored, and a
        ``nan`` spread never selects.

    Returns
    -------
    ndarray
        Boolean mask, true where ``rho >= threshold`` for at least one name.
    """
    mask = None
    for name, thr in thresholds.items():
        if name not in rhos:
            continue
        r = np.asarray(rhos[name], dtype=float)
        hit = np.where(np.isnan(r), False, r >= thr)
        mask = hit if mask is None else (mask | hit)
    if mask is None:
        raise ValueError(
            f"no threshold matches a spread: {sorted(thresholds)} vs {sorted(rhos)}"
        )
    return mask
