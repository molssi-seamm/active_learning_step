#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The fold plans of ``seamm-al train``: plain k-fold and stratified."""

import numpy as np
import pytest

from active_learning_step.thresholds import kfold_indices
from active_learning_step.train import fold_plan, stratified_fold_plan


def _strata():
    # wave-1-like: few cells per liquid, many clusters
    return ["cell|EC"] * 9 + ["cell|FEC"] * 20 + ["cluster|"] * 301 + ["cell|DMC"] * 3


@pytest.mark.parametrize("k", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_every_frame_is_held_out_exactly_once(k, seed):
    strata = _strata()
    fold = stratified_fold_plan(strata, k, seed)
    held = [np.flatnonzero(fold == m) for m in range(k)]
    every = np.concatenate(held)
    assert sorted(every.tolist()) == list(range(len(strata)))  # once each, none missing
    for m in range(k):
        train = np.flatnonzero(fold != m)
        assert len(train) + len(held[m]) == len(strata)
        assert not set(train.tolist()) & set(held[m].tolist())


@pytest.mark.parametrize("k", [2, 3, 5, 7])
def test_strata_and_folds_are_split_evenly(k):
    strata = np.array(_strata(), dtype=object)
    fold = stratified_fold_plan(strata, k, 3)
    sizes = np.bincount(fold, minlength=k)
    assert sizes.max() - sizes.min() <= 1
    for label in set(strata):
        counts = np.bincount(fold[strata == label], minlength=k)
        assert counts.max() - counts.min() <= 1, label


def test_plain_kfold_would_be_uneven_for_the_cells():
    # the reason for stratifying: 9 EC cells under plain k-fold, seed 0
    strata = np.array(_strata(), dtype=object)
    plain = fold_plan(len(strata), 5, 0)
    counts = np.bincount(plain[strata == "cell|EC"], minlength=5)
    strat = np.bincount(
        stratified_fold_plan(strata, 5, 0)[strata == "cell|EC"], minlength=5
    )
    assert strat.max() - strat.min() <= 1
    assert counts.max() - counts.min() >= strat.max() - strat.min()


def test_seeded_and_reproducible():
    strata = _strata()
    a = stratified_fold_plan(strata, 5, 4)
    assert np.array_equal(a, stratified_fold_plan(strata, 5, 4))
    assert not np.array_equal(a, stratified_fold_plan(strata, 5, 5))


def test_plain_plan_is_the_water_campaigns_kfold():
    fold = fold_plan(23, 5, 1)
    for member, (train, held) in enumerate(kfold_indices(23, 5, 1)):
        assert np.array_equal(np.flatnonzero(fold == member), held)
        assert np.array_equal(np.flatnonzero(fold != member), train)


def test_too_many_folds():
    with pytest.raises(ValueError):
        stratified_fold_plan(["a", "b", "b"], 4, 0)
    with pytest.raises(ValueError):
        stratified_fold_plan(["a", "b", "b"], 1, 0)
