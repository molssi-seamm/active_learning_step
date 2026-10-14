#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The batch samplers, with the water campaign's tests (Sina Mostafanejad)."""

import pytest

pytest.importorskip("torch")

from active_learning_step.samplers import (  # noqa: E402
    KindBatchSampler,
    ShareBatchSampler,
    kinds_of,
)

SIZES = {"cluster": 32, "cell": 4}
SHARE = 0.15


def _kinds(n_cluster, n_cell):
    return ["cluster"] * n_cluster + ["cell"] * n_cell


@pytest.mark.parametrize("n_cluster,n_cell", [(8000, 180), (120000, 80), (5000, 600)])
def test_cell_batches_take_the_share(n_cluster, n_cell):
    """Cell batches make the requested share of an epoch, clusters one pass each."""
    s = ShareBatchSampler(_kinds(n_cluster, n_cell), SIZES, True, seed=3, share=SHARE)
    batches = list(s)
    cells = [b for b in batches if b[0] >= n_cluster]
    clusters = sorted(i for b in batches if b[0] < n_cluster for i in b)
    assert len(batches) == len(s)
    assert abs(len(cells) / len(batches) - SHARE) < 2.0 / len(batches)
    assert clusters == list(range(n_cluster))
    assert all(len(b) == 4 and all(i >= n_cluster for i in b) for b in cells)


def test_few_cells_are_repeated_evenly():
    """With fewer cells than the share needs, every cell recurs about as often."""
    n_cluster, n_cell = 120000, 80
    s = ShareBatchSampler(_kinds(n_cluster, n_cell), SIZES, True, seed=1, share=SHARE)
    seen = [i for b in s if b[0] >= n_cluster for i in b]
    counts = [seen.count(n_cluster + c) for c in range(n_cell)]
    assert max(counts) - min(counts) <= 1


def test_epochs_differ_and_repeat_by_seed():
    """Each epoch draws a new order; the same seed and epoch draw the same one."""
    a = ShareBatchSampler(_kinds(3000, 100), SIZES, True, seed=5, share=SHARE)
    b = ShareBatchSampler(_kinds(3000, 100), SIZES, True, seed=5, share=SHARE)
    e0 = list(a)
    a.set_epoch(1)
    assert list(a) != e0
    assert list(b) == e0


def test_the_share_must_be_given():
    with pytest.raises(TypeError):
        ShareBatchSampler(_kinds(100, 10), SIZES, True)
    with pytest.raises(ValueError):
        ShareBatchSampler(_kinds(100, 10), SIZES, True, share=1.0)


def test_no_cells():
    """A selection without cells gives the cluster batches only."""
    s = ShareBatchSampler(_kinds(1000, 0), SIZES, True, share=SHARE)
    assert len(s) == len(list(s)) == 32


@pytest.mark.parametrize(
    "make",
    [
        lambda: KindBatchSampler(_kinds(3000, 100), SIZES, True, 7, 0, 1),
        lambda: ShareBatchSampler(_kinds(3000, 100), SIZES, True, seed=7, share=SHARE),
    ],
)
def test_passes_reshuffle_without_set_epoch(make):
    """Each pass draws the next epoch's order without set_epoch, the order it gives."""
    s = make()
    passes = [list(s) for _ in range(3)]
    assert passes[0] != passes[1] != passes[2] != passes[0]
    for epoch, order in enumerate(passes):
        told = make()
        told.set_epoch(epoch)
        assert list(told) == order


def test_unshuffled_passes_repeat():
    """A validation sampler (no shuffle) gives the same batches every pass."""
    s = KindBatchSampler(_kinds(300, 20), SIZES, False, 7, 0, 1)
    assert list(s) == list(s)


def test_ranks_keep_the_same_epoch():
    """Ranks advancing on their own stay on the same global batches."""
    ranks = [KindBatchSampler(_kinds(500, 40), SIZES, True, 7, r, 2) for r in (0, 1)]
    seen = []
    for _ in range(3):
        g0, g1 = (r._global_batches() for r in ranks)
        assert g0 == g1
        seen.append(g0)
        a, b = (list(r) for r in ranks)
        assert [x + y for x, y in zip(a, b)] == [c[0::2] + c[1::2] for c in g0]
    assert seen[0] != seen[1] != seen[2]


def test_kinds_and_missing_batch_size():
    assert kinds_of(
        [{"cell": None}, {"cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, {}]
    ) == [
        "cluster",
        "cell",
        "cluster",
    ]
    with pytest.raises(ValueError):
        KindBatchSampler(["cluster", "cell"], {"cluster": 2}, True)
