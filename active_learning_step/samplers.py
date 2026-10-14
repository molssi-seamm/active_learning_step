# -*- coding: utf-8 -*-

"""Batch samplers for training committee members: one kind per batch, a set cell share.

Adapted from code by Sina Mostafanejad, with the kinds named as in ``scores.csv``
(``cell`` and ``cluster``).

* :class:`KindBatchSampler` never mixes clusters and periodic cells in a batch, and
  gives each kind its own batch size: a cell holds about 20 times the atoms and
  neighbours of a cluster, so one size cannot suit both.
* :class:`ShareBatchSampler` keeps one pass over the clusters per epoch and draws the
  cell batches so that they make a fixed share of the epoch's batches, cycling
  through the cells in a fresh order. Without it, the share of optimiser steps spent
  on cells is set by the selection, and a selection of many clusters and few cells
  hardly trains on cells.

Both are DDP-aware (every rank builds the same global batches from the same seed and
takes its own slice of each) and reshuffle every epoch under any caller: the trainer
tells them the epoch with ``set_epoch`` (xnn 0.12 does in every launch mode), and they
also advance their own epoch after each pass, giving the order ``set_epoch`` would.
So a sampler passed to an xnn 0.11 ``Trainer``, which calls ``set_epoch`` only in
distributed runs, still reshuffles.

Only torch is needed.
"""

import torch

__all__ = ["KindBatchSampler", "ShareBatchSampler", "kinds_of"]


def kinds_of(structures):
    """``"cell"`` for periodic structures, ``"cluster"`` for the others.

    Parameters
    ----------
    structures : sequence of dict
        xnn structure dicts.

    Returns
    -------
    list of str
    """
    return ["cell" if s.get("cell") is not None else "cluster" for s in structures]


class KindBatchSampler:
    """Batches of one kind each, shuffled per epoch, with a batch size per kind.

    Parameters
    ----------
    kinds : sequence of str
        The kind of every dataset index.
    batch_sizes : dict of str to int
        The global batch size of each kind.
    shuffle : bool
        Shuffle within kinds and the order of the batches (training); a validation
        sampler gives the same batches every pass.
    seed : int, optional
        The seed; epoch ``e`` uses ``seed + 1000003 * e``.
    rank, world : int, optional
        This rank and the number of ranks of a distributed run.
    """

    def __init__(self, kinds, batch_sizes, shuffle, seed=0, rank=0, world=1):
        self.groups = {}
        for i, k in enumerate(kinds):
            self.groups.setdefault(k, []).append(i)
        missing = sorted(set(self.groups) - set(batch_sizes))
        if missing:
            raise ValueError(f"no batch size for the kinds {missing}")
        self.batch_sizes, self.shuffle, self.seed = batch_sizes, shuffle, seed
        self.rank, self.world, self.epoch = rank, world, 0

    def set_epoch(self, epoch):
        """Set the epoch whose order the next pass gives."""
        self.epoch = epoch

    def _generator(self):
        return torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)

    def _kind_batches(self, k, g):
        idx = self.groups[k]
        if self.shuffle:
            idx = [idx[j] for j in torch.randperm(len(idx), generator=g).tolist()]
        b = self.batch_sizes[k]
        batches = []
        for s in range(0, len(idx), b):
            chunk = idx[s : s + b]
            while len(chunk) % self.world:  # every rank needs a share: wrap in the kind
                chunk.append(idx[len(chunk) % len(idx)])
            batches.append(chunk)
        return batches

    def _global_batches(self):
        g = self._generator()
        batches = []
        for k in sorted(self.groups):
            batches += self._kind_batches(k, g)
        if self.shuffle:
            batches = [
                batches[j] for j in torch.randperm(len(batches), generator=g).tolist()
            ]
        return batches

    def __iter__(self):
        batches = self._global_batches()
        self.epoch += 1  # the next pass reshuffles; set_epoch still overrides
        for chunk in batches:
            yield chunk[self.rank :: self.world]

    def __len__(self):
        return sum(-(-len(v) // self.batch_sizes[k]) for k, v in self.groups.items())


class ShareBatchSampler(KindBatchSampler):
    """Like :class:`KindBatchSampler`, with cell batches a fixed share of every epoch.

    The cluster batches of an epoch are one pass over the clusters; the cell batches
    are drawn from the cells in a fresh random order, repeated or left out across
    epoch boundaries as the share requires, so that every cell recurs as evenly as
    possible.

    Parameters
    ----------
    kinds, batch_sizes, shuffle, seed
        As in :class:`KindBatchSampler` (single process only).
    share : float
        The share of an epoch's batches that are cell batches, in (0, 1).
    cell_kind : str, optional
        The kind that is cells, by default ``"cell"``.
    """

    def __init__(self, kinds, batch_sizes, shuffle, seed=0, *, share, cell_kind="cell"):
        super().__init__(kinds, batch_sizes, shuffle, seed, 0, 1)
        if not 0.0 < share < 1.0:
            raise ValueError(f"share must lie in (0, 1), got {share}")
        self.share, self.cell_kind = share, cell_kind
        n_other = sum(
            -(-len(v) // batch_sizes[k])
            for k, v in self.groups.items()
            if k != cell_kind
        )
        n_cells = len(self.groups.get(cell_kind, []))
        self.n_cell_batches = (
            0 if n_cells == 0 else max(1, round(share / (1.0 - share) * n_other))
        )
        self.n_other_batches = n_other

    def _global_batches(self):
        g = self._generator()
        batches = []
        for k in sorted(self.groups):
            if k == self.cell_kind:
                continue
            idx = self.groups[k]
            idx = [idx[j] for j in torch.randperm(len(idx), generator=g).tolist()]
            b = self.batch_sizes[k]
            batches += [idx[s : s + b] for s in range(0, len(idx), b)]
        cells = self.groups.get(self.cell_kind, [])
        b = self.batch_sizes.get(self.cell_kind, 1)
        need, stream = self.n_cell_batches * b, []
        while len(stream) < need:
            stream += [
                cells[j] for j in torch.randperm(len(cells), generator=g).tolist()
            ]
        batches += [stream[s : s + b] for s in range(0, need, b)]
        return [batches[j] for j in torch.randperm(len(batches), generator=g).tolist()]

    def __len__(self):
        return self.n_other_batches + self.n_cell_batches
