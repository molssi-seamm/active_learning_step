# -*- coding: utf-8 -*-

"""``seamm-al train``: the k members of a committee, each on k - 1 folds of the data.

The labelled frames of the data files are cut into ``k`` folds
(:func:`~active_learning_step.thresholds.kfold_indices`, seeded by the data seed);
member ``l`` trains on every fold but ``l`` and validates on fold ``l``, from the
initialisation seed ``init_seed + l``. The recipe (model, loss weights, optimiser,
epochs, the recorded subtracted dispersion) is an ordinary xnn YAML config; only the
data, the folds, the seeds, the output directory and the batching come from here.

Batches never mix clusters and cells (:mod:`~active_learning_step.samplers`): clusters
use the config's ``data.batch_size``, cells ``cell_batch_size``. With ``cell_share``
the cell batches make that share of every epoch. With ``steps`` every member trains
for that many optimiser steps whatever the size of the data (the epochs are the steps
divided by the batches per epoch), so committees trained on different selections
differ only in their data.

The output directory holds

* ``manifest.json``: the folds' sizes, the seeds, the data files with their SHA-256,
  a hash of the ordered frames, the recipe, the batching and the code versions. Every
  member checks it before training, so members trained in separate jobs are certain
  to share one fold plan; a different plan in the same directory is refused;
* ``frames.csv``: every frame with its file, index, configuration name, kind and fold;
* ``m<l>/``: each member's ``best.pt``, ``last.pt``, ``history.json`` and
  ``member.json`` (its seeds, sizes, steps, timing and versions).

A member whose ``last.pt`` exists is skipped, so a job can be rerun after a failure.
"""

import csv
import hashlib
import json
import logging
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

from .pool import configuration_name, iter_frames
from .thresholds import kfold_indices

logger = logging.getLogger(__name__)

__all__ = [
    "file_sha256",
    "load_frames",
    "fold_plan",
    "make_trainer",
    "require_training_xnn",
    "source_sha256",
    "stratified_fold_plan",
    "train_committee",
]


#: The oldest xnn that ``seamm-al train`` trains with: 0.13 chooses ``best.pt``,
#: drives the plateau learning rate and logs on dataset-averaged epoch metrics, takes
#: batch samplers (``set_epoch`` every epoch) and has the callback API used here.
#: Sina (xnn's author): older releases come with no guarantees. Inference (score,
#: check, engine) works with older releases.
MIN_TRAIN_XNN = (0, 13, 0)


def _version_tuple(text):
    parts = []
    for p in str(text).split("+")[0].split(".")[:3]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def require_training_xnn():
    """Refuse to train with an xnn older than :data:`MIN_TRAIN_XNN`."""
    import xnn

    if _version_tuple(xnn.__version__) < MIN_TRAIN_XNN:
        need = ".".join(str(v) for v in MIN_TRAIN_XNN)
        raise ValueError(
            f"xnn >= {need} is required to train (found {xnn.__version__}); "
            "score, check and engine work with older releases"
        )


def history_callback(history, epochs, t0):
    """An xnn trainer callback that records every epoch in ``history``.

    ``history`` gets ``epoch``, ``lr``, ``train`` and ``val`` (dicts of the
    epoch's dataset-level metrics, without the ``train/`` and ``val/`` prefixes;
    ``val`` is empty on epochs without validation) and ``seconds``.
    """
    from xnn.common.train.callbacks import Callback

    class _History(Callback):
        def __init__(self):
            self.last = t0

        def on_epoch_end(self, trainer, epoch, metrics):
            now = time.time()
            tr = {k[6:]: v for k, v in metrics.items() if k.startswith("train/")}
            va = {k[4:]: v for k, v in metrics.items() if k.startswith("val/")}
            history["epoch"].append(int(epoch))
            history["lr"].append(metrics.get("lr"))
            history["train"].append(tr)
            history["val"].append(va)
            history["seconds"].append(now - self.last)
            self.last = now
            if epoch < 2 or epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                print(
                    f"  epoch {epoch:>4d}  train {tr.get('loss', float('nan')):.4e}"
                    f"  val {va.get('loss', float('nan')):.4e}  ({now - t0:.0f} s)",
                    flush=True,
                )

    return _History()


def make_trainer(Trainer, cfg, train_set, val_set, sampler, val_sampler, callbacks=()):
    """An xnn Trainer that batches with our samplers and runs our callbacks.

    The samplers go to the trainer's ``train_batch_sampler`` and
    ``val_batch_sampler``; it tells them the epoch before every pass. Needs xnn
    0.13 or later (:func:`require_training_xnn`).
    """
    require_training_xnn()
    return Trainer(
        cfg,
        train_set,
        val_set,
        train_batch_sampler=sampler,
        val_batch_sampler=val_sampler,
        callbacks=list(callbacks),
    )


def file_sha256(path, chunk=1 << 20):
    """The SHA-256 of a file, hex."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_frames(
    paths, energy_key="energy", forces_key="forces", stress_key="stress", columns=()
):
    """The labelled structures of data files, with where each came from.

    Parameters
    ----------
    paths : sequence of str
        extxyz files (or SDF from SEAMM; their labels must be in xnn's keys).
    energy_key, forces_key, stress_key : str, optional
        Where the labels are, as xnn's ``data.energy_key`` etc.
    columns : sequence of str, optional
        ``info`` keys to keep for every frame (as strings; ``""`` when absent).

    Returns
    -------
    (list of dict, list of dict)
        The xnn structure dicts, in file order, and for each the ``file``,
        ``index``, ``configuration`` and the requested columns.
    """
    from xnn.common.data.ase_io import atoms_to_structure

    structures, frames = [], []
    for path in paths:
        fname = str(Path(path).resolve())
        for index, atoms in iter_frames(path):
            structures.append(
                atoms_to_structure(
                    atoms,
                    energy_key=energy_key,
                    forces_key=forces_key,
                    stress_key=stress_key,
                )
            )
            frames.append(
                {
                    "file": fname,
                    "index": index,
                    "configuration": configuration_name(atoms),
                    **{c: _text(atoms.info.get(c)) for c in columns},
                }
            )
    return structures, frames


def _text(value):
    return "" if value is None else str(value)


def fold_plan(n, k, data_seed):
    """The fold of every frame: ``fold[i]`` is the member that holds frame ``i`` out.

    Plain k-fold: :func:`~active_learning_step.thresholds.kfold_indices`, the water
    campaign's split.
    """
    fold = np.empty(n, dtype=int)
    for member, (_, held) in enumerate(kfold_indices(n, k, data_seed)):
        fold[held] = member
    return fold


def stratified_fold_plan(strata, k, data_seed):
    """Folds that split every stratum as evenly as possible.

    The strata are taken in sorted order; the frames of each are shuffled (one
    generator seeded by ``data_seed``) and dealt to the folds in turn, each stratum
    starting at the fold after the one the previous stratum ended on. So within a
    stratum the folds' counts differ by at most one, and so do the folds' sizes.

    Parameters
    ----------
    strata : sequence of str
        The stratum of every frame.
    k : int
        Number of folds.
    data_seed : int
        Seed of the shuffles.

    Returns
    -------
    numpy.ndarray of int
        The fold of every frame.

    Raises
    ------
    ValueError
        If ``k`` is smaller than 2 or larger than the number of frames.
    """
    n = len(strata)
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]; got k={k}, n={n}")
    rng = np.random.default_rng(data_seed)
    strata = np.asarray(strata, dtype=object)
    fold = np.empty(n, dtype=int)
    start = 0
    for label in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == label)
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = (start + np.arange(len(idx))) % k
        start = (start + len(idx)) % k
    return fold


def source_sha256():
    """A SHA-256 over the package's Python sources (name and contents, sorted).

    It identifies the code however it was deployed, including a copy without git
    history, where the version reads ``0+unknown``.
    """
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        if path.name == "_version.py":
            continue
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _versions():
    import torch
    import xnn

    import active_learning_step

    return {
        "active_learning_step": getattr(active_learning_step, "__version__", "unknown"),
        "active_learning_step_revision": getattr(
            active_learning_step, "__git_revision__", None
        ),
        "active_learning_step_source_sha256": source_sha256(),
        "xnn": xnn.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _plain(value):
    """A JSON-safe copy of a config section."""
    return json.loads(
        json.dumps(value, default=lambda o: getattr(o, "__dict__", str(o)))
    )


def _write_json(path, data):
    tmp = Path(f"{path}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


def train_committee(
    config,
    data,
    out,
    k=5,
    members=None,
    data_seed=0,
    init_seed=1000,
    cell_share=None,
    cell_batch_size=4,
    steps=None,
    device=None,
    overrides=(),
    stratify="kind",
    stratify_columns=(),
):
    """Train members of a k-fold committee.

    Parameters
    ----------
    config : str
        The xnn YAML config: the recipe.
    data : sequence of str
        The labelled data files, in order.
    out : str
        The committee's directory.
    k : int, optional
        Number of folds and members, by default 5.
    members : sequence of int, optional
        Which members to train here, by default all.
    data_seed : int, optional
        Seed of the fold permutation, by default 0.
    init_seed : int, optional
        Member ``l`` is initialised (and its batches shuffled) with ``init_seed + l``;
        by default 1000, as in the water campaign.
    cell_share : float, optional
        Cell batches as this share of every epoch; by default one pass over the
        cells.
    cell_batch_size : int, optional
        Cells per batch, by default 4.
    steps : int, optional
        Train for this many optimiser steps instead of the config's epochs.
    device : str, optional
        Overrides the config's device.
    overrides : sequence of str, optional
        xnn-style ``a.b=value`` overrides of the config.
    stratify : str, optional
        ``"kind"`` (the default) splits clusters and cells separately, so every
        fold holds out its share of each; ``"none"`` is the water campaign's plain
        k-fold, exactly.
    stratify_columns : sequence of str, optional
        ``info`` keys whose values also define strata (combined with the kind
        unless ``stratify="none"``), e.g. ``liquid``.

    Returns
    -------
    list of str
        The directories of the members trained.
    """
    import torch
    from xnn.common.config import apply_overrides, from_yaml

    require_training_xnn()
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    from .samplers import KindBatchSampler, ShareBatchSampler, kinds_of

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg0 = from_yaml(config)
    if overrides:
        apply_overrides(cfg0, list(overrides))
    keys = {
        "energy_key": cfg0.data.energy_key,
        "forces_key": cfg0.data.forces_key,
        "stress_key": cfg0.data.stress_key,
    }
    if stratify not in ("kind", "none"):
        raise ValueError(f"stratify must be 'kind' or 'none', not {stratify!r}")
    stratify_columns = list(stratify_columns)
    structures, frames = load_frames(data, columns=stratify_columns, **keys)
    n = len(structures)
    kinds = kinds_of(structures)
    by = (["kind"] if stratify == "kind" else []) + stratify_columns
    if by:
        strata = [
            "|".join(
                [kinds[i]] * (stratify == "kind") + [f[c] for c in stratify_columns]
            )
            for i, f in enumerate(frames)
        ]
        fold = stratified_fold_plan(strata, k, data_seed)
    else:
        strata = [""] * n
        fold = fold_plan(n, k, data_seed)
    batch_sizes = {"cluster": int(cfg0.data.batch_size), "cell": int(cell_batch_size)}

    order = hashlib.sha256()
    for f in frames:
        order.update(f"{f['file']}\0{f['index']}\0{f['configuration']}\n".encode())
    files = {}
    for f in frames:
        files.setdefault(f["file"], 0)
        files[f["file"]] += 1
    manifest = {
        "k": k,
        "data_seed": data_seed,
        "init_seeds": [init_seed + member for member in range(k)],
        "data": [
            {"path": p, "sha256": file_sha256(p), "n_frames": m}
            for p, m in files.items()
        ],
        "n_frames": n,
        "frames_sha256": order.hexdigest(),
        "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
        "folds": [
            {
                "member": member,
                "n_train": int((fold != member).sum()),
                "n_held": int((fold == member).sum()),
                "held_kinds": {
                    kind: int(
                        sum(
                            1
                            for i in np.flatnonzero(fold == member)
                            if kinds[i] == kind
                        )
                    )
                    for kind in sorted(set(kinds))
                },
            }
            for member in range(k)
        ],
        "stratification": {
            "by": by,
            "strata": (
                {
                    label: [
                        int(
                            sum(
                                1
                                for i in range(n)
                                if strata[i] == label and fold[i] == m
                            )
                        )
                        for m in range(k)
                    ]
                    for label in sorted(set(strata))
                }
                if by
                else None
            ),
        },
        "batching": {
            "batch_sizes": batch_sizes,
            "cell_share": cell_share,
            "steps": steps,
        },
        "config": str(Path(config).resolve()),
        "config_sha256": file_sha256(config),
        "overrides": list(overrides),
        "recipe": _plain(cfg0.__dict__),
    }
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        same = {key: old.get(key) == manifest[key] for key in manifest}
        if not all(same.values()):
            differ = sorted(key for key, ok in same.items() if not ok)
            raise ValueError(
                f"{path} describes a different committee (differs in {differ}); "
                "use another directory"
            )
    else:
        manifest["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["code"] = _versions()
        _write_json(path, manifest)
        with open(out / "frames.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["frame", "file", "index", "configuration", "kind", *stratify_columns]
                + ["stratum", "fold"]
            )
            for i, f in enumerate(frames):
                writer.writerow(
                    [i, f["file"], f["index"], f["configuration"], kinds[i]]
                    + [f[c] for c in stratify_columns]
                    + [strata[i], fold[i]]
                )

    done = []
    for member in range(k) if members is None else members:
        if not 0 <= member < k:
            raise ValueError(f"member {member} is not in 0..{k - 1}")
        mdir = out / f"m{member}"
        if (mdir / "last.pt").exists():
            print(f"{mdir}/last.pt exists, skipping member {member}", flush=True)
            continue
        mdir.mkdir(exist_ok=True)
        seed = init_seed + member
        train_idx = np.flatnonzero(fold != member)
        held_idx = np.flatnonzero(fold == member)
        train_s = [structures[i] for i in train_idx]
        val_s = [structures[i] for i in held_idx]

        cfg = from_yaml(config)
        if overrides:
            apply_overrides(cfg, list(overrides))
        cfg.seed = seed
        cfg.output_dir = str(mdir)
        if device is not None:
            cfg.device = device
        torch.manual_seed(seed)
        ds_tr = AtomicDataset(train_s, cfg.data.cutoff)
        ds_va = AtomicDataset(val_s, cfg.data.cutoff)
        extra = cfg.model.extra if cfg.model.extra is not None else {}
        avg_nn = None
        if extra.get("avg_num_neighbors") == "auto":
            # as the water campaign: the mean neighbour count of the training set
            edges = sum(ds_tr[i].num_edges for i in range(len(ds_tr)))
            nodes = sum(ds_tr[i].num_nodes for i in range(len(ds_tr)))
            avg_nn = float(edges / nodes)
            extra["avg_num_neighbors"] = avg_nn
        tr_kinds, va_kinds = kinds_of(train_s), kinds_of(val_s)
        if cell_share is None or "cell" not in tr_kinds:
            sampler = KindBatchSampler(tr_kinds, batch_sizes, True, seed)
        else:
            sampler = ShareBatchSampler(tr_kinds, batch_sizes, True, seed, cell_share)
        val_sampler = KindBatchSampler(va_kinds, batch_sizes, False, seed)
        per_epoch = len(sampler)
        if steps is not None:
            cfg.optim.epochs = max(1, int(round(steps / per_epoch)))
        epochs = int(cfg.optim.epochs)
        print(
            f"member {member} of {k}: train {len(train_s)} val {len(val_s)}, "
            f"{per_epoch} batches x {epochs} epochs = {per_epoch * epochs} steps, "
            f"seed {seed}, data seed {data_seed}"
            + (f", avg neighbours {avg_nn:.3f}" if avg_nn is not None else ""),
            flush=True,
        )
        torch.manual_seed(seed)
        history = {"epoch": [], "lr": [], "train": [], "val": [], "seconds": []}
        t0 = time.time()
        trainer = make_trainer(
            Trainer,
            cfg,
            ds_tr,
            ds_va,
            sampler,
            val_sampler,
            callbacks=[history_callback(history, epochs, t0)],
        )
        trainer.fit()
        _write_json(mdir / "history.json", history)
        _write_json(
            mdir / "member.json",
            {
                "member": member,
                "k": k,
                "data_seed": data_seed,
                "init_seed": seed,
                "n_train": len(train_s),
                "n_val": len(val_s),
                "train_kinds": {kd: tr_kinds.count(kd) for kd in sorted(set(tr_kinds))},
                "batches_per_epoch": per_epoch,
                "epochs": epochs,
                "steps": per_epoch * epochs,
                "cell_share": cell_share,
                "avg_num_neighbors": avg_nn,
                "seconds": time.time() - t0,
                "device": str(trainer.device),
                "host": socket.gethostname(),
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "code": _versions(),
                "manifest_frames_sha256": manifest["frames_sha256"],
            },
        )
        print(
            f"member {member}: done in {time.time() - t0:.0f} s -> {mdir}", flush=True
        )
        done.append(str(mdir))
    return done
