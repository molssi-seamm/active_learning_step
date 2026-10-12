#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""``seamm-al train``: k-fold members, the manifest, and reshuffling in xnn's Trainer.

Needs torch and xnn; skipped where they are absent. Tiny SchNet models on CPU.
"""

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xnn")

from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402

from active_learning_step.cli import main  # noqa: E402
from active_learning_step.committee import Committee  # noqa: E402
from active_learning_step.samplers import KindBatchSampler, kinds_of  # noqa: E402
from active_learning_step.train import make_trainer  # noqa: E402

RECIPE = """\
model:
  name: schnet
  cutoff: 4.0
  n_features: 8
  n_interactions: 1
  n_rbf: 6
  extra:
    species: [1, 8]
data:
  cutoff: 4.0
  batch_size: 3
optim:
  lr: 0.001
  epochs: 2
  energy_weight: 1.0
  force_weight: 10.0
  stress_weight: 0.0
  scheduler: none
device: cpu
seed: 0
output_dir: unused
"""


def _data(path, n=14, seed=0):
    """Labelled H/O frames, every third one periodic."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        k = 6 if i % 2 else 3
        a = Atoms("OHH" * (k // 3), positions=rng.uniform(0, 4, (k, 3)))
        if i % 3 == 0:
            a.set_cell(np.eye(3) * 6.0)
            a.set_pbc(True)
        a.info["energy"] = float(rng.normal())
        a.arrays["forces"] = rng.normal(0, 0.5, (k, 3))
        a.info["SEAMM/configuration_name"] = f"frame {i}"
        frames.append(a)
    write(path, frames, format="extxyz")
    return frames


@pytest.fixture
def setup(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE)
    data = tmp_path / "data.extxyz"
    _data(data)
    return tmp_path, recipe, data


def _train(tmp_path, recipe, data, *extra):
    return main(
        [
            "train",
            str(data),
            "-c",
            str(recipe),
            "-o",
            str(tmp_path / "com"),
            "-k",
            "3",
            *extra,
        ]
    )


def test_train_committee_members_manifest_and_frames(setup):
    tmp_path, recipe, data = setup
    assert _train(tmp_path, recipe, data, "--cell-share", "0.3") == 0
    com = tmp_path / "com"
    manifest = json.loads((com / "manifest.json").read_text())
    assert manifest["k"] == 3 and manifest["n_frames"] == 14
    assert manifest["init_seeds"] == [1000, 1001, 1002]
    assert sum(f["n_held"] for f in manifest["folds"]) == 14
    assert (
        manifest["data"][0]["n_frames"] == 14
        and len(manifest["data"][0]["sha256"]) == 64
    )
    assert manifest["batching"]["cell_share"] == 0.3
    assert "xnn" in manifest["code"]
    assert len(manifest["code"]["active_learning_step_source_sha256"]) == 64
    with open(com / "frames.csv") as fh:
        frames = list(csv.DictReader(fh))
    assert [f["configuration"] for f in frames] == [f"frame {i}" for i in range(14)]
    assert sorted({f["fold"] for f in frames}) == ["0", "1", "2"]
    for member in range(3):
        mdir = com / f"m{member}"
        for name in ("best.pt", "last.pt", "history.json", "member.json"):
            assert (mdir / name).exists(), name
        info = json.loads((mdir / "member.json").read_text())
        assert info["init_seed"] == 1000 + member and info["epochs"] == 2
        held = sum(1 for f in frames if f["fold"] == str(member))
        assert info["n_val"] == held and info["n_train"] == 14 - held
        assert len(json.loads((mdir / "history.json").read_text())["epoch"]) == 2
    # the members are a committee
    committee = Committee.from_checkpoints(
        [str(com / f"m{m}" / "best.pt") for m in range(3)]
    )
    assert committee.size == 3 and committee.cutoff == 4.0


def test_rerun_skips_and_a_different_plan_is_refused(setup, capsys):
    tmp_path, recipe, data = setup
    assert _train(tmp_path, recipe, data, "--member", "1") == 0
    assert (tmp_path / "com" / "m1" / "last.pt").exists()
    assert not (tmp_path / "com" / "m0").exists()
    stamp = (tmp_path / "com" / "m1" / "last.pt").stat().st_mtime_ns
    assert _train(tmp_path, recipe, data, "--member", "1") == 0
    assert (tmp_path / "com" / "m1" / "last.pt").stat().st_mtime_ns == stamp
    assert "skipping" in capsys.readouterr().out
    assert _train(tmp_path, recipe, data, "--data-seed", "4") == 2
    assert "different committee" in capsys.readouterr().err


def test_steps_set_the_epochs(setup):
    tmp_path, recipe, data = setup
    assert _train(tmp_path, recipe, data, "--member", "0", "--steps", "9") == 0
    info = json.loads((tmp_path / "com" / "m0" / "member.json").read_text())
    assert info["epochs"] == max(1, round(9 / info["batches_per_epoch"]))


class _Recorder(KindBatchSampler):
    """A sampler that records the order of every pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passes = []

    def __iter__(self):
        batches = list(super().__iter__())
        self.passes.append(batches)
        return iter(batches)


def test_the_trainer_sees_a_new_order_every_epoch(tmp_path):
    """Under the installed xnn, the training batches are reshuffled every epoch."""
    from xnn.common.config import from_yaml
    from xnn.common.data import AtomicDataset
    from xnn.common.data.ase_io import atoms_to_structure
    from xnn.common.train import Trainer

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE.replace("epochs: 2", "epochs: 4"))
    cfg = from_yaml(str(recipe))
    cfg.output_dir = str(tmp_path)
    structures = [atoms_to_structure(a) for a in _data(tmp_path / "d.extxyz", n=24)]
    sizes = {"cluster": 3, "cell": 2}
    sampler = _Recorder(kinds_of(structures), sizes, True, 11)
    val = KindBatchSampler(kinds_of(structures[:6]), sizes, False, 11)
    trainer, hook = make_trainer(
        Trainer,
        cfg,
        AtomicDataset(structures, 4.0),
        AtomicDataset(structures[:6], 4.0),
        sampler,
        val,
    )
    trainer.fit()
    print(f"samplers attached by {hook}")
    assert len(sampler.passes) == 4
    assert len({str(p) for p in sampler.passes}) == 4
    for epoch, order in enumerate(sampler.passes):
        told = KindBatchSampler(kinds_of(structures), sizes, True, 11)
        told.set_epoch(epoch)
        assert list(told) == order


MACE = """\
model:
  name: mace
  cutoff: 4.0
  n_features: 8
  n_interactions: 1
  n_rbf: 4
  avg_num_neighbors: auto
  species: [1, 8]
  max_ell: 1
  correlation: 2
  hidden_irreps: 8x0e
  MLP_irreps: 4x0e
  radial_MLP: [8]
  atomic_energies: [-13.6, -2041.0]
data:
  cutoff: 4.0
  batch_size: 3
optim:
  epochs: 1
  scheduler: none
device: cpu
"""


def test_avg_num_neighbors_auto_is_the_training_folds_mean(tmp_path):
    recipe = tmp_path / "mace.yaml"
    recipe.write_text(MACE)
    data = tmp_path / "data.extxyz"
    _data(data)
    assert (
        main(
            [
                "train",
                str(data),
                "-c",
                str(recipe),
                "-o",
                str(tmp_path / "c"),
                "-k",
                "2",
                "--member",
                "0",
            ]
        )
        == 0
    )
    info = json.loads((tmp_path / "c" / "m0" / "member.json").read_text())
    assert info["avg_num_neighbors"] > 0
    ck = torch.load(tmp_path / "c" / "m0" / "last.pt", weights_only=False)
    assert ck["cfg"].model.extra["avg_num_neighbors"] == pytest.approx(
        info["avg_num_neighbors"]
    )
