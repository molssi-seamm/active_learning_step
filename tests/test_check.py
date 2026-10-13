#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""``seamm-al check``: out-of-fold errors, spreads, correlations and calibration.

The error definitions are checked against each member evaluated on its own. Needs
torch and xnn; skipped where they are absent. Tiny SchNet models on CPU.
"""

import csv
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xnn")

from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402

from active_learning_step.check import _ranks, read_folds, spearman  # noqa: E402
from active_learning_step.cli import main  # noqa: E402
from active_learning_step.committee import (  # noqa: E402
    ATM_PER_EV_A3,
    molecular_pressure,
    structure_from_atoms,
)
from active_learning_step.molecules import masses, molecules  # noqa: E402
from active_learning_step.thresholds import calibrate_threshold  # noqa: E402

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
  epochs: 2
  scheduler: none
device: cpu
"""


def _data(path, n=15, seed=0):
    """Labelled waters, every third frame a cell with a stress; REF_* keys, tags."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        k = 6 if i % 2 else 3
        a = Atoms("OHH" * (k // 3), positions=rng.uniform(0, 4, (k, 3)))
        if i % 3 == 0:
            a.set_cell(np.eye(3) * 6.0)
            a.set_pbc(True)
            a.info["REF_stress"] = rng.normal(0, 0.01, 9)  # a 9-vector, as mlff's
        a.info["REF_energy"] = float(rng.normal())
        a.arrays["REF_forces"] = rng.normal(0, 0.5, (k, 3))
        a.info["SEAMM/configuration_name"] = f"frame {i}"
        a.info["frame"] = f"T{i:02d}"
        frames.append(a)
    write(path, frames, format="extxyz")
    return frames


@pytest.fixture(scope="module")
def committee(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("check")
    (tmp / "recipe.yaml").write_text(RECIPE)
    frames = _data(tmp / "data.extxyz")
    assert (
        main(
            [
                "train",
                str(tmp / "data.extxyz"),
                "-c",
                str(tmp / "recipe.yaml"),
                "-o",
                str(tmp / "com"),
                "-k",
                "3",
                "--set",
                "data.energy_key=REF_energy",
                "--set",
                "data.forces_key=REF_forces",
            ]
        )
        == 0
    )
    return tmp, frames


def _member_outputs(path, atoms):
    """One member alone: energy, forces, pressure and molecular pressure (atm)."""
    from xnn.common.data import collate, structure_to_graph
    from xnn.common.models.hub import build_potential, load_checkpoint

    ck = load_checkpoint(path)
    m = build_potential(
        ck.config,
        ck.state_dict,
        dispersion=False,
        compute_forces=True,
        compute_stress=True,
    )
    s = structure_from_atoms(atoms)
    g = collate([structure_to_graph(s, float(m.cutoff))])
    out = m(g)
    res = {"E": float(out["energy"][0]), "F": out["forces"].detach().double().numpy()}
    if s["cell"] is not None:
        st = out["stress"].detach()
        mol = torch.as_tensor(molecules(s))
        mas = torch.as_tensor(masses(s), dtype=g.pos.dtype)
        res["P"] = float(-st[0].diagonal().sum() / 3) * ATM_PER_EV_A3
        res["Pm"] = (
            float(
                molecular_pressure(
                    st,
                    g.pos.detach(),
                    out["forces"].detach(),
                    g.cell,
                    g.batch,
                    mol,
                    g.pbc,
                    mas,
                )[0]
            )
            * ATM_PER_EV_A3
        )
    return res


def _label_pressures(atoms):
    from xnn.common.data import collate, structure_to_graph

    s = structure_from_atoms(atoms)
    g = collate([structure_to_graph(s, 4.0)])
    st = torch.as_tensor(
        np.reshape(atoms.info["REF_stress"], (3, 3)), dtype=g.pos.dtype
    )[None]
    mol = torch.as_tensor(molecules(s))
    mas = torch.as_tensor(masses(s), dtype=g.pos.dtype)
    f = torch.as_tensor(atoms.arrays["REF_forces"], dtype=g.pos.dtype)
    p = float(-st[0].diagonal().sum() / 3) * ATM_PER_EV_A3
    pm = (
        float(molecular_pressure(st, g.pos, f, g.cell, g.batch, mol, g.pbc, mas)[0])
        * ATM_PER_EV_A3
    )
    return p, pm


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_train_layout_errors_are_the_holding_members(committee):
    tmp, frames = committee
    out = tmp / "check.csv"
    assert (
        main(
            [
                "check",
                str(tmp / "com"),
                str(tmp / "data.extxyz"),
                "-o",
                str(out),
                "--summary",
                str(tmp / "summary.json"),
            ]
        )
        == 0
    )
    rows = _rows(out)
    folds = {
        r["configuration"]: int(r["fold"]) for r in _rows(tmp / "com" / "frames.csv")
    }
    members = [str(tmp / "com" / f"m{k}" / "best.pt") for k in range(3)]
    assert len(rows) == 15 and all(r["role"] == "held" for r in rows)
    for row, atoms in zip(rows, frames):
        k = folds[row["tag"]]
        assert row["oof_members"] == str(k)
        alone = _member_outputs(members[k], atoms)
        fr = atoms.arrays["REF_forces"]
        rmse = np.sqrt(((alone["F"] - fr) ** 2).sum(-1).mean())
        assert float(row["err_forces"]) == pytest.approx(rmse, rel=1e-5)
        e = abs(alone["E"] - atoms.info["REF_energy"]) / len(atoms)
        assert float(row["err_energy"]) == pytest.approx(e, rel=1e-5, abs=1e-7)
        if row["kind"] == "cell":
            p, pm = _label_pressures(atoms)
            assert float(row["err_pressure"]) == pytest.approx(
                abs(alone["P"] - p), rel=1e-4
            )
            assert float(row["err_pressure_molecular"]) == pytest.approx(
                abs(alone["Pm"] - pm), rel=1e-4
            )
            worst = max(abs(_member_outputs(m, atoms)["Pm"] - pm) for m in members)
            assert float(row["eps_pressure_molecular"]) == pytest.approx(
                worst, rel=1e-4
            )
        worst_f = max(
            np.sqrt(((_member_outputs(m, atoms)["F"] - fr) ** 2).sum(-1).mean())
            for m in members
        )
        assert float(row["eps_forces"]) == pytest.approx(worst_f, rel=1e-5)
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["roles"] == {"held": 15, "unseen": 0, "trained": 0}
    assert (
        "cell/all/rho_pressure_molecular~err_pressure_molecular" in summary["spearman"]
    )
    assert summary["spearman"]["cluster/held/rho_forces~err_forces"]["n"] == 10


def test_fold_manifest_layout_held_unseen_and_trained(committee, tmp_path):
    tmp, frames = committee
    com = tmp_path / "mlff"
    for k in range(3):
        (com / f"m{k}").mkdir(parents=True)
        shutil.copy(tmp / "com" / f"m{k}" / "best.pt", com / f"m{k}" / "best.pt")
        manifest = {
            "member": k,
            "held_out": [f"T{i:02d}" for i in range(9) if i % 3 == k],
            "validation": ["T09", "T10"],
            "test": ["T11", "T12"],
            "trained_on": [],
        }
        (com / f"m{k}" / "fold_manifest.json").write_text(json.dumps(manifest))
    paths, roles, layout = read_folds(com)
    assert layout == "fold_manifest" and roles["T04"] == ("held", [1])
    assert roles["T10"] == ("unseen", [0, 1, 2])
    out = tmp_path / "c.csv"
    assert (
        main(
            [
                "check",
                str(com),
                str(tmp / "data.extxyz"),
                "-o",
                str(out),
                "--tag-key",
                "frame",
            ]
        )
        == 0
    )
    rows = {r["tag"]: r for r in _rows(out)}
    assert rows["T04"]["role"] == "held" and rows["T04"]["oof_members"] == "1"
    assert rows["T13"]["role"] == "trained" and rows["T13"]["err_forces"] == ""
    # an unseen frame: the mean of every member's error
    atoms = frames[10]
    fr = atoms.arrays["REF_forces"]
    mean = np.mean(
        [
            np.sqrt(((_member_outputs(p, atoms)["F"] - fr) ** 2).sum(-1).mean())
            for p in paths
        ]
    )
    assert rows["T10"]["role"] == "unseen" and rows["T10"]["oof_members"] == "0;1;2"
    assert float(rows["T10"]["err_forces"]) == pytest.approx(mean, rel=1e-5)


def test_calibration_and_its_warning(committee, tmp_path):
    from active_learning_step.check import check_committee

    tmp, _ = committee
    rows, summary = check_committee(
        tmp / "com",
        [str(tmp / "data.extxyz")],
        out_csv=tmp_path / "c.csv",
        calibrate=["rho_forces:0:0.9"],
    )
    clusters = [r for r in rows if r["kind"] == "cluster"]
    level = float(np.median([r["eps_forces"] for r in clusters]))
    rows, summary = check_committee(
        tmp / "com",
        [str(tmp / "data.extxyz")],
        out_csv=tmp_path / "c.csv",
        calibrate=[f"rho_forces:{level}:0.9"],
    )
    entry = summary["calibration"]["cluster/rho_forces"]
    rho = np.array([r["rho_forces"] for r in clusters])
    eps = np.array([r["eps_forces"] for r in clusters])
    assert entry["n_fail"] == int((eps > level).sum()) < 10
    assert entry["threshold"] == pytest.approx(
        calibrate_threshold(rho, eps, level, 0.9)
    )
    assert "not to be trusted" in entry["warning"]
    with pytest.raises(ValueError):
        check_committee(
            tmp / "com",
            [str(tmp / "data.extxyz")],
            out_csv=tmp_path / "c.csv",
            calibrate=["rho_nonesuch:1"],
        )


def test_spearman_with_ties_matches_scipy():
    x = [1.0, 2.0, 2.0, 3.0, 5.0, 4.0, np.nan]
    y = [2.0, 1.0, 3.0, 3.0, 6.0, 5.0, 1.0]
    rho, p, n = spearman(x, y)
    assert n == 6
    rx, ry = _ranks(x[:6]), _ranks(y[:6])
    assert rho == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    scipy = pytest.importorskip("scipy.stats")
    ref = scipy.spearmanr(x[:6], y[:6])
    assert rho == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
