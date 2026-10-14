#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The committee MDI engine: the mean potential and the spreads it records.

Needs torch and xnn (not pymdi: the engine is driven here without MDI). Tiny SchNet
members on CPU, in float64.
"""

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xnn")

from active_learning_step.committee import Committee, masses, molecules  # noqa: E402
from active_learning_step.engine import (  # noqa: E402
    CommitteeEngine,
    CommitteeMeanCore,
    build_committee,
    committee_mean_potential,
)
from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _cfg():
    return from_dict(
        {
            "model": {
                "name": "schnet",
                "n_features": 8,
                "n_interactions": 1,
                "n_rbf": 6,
                "cutoff": 4.0,
                "extra": {"species": [1, 8]},
            }
        }
    )


@pytest.fixture
def paths(tmp_path):
    out = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = ForceStressOutput(build_model(_cfg().model), compute_stress=True)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.3 * torch.randn_like(p))
        path = tmp_path / f"m{seed}.pt"
        torch.save({"model": m.state_dict(), "cfg": _cfg()}, path)
        out.append(str(path))
    return out


def _water_cell(seed=0, n=4, box=7.0):
    rng = np.random.default_rng(seed)
    pos = []
    for k in range(n):
        o = rng.uniform(0, box, 3)
        pos += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
    return {
        "pos": np.array(pos),
        "atomic_numbers": [8, 1, 1] * n,
        "cell": np.eye(3) * box,
        "pbc": [True] * 3,
    }


def test_mean_core_shares_the_members_and_is_their_mean(paths):
    members, records = build_committee(paths, dtype=torch.float64)
    mean = committee_mean_potential(members=members, records=records, dispersion=False)
    core = mean.model
    assert isinstance(core, CommitteeMeanCore) and core.size == 3
    assert all(core.members[k] is members[k].model for k in range(3))
    com = Committee(members, subtracted_dispersion=records)
    for s in (
        _water_cell(1),
        {k: v for k, v in _water_cell(2).items() if k not in ("cell", "pbc")},
    ):
        g = collate([structure_to_graph(s, 4.0)])
        out = mean(g)
        ref = com.predict(collate([structure_to_graph(s, 4.0)]))
        assert torch.allclose(out["energy"], ref["energy"], rtol=1e-12, atol=1e-12)
        assert torch.allclose(out["forces"], ref["forces"], rtol=1e-10, atol=1e-12)
        if ref["stress"] is not None:
            assert torch.allclose(out["stress"], ref["stress"], rtol=1e-10, atol=1e-12)


def test_mean_forces_are_the_energy_gradient(paths):
    mean = committee_mean_potential(paths, dtype=torch.float64, dispersion=False)
    s = _water_cell(3)
    f = mean(collate([structure_to_graph(s, 4.0)]))["forces"].detach().numpy()
    h = 1e-5
    for atom, axis in ((0, 0), (4, 2), (7, 1)):
        e = []
        for sign in (1, -1):
            t = {**s, "pos": s["pos"].copy()}
            t["pos"][atom, axis] += sign * h
            e.append(float(mean(collate([structure_to_graph(t, 4.0)]))["energy"][0]))
        assert -(e[0] - e[1]) / (2 * h) == pytest.approx(
            f[atom, axis], rel=1e-6, abs=1e-8
        )


def _drive(engine, structures):
    """Feed structures to the engine as an MDI driver would, one step each."""
    for s in structures:
        engine.atomic_numbers = np.asarray(s["atomic_numbers"])
        engine.natoms = len(engine.atomic_numbers)
        engine.coords_bohr = np.asarray(s["pos"]) / BOHR_TO_ANGSTROM
        engine.cell_bohr = np.asarray(s["cell"]) / BOHR_TO_ANGSTROM
        engine.calculate()


def test_engine_records_the_spreads_of_the_committee(paths, tmp_path):
    members, records = build_committee(paths, dtype=torch.float64)
    mean = committee_mean_potential(members=members, records=records, dispersion=False)
    com = Committee(members, subtracted_dispersion=records)
    engine = CommitteeEngine(
        mean,
        cutoff=4.0,
        committee=com,
        every=2,
        spreads=str(tmp_path / "s.csv"),
        flags={"rho_forces": 0.0},
        flagged=str(tmp_path / "flagged.extxyz"),
        frames=str(tmp_path / "frames.extxyz"),
    )
    structures = []
    for step in range(5):  # the same molecules, moved a little each step
        s = _water_cell(4)
        s["pos"] = s["pos"] + 0.02 * step
        structures.append(s)
    _drive(engine, structures)
    engine.close()
    with open(tmp_path / "s.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["step"]) for r in rows] == [0, 2, 4]
    for r in rows:
        s = structures[int(r["step"])]
        ref = com.score([s], batch_size=1, molecule=molecules, masses=masses)
        for k in (
            "rho_energy",
            "rho_forces",
            "rho_forces_max_atom",
            "rho_pressure",
            "rho_pressure_molecular",
        ):
            assert float(r[k]) == pytest.approx(float(ref[k][0]), rel=1e-9), k
        assert float(r["max_dF"]) < 1e-10
        assert float(r["engine_energy"]) == pytest.approx(
            float(r["energy_mean"]), rel=1e-12
        )
    from ase.io import read

    flagged = read(tmp_path / "flagged.extxyz", index=":")
    assert [a.info["AL/step"] for a in flagged] == [0, 2, 4]
    assert flagged[1].arrays["rho_forces_atom"].shape == (12,)
    assert np.allclose(flagged[1].positions, structures[2]["pos"])
    assert "rho_forces" in flagged[0].info["AL/flags"]
    assert len(read(tmp_path / "frames.extxyz", index=":")) == 3
    # the engine's own results are the mean, in MDI units
    f_ev = engine.forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
    g = collate([structure_to_graph(structures[-1], 4.0)])
    assert np.allclose(f_ev, mean(g)["forces"].detach().numpy(), rtol=1e-10, atol=1e-12)


def test_flags_must_name_a_spread():
    from active_learning_step.engine import _flags

    assert _flags(["rho_forces=0.1", "rho_pressure_molecular=500"]) == {
        "rho_forces": 0.1,
        "rho_pressure_molecular": 500.0,
    }
    with pytest.raises(ValueError):
        _flags(["temperature=300"])
    with pytest.raises(ValueError):
        _flags(["rho_forces"])
