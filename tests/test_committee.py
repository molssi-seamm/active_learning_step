"""The committee module, ported from the water campaign.

Sina's tests plus tests of the generalisations.

Needs torch and xnn; skipped where they are absent (CI, the Mac).
"""

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xnn")

from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402

from active_learning_step.committee import (  # noqa: E402
    ATM_PER_EV_A3,
    Committee,
    calibrate_threshold,  # noqa: E402
    disagreement,
    kfold_indices,
    masses,
    molecular_pressure,
    molecular_virial,
    molecules,
    pressure,
    select,
    split_statistics,
    structure_from_atoms,
    worst_error,
)
from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import AtomicDataset, collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structs(n=6, seed=0, periodic=True):
    """Small H/O structures, every other one periodic, with labels."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 6 + 3 * (i % 2)
        s = {
            "pos": rng.uniform(0, 5, (k, 3)),
            "atomic_numbers": [8, 1, 1] * (k // 3),
            "energy": float(rng.normal()),
            "forces": rng.normal(0, 1, (k, 3)),
        }
        if periodic and i % 2 == 0:
            s["cell"] = np.eye(3) * 7.0
            s["pbc"] = [True, True, True]
            s["stress"] = rng.normal(0, 0.01, (3, 3))
        out.append(s)
    return out


def _cfg(cutoff=4.0):
    return from_dict(
        {
            "model": {
                "name": "schnet",
                "n_features": 8,
                "n_interactions": 1,
                "n_rbf": 6,
                "cutoff": cutoff,
                "extra": {"species": [1, 8]},
            }
        }
    )


def _member(seed, cutoff=4.0):
    torch.manual_seed(seed)
    m = ForceStressOutput(build_model(_cfg(cutoff).model), compute_stress=True)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m.eval()


def _committee(n=3, cutoff=4.0):
    return Committee([_member(s, cutoff) for s in range(n)])


# folds
def test_kfold_indices_are_disjoint_and_cover_everything():
    folds = kfold_indices(23, 5, seed=1)
    assert len(folds) == 5
    held = np.concatenate([h for _, h in folds])
    assert sorted(held.tolist()) == list(range(23))
    for train, h in folds:
        assert len(train) + len(h) == 23
        assert not set(train) & set(h)
    assert np.array_equal(folds[0][1], kfold_indices(23, 5, seed=1)[0][1])
    assert not np.array_equal(folds[0][1], kfold_indices(23, 5, seed=2)[0][1])
    with pytest.raises(ValueError):
        kfold_indices(5, 1)


# pressures
def test_pressure_of_hydrostatic_stress():
    s = -0.02 * torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(pressure(s), torch.full((4,), 0.02))


def test_molecular_virial_single_atoms_vanish_and_diatomic_is_analytic():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    F = torch.tensor([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    batch = torch.zeros(3, dtype=torch.long)
    assert float(molecular_virial(pos, F, batch, torch.arange(3))) == pytest.approx(0.0)
    # atoms 0 and 1 form a molecule: R = (0.5, 0, 0);
    # (x0 - R).F0 + (x1 - R).F1 = -0.5 - 1.0
    w = molecular_virial(pos, F, batch, torch.tensor([0, 0, 1]))
    assert float(w) == pytest.approx(-1.5)
    # mass-weighted centre: masses 3 and 1 put R at (0.25, 0, 0)
    w = molecular_virial(
        pos, F, batch, torch.tensor([0, 0, 1]), masses=torch.tensor([3.0, 1.0, 1.0])
    )
    assert float(w) == pytest.approx(-0.25 - 1.5)


def test_molecular_virial_uses_minimum_image_across_the_boundary():
    cell = torch.eye(3)[None] * 10.0
    pos = torch.tensor(
        [[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]]
    )  # 0.4 A apart through the wall
    F = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    mol = torch.zeros(2, dtype=torch.long)
    w = molecular_virial(pos, F, batch, mol, cell, torch.ones(1, 3, dtype=torch.bool))
    assert float(w) == pytest.approx(
        0.2 * 1.0 + (-0.2) * (-1.0)
    )  # whole molecule at 0.2, -0.2
    w_raw = molecular_virial(pos, F, batch, mol)  # no cell: no wrapping
    assert float(w_raw) == pytest.approx((-4.8) * 1.0 + 4.8 * (-1.0))


def test_molecular_pressure_is_atomic_for_single_atoms_and_nan_without_cell():
    cell = torch.stack([torch.eye(3) * 8.0, torch.zeros(3, 3)])
    pbc = torch.tensor([[True] * 3, [False] * 3])
    stress = torch.stack([-0.01 * torch.eye(3), torch.zeros(3, 3)])
    pos = torch.rand(7, 3) * 8
    F = torch.rand(7, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    p = molecular_pressure(stress, pos, F, cell, batch, torch.arange(7), pbc)
    assert float(p[0]) == pytest.approx(0.01)
    assert torch.isnan(p[1])


# spreads and errors on hand-built member outputs
def test_disagreement_formulas():
    e = torch.tensor([[1.0, 2.0], [3.0, 2.0], [2.0, 2.0]])  # L=3, B=2
    f = torch.zeros(3, 5, 3)
    f[:, 0, 0] = torch.tensor([1.0, -1.0, 0.0])  # atom 0 of structure 0
    n_atoms = torch.tensor([2, 3])
    batch = torch.tensor([0, 0, 1, 1, 1])
    r = disagreement(
        e,
        f,
        n_atoms,
        batch,
        {"pressure": torch.tensor([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])},
    )
    assert float(r["rho_energy"][0]) == pytest.approx(np.std([1, 3, 2]) / np.sqrt(2))
    assert float(r["rho_energy"][1]) == pytest.approx(0.0)
    sd0 = np.std([1, -1, 0])
    assert float(r["rho_forces"][0]) == pytest.approx(np.sqrt(sd0**2 / 2))
    assert float(r["rho_forces"][1]) == pytest.approx(0.0)
    assert float(r["rho_pressure"][0]) == pytest.approx(np.std([1, 3, 2]))
    assert float(r["rho_pressure"][1]) == pytest.approx(0.0)


def test_worst_error_formulas():
    e = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    e_ref = torch.tensor([2.0, 2.5])
    f = torch.zeros(2, 4, 3)
    f[1, 3, 1] = 2.0
    f_ref = torch.zeros(4, 3)
    n_atoms = torch.tensor([1, 3])
    batch = torch.tensor([0, 1, 1, 1])
    r = worst_error(
        e,
        f,
        n_atoms,
        batch,
        e_ref,
        f_ref,
        {"pressure": torch.tensor([[1.0, 2.0], [4.0, 2.0]])},
        {"pressure": torch.tensor([2.0, 0.0])},
    )
    assert float(r["eps_energy"][0]) == pytest.approx(1.0)
    assert float(r["eps_energy"][1]) == pytest.approx(0.5 / np.sqrt(3))
    assert float(r["eps_forces"][0]) == pytest.approx(0.0)
    assert float(r["eps_forces"][1]) == pytest.approx(np.sqrt(4.0 / 3))
    assert r["eps_pressure"].tolist() == [2.0, 2.0]


# thresholds
def test_calibrate_threshold_captures_the_fraction_and_is_the_largest_such():
    rng = np.random.default_rng(0)
    eps = rng.uniform(0, 3, 500)
    rho = 0.2 * eps + rng.normal(0, 0.1, 500)
    thr = calibrate_threshold(rho, eps, fail_level=1.5, capture=0.98)
    fail = eps > 1.5
    assert (rho[fail] >= thr).mean() >= 0.98
    bigger = np.min(rho[fail][rho[fail] > thr])  # the next failure up
    assert (rho[fail] >= bigger).mean() < 0.98
    assert calibrate_threshold(rho, eps, 1.5, capture=1.0) == pytest.approx(
        rho[fail].min()
    )
    with pytest.raises(ValueError):
        calibrate_threshold(rho, eps, fail_level=10.0)


def test_split_statistics_and_select():
    rho = np.array([0.1, 0.5, 0.9, np.nan])
    err = np.array([1.0, 2.0, 3.0, 4.0])
    st = split_statistics(rho[:3], err[:3], 0.5)
    assert st["n_above"] == 2 and st["n_below"] == 1
    assert st["rmse_above"] == pytest.approx(np.sqrt((4 + 9) / 2))
    assert st["rmse_below"] == pytest.approx(1.0)
    mask = select(
        {"rho_energy": rho, "rho_forces": np.array([0, 0, 0, 1.0])},
        {"rho_energy": 0.5, "rho_forces": 0.5, "rho_pressure": 1.0},
    )
    assert mask.tolist() == [False, True, True, True]
    with pytest.raises(ValueError):
        select({"rho_energy": rho}, {"rho_pressure": 1.0})


# the Committee
def test_committee_rejects_single_member_and_mixed_cutoffs():
    with pytest.raises(ValueError):
        Committee([_member(0)])
    with pytest.raises(ValueError):
        Committee([_member(0, 4.0), _member(1, 4.5)])


def test_committee_forward_predict_and_disagreement_on_a_mixed_batch():
    com = _committee(3)
    structs = _structs(4)
    data = collate([structure_to_graph(s, com.cutoff) for s in structs])
    pos_before = data.pos.clone()
    out = com(data)
    assert out["energy"].shape == (3, 4)
    assert out["forces"].shape == (3, data.num_nodes, 3)
    assert out["stress"].shape == (3, 4, 3, 3)
    assert torch.equal(data.pos, pos_before) and not data.pos.requires_grad
    # members differ, so the spread is non-zero; the mean is the mean
    assert (out["energy"].std(0) > 0).all()
    mean = com.predict(data)
    assert torch.allclose(mean["energy"], out["energy"].mean(0))
    assert torch.allclose(mean["forces"], out["forces"].mean(0))
    # each member alone agrees with its row
    single = com.members[1](
        collate([structure_to_graph(s, com.cutoff) for s in structs])
    )
    assert torch.allclose(single["energy"], out["energy"][1])
    assert torch.allclose(single["forces"], out["forces"][1])
    mol = torch.as_tensor(
        np.concatenate(
            [np.arange(len(s["pos"])) // 3 + 100 * i for i, s in enumerate(structs)]
        )
    )
    r = com.disagreement(data, mol)
    for k in (
        "rho_energy",
        "rho_forces",
        "rho_pressure",
        "rho_pressure_molecular",
        "mean_energy",
        "n_atoms",
    ):
        assert r[k].shape == (4,)
    assert torch.allclose(
        r["rho_energy"],
        out["energy"].std(0, unbiased=False) / data.n_atoms.double().sqrt(),
    )
    # molecular structures of the mixed batch carry no pressure
    assert torch.isnan(r["rho_pressure"][1]) and torch.isnan(r["rho_pressure"][3])
    assert torch.isfinite(r["rho_pressure"][0]) and torch.isfinite(
        r["rho_pressure_molecular"][0]
    )
    p0 = pressure(out["stress"][:, 0]) * ATM_PER_EV_A3
    assert torch.allclose(r["pressure"][:, 0], p0)
    assert float(r["rho_pressure"][0]) == pytest.approx(float(p0.std(unbiased=False)))


def test_committee_identical_members_have_zero_spread():
    m = _member(0)
    com = Committee([m, m, m])
    r = com.disagreement(collate([structure_to_graph(s, 4.0) for s in _structs(2)]))
    assert torch.allclose(r["rho_energy"], torch.zeros(2))
    assert torch.allclose(r["rho_forces"], torch.zeros(2))


def test_committee_score_concatenates_batches_and_matches_disagreement():
    com = _committee(2)
    structs = _structs(5)
    sc = com.score(
        structs, batch_size=2, molecule=lambda s: np.arange(len(s["pos"])) // 3
    )
    assert set(sc) >= {
        "rho_energy",
        "rho_forces",
        "rho_pressure",
        "rho_pressure_molecular",
        "n_atoms",
    }
    assert all(v.shape == (5,) for v in sc.values())
    one = com.disagreement(
        structure_to_graph(structs[2], com.cutoff).to("cpu"),
        torch.as_tensor(np.arange(len(structs[2]["pos"])) // 3),
    )
    assert sc["rho_energy"][2] == pytest.approx(float(one["rho_energy"][0]))
    assert sc["rho_pressure_molecular"][2] == pytest.approx(
        float(one["rho_pressure_molecular"][0])
    )
    # a dataset is accepted as is, and molecular-only input has no pressure keys
    sc2 = com.score(AtomicDataset(_structs(3, periodic=False), com.cutoff))
    assert "rho_pressure" not in sc2 and sc2["rho_energy"].shape == (3,)


def test_committee_from_checkpoints_round_trip(tmp_path):
    members = [_member(s) for s in range(2)]
    paths = []
    for i, m in enumerate(members):
        p = tmp_path / f"m{i}.pt"
        torch.save({"model": m.state_dict(), "cfg": _cfg()}, p)
        paths.append(str(p))
    com = Committee.from_checkpoints(paths, dtype=torch.float64)
    assert com.size == 2 and com.cutoff == 4.0
    data = collate([structure_to_graph(s, 4.0) for s in _structs(2)])
    out = com(data)
    ref = Committee(members)(collate([structure_to_graph(s, 4.0) for s in _structs(2)]))
    assert torch.allclose(out["energy"], ref["energy"])
    assert torch.allclose(out["forces"], ref["forces"])


# the generalisations
def test_disagreement_per_atom_force_spread_and_its_maximum():
    f = torch.zeros(3, 5, 3)
    f[:, 0, 0] = torch.tensor([1.0, -1.0, 0.0])
    f[:, 3, 1] = torch.tensor([2.0, 0.0, -2.0])
    batch = torch.tensor([0, 0, 1, 1, 1])
    r = disagreement(torch.zeros(3, 2), f, torch.tensor([2, 3]), batch)
    sd = r["rho_forces_atom"]
    assert sd.shape == (5,)
    assert float(sd[0]) == pytest.approx(np.std([1, -1, 0]))
    assert float(sd[3]) == pytest.approx(np.std([2, 0, -2]))
    assert r["rho_forces_max_atom"].tolist() == pytest.approx(
        [float(sd[0]), float(sd[3])]
    )
    # the RMS of the per-atom spreads is rho_forces
    assert float(r["rho_forces"][1]) == pytest.approx(
        float(sd[2:].pow(2).mean().sqrt())
    )


def test_committee_rejects_different_subtracted_dispersions(tmp_path):
    with pytest.raises(ValueError):
        Committee([_member(0), _member(1)], subtracted_dispersion=[None, "d4"])
    com = Committee([_member(0), _member(1)], subtracted_dispersion=["d4", "d4"])
    assert com.subtracted_dispersion == "d4"
    paths = []
    for i, record in enumerate((None, {"name": "d4", "s6": 1.0})):
        cfg = _cfg()
        cfg.subtracted_dispersion = record
        p = tmp_path / f"m{i}.pt"
        torch.save({"model": _member(i).state_dict(), "cfg": cfg}, p)
        paths.append(str(p))
    with pytest.raises(ValueError, match="dispersion"):
        Committee.from_checkpoints(paths, dtype=torch.float64)


class _Spy(torch.nn.Module):
    """A member that records the total charge it is given."""

    cutoff = 4.0

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, data):
        self.seen.append(
            None if data.total_charge is None else data.total_charge.tolist()
        )
        n = data.num_graphs
        return {
            "energy": self.w * torch.zeros(n),
            "forces": torch.zeros_like(data.pos),
            "stress": None,
        }


def test_total_charge_reaches_the_members():
    a = Atoms("LiOHH", positions=[(0, 0, 0), (2, 0, 0), (2.6, 0.6, 0), (2.6, -0.6, 0)])
    a.info["total_charge"] = 1
    b = a.copy()
    b.info.pop("total_charge")
    s = structure_from_atoms(a)
    assert s["total_charge"] == 1.0 and s["cell"] is None
    assert "total_charge" not in structure_from_atoms(b)
    spies = [_Spy(), _Spy()]
    Committee(spies).score([s, structure_from_atoms(b)], batch_size=2)
    assert spies[0].seen == [[1.0, 0.0]]  # missing means neutral in a batch


def test_score_returns_per_atom_spreads_in_order():
    com = _committee(2)
    structs = _structs(5)
    keys = ("rho_forces", "rho_forces_max_atom", "n_atoms", "rho_forces_atom")
    sc = com.score(structs, batch_size=2, molecule=lambda s: molecules(s), keys=keys)
    n = sc["n_atoms"]
    assert sc["rho_forces_atom"].shape == (int(n.sum()),)
    offsets = np.concatenate([[0], np.cumsum(n)])
    for k in range(5):
        part = sc["rho_forces_atom"][offsets[k] : offsets[k + 1]]
        assert sc["rho_forces_max_atom"][k] == pytest.approx(part.max())
        assert sc["rho_forces"][k] == pytest.approx(np.sqrt(np.mean(part**2)))


def test_seamm_al_score_end_to_end(tmp_path):
    from active_learning_step.cli import main

    paths = []
    for i in range(3):
        p = tmp_path / f"m{i}.pt"
        torch.save({"model": _member(i).state_dict(), "cfg": _cfg()}, p)
        paths.append(str(p))
    frames = []
    for i, s in enumerate(_structs(5)):
        a = Atoms(numbers=s["atomic_numbers"], positions=s["pos"])
        if s.get("cell") is not None:
            a.set_cell(s["cell"])
            a.set_pbc(True)
        a.info["SEAMM/configuration_name"] = f"frame {i}"
        a.info["temperature"] = 300 + i
        frames.append(a)
    pool = tmp_path / "pool.extxyz"
    write(pool, frames, format="extxyz")
    out = tmp_path / "scores.csv"
    assert (
        main(
            [
                "score",
                str(pool),
                "-m",
                *paths,
                "-o",
                str(out),
                "--carry",
                "temperature",
                "--atoms",
                str(tmp_path / "atoms.extxyz"),
                "--dtype",
                "float64",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["configuration"] for r in rows] == [f"frame {i}" for i in range(5)]
    assert [r["kind"] for r in rows] == ["cell", "cluster", "cell", "cluster", "cell"]
    assert [r["temperature"] for r in rows] == ["300", "301", "302", "303", "304"]
    assert rows[1]["rho_pressure"] == "" and rows[0]["rho_pressure_molecular"] != ""
    # the same numbers as the library on the structures as written (8 decimals)
    from ase.io import read

    com = Committee.from_checkpoints(paths, dtype=torch.float64)
    back = [structure_from_atoms(a) for a in read(pool, index=":")]
    ref = com.score(back, batch_size=1, molecule=molecules, masses=masses)
    for k, r in enumerate(rows):
        assert float(r["rho_forces"]) == pytest.approx(ref["rho_forces"][k], rel=1e-10)
        assert float(r["rho_energy"]) == pytest.approx(ref["rho_energy"][k], rel=1e-10)
        if r["kind"] == "cell":
            assert float(r["rho_pressure_molecular"]) == pytest.approx(
                ref["rho_pressure_molecular"][k], rel=1e-10
            )
    frames = read(tmp_path / "atoms.extxyz", index=":")
    assert len(frames) == 5 and frames[0].arrays["rho_forces_atom"].shape == (
        len(frames[0]),
    )


def test_score_keeps_pressures_aligned_in_mixed_pools():
    """Cluster-only batches give nan pressures, not shorter arrays."""
    com = _committee(2)
    structs = _structs(5)  # cell, cluster, cell, cluster, cell
    sc = com.score(structs, batch_size=1, molecule=molecules, masses=masses)
    assert all(v.shape == (5,) for v in sc.values())
    assert np.isnan(sc["rho_pressure"][[1, 3]]).all()
    assert np.isfinite(sc["rho_pressure"][[0, 2, 4]]).all()
    whole = com.score(structs, batch_size=5, molecule=molecules, masses=masses)
    for k in ("rho_pressure", "rho_pressure_molecular", "rho_forces"):
        assert np.allclose(sc[k], whole[k], equal_nan=True, rtol=1e-10)
