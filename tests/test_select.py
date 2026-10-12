#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pool records and ``seamm-al select``: ranking, budget, caps and verbatim output."""

import csv
from pathlib import Path

import numpy as np
import pytest

from active_learning_step.cli import main
from active_learning_step.pool import iter_frames, iter_records
from active_learning_step.selection import choose, parse_costs, parse_thresholds

DATA = Path(__file__).parent / "data"

# Frames as SEAMM writes them: DDEC6 charges, header keys ASE does not know.
_HEADER = (
    "Properties=species:S:1:pos:R:3:REF_forces:R:3:charge:R:1 REF_energy={e} "
    'model=ORCA#DFT@REVDSD-PBEP86-D4_2021/def2-TZVPPD SEAMM/system_name="EC liquid" '
    'SEAMM/configuration_name="{name}" d4_params="PBE0-D4 bj-eeq-atm s6=1.0" '
    "total_charge={q} {cell}"
)


def _frame(name, n, e, q=0.0, cell=True, seed=0):
    rng = np.random.default_rng(seed)
    lattice = (
        'Lattice="9.0 0.0 0.0 0.0 9.0 0.0 0.0 0.0 9.0" pbc="T T T"'
        if cell
        else 'pbc="F F F"'
    )
    lines = [f"{n}", _HEADER.format(e=e, name=name, q=q, cell=lattice)]
    for k in range(n):
        x = rng.uniform(0, 9, 3)
        f = rng.normal(0, 0.1, 3)
        sym = "O" if k % 3 == 0 else "H"
        lines.append(
            f"{sym:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}   "
            f"{f[0]: .6e} {f[1]: .6e} {f[2]: .6e}  "
            f"{rng.normal(0, 0.4): .5f}"
        )
    return "\n".join(lines) + "\n"


@pytest.fixture
def xyz_pool(tmp_path):
    path = tmp_path / "pool.extxyz"
    frames = [
        _frame("cell_EC_300", 6, -12.5, seed=1),
        _frame("cell_FEC_300", 6, -13.5, seed=2),
        _frame("cluster_1", 3, -1.25, cell=False, seed=3),
        _frame("cell_EC_350", 6, -12.25, seed=4),
        _frame("cell_FEC_350", 6, -13.25, q=-1.0, seed=5),
        _frame("cluster_2", 3, -1.5, cell=False, seed=6),
    ]
    path.write_text("".join(frames))
    return path


def _rows(path, values):
    """Score rows for the frames of a pool, with the spreads given."""
    rows = []
    for (index, atoms), (rho_p, rho_f) in zip(iter_frames(path), values):
        rows.append(
            {
                "file": str(path),
                "index": str(index),
                "configuration": atoms.info["SEAMM/configuration_name"],
                "kind": "cell" if atoms.pbc.any() else "cluster",
                "n_atoms": str(len(atoms)),
                "formula": atoms.get_chemical_formula(mode="hill"),
                "liquid": atoms.info["SEAMM/configuration_name"].split("_")[1],
                "temperature": atoms.info["SEAMM/configuration_name"].split("_")[-1],
                "rho_pressure_molecular": "" if rho_p is None else str(rho_p),
                "rho_forces": str(rho_f),
            }
        )
    return rows


def _write_scores(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


VALUES = [
    (300.0, 0.020),
    (500.0, 0.010),
    (None, 0.030),
    (100.0, 0.040),
    (400.0, 0.030),
    (None, 0.005),
]


def test_records_and_frames_of_extxyz(xyz_pool):
    records = list(iter_records(xyz_pool))
    assert len(records) == 6
    assert b"".join(records) == xyz_pool.read_bytes()
    frames = list(iter_frames(xyz_pool))
    assert [i for i, _ in frames] == list(range(6))
    assert frames[4][1].info["total_charge"] == -1.0
    assert frames[2][1].pbc.any() == np.False_


def test_seamm_sdf_reader():
    path = DATA / "seamm_pool.sdf"
    records = list(iter_records(path))
    assert len(records) == 2
    assert b"".join(records) == path.read_bytes()
    (_, cell), (_, cluster) = list(iter_frames(path))
    assert cell.info["SEAMM/configuration_name"] == "EC_cell_300K"
    assert cell.pbc.all() and np.allclose(cell.cell.lengths(), [10.0, 10.5, 11.0])
    assert cell.positions[0, 0] == pytest.approx(
        2.76360062189
    )  # full precision from the JSON
    assert cell.info["total_charge"] == 0.0
    assert cluster.info["SEAMM/configuration_name"] == "LiEC_1"
    assert cluster.info["total_charge"] == 1.0
    assert not cluster.pbc.any()
    assert cluster.get_chemical_symbols()[0] == "Li"


def test_rank_sum_order_and_budget(xyz_pool):
    rows = _rows(xyz_pool, VALUES)
    costs = parse_costs(["cell=2800", "cluster=5"])
    sel, rej = choose([r for r in rows if r["kind"] == "cell"], costs, budget=3 * 2800)
    # cells: P ranks 500>400>300>100 -> FEC_300 1, FEC_350 2, EC_300 3, EC_350 4;
    #        F ranks 0.04>0.03>0.02>0.01 -> EC_350 1, FEC_350 2, EC_300 3, FEC_300 4
    # sums: FEC_350 4, FEC_300 5, EC_350 5, EC_300 6; ties broken by row order
    assert [r["configuration"] for r in sel] == [
        "cell_FEC_350",
        "cell_FEC_300",
        "cell_EC_350",
    ]
    assert [r["running_cost"] for r in sel] == [2800, 5600, 8400]
    assert [r["configuration"] for r in rej] == ["cell_EC_300"]
    assert rej[0]["reason"].startswith("over budget")
    # where each frame stands in its kind, so the budget cut can be read off
    assert [r["rank_in_kind"] for r in sel] == [1, 2, 3]
    assert [r["rank_sum"] for r in sel] == [4, 5, 5]
    assert rej[0]["rank_in_kind"] == 4 and rej[0]["n_in_kind"] == 4
    assert rej[0]["cost"] == 2800


def test_group_costs_caps_exclude_and_fill(xyz_pool):
    rows = [r for r in _rows(xyz_pool, VALUES) if r["kind"] == "cell"]
    costs = parse_costs(["cell:FEC=3900", "cell:EC=2800"])
    sel, _ = choose(rows, costs, budget=7000, group_column="liquid")
    assert [r["configuration"] for r in sel] == [
        "cell_FEC_350"
    ]  # the next FEC does not fit
    sel, _ = choose(rows, costs, budget=7000, group_column="liquid", fill=True)
    assert [r["configuration"] for r in sel] == ["cell_FEC_350", "cell_EC_350"]
    assert sel[-1]["running_cost"] == 6700
    sel, rej = choose(rows, costs, group_column="liquid", caps={"liquid": 1})
    assert [r["configuration"] for r in sel] == ["cell_FEC_350", "cell_EC_350"]
    assert all(r["reason"].startswith("cap") for r in rej)
    sel, rej = choose(rows, costs, group_column="liquid", exclude={"cell_FEC_350"})
    # the three left tie on rank-sum (4 each), so they keep their order in the file
    assert [r["configuration"] for r in sel] == [
        "cell_EC_300",
        "cell_FEC_300",
        "cell_EC_350",
    ]
    assert any(r["reason"].startswith("excluded") for r in rej)
    with pytest.raises(ValueError):
        choose(rows, parse_costs(["cluster=1"]))


def test_thresholds_any_and_all(xyz_pool):
    rows = _rows(xyz_pool, VALUES)
    costs = parse_costs(["cell=2800", "cluster=5"])
    th = parse_thresholds(["rho_forces=0.025", "cell:rho_pressure_molecular=350"])
    sel, _ = choose(rows, costs, thresholds=th)
    assert sorted(r["configuration"] for r in sel) == [
        "cell_EC_350",
        "cell_FEC_300",
        "cell_FEC_350",
        "cluster_1",
    ]
    sel, _ = choose(rows, costs, thresholds=th, combine="all")
    assert sorted(r["configuration"] for r in sel) == ["cell_FEC_350", "cluster_1"]
    assert (
        "rho_forces 0.03 >= 0.025" in sel[0]["reason"]
        or "rho_forces 0.03 >= 0.025" in sel[1]["reason"]
    )


def _round_trip(tmp_path, pool, rows, n_expected, suffix):
    scores = tmp_path / "scores.csv"
    _write_scores(scores, rows)
    out = tmp_path / "selection.csv"
    prefix = tmp_path / "chosen"
    assert (
        main(
            [
                "select",
                str(scores),
                "--cost",
                "cell=1",
                "--cost",
                "cluster=1",
                "-o",
                str(out),
                "--output",
                str(prefix),
            ]
        )
        == 0
    )
    with open(out) as fh:
        selected = list(csv.DictReader(fh))
    assert len(selected) == n_expected
    written = list(iter_records(f"{prefix}.{suffix}"))
    original = list(iter_records(pool))
    assert written == [original[int(r["index"])] for r in selected]
    # nothing but the records: the file is their concatenation
    assert Path(f"{prefix}.{suffix}").read_bytes() == b"".join(written)


def test_select_round_trip_extxyz_is_byte_identical(tmp_path, xyz_pool):
    _round_trip(tmp_path, xyz_pool, _rows(xyz_pool, VALUES), 6, "extxyz")


def test_select_round_trip_sdf_is_byte_identical(tmp_path):
    pool = DATA / "seamm_pool.sdf"
    rows = []
    for (index, atoms), rho in zip(iter_frames(pool), (0.01, 0.02)):
        rows.append(
            {
                "file": str(pool),
                "index": str(index),
                "configuration": atoms.info["SEAMM/configuration_name"],
                "kind": "cell" if atoms.pbc.any() else "cluster",
                "rho_pressure_molecular": "100.0",
                "rho_forces": str(rho),
            }
        )
    _round_trip(tmp_path, pool, rows, 2, "sdf")
    text = (tmp_path / "chosen.sdf").read_text()
    # every property keeps its terminating blank line, including the last before $$$$
    assert text.count("\n\n$$$$\n") == 2


def test_select_refuses_a_changed_pool(tmp_path, xyz_pool):
    rows = _rows(xyz_pool, VALUES)
    rows[0]["configuration"] = "something else"
    scores = tmp_path / "scores.csv"
    _write_scores(scores, rows)
    assert (
        main(
            [
                "select",
                str(scores),
                "--cost",
                "cell=1",
                "--cost",
                "cluster=1",
                "-o",
                str(tmp_path / "s.csv"),
                "--output",
                str(tmp_path / "x"),
            ]
        )
        == 2
    )
