#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""molecules() gives what molsystem's perceive_bonds + find_molecules give.

Skipped where molsystem is not installed (the MLFF environments).
"""

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("molsystem")

from active_learning_step.molecules import (  # noqa: E402
    bonds,
    molecules,
    molecules_of_configuration,
)

from .test_molecules import DMC, EC, FEC, WATER  # noqa: E402


@pytest.fixture(scope="module")
def db():
    from molsystem import SystemDB

    db = SystemDB(filename="file:al_rule_db?mode=memory&cache=shared")
    yield db
    db.close()


_count = [0]


def _configuration(db, atoms):
    """A molsystem configuration with the atoms (and cell) of an ASE Atoms."""
    _count[0] += 1
    system = db.create_system(name=f"s{_count[0]}")
    c = system.create_configuration(name="c")
    if atoms.pbc.any():
        c.periodicity = 3
        c.coordinate_system = "Cartesian"
        c.cell.parameters = list(atoms.cell.cellpar())
    xyz = atoms.positions
    c.atoms.append(
        x=xyz[:, 0].tolist(),
        y=xyz[:, 1].tolist(),
        z=xyz[:, 2].tolist(),
        symbol=atoms.get_chemical_symbols(),
    )
    return c


def _molsystem_molecules(c):
    c.perceive_bonds(replace=True)
    label = np.empty(c.atoms.n_atoms, dtype=int)
    for k, members in enumerate(sorted(c.find_molecules(as_indices=True), key=min)):
        label[members] = k
    return label, c.bonds.n_bonds


def _atoms(*parts, cell=None):
    symbols, positions = "", []
    for sym, pos, shift in parts:
        symbols += sym
        positions.extend(np.asarray(pos) + np.asarray(shift))
    a = Atoms(symbols, positions=positions)
    if cell is not None:
        # an orthorhombic box: molsystem's cell parameters give the same vectors
        a.set_cell(cell)
        a.set_pbc(True)
        a.wrap()
    return a


def _jittered_box(seed, n=27, box=9.6, jitter=0.12, liquids=(WATER, EC)):
    """A dense box of molecules on a grid, randomly rotated, atoms jittered."""
    rng = np.random.default_rng(seed)
    parts = []
    k = int(round(n ** (1 / 3)))
    step = box / k
    for a in range(k):
        for b in range(k):
            for c in range(k):
                sym, pos = liquids[(a + b + c) % len(liquids)]
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                pos = (np.asarray(pos) - np.mean(pos, axis=0)) @ q.T
                pos = pos + rng.normal(0, jitter, pos.shape)
                parts.append((sym, pos, np.array((a, b, c)) * step + 0.3))
    return _atoms(*parts, cell=[box] * 3)


CASES = {
    "carbonate cluster": lambda: _atoms(
        (EC[0], EC[1], (0, 0, 0)),
        (FEC[0], FEC[1], (0, 5, 0)),
        (DMC[0], DMC[1], (0, 0, 5)),
    ),
    "cell across the walls": lambda: _atoms(
        (EC[0], EC[1], (0.5, 3, 3)), (FEC[0], FEC[1], (5, 7, 7)), cell=[10.0] * 3
    ),
    "LiPF6 + EC cell": lambda: _atoms(
        ("Li", [(0.0, 0.0, 3.6)], (5, 5, 5)),
        (
            "PF6",
            [(0, 0, 0)] + [tuple(1.60 * s * e) for e in np.eye(3) for s in (1, -1)],
            (5, 5, 5),
        ),
        (EC[0], EC[1], (5, 5, 10)),
        cell=[12.0] * 3,
    ),
    "NaCl in water": lambda: _atoms(
        *[(WATER[0], WATER[1], (6 * k, 0, 0)) for k in range(4)],
        ("Na", [(0.0068, 0.3978, 2.4)], (0, 0, 0)),
        ("Cl", [(15.5, -1.5, 0.0)], (0, 0, 0)),
        cell=[24.0, 9.0, 9.0],
    ),
    "stretched C-H (Wigner sampling)": lambda: _atoms(
        (
            "CHHHH",
            [
                (0, 0, 0),
                (1.33, 0, 0),
                (-0.36, 1.03, 0),
                (-0.36, -0.51, 0.89),
                (-0.36, -0.51, -0.89),
            ],
            (0, 0, 0),
        )
    ),
}
CASES.update({f"jittered box {s}": (lambda s=s: _jittered_box(s)) for s in range(3)})
CASES["jittered water box"] = lambda: _jittered_box(
    7, n=64, box=12.4, jitter=0.15, liquids=(WATER,)
)


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_molecules_and_bonds_as_molsystem(db, name):
    atoms = CASES[name]()
    label, n_bonds = _molsystem_molecules(_configuration(db, atoms))
    assert molecules(atoms).tolist() == label.tolist()
    assert len(bonds(atoms)[0]) == n_bonds


def test_stretched_ch_is_not_a_bond_in_either():
    atoms = CASES["stretched C-H (Wigner sampling)"]()
    assert molecules(atoms).tolist() == [0, 1, 0, 0, 0]
    # mbe's observation: a looser tolerance bonds it, in both
    assert molecules(atoms, tolerance=1.35).tolist() == [0] * 5


def test_configuration_uses_its_own_bonds(db):
    atoms = CASES["carbonate cluster"]()
    c = _configuration(db, atoms)
    # no bonds: perceived with molsystem's rule, and the configuration is untouched
    assert c.bonds.n_bonds == 0
    assert molecules_of_configuration(c).tolist() == molecules(atoms).tolist()
    assert c.bonds.n_bonds == 0
    # with bonds of its own, those are what count: delete the carbonyl C=O of the
    # EC, and its O becomes a molecule of its own, as molsystem says
    c.perceive_bonds()
    ids = c.atoms.ids
    n_bonds = c.bonds.n_bonds
    c.bonds.delete_bond(ids[0], ids[1])
    assert c.bonds.n_bonds == n_bonds - 1
    label = molecules_of_configuration(c)
    assert label[0] != label[1] and len(set(label[:10].tolist())) == 2
    assert label.tolist() == _molsystem_from_existing(c).tolist()
    assert label.tolist() != molecules(atoms).tolist()


def _molsystem_from_existing(c):
    label = np.empty(c.atoms.n_atoms, dtype=int)
    for k, members in enumerate(sorted(c.find_molecules(as_indices=True), key=min)):
        label[members] = k
    return label
