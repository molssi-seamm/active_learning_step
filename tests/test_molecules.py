#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Molecules from the bond graph: carbonates, Li salts, NaCl in water, cells."""

import numpy as np
from ase import Atoms

from active_learning_step.molecules import masses, molecules

# MMFF geometries from RDKit (seed 7)
EC = (
    "OCOCCOHHHH",
    [
        (2.6904, -0.1264, -0.0178),
        (1.4731, -0.0692, -0.0098),
        (0.6530, -1.0957, 0.3667),
        (-0.6600, -0.7167, -0.0007),
        (-0.5898, 0.7754, 0.0090),
        (0.7479, 1.0299, -0.3760),
        (-1.3942, -1.1067, 0.7094),
        (-0.8744, -1.1196, -0.9963),
        (-1.2936, 1.2330, -0.6916),
        (-0.7524, 1.1960, 1.0071),
    ],
)
FEC = (
    "OCOCCFOHHH",
    [
        (2.6739, -0.2062, 0.3828),
        (1.4713, -0.1075, 0.2115),
        (0.8544, 1.0238, -0.2431),
        (-0.5370, 0.8014, -0.1291),
        (-0.6433, -0.6868, -0.1092),
        (-1.6350, -1.0820, 0.7226),
        (0.5800, -1.1106, 0.4435),
        (-0.9043, 1.2640, 0.7929),
        (-1.0696, 1.2419, -0.9766),
        (-0.7905, -1.1380, -1.0953),
    ],
)
DMC = (
    "COCOOCHHHHHH",
    [
        (2.3296, 0.0648, 0.1147),
        (1.1281, -0.5841, 0.5247),
        (0.0001, -0.0019, 0.0467),
        (-0.0475, 0.9857, -0.6710),
        (-1.0660, -0.7055, 0.5032),
        (-2.3234, -0.1926, 0.0689),
        (2.3550, 1.0954, 0.4825),
        (3.1738, -0.4775, 0.5496),
        (2.4278, 0.0395, -0.9753),
        (-3.1110, -0.8251, 0.4878),
        (-2.3968, -0.2273, -1.0227),
        (-2.4696, 0.8286, 0.4351),
    ],
)
WATER = (
    "OHH",
    [(0.0068, 0.3978, 0.0), (-0.7668, -0.1857, 0.0), (0.7599, -0.2120, 0.0)],
)


def _atoms(*parts):
    """Atoms from (symbols, positions, shift) parts."""
    symbols, positions = "", []
    for sym, pos, shift in parts:
        symbols += sym
        positions.extend(np.asarray(pos) + np.asarray(shift))
    return Atoms(symbols, positions=positions)


def _sizes(mol):
    return np.bincount(mol).tolist()


def test_single_carbonates_are_one_molecule():
    for sym, pos in (EC, FEC, DMC):
        mol = molecules(_atoms((sym, pos, (0, 0, 0))))
        assert mol.tolist() == [0] * len(sym)


def test_carbonate_mixture_cluster():
    a = _atoms(
        (EC[0], EC[1], (0, 0, 0)),
        (FEC[0], FEC[1], (0, 5, 0)),
        (DMC[0], DMC[1], (0, 0, 5)),
    )
    assert molecules(a).tolist() == [0] * 10 + [1] * 10 + [2] * 12


def test_molecule_crossing_the_periodic_boundary():
    a = _atoms((EC[0], EC[1], (0.5, 3, 3)), (FEC[0], FEC[1], (5, 7, 7)))
    a.set_cell([10.0, 10.0, 10.0])
    a.set_pbc(True)
    a.wrap()
    # the EC straddles x = 0 and the FEC straddles y = 10 after wrapping
    assert a.positions[:10, 0].min() < 2 and a.positions[:10, 0].max() > 8
    assert molecules(a).tolist() == [0] * 10 + [1] * 10
    # without periodicity the same coordinates break into pieces
    b = a.copy()
    b.set_pbc(False)
    assert molecules(b).max() > 1


def test_lipf6_li_alone_and_pf6_one_molecule():
    pf6 = [(0, 0, 0)] + [tuple(1.60 * s * e) for e in np.eye(3) for s in (1, -1)]
    li = [(0.0, 0.0, 3.6)]  # 2.0 A from an F: bonded but for the ion rule
    a = _atoms(("Li", li, (0, 0, 0)), ("PF6", pf6, (0, 0, 0)))
    mol = molecules(a)
    assert mol.tolist() == [0] + [1] * 7
    # and in a periodic cell with an EC solvent molecule
    c = _atoms(
        ("Li", li, (5, 5, 5)), ("PF6", pf6, (5, 5, 5)), (EC[0], EC[1], (5, 5, 10))
    )
    c.set_cell([12.0, 12.0, 12.0])
    c.set_pbc(True)
    c.wrap()
    assert _sizes(molecules(c)) == [1, 7, 10]


def test_libf4_bf4_one_molecule_of_five():
    t = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / np.sqrt(3)
    bf4 = [(0, 0, 0)] + [tuple(1.40 * v) for v in t]
    li = [tuple((1.40 + 1.9) * t[0])]
    a = _atoms(("BF4", bf4, (0, 0, 0)), ("Li", li, (0, 0, 0)))
    assert molecules(a).tolist() == [0] * 5 + [1]


def test_nacl_in_water():
    o, h1 = np.array(WATER[1][0]), np.array(WATER[1][1])
    parts = [(WATER[0], WATER[1], (6 * k, 0, 0)) for k in range(4)]
    na = o + np.array([0, 0, 2.4])  # 2.4 A over water 0's O
    oh = (h1 - o) / np.linalg.norm(h1 - o)
    cl = h1 + np.array([18.0, 0, 0]) + 2.2 * oh  # 2.2 A out along water 3's O-H
    a = _atoms(*parts, ("Na", [na], (0, 0, 0)), ("Cl", [cl], (0, 0, 0)))
    assert molecules(a).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
    # periodic, as a solution cell
    a.set_cell([24.0, 9.0, 9.0])
    a.set_pbc(True)
    a.wrap()
    assert molecules(a).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5]


def test_water_cell_is_index_div_3():
    rng = np.random.default_rng(3)
    parts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # randomly rotated waters on a 3.1 A grid shifted across the walls
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                pos = np.asarray(WATER[1]) @ q.T
                parts.append((WATER[0], pos, np.array((i, j, k)) * 3.1 - 1.0))
    a = _atoms(*parts)
    a.set_cell([9.3, 9.3, 9.3])
    a.set_pbc(True)
    a.wrap()
    assert np.array_equal(molecules(a), np.arange(len(a)) // 3)


def test_structure_dict_and_masses():
    s = {"pos": np.asarray(WATER[1]), "atomic_numbers": [8, 1, 1]}
    assert molecules(s).tolist() == [0, 0, 0]
    assert np.allclose(masses(s), [15.999, 1.008, 1.008])
