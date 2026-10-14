# -*- coding: utf-8 -*-

"""Molecules of a structure from its covalent bonds, with molsystem's rule.

The committee's molecular pressure removes the intramolecular virial, so it needs to
know which atoms form each molecule. SEAMM defines molecules through bonds:
``molsystem``'s ``Configuration.perceive_bonds()`` and ``find_molecules()``. This
module applies **the same rule** to coordinates alone, so that the command line runs in
an MLFF environment that has neither SEAMM nor molsystem, and gives the same molecules:

* two atoms are bonded when ``min_distance < d < tolerance * (r_i + r_j)``, with the
  Pyykkö covalent radii molsystem uses (:mod:`~active_learning_step.covalent_radii`)
  and ``tolerance`` 1.2, using every periodic image (ASE's neighbour list);
* the alkali and alkaline-earth metals (:data:`IONIC_ELEMENTS`) are never bonded, so
  each cation is a molecule of its own; halide anions come out as single atoms;
* an atom has at most ``max_bonds`` bonds by element (H: 1), the shortest kept;
* the molecules are the connected components, numbered in the order of their first
  atom, so ``O H H O H H ...`` water gives ``index // 3``.

Inside SEAMM, :func:`molecules_of_configuration` uses a configuration's own bonds when
it has them. The test suite checks this module against molsystem wherever molsystem is
installed, so the two cannot drift apart.

Only ASE and numpy are needed.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.neighborlist import neighbor_list

from .covalent_radii import COVALENT_RADII

__all__ = [
    "IONIC_ELEMENTS",
    "MAX_BONDS",
    "BOND_TOLERANCE",
    "MIN_DISTANCE",
    "molecules",
    "molecules_of_configuration",
    "masses",
    "as_atoms",
]

#: Elements never bonded: the alkali and alkaline-earth metals, ions in molecular
#: systems (as ``molsystem.topology.IONIC_ELEMENTS``).
IONIC_ELEMENTS = ("Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba", "Ra")

#: The most bonds an atom of an element may have (as ``molsystem.topology.MAX_BONDS``).
MAX_BONDS = {"H": 1}

#: Bonds are distances below this multiple of the sum of the covalent radii.
BOND_TOLERANCE = 1.2

#: Pairs closer than this (Å) are ignored rather than bonded.
MIN_DISTANCE = 0.4


def as_atoms(structure):
    """An ASE Atoms from Atoms or an xnn structure dict.

    Parameters
    ----------
    structure : ase.Atoms or dict
        The structure; a dict needs ``pos`` and ``atomic_numbers`` and may have
        ``cell`` and ``pbc``.

    Returns
    -------
    ase.Atoms
    """
    if isinstance(structure, Atoms):
        return structure
    cell = structure.get("cell")
    pbc = structure.get("pbc")
    if pbc is None:
        pbc = cell is not None
    return Atoms(
        numbers=np.asarray(structure["atomic_numbers"]),
        positions=np.asarray(structure["pos"], dtype=float),
        cell=None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3),
        pbc=pbc,
    )


def _components(n, i, j):
    """Molecule index per atom from bonds (i, j): first-atom order."""
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i, j):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(a) for a in range(n)])
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse].astype(np.int64)


def bonds(
    structure,
    tolerance=BOND_TOLERANCE,
    radii=None,
    max_bonds=None,
    exclude=IONIC_ELEMENTS,
    min_distance=MIN_DISTANCE,
):
    """The bonded pairs of a structure, by molsystem's ``perceive_bonds`` rule.

    Parameters are those of ``molsystem``'s ``Configuration.perceive_bonds``:
    ``tolerance`` times the sum of the covalent radii, ``radii`` overriding them by
    element symbol, ``max_bonds`` by element symbol (over the default H: 1),
    ``exclude`` the elements never bonded, ``min_distance`` the closest pair bonded.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Atom indices ``i`` and ``j`` of every bond (``i <= j``; one entry per
        bond, through whichever periodic image) and their lengths.
    """
    atoms = as_atoms(structure)
    n = len(atoms)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    symbols = atoms.get_chemical_symbols()
    radius = {s: COVALENT_RADII[atoms.numbers[k]] for k, s in enumerate(symbols)}
    if radii is not None:
        radius.update({k: float(v) for k, v in radii.items()})
    for s, r in radius.items():
        if r is None:
            raise ValueError(f"No covalent radius available for {s}")
    limits = dict(MAX_BONDS)
    if max_bonds is not None:
        limits.update(max_bonds)
    excluded = set(exclude) if exclude is not None else set()

    r_atom = np.array([radius[s] for s in symbols])
    active = np.array([s not in excluded for s in symbols])
    if not active.any():
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    if not atoms.pbc.any():
        # neighbor_list needs a cell to hold a non-periodic structure
        atoms = atoms.copy()
        atoms.set_cell(np.eye(3))
    # every pair within the largest possible bond, then molsystem's criterion
    cutoff = tolerance * 2.0 * r_atom[active].max()
    i, j, d, shift = neighbor_list("ijdS", atoms, cutoff, self_interaction=False)
    # each pair appears twice (i, j) and (j, i); keep one, as molsystem's pair list
    one = (i < j) | ((i == j) & _positive(shift))
    i, j, d = i[one], j[one], d[one]
    keep = (
        active[i]
        & active[j]
        & (d > min_distance)
        & (d < tolerance * (r_atom[i] + r_atom[j]))
    )
    i, j, d = i[keep], j[keep], d[keep]

    # the per-element limits, keeping the shortest bonds
    order = np.argsort(d, kind="stable")
    count = np.zeros(n, dtype=int)
    limit = np.array([limits.get(s, 10**9) for s in symbols])
    out = []
    for k in order:
        a, b = int(i[k]), int(j[k])
        if a == b:
            if count[a] + 2 > limit[a]:
                continue
            count[a] += 2
        else:
            if count[a] >= limit[a] or count[b] >= limit[b]:
                continue
            count[a] += 1
            count[b] += 1
        out.append(k)
    out = np.array(out, dtype=int)
    return i[out], j[out], d[out]


def _positive(shift):
    """Whether each cell shift is the 'positive' one of its pair (first nonzero > 0)."""
    s = np.asarray(shift)
    first = np.argmax(s != 0, axis=1)
    return s[np.arange(len(s)), first] > 0


def molecules(structure, **rule):
    """The molecule index of every atom: connected components of the bonds.

    Parameters
    ----------
    structure : ase.Atoms or dict
        The structure (an xnn structure dict is accepted too). Periodic directions
        use every image.
    **rule
        ``tolerance``, ``radii``, ``max_bonds``, ``exclude``, ``min_distance``, as
        in :func:`bonds` (molsystem's ``perceive_bonds``).

    Returns
    -------
    numpy.ndarray of int
        ``(N,)`` molecule index per atom, numbered 0, 1, ... in the order of each
        molecule's first atom.
    """
    n = len(as_atoms(structure))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    i, j, _ = bonds(structure, **rule)
    return _components(n, i.tolist(), j.tolist())


def molecules_of_configuration(configuration, **rule):
    """The molecule index of every atom of a SEAMM (molsystem) configuration.

    Uses the configuration's own bonds when it has any (``find_molecules``), so a
    structure keeps the molecules SEAMM gave it. Otherwise the bonds are perceived
    with molsystem's rule (:func:`bonds`), without adding them to the configuration.

    Parameters
    ----------
    configuration : molsystem configuration
        The structure.
    **rule
        Options of :func:`bonds`, used only when the configuration has no bonds.

    Returns
    -------
    numpy.ndarray of int
        ``(N,)`` molecule index per atom, in the order of each molecule's first atom.
    """
    n = configuration.atoms.n_atoms
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if configuration.bonds.n_bonds > 0:
        label = np.empty(n, dtype=np.int64)
        groups = sorted(configuration.find_molecules(as_indices=True), key=min)
        for k, members in enumerate(groups):
            label[list(members)] = k
        return label
    xyz = np.asarray(
        configuration.atoms.get_coordinates(fractionals=False), dtype=float
    )
    numbers = np.array([chemical_symbols.index(s) for s in configuration.atoms.symbols])
    structure = {"pos": xyz, "atomic_numbers": numbers}
    if configuration.periodicity == 3:
        structure["cell"] = np.asarray(configuration.cell.vectors(), dtype=float)
        structure["pbc"] = [True, True, True]
    elif configuration.periodicity != 0:
        raise ValueError("only molecular and 3-D periodic configurations are handled")
    return molecules(structure, **rule)


def masses(structure):
    """The atomic masses (amu) of a structure, from ASE's table.

    Parameters
    ----------
    structure : ase.Atoms or dict
        The structure.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` masses.
    """
    if isinstance(structure, Atoms):
        z = structure.numbers
    else:
        z = np.asarray(structure["atomic_numbers"])
    return atomic_masses[z]
