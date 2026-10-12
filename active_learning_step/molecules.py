# -*- coding: utf-8 -*-

"""Molecules of a structure from its covalent bond graph.

The committee's molecular pressure removes the intramolecular virial, so it needs to
know which atoms form each molecule. For water that was ``atom index // 3``; for
electrolytes it is found here from the geometry:

* two atoms are bonded when their distance is at most ``scale`` times the sum of their
  covalent radii (ASE's table), using the minimum image in periodic cells (ASE's
  neighbour list);
* alkali and alkaline-earth atoms (Li, Na, K, Mg, Ca, ...) are never bonded, so each
  cation is a molecule of its own; halide anions come out as single atoms naturally;
* the molecules are the connected components of the bond graph, numbered in the order
  of their first atom, so ``O H H O H H ...`` water gives ``index // 3``.

Only ASE and numpy are needed, so this works in a SEAMM environment as well as in an
MLFF one.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_masses, covalent_radii
from ase.neighborlist import neighbor_list

__all__ = ["IONS", "BOND_SCALE", "molecules", "masses", "as_atoms"]

#: Atomic numbers never bonded: the alkali (Li to Fr) and alkaline-earth (Be to Ra)
#: metals, which are present as cations in electrolytes.
IONS = frozenset((3, 11, 19, 37, 55, 87, 4, 12, 20, 38, 56, 88))

#: Bonds are distances up to this multiple of the sum of the covalent radii.
BOND_SCALE = 1.2


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


def molecules(structure, scale=BOND_SCALE, ions=IONS):
    """The molecule index of every atom: connected components of the bond graph.

    Parameters
    ----------
    structure : ase.Atoms or dict
        The structure (an xnn structure dict is accepted too). Periodic directions
        use the minimum image.
    scale : float, optional
        Two atoms are bonded when closer than ``scale * (r_i + r_j)``, the covalent
        radii from :data:`ase.data.covalent_radii`; by default 1.2.
    ions : collection of int, optional
        Atomic numbers that are never bonded, by default :data:`IONS`.

    Returns
    -------
    numpy.ndarray of int
        ``(N,)`` molecule index per atom, numbered 0, 1, ... in the order of each
        molecule's first atom.
    """
    atoms = as_atoms(structure)
    n = len(atoms)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    z = atoms.numbers
    radii = scale * covalent_radii[z]
    # an ion gets a zero radius; it still bonds to nothing because of the mask below
    is_ion = np.isin(z, list(ions))
    radii = np.where(is_ion, 0.0, radii)
    if not atoms.pbc.any():
        # neighbor_list needs a cell large enough to hold a non-periodic structure
        atoms = atoms.copy()
        atoms.set_cell(np.eye(3))
    i, j = neighbor_list("ij", atoms, radii, self_interaction=False)
    keep = ~(is_ion[i] | is_ion[j])
    i, j = i[keep], j[keep]

    # union-find over the bonds
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i.tolist(), j.tolist()):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(a) for a in range(n)])
    # renumber in the order of each molecule's first atom
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse].astype(np.int64)


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
