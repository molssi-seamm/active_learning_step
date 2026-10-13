# -*- coding: utf-8 -*-

"""``seamm-al score``: the committee's spreads for every frame of a candidate pool.

Each frame of the pool files (extxyz, or SDF from SEAMM) becomes one row of
``scores.csv``:

* where it is: ``file``, ``index`` (0-based, in the file), ``configuration``;
* what it is: ``kind`` (``cell`` if periodic, else ``cluster``), ``n_atoms``,
  ``formula``, ``total_charge``, ``n_molecules`` (cells);
* the spreads: ``rho_energy`` (eV), ``rho_forces`` and ``rho_forces_max_atom``
  (eV/A), and for cells ``rho_pressure`` and ``rho_pressure_molecular`` (atm);
* the committee average: ``energy_mean`` (eV), and for cells ``pressure_mean`` and
  ``pressure_molecular_mean`` (atm);
* any ``info`` keys asked for with ``carry``.

Clusters are scored in batches; cells one at a time, with their molecules from the
bond graph (:func:`~active_learning_step.molecules.molecules`). Frames are read and
written in order, so memory stays small however large the pool.
"""

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

from .molecules import masses, molecules
from .pool import configuration_name, iter_frames, total_charge

logger = logging.getLogger(__name__)

__all__ = ["COLUMNS", "score_pool"]

#: The columns of scores.csv, before any carried keys.
COLUMNS = (
    "file",
    "index",
    "configuration",
    "kind",
    "n_atoms",
    "formula",
    "total_charge",
    "n_molecules",
    "rho_energy",
    "rho_forces",
    "rho_forces_max_atom",
    "rho_pressure",
    "rho_pressure_molecular",
    "energy_mean",
    "pressure_mean",
    "pressure_molecular_mean",
)

# committee.score keys -> scores.csv columns
_KEYS = {
    "rho_energy": "rho_energy",
    "rho_forces": "rho_forces",
    "rho_forces_max_atom": "rho_forces_max_atom",
    "rho_pressure": "rho_pressure",
    "rho_pressure_molecular": "rho_pressure_molecular",
    "mean_energy": "energy_mean",
    "mean_pressure": "pressure_mean",
    "mean_pressure_molecular": "pressure_molecular_mean",
}


# what Committee.score collects: the columns, n_atoms and the per-atom spread
_SCORE_KEYS = (*_KEYS, "n_atoms", "rho_forces_atom")


def _fmt(value):
    """A CSV cell: full precision for floats, empty for missing or nan."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else repr(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return str(value)


def _chunks(path, size):
    """Lists of up to ``size`` (index, atoms) frames of a pool file, in order."""
    chunk = []
    for frame in iter_frames(path):
        chunk.append(frame)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def score_pool(
    paths,
    models,
    out_csv="scores.csv",
    atoms_out=None,
    carry=(),
    dispersion=False,
    device="cpu",
    dtype=None,
    batch_size=16,
):
    """Score the frames of pool files with a committee and write ``scores.csv``.

    Parameters
    ----------
    paths : sequence of str
        The pool files (``.extxyz`` or ``.sdf``).
    models : sequence of str or None
        The members' checkpoints. ``None`` (``seamm-al score --no-models``)
        writes only the identity and carried columns, without torch or xnn: a
        pool listing for a random draw (``seamm-al select --random``) before any
        committee exists.
    out_csv : str, optional
        The scores file.
    atoms_out : str, optional
        If given, an extxyz file of every frame with its per-atom force spread
        (``rho_forces_atom``, eV/A) as a per-atom array, the molecule index of
        cells (``molecule``), and the frame's scores in its info.
    carry : sequence of str, optional
        ``info`` keys copied into columns of the same name.
    dispersion : bool, optional
        Add the dispersion the checkpoints record as subtracted, by default
        ``False``: it is shared by every member, so it adds nothing to the
        spreads (route B, as the water campaign scores).
    device : str, optional
        Torch device, by default ``"cpu"``.
    dtype : str, optional
        ``"float32"`` or ``"float64"``; by default the weights' own.
    batch_size : int, optional
        Clusters per batch, by default 16. Cells are scored one at a time.

    Returns
    -------
    int
        The number of frames scored.
    """
    committee = None
    if models:
        import torch

        from .committee import Committee

        if "SLURM_CPUS_PER_TASK" in os.environ:
            torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
        if dtype is not None and isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        committee = Committee.from_checkpoints(
            list(models),
            device=device,
            dtype=dtype,
            dispersion=None if dispersion else False,
        )
        logger.info(
            f"Committee of {committee.size} members, cutoff {committee.cutoff} A, "
            f"subtracted dispersion {committee.subtracted_dispersion}, "
            f"dispersion {'added' if dispersion else 'not added'}"
        )
    elif atoms_out is not None:
        raise ValueError("--atoms needs a committee")

    columns = list(COLUMNS) + [c for c in carry if c not in COLUMNS]
    atoms_fh = open(atoms_out, "w") if atoms_out is not None else None
    n_scored = 0
    try:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for path in paths:
                fname = str(Path(path).resolve())
                for chunk in _chunks(path, batch_size):
                    rows = {}
                    per_atom = {}
                    cluster_frames = []
                    for index, atoms in chunk:
                        charge = total_charge(atoms)
                        row = {
                            "file": fname,
                            "index": index,
                            "configuration": configuration_name(atoms),
                            "kind": "cell" if atoms.pbc.any() else "cluster",
                            "n_atoms": len(atoms),
                            "formula": atoms.get_chemical_formula(mode="hill"),
                            "total_charge": "" if charge is None else charge,
                        }
                        for key in carry:
                            row[key] = atoms.info.get(key, "")
                        rows[index] = row
                        if committee is None:
                            continue
                        from .committee import structure_from_atoms

                        s = structure_from_atoms(atoms)
                        if row["kind"] == "cell":
                            mol = molecules(atoms)
                            mas = masses(atoms)
                            row["n_molecules"] = int(mol.max()) + 1
                            res = committee.score(
                                [s],
                                batch_size=1,
                                molecule=lambda _, m=mol: m,
                                masses=lambda _, m=mas: m,
                                device=device,
                                keys=_SCORE_KEYS,
                            )
                            for key, column in _KEYS.items():
                                if key in res:
                                    row[column] = float(res[key][0])
                            per_atom[index] = (res["rho_forces_atom"], mol)
                        else:
                            cluster_frames.append((index, s))
                    if cluster_frames:
                        res = committee.score(
                            [s for _, s in cluster_frames],
                            batch_size=len(cluster_frames),
                            device=device,
                            keys=_SCORE_KEYS,
                        )
                        offsets = np.concatenate([[0], np.cumsum(res["n_atoms"])])
                        for k, (index, _) in enumerate(cluster_frames):
                            row = rows[index]
                            for key in (
                                "rho_energy",
                                "rho_forces",
                                "rho_forces_max_atom",
                            ):
                                row[_KEYS[key]] = float(res[key][k])
                            row["energy_mean"] = float(res["mean_energy"][k])
                            per_atom[index] = (
                                res["rho_forces_atom"][offsets[k] : offsets[k + 1]],
                                None,
                            )
                    for index, atoms in chunk:
                        row = rows[index]
                        writer.writerow({c: _fmt(row.get(c)) for c in columns})
                        if atoms_fh is not None:
                            _write_atoms(atoms_fh, atoms, row, *per_atom[index])
                    n_scored += len(chunk)
                    fh.flush()
                    logger.info(f"{fname}: {n_scored} frames scored")
    finally:
        if atoms_fh is not None:
            atoms_fh.close()
    return n_scored


def _write_atoms(fh, atoms, row, rho_atom, mol):
    """Append a frame with its per-atom spread to the extxyz of ``--atoms``."""
    from ase.io import write

    a = atoms  # keeps the frame's info, arrays and labels
    a.arrays["rho_forces_atom"] = np.asarray(rho_atom, dtype=float)
    if mol is not None:
        a.arrays["molecule"] = np.asarray(mol, dtype=int)
    for key in _KEYS.values():
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            a.info[f"AL/{key}"] = value
    write(fh, a, format="extxyz")
