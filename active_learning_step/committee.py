"""Query-by-committee tools: ensembles of potentials, disagreement, thresholds.

The active-learning recipe of Smith *et al.* (J. Chem. Phys. 148, 241733,
2018): a committee of ``L`` potentials trained on different folds of the same
data with different initialisations; a structure ``i`` is uncertain when the
spread of the committee's energies

.. math:: \\rho_i = \\sigma_i / \\sqrt{N_i}

(their eq 1; :math:`\\sigma_i` the standard deviation of the ``L`` energies,
:math:`N_i` the atom count) exceeds a threshold :math:`\\hat\\rho`. The
threshold is calibrated on a labelled benchmark (their fig 1): it is the value
that captures a chosen fraction (98 %) of the structures whose largest
per-member error :math:`\\epsilon_i = \\max_l |E_l - E_{ref}| / \\sqrt{N_i}`
exceeds a failure level. The mean of the ``L`` predictions is the committee's
prediction (their sec II E).

Condensed phases need more than the energy. In liquid water the energy error
of clusters never predicted the pressure of the bulk, so force and pressure
spreads are provided alongside, calibrated the same way:

* :math:`\\rho_F`: the root mean square over the atoms of the committee
  standard deviation of the force (eV/A);
* :math:`\\rho_P`: the committee standard deviation of the pressure (atm),
  atomic (from the stress) or molecular (the intramolecular virial removed,
  the pressure a rigid-molecule barostat feels).

Nothing here trains. The members are trained by the
:class:`~xnn.common.train.Trainer`, one run per member, on the folds of
:func:`kfold_indices`; :class:`Committee` loads the resulting checkpoints and
scores structures.

Origin and changes
------------------
Ported from the water campaign's ``campaign/committee.py`` (Sina Mostafanejad,
2026-10, written against xnn 0.8.0), which he agreed should live in this
plug-in. The API and the formulas are his; the changes generalise it beyond
water:

* the molecules come from a covalent bond graph (:func:`molecules`, in
  :mod:`active_learning_step.molecules`) rather than ``atom index // 3``; ions
  such as Li+ are molecules of their own, and the masses come from the
  elements (:func:`masses`);
* a structure's ``total_charge`` reaches the models (:func:`structure_from_atoms`
  puts it in the structure dict that xnn's graph builder reads);
* :meth:`Committee.from_checkpoints` refuses members that record different
  subtracted dispersions, as well as different cutoffs;
* the numpy-only functions (:func:`kfold_indices`, :func:`calibrate_threshold`,
  :func:`split_statistics`, :func:`select`) live in
  :mod:`active_learning_step.thresholds`, so that selecting needs no torch; they
  are re-exported here unchanged;
* besides ``rho_forces``, :func:`disagreement` returns the per-atom spread
  ``rho_forces_atom`` and its maximum ``rho_forces_max_atom``, because one bad
  ion environment is diluted in the RMS over a large cell.

xnn interface
-------------
Written for xnn 0.11.x. The xnn names used, all public:

* ``xnn.common.data``: ``AtomicDataset``, ``AtomicGraph``, ``collate``;
* ``xnn.common.models.hub``: ``load_checkpoint``, ``build_potential``
  (in :meth:`Committee.from_checkpoints`);
* the checkpoint's ``Config.subtracted_dispersion`` attribute, read with a
  default as ``xnn.common.models.registry.recorded_dispersion`` does;
* members are ``xnn.common.models.ForceStressOutput`` objects, as
  ``build_potential`` returns them.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from xnn.common.data import AtomicDataset, AtomicGraph, collate

from .molecules import masses, molecules  # noqa: F401
from .thresholds import (
    calibrate_threshold,
    kfold_indices,
    select,
    split_statistics,
)  # noqa: F401

#: atm per eV/A^3 (CODATA 2018 electronvolt, 101 325 Pa per atm).
ATM_PER_EV_A3 = 1.602176634e6 / 1.01325

__all__ = [
    "ATM_PER_EV_A3",
    "Committee",
    "kfold_indices",
    "pressure",
    "molecular_virial",
    "molecular_pressure",
    "disagreement",
    "worst_error",
    "calibrate_threshold",
    "split_statistics",
    "select",
    "molecules",
    "masses",
    "structure_from_atoms",
]


def structure_from_atoms(atoms, total_charge=None) -> dict:
    """The xnn structure dict of an ASE Atoms, for scoring.

    Only the geometry and the total charge are taken: no labels, so nothing in
    the frame's ``info`` or ``arrays`` can change what the models see.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure. It is periodic when any direction is.
    total_charge : float, optional
        The net charge; by default ``atoms.info["total_charge"]`` (or
        ``"charge"``), and none (neutral to the models) when absent.

    Returns
    -------
    dict
        ``pos``, ``atomic_numbers``, ``cell`` and ``pbc`` (both ``None`` for a
        non-periodic structure) and ``total_charge`` when known.
    """
    periodic = bool(atoms.pbc.any())
    s = {
        "pos": np.asarray(atoms.get_positions(), dtype=float),
        "atomic_numbers": np.asarray(atoms.get_atomic_numbers()),
        "cell": np.asarray(atoms.get_cell()[:], dtype=float) if periodic else None,
        "pbc": np.asarray(atoms.pbc, dtype=bool) if periodic else None,
    }
    if total_charge is None:
        for key in ("total_charge", "charge"):
            value = atoms.info.get(key)
            if value is not None and np.ndim(value) == 0:
                total_charge = float(value)
                break
    if total_charge is not None:
        s["total_charge"] = float(total_charge)
    return s


def pressure(stress: Tensor) -> Tensor:
    """The isotropic pressure of a stress tensor, ``-tr(sigma) / 3``.

    Parameters
    ----------
    stress : Tensor
        Stress of shape ``(..., 3, 3)`` in eV/A^3 with the ASE / xnn sign
        (``sigma = -P`` under hydrostatic pressure).

    Returns
    -------
    Tensor
        Pressure of shape ``(...)`` in eV/A^3 (multiply by
        :data:`ATM_PER_EV_A3` for atm).
    """
    return -stress.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0


def _first_atom(molecule: Tensor) -> Tensor:
    """Index of the first atom of each molecule, indexed by molecule id."""
    n_mol = int(molecule.max()) + 1
    first = torch.full(
        (n_mol,), molecule.numel(), dtype=torch.long, device=molecule.device
    )
    idx = torch.arange(molecule.numel(), device=molecule.device)
    first.scatter_reduce_(0, molecule, idx, reduce="amin")
    return first


def molecular_virial(
    pos: Tensor,
    forces: Tensor,
    batch: Tensor,
    molecule: Tensor,
    cell: Optional[Tensor] = None,
    pbc: Optional[Tensor] = None,
    masses: Optional[Tensor] = None,
    n_structures: Optional[int] = None,
) -> Tensor:
    """The intramolecular virial ``sum_i (x_i - R_m) . F_i`` per structure.

    Each molecule is made whole by the minimum image of every atom with respect
    to its first atom (periodic structures only), ``R_m`` is its centre of mass
    (its geometric centre without ``masses``), and the trace of the virial is
    summed over the molecules of each structure.

    Parameters
    ----------
    pos : Tensor
        Positions ``(N, 3)`` in A.
    forces : Tensor
        Forces ``(N, 3)`` in eV/A.
    batch : Tensor
        Structure index per atom ``(N,)``.
    molecule : Tensor
        Molecule index per atom ``(N,)``, unique across the batch.
    cell : Tensor, optional
        Lattice vectors as rows ``(B, 3, 3)``; needed for the minimum image.
    pbc : Tensor, optional
        Periodic flags ``(B, 3)``; a structure is wrapped only when all three
        are set. ``None`` with a cell means every structure is periodic.
    masses : Tensor, optional
        Atomic masses ``(N,)`` for the centre of mass. ``None`` uses the
        geometric centre.
    n_structures : int, optional
        Number of structures ``B``; by default ``batch.max() + 1``.

    Returns
    -------
    Tensor
        The virial trace ``(B,)`` in eV.
    """
    n_b = int(batch.max()) + 1 if n_structures is None else n_structures
    first = _first_atom(molecule)
    d = pos - pos[first[molecule]]
    if cell is not None:
        periodic = (
            torch.ones(n_b, dtype=torch.bool, device=pos.device)
            if pbc is None
            else pbc.all(-1)
        )
        if bool(periodic.any()):
            inv = torch.zeros_like(cell)
            inv[periodic] = torch.linalg.inv(cell[periodic])
            frac = torch.einsum("ni,nij->nj", d, inv[batch])
            d = d - torch.einsum("ni,nij->nj", torch.round(frac), cell[batch])
    whole = pos[first[molecule]] + d
    n_mol = first.numel()
    w = (
        torch.ones(pos.shape[0], dtype=pos.dtype, device=pos.device)
        if masses is None
        else masses.to(pos.dtype)
    )
    wsum = torch.zeros(n_mol, dtype=pos.dtype, device=pos.device).index_add_(
        0, molecule, w
    )
    com = torch.zeros(n_mol, 3, dtype=pos.dtype, device=pos.device).index_add_(
        0, molecule, whole * w[:, None]
    )
    com = com / wsum[:, None]
    per_atom = ((whole - com[molecule]) * forces).sum(-1)
    return torch.zeros(n_b, dtype=pos.dtype, device=pos.device).index_add_(
        0, batch, per_atom
    )


def molecular_pressure(
    stress: Tensor,
    pos: Tensor,
    forces: Tensor,
    cell: Tensor,
    batch: Tensor,
    molecule: Tensor,
    pbc: Optional[Tensor] = None,
    masses: Optional[Tensor] = None,
) -> Tensor:
    """The molecular pressure ``P_atom - tr(W_intra) / (3 V)`` per structure.

    The pressure of a rigid-molecule description of the same structure: the
    atomic pressure of the stress with the intramolecular virial of
    :func:`molecular_virial` removed. It is what the labelling pipeline of the
    water work reports and the quantity the committee spread is taken over in
    :func:`disagreement` when molecules are given.

    Parameters
    ----------
    stress : Tensor
        Stress ``(B, 3, 3)`` in eV/A^3.
    pos, forces, cell, batch, molecule, pbc, masses
        As in :func:`molecular_virial`.

    Returns
    -------
    Tensor
        Molecular pressure ``(B,)`` in eV/A^3. Non-periodic structures of a
        mixed batch get ``nan``.
    """
    n_b = stress.shape[0]
    volume = torch.linalg.det(cell)
    w = molecular_virial(pos, forces, batch, molecule, cell, pbc, masses, n_b)
    periodic = (
        torch.ones(n_b, dtype=torch.bool, device=pos.device)
        if pbc is None
        else pbc.all(-1)
    )
    p = pressure(stress) - w / (3.0 * volume.clamp_min(torch.finfo(volume.dtype).tiny))
    return torch.where(periodic, p, torch.full_like(p, float("nan")))


def _scatter_mean(values: Tensor, index: Tensor, n: int) -> Tensor:
    out = torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(
        0, index, values
    )
    cnt = torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(
        0, index, torch.ones_like(values)
    )
    return out / cnt.clamp_min(1)


def disagreement(
    energy: Tensor,
    forces: Tensor,
    n_atoms: Tensor,
    batch: Tensor,
    pressures: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """The committee spreads of a batch of structures.

    The standard deviations are population ones (``ddof = 0``), as in the
    reference implementation of Smith *et al.*

    Parameters
    ----------
    energy : Tensor
        Member energies ``(L, B)`` in eV.
    forces : Tensor
        Member forces ``(L, N, 3)`` in eV/A.
    n_atoms : Tensor
        Atoms per structure ``(B,)``.
    batch : Tensor
        Structure index per atom ``(N,)``.
    pressures : dict of str to Tensor, optional
        Named member pressures ``(L, B)`` in atm, for example
        ``{"pressure": ..., "pressure_molecular": ...}``; each adds a
        ``rho_<name>`` entry.

    Returns
    -------
    dict of str to Tensor
        ``rho_energy`` ``(B,)`` in eV: :math:`\\sigma_E / \\sqrt{N}`;
        ``rho_forces`` ``(B,)`` in eV/A: the RMS over atoms of the per-atom
        standard deviation of the force vector; ``rho_forces_atom`` ``(N,)``
        in eV/A: that per-atom standard deviation, ``sqrt(sum_xyz var)``;
        ``rho_forces_max_atom`` ``(B,)``: its largest value in each
        structure; and ``rho_<name>`` ``(B,)`` per pressure given, in atm.
    """
    n_b = energy.shape[1]
    out = {
        "rho_energy": energy.std(0, unbiased=False) / n_atoms.to(energy.dtype).sqrt()
    }
    var_f = forces.var(0, unbiased=False).sum(-1)  # (N,)
    out["rho_forces"] = _scatter_mean(var_f, batch, n_b).sqrt()
    sd_f = var_f.sqrt()
    out["rho_forces_atom"] = sd_f
    out["rho_forces_max_atom"] = torch.zeros(
        n_b, dtype=sd_f.dtype, device=sd_f.device
    ).scatter_reduce_(0, batch, sd_f, reduce="amax", include_self=True)
    for name, p in (pressures or {}).items():
        out[f"rho_{name}"] = p.std(0, unbiased=False)
    return out


def worst_error(
    energy: Tensor,
    forces: Tensor,
    n_atoms: Tensor,
    batch: Tensor,
    energy_ref: Tensor,
    forces_ref: Tensor,
    pressures: Optional[Dict[str, Tensor]] = None,
    pressures_ref: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """The largest per-member error of each structure (the paper's epsilon).

    Parameters
    ----------
    energy, forces, n_atoms, batch, pressures
        As in :func:`disagreement`.
    energy_ref : Tensor
        Reference energies ``(B,)``.
    forces_ref : Tensor
        Reference forces ``(N, 3)``.
    pressures_ref : dict of str to Tensor, optional
        Reference pressures ``(B,)`` per name in ``pressures``.

    Returns
    -------
    dict of str to Tensor
        ``eps_energy`` ``(B,)``: :math:`\\max_l |E_l - E_{ref}| / \\sqrt{N}`;
        ``eps_forces`` ``(B,)``: the largest over members of the RMS force
        error of the structure; ``eps_<name>`` ``(B,)``: the largest absolute
        pressure error.
    """
    n_b = energy.shape[1]
    out = {
        "eps_energy": (energy - energy_ref).abs().max(0).values
        / n_atoms.to(energy.dtype).sqrt()
    }
    err = ((forces - forces_ref) ** 2).sum(-1)  # (L, N)
    rms = torch.stack([_scatter_mean(e, batch, n_b) for e in err]).sqrt()
    out["eps_forces"] = rms.max(0).values
    for name, p in (pressures or {}).items():
        out[f"eps_{name}"] = (p - pressures_ref[name]).abs().max(0).values
    return out


#: Entries of :func:`disagreement` with one value per atom rather than per structure.
_PER_ATOM = ("rho_forces_atom",)


class Committee(nn.Module):
    """A committee of potentials evaluated together.

    Every member sees its own shallow copy of the batch, because the
    force / stress wrapper displaces positions and cell by the strain it
    differentiates, and returns detached predictions.

    Parameters
    ----------
    members : sequence of torch.nn.Module
        The potentials, each returning ``energy`` ``(B,)``, ``forces``
        ``(N, 3)`` and, for periodic batches, ``stress`` ``(B, 3, 3)`` (a
        :class:`~xnn.common.models.ForceStressOutput` built with
        ``compute_stress=True``). They must share one cutoff, since one
        neighbour list serves all of them.
    subtracted_dispersion : sequence, optional
        Each member's ``subtracted_dispersion`` record (what its training
        labels had removed); :meth:`from_checkpoints` reads them from the
        checkpoints. They must all be equal: members of different label spaces
        disagree for a reason that is not uncertainty.

    Attributes
    ----------
    members : torch.nn.ModuleList
        The potentials.
    cutoff : float
        Their common neighbour-list cutoff.
    subtracted_dispersion : dict, str or None
        The members' common record (``None`` when not given or not recorded).

    Raises
    ------
    ValueError
        With fewer than two members, with differing cutoffs, or with differing
        subtracted dispersions.
    """

    def __init__(
        self,
        members: Sequence[nn.Module],
        subtracted_dispersion: Optional[Sequence] = None,
    ):
        super().__init__()
        if len(members) < 2:
            raise ValueError("a committee needs at least two members")
        cutoffs = {round(float(getattr(m, "cutoff")), 9) for m in members}
        if len(cutoffs) != 1:
            raise ValueError(f"members must share one cutoff; got {sorted(cutoffs)}")
        records = (
            list(subtracted_dispersion) if subtracted_dispersion is not None else [None]
        )
        if subtracted_dispersion is not None and len(records) != len(members):
            raise ValueError(
                f"{len(records)} dispersion records for {len(members)} members"
            )
        if any(r != records[0] for r in records):
            raise ValueError(
                f"the members record different subtracted dispersions: {records}"
            )
        self.members = nn.ModuleList(members)
        self.cutoff = float(cutoffs.pop())
        self.subtracted_dispersion = records[0]
        self.eval()

    @classmethod
    def from_checkpoints(
        cls,
        paths: Iterable[str],
        device=None,
        dtype=None,
        dispersion=False,
        compute_stress: bool = True,
        **build_options,
    ) -> "Committee":
        """Load the members from trainer checkpoints or model directories.

        Parameters
        ----------
        paths : iterable of str
            Anything :func:`~xnn.common.models.hub.load_checkpoint` reads.
        device : torch.device or str, optional
            Where to serve the members.
        dtype : torch.dtype or str, optional
            Dtype to serve in; by default that of each checkpoint's weights.
        dispersion : dict, str, bool or None, optional
            Passed to :func:`~xnn.common.models.hub.build_potential`. By
            default ``False``: a recorded subtracted dispersion is a term
            shared by every member, so it adds nothing to the spread and is
            left out. Pass ``None`` to serve the full energy.
        compute_stress : bool, optional
            Build the stress head (needed for the pressure spread), by
            default ``True``.
        **build_options
            Further keywords of :func:`~xnn.common.models.hub.build_potential`.

        Returns
        -------
        Committee

        Raises
        ------
        ValueError
            When the checkpoints record different subtracted dispersions or
            the members differ in cutoff.
        """
        from xnn.common.models.hub import build_potential, load_checkpoint

        members, records = [], []
        for p in paths:
            ck = load_checkpoint(p)
            # as xnn.common.models.registry.recorded_dispersion: older configs lack it
            records.append(getattr(ck.config, "subtracted_dispersion", None))
            if any(r != records[0] for r in records):
                raise ValueError(
                    f"the members record different subtracted dispersions: {records}"
                )
            m = build_potential(
                ck.config,
                ck.state_dict,
                dtype=dtype,
                dispersion=dispersion,
                compute_forces=True,
                compute_stress=compute_stress,
                label=str(p),
                **build_options,
            )
            members.append(m.to(device) if device is not None else m)
        return cls(members, subtracted_dispersion=records)

    @property
    def size(self) -> int:
        """Number of members ``L``."""
        return len(self.members)

    def forward(self, data: AtomicGraph) -> Dict[str, Optional[Tensor]]:
        """Every member's prediction of a batch.

        Parameters
        ----------
        data : AtomicGraph
            A structure or batch on the members' device.

        Returns
        -------
        dict of str to Tensor
            ``energy`` ``(L, B)``, ``forces`` ``(L, N, 3)`` and ``stress``
            ``(L, B, 3, 3)`` (``None`` when the batch has no cell), detached.
        """
        energies, forces, stresses = [], [], []
        with torch.enable_grad():
            for m in self.members:
                out = m(copy.copy(data))
                energies.append(out["energy"].detach())
                forces.append(out["forces"].detach())
                stresses.append(
                    out["stress"].detach() if out.get("stress") is not None else None
                )
        stress = None if any(s is None for s in stresses) else torch.stack(stresses)
        return {
            "energy": torch.stack(energies),
            "forces": torch.stack(forces),
            "stress": stress,
        }

    def predict(self, data: AtomicGraph) -> Dict[str, Optional[Tensor]]:
        """The committee's prediction: the mean over the members.

        Parameters
        ----------
        data : AtomicGraph
            A structure or batch on the members' device.

        Returns
        -------
        dict of str to Tensor
            ``energy`` ``(B,)``, ``forces`` ``(N, 3)``, ``stress``
            ``(B, 3, 3)`` or ``None``.
        """
        out = self(data)
        return {k: (v.mean(0) if v is not None else None) for k, v in out.items()}

    def disagreement(
        self,
        data: AtomicGraph,
        molecule: Optional[Tensor] = None,
        masses: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """The spreads of a batch, with the members' and the mean prediction.

        Parameters
        ----------
        data : AtomicGraph
            A structure or batch on the members' device.
        molecule : Tensor, optional
            Molecule index per atom ``(N,)``, unique across the batch. When
            given and the batch has a cell, the molecular pressure spread is
            added.
        masses : Tensor, optional
            Atomic masses ``(N,)`` for the molecular centres of mass.

        Returns
        -------
        dict of str to Tensor
            The entries of :func:`disagreement` (``rho_energy``,
            ``rho_forces``, and ``rho_pressure`` / ``rho_pressure_molecular``
            in atm for periodic batches), the members' ``energy`` ``(L, B)``,
            ``forces`` ``(L, N, 3)``, ``stress``, ``pressure`` ``(L, B)`` and
            ``pressure_molecular`` ``(L, B)`` in atm, their means under
            ``mean_<name>``, and ``n_atoms`` ``(B,)``.
        """
        out = self(data)
        pressures: Dict[str, Tensor] = {}
        if out["stress"] is not None:
            p = pressure(out["stress"]) * ATM_PER_EV_A3
            if data.pbc is not None:
                p = torch.where(
                    data.pbc.all(-1)[None], p, torch.full_like(p, float("nan"))
                )
            pressures["pressure"] = p
            if molecule is not None:
                pressures["pressure_molecular"] = torch.stack(
                    [
                        molecular_pressure(
                            s,
                            data.pos.detach(),
                            f,
                            data.cell.detach(),
                            data.batch,
                            molecule,
                            data.pbc,
                            masses,
                        )
                        * ATM_PER_EV_A3
                        for s, f in zip(out["stress"], out["forces"])
                    ]
                )
        res = disagreement(
            out["energy"], out["forces"], data.n_atoms, data.batch, pressures
        )
        res.update(out)
        res.update(pressures)
        for name in ("energy", "forces", "stress", *pressures):
            v = res[name]
            res[f"mean_{name}"] = v.mean(0) if v is not None else None
        res["n_atoms"] = data.n_atoms
        return res

    @torch.no_grad()
    def score(
        self,
        structures,
        batch_size: int = 8,
        molecule: Optional[Callable[[dict], np.ndarray]] = None,
        masses: Optional[Callable[[dict], np.ndarray]] = None,
        device=None,
        keys: Sequence[str] = (
            "rho_energy",
            "rho_forces",
            "rho_forces_max_atom",
            "rho_pressure",
            "rho_pressure_molecular",
            "mean_energy",
            "mean_pressure",
            "mean_pressure_molecular",
            "n_atoms",
        ),
    ) -> Dict[str, np.ndarray]:
        """Score many structures: per-structure spreads as numpy arrays.

        Parameters
        ----------
        structures : AtomicDataset or list of dict
            The structures; a list is wrapped in an
            :class:`~xnn.common.data.AtomicDataset` at the committee's cutoff.
        batch_size : int, optional
            Structures per batch, by default 8.
        molecule : callable, optional
            ``f(structure) -> (n_i,)`` molecule index per atom of one
            structure, in general :func:`molecules` (or, for OHH waters only,
            ``lambda s: np.arange(len(s["pos"])) // 3``). Enables the
            molecular pressure spread.
        masses : callable, optional
            ``f(structure) -> (n_i,)`` atomic masses, with ``molecule``; in
            general the module's :func:`masses`.
        device : torch.device or str, optional
            Where the batches go; by default the first member's device.
        keys : sequence of str, optional
            Which entries of :meth:`disagreement` to collect: per-structure
            ones by default; add ``"rho_forces_atom"`` for the per-atom spread.

        Returns
        -------
        dict of str to ndarray
            Each requested key present in the output, concatenated over the
            structures in order: ``(n_structures,)`` for the per-structure
            entries, ``(total atoms,)`` for ``rho_forces_atom`` (split it with
            ``n_atoms``). Pressure keys are absent when no structure has a
            cell, and ``nan`` for the structures of batches without one.
        """
        if isinstance(structures, AtomicDataset):
            ds, raw = structures, structures.structures
        else:
            raw = list(structures)
            ds = AtomicDataset(raw, self.cutoff)
        if device is None:
            device = next(self.parameters()).device
        cols: Dict[str, Dict[int, np.ndarray]] = {}
        batches = []
        for start in range(0, len(ds), batch_size):
            idx = range(start, min(start + batch_size, len(ds)))
            data = collate([ds[i] for i in idx]).to(device)
            mol = mas = None
            if molecule is not None:
                parts, offset = [], 0
                for i in idx:
                    m = np.asarray(molecule(raw[i]), dtype=np.int64)
                    parts.append(m + offset)
                    offset += int(m.max()) + 1
                mol = torch.as_tensor(np.concatenate(parts), device=device)
                if masses is not None:
                    mas = torch.as_tensor(
                        np.concatenate(
                            [np.asarray(masses(raw[i]), dtype=float) for i in idx]
                        ),
                        device=device,
                    )
            res = self.disagreement(data, mol, mas)
            batches.append((len(idx), int(data.num_nodes)))
            for k in keys:
                if res.get(k) is not None:
                    cols.setdefault(k, {})[len(batches) - 1] = (
                        res[k].detach().cpu().numpy()
                    )
        # A batch without cells has no pressures; fill those structures with nan
        # so every key stays aligned with the structures. (The original dropped
        # them, misaligning mixed pools scored in several batches.)
        out = {}
        for k, parts in cols.items():
            per_atom = k in _PER_ATOM
            out[k] = np.concatenate(
                [
                    parts.get(b, np.full(n_atoms if per_atom else n_structs, np.nan))
                    for b, (n_structs, n_atoms) in enumerate(batches)
                ]
            )
        return out
