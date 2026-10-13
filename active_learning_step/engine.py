# -*- coding: utf-8 -*-

"""``seamm-al engine``: a committee's mean as one MDI engine, with its spreads.

The committee mean is served to an MDI driver (LAMMPS, ``fix mdi/qm``) as a single
potential, from the water campaign's ``campaign/liquid.py`` (Sina Mostafanejad): the
mean of the members' short-range cores is one
:class:`~xnn.common.models.InteratomicPotential` (:class:`CommitteeMeanCore`); the
dispersion recorded in the checkpoints is added once on top (the same term in every
member, so it is not averaged); and the force and stress heads differentiate the sum.
The mean of the members' energies has the mean of their forces as its gradient, so
this is the committee's prediction at the cost of one backward pass.

What is added here: every ``every`` evaluations (``--spreads FILE --every N``) the
engine also evaluates each member on its own (one forward and backward pass per
member, only on the steps written) and appends the committee spreads to a CSV:
``rho_energy``, ``rho_forces``, ``rho_forces_max_atom`` and, for periodic systems,
``rho_pressure`` and ``rho_pressure_molecular`` (molecules from the bond graph of the
first frame written). Each row also carries ``max_dF``, the largest difference between
the engine's forces and the mean of the members' forces (zero to rounding when no
dispersion is added; the check that the mean core is the members' mean). With
``--flag NAME=VALUE`` a frame whose spread reaches a value is written, with its step
and spreads and the per-atom force spread, to an extxyz (``--flagged``);
``--frames`` writes every frame recorded.

The members and the mean share their modules: each member is built once and the
mean core holds the members' cores.

xnn interface (public names): ``xnn.common.models.InteratomicPotential``,
``ForceStressOutput``, ``add_dispersion``; ``xnn.common.models.hub.load_checkpoint``,
``build_potential``; ``xnn.common.models.registry.resolve_dispersion``;
``xnn.common.models.d4.enable_eeq_reuse``; ``xnn.common.deploy.mdi_engine.MDIEngine``,
``BOHR_TO_ANGSTROM``, ``HARTREE_TO_EV``; ``xnn.common.data.structure_to_graph``,
``collate``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from xnn.common.data import AtomicGraph
from xnn.common.models import ForceStressOutput, InteratomicPotential

logger = logging.getLogger(__name__)

__all__ = [
    "CommitteeMeanCore",
    "committee_mean_potential",
    "build_committee",
    "CommitteeEngine",
    "main",
]

SPREAD_COLUMNS = (
    "step",
    "n_atoms",
    "rho_energy",
    "rho_forces",
    "rho_forces_max_atom",
    "rho_pressure",
    "rho_pressure_molecular",
    "energy_mean",
    "engine_energy",
    "max_dF",
    "seconds",
)


class CommitteeMeanCore(InteratomicPotential):
    """The mean energy of several short-range cores, differentiable.

    Parameters
    ----------
    cores : sequence of torch.nn.Module
        The members' cores (the model inside each
        :class:`~xnn.common.models.ForceStressOutput`), returning ``node_energy``
        and ``energy``; one cutoff for all.

    Attributes
    ----------
    members : torch.nn.ModuleList
        The cores.
    cutoff : float
        Their common neighbour-list cutoff.
    node_feature_dim : int
        That of the first core (the long-range and dispersion wrappers read it;
        the mean core exposes no features of its own).

    Raises
    ------
    ValueError
        With fewer than two cores or differing cutoffs.
    """

    def __init__(self, cores: Sequence[nn.Module]):
        super().__init__()
        if len(cores) < 2:
            raise ValueError("a committee needs at least two members")
        cutoffs = {round(float(getattr(m, "cutoff")), 9) for m in cores}
        if len(cutoffs) != 1:
            raise ValueError(f"members must share one cutoff; got {sorted(cutoffs)}")
        self.members = nn.ModuleList(cores)
        self.cutoff = float(cutoffs.pop())
        self.node_feature_dim = int(getattr(cores[0], "node_feature_dim", 0))

    @property
    def size(self) -> int:
        """Number of members."""
        return len(self.members)

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Mean ``node_energy`` and ``energy`` over the members.

        Every member sees a shallow copy of the graph, as in
        :class:`~active_learning_step.committee.Committee`. The result stays
        attached to the graph's positions and cell, so a wrapping
        :class:`~xnn.common.models.ForceStressOutput` differentiates it.
        """
        node, total = [], []
        for m in self.members:
            out = m(copy.copy(data))
            node.append(out["node_energy"])
            total.append(out["energy"])
        return {
            "node_energy": torch.stack(node).mean(0),
            "energy": torch.stack(total).mean(0),
        }


def build_committee(paths, device=None, dtype=None):
    """The members, each built once with forces and stress and no dispersion.

    Returns
    -------
    (list of ForceStressOutput, list)
        The members and their ``subtracted_dispersion`` records.

    Raises
    ------
    ValueError
        When the records differ.
    """
    from xnn.common.models.hub import build_potential

    from .committee import read_checkpoint

    members, records = [], []
    for p in paths:
        ck = read_checkpoint(p)
        records.append(getattr(ck.config, "subtracted_dispersion", None))
        m = build_potential(
            ck.config,
            ck.state_dict,
            dtype=dtype,
            dispersion=False,
            compute_forces=True,
            compute_stress=True,
            label=str(p),
        )
        members.append(m if device is None else m.to(device))
    if any(r != records[0] for r in records):
        raise ValueError(
            f"the members record different subtracted dispersions: {records}"
        )
    return members, records


def committee_mean_potential(
    paths=None,
    device=None,
    dtype=None,
    dispersion=None,
    eeq_reuse: bool = False,
    compute_stress: bool = True,
    members=None,
    records=None,
) -> ForceStressOutput:
    """Build the committee mean as one potential with forces and stress.

    Parameters
    ----------
    paths : iterable of str, optional
        The members' checkpoints; or give ``members`` (and ``records``) from
        :func:`build_committee`, whose cores the mean then shares.
    device : torch.device or str, optional
        Where to serve.
    dtype : torch.dtype or str, optional
        Dtype to serve in; by default that of the first member's weights.
    dispersion : dict, str, bool or None, optional
        The term added once on top of the mean, resolved against the
        checkpoints' record as :func:`~xnn.common.models.registry.resolve_dispersion`
        does: ``None`` adds the recorded term, ``False`` none (the route-B space).
    eeq_reuse : bool, optional
        Carry the D4 EEQ solve between steps (molecular dynamics).
    compute_stress : bool, optional
        Build the stress head, by default ``True``.
    members, records : optional
        Prebuilt members and their dispersion records.

    Returns
    -------
    ForceStressOutput
        ``ForceStressOutput(dispersion(CommitteeMeanCore(cores)))`` in eval mode;
        its ``cutoff`` is the dispersion term's when one is added.
    """
    from xnn.common.models import add_dispersion
    from xnn.common.models.registry import resolve_dispersion

    if members is None:
        members, records = build_committee(paths, device=device, dtype=dtype)
    serve_dtype = next(members[0].parameters()).dtype
    core = CommitteeMeanCore([m.model for m in members])
    spec = resolve_dispersion(dispersion, records[0] if records else None)
    if spec is not None:
        # the term's constant tables are built in float64 and cast once, as
        # build_potential does
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            model = add_dispersion(core, spec)
        finally:
            torch.set_default_dtype(prev)
        logger.info(
            f"committee mean of {core.size} members with {type(model).__name__} on top"
        )
    else:
        model = core
        logger.info(f"committee mean of {core.size} members, no dispersion (route B)")
    potential = ForceStressOutput(
        model, compute_forces=True, compute_stress=compute_stress
    ).to(serve_dtype)
    if device is not None:
        potential = potential.to(device)
    if eeq_reuse:
        from xnn.common.models.d4 import enable_eeq_reuse

        enable_eeq_reuse(potential)
    return potential.eval()


def _flags(specs):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--flag needs NAME=VALUE, not {spec!r}")
        name, value = spec.split("=", 1)
        if name not in SPREAD_COLUMNS[2:7]:
            raise ValueError(f"cannot flag on {name!r}; one of {SPREAD_COLUMNS[2:7]}")
        out[name] = float(value)
    return out


def _engine_class():
    from xnn.common.deploy.mdi_engine import MDIEngine

    class CommitteeEngine(MDIEngine):
        """xnn's MDI engine serving a committee mean, writing its spreads.

        Parameters
        ----------
        model : torch.nn.Module
            The committee mean (:func:`committee_mean_potential`).
        cutoff : float
            Its neighbour-list cutoff.
        device : str
            Torch device.
        total_charge : float
            The system's net charge.
        committee : active_learning_step.committee.Committee, optional
            The members, for the spreads.
        every : int, optional
            Record the spreads every ``every`` evaluations (0: never).
        spreads : str, optional
            The CSV the spreads are appended to.
        flags : dict, optional
            Spread name -> value; a frame reaching any is written to ``flagged``.
        flagged, frames : str, optional
            extxyz files for the flagged frames and for every frame recorded.
        dispersion_added : bool, optional
            Whether the mean carries a dispersion term (``max_dF`` is then not
            reported).
        """

        def __init__(
            self,
            model,
            cutoff,
            device="cpu",
            total_charge=0.0,
            committee=None,
            every=0,
            spreads=None,
            flags=None,
            flagged=None,
            frames=None,
            dispersion_added=False,
        ):
            super().__init__(
                model, cutoff=cutoff, device=device, total_charge=total_charge
            )
            self.committee = committee
            if committee is not None:
                for p in committee.parameters():
                    p.requires_grad_(False)
            self.every = int(every)
            self.flags = flags or {}
            self.flagged = flagged
            self.frames = frames
            self.dispersion_added = dispersion_added
            self.molecule = None
            self._spreads_fh = None
            self._writer = None
            if spreads is not None:
                self._spreads_fh = open(spreads, "w", newline="")
                self._writer = csv.DictWriter(
                    self._spreads_fh, fieldnames=SPREAD_COLUMNS
                )
                self._writer.writeheader()
            self._n_recorded = 0
            self._t_record = 0.0

        def calculate(self) -> None:
            super().calculate()
            step = self._n_calc - 1
            if self.committee is not None and self.every > 0 and step % self.every == 0:
                self.record(step)

        def record(self, step: int) -> dict:
            """The committee spreads of the current structure, written out."""
            from xnn.common.data import collate, structure_to_graph
            from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV

            from .molecules import masses, molecules

            t0 = time.perf_counter()
            pos = self.coords_bohr * BOHR_TO_ANGSTROM
            cell = None if self.cell_bohr is None else self.cell_bohr * BOHR_TO_ANGSTROM
            s = {
                "pos": torch.as_tensor(pos, dtype=torch.float64, device=self.device),
                "atomic_numbers": torch.as_tensor(
                    self.atomic_numbers, dtype=torch.long, device=self.device
                ),
                "cell": (
                    None
                    if cell is None
                    else torch.as_tensor(cell, dtype=torch.float64, device=self.device)
                ),
                "pbc": (
                    None
                    if cell is None
                    else torch.ones(3, dtype=torch.bool, device=self.device)
                ),
                "total_charge": self.total_charge,
            }
            g = collate(
                [structure_to_graph(s, self.committee.cutoff, device=self.device)]
            )
            g.compute_dtype = self.dtype
            mol = mas = None
            if cell is not None:
                if self.molecule is None:
                    plain = {
                        "pos": pos,
                        "atomic_numbers": self.atomic_numbers,
                        "cell": cell,
                        "pbc": [True] * 3,
                    }
                    self.molecule = torch.as_tensor(
                        molecules(plain), device=self.device
                    )
                    self._masses = torch.as_tensor(
                        masses(plain), dtype=torch.float64, device=self.device
                    )
                mol, mas = self.molecule, self._masses
            r = self.committee.disagreement(g, mol, mas)
            n = len(self.atomic_numbers)
            row = {
                "step": step,
                "n_atoms": n,
                "rho_energy": float(r["rho_energy"][0]),
                "rho_forces": float(r["rho_forces"][0]),
                "rho_forces_max_atom": float(r["rho_forces_max_atom"][0]),
                "energy_mean": float(r["mean_energy"][0]),
                "engine_energy": self.energy * HARTREE_TO_EV,
            }
            if cell is not None and r.get("rho_pressure") is not None:
                row["rho_pressure"] = float(r["rho_pressure"][0])
                row["rho_pressure_molecular"] = float(r["rho_pressure_molecular"][0])
            if not self.dispersion_added:
                f_engine = self.forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
                f_mean = r["mean_forces"].detach().double().cpu().numpy()
                row["max_dF"] = float(np.abs(f_engine - f_mean).max())
            self._t_record += time.perf_counter() - t0
            self._n_recorded += 1
            row["seconds"] = time.perf_counter() - t0
            if self._writer is not None:
                self._writer.writerow({k: _cell(row.get(k)) for k in SPREAD_COLUMNS})
                self._spreads_fh.flush()
            hit = [
                f"{name} {row[name]:.4g} >= {value:.4g}"
                for name, value in self.flags.items()
                if row.get(name) is not None and row[name] >= value
            ]
            if (hit and self.flagged) or self.frames:
                atom_sd = r["rho_forces_atom"].detach().double().cpu().numpy()
                if hit and self.flagged:
                    self._write_frame(self.flagged, pos, cell, row, atom_sd, hit)
                if self.frames:
                    self._write_frame(self.frames, pos, cell, row, atom_sd, hit)
            return row

        def _write_frame(self, path, pos, cell, row, atom_sd, hit):
            from ase import Atoms
            from ase.io import write

            a = Atoms(numbers=self.atomic_numbers, positions=pos)
            if cell is not None:
                a.set_cell(cell)
                a.set_pbc(True)
            for k, v in row.items():
                if v is not None:
                    a.info[f"AL/{k}"] = v
            a.info["SEAMM/configuration_name"] = f"step {row['step']}"
            if self.total_charge:
                a.info["total_charge"] = self.total_charge
            if hit:
                a.info["AL/flags"] = "; ".join(hit)
            a.arrays["rho_forces_atom"] = atom_sd
            write(path, a, format="extxyz", append=True)

        def close(self):
            if self._spreads_fh is not None:
                self._spreads_fh.close()
                self._spreads_fh = None
            if self._n_recorded:
                logger.info(
                    f"spreads recorded on {self._n_recorded} steps, "
                    f"{1000 * self._t_record / self._n_recorded:.1f} ms each"
                )

    return CommitteeEngine


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def CommitteeEngine(*args, **kwargs):
    """Build the committee MDI engine (see :func:`_engine_class`)."""
    return _engine_class()(*args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Serve a committee mean to an MDI driver, recording its spreads."""
    p = argparse.ArgumentParser(
        prog="seamm-al engine",
        description="MDI engine serving the mean of a committee of xnn checkpoints",
    )
    p.add_argument(
        "--members", nargs="+", required=True, help="the members' checkpoints"
    )
    p.add_argument(
        "--drop-member",
        type=int,
        action="append",
        default=[],
        metavar="K",
        help="leave out the K-th member, counting from 0 (repeatable)",
    )
    p.add_argument(
        "-mdi",
        "--mdi",
        dest="mdi_options",
        required=True,
        help='MDI options, e.g. "-role ENGINE -name xnn -method MPI"',
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, choices=("float32", "float64"))
    p.add_argument(
        "--no-dispersion",
        action="store_true",
        help="serve the route-B energy without the recorded dispersion",
    )
    p.add_argument(
        "--eeq-reuse", action="store_true", help="carry the D4 EEQ solve between steps"
    )
    p.add_argument("--total-charge", type=float, default=0.0)
    p.add_argument("--spreads", default=None, help="CSV of the spreads")
    p.add_argument(
        "--every",
        type=int,
        default=10,
        help="record the spreads every N evaluations (default 10)",
    )
    p.add_argument(
        "--flag",
        action="append",
        default=[],
        help="NAME=VALUE: write frames whose spread reaches it (repeatable)",
    )
    p.add_argument(
        "--flagged",
        default="flagged.extxyz",
        help="extxyz of the flagged frames (default flagged.extxyz)",
    )
    p.add_argument("--frames", default=None, help="extxyz of every frame recorded")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .committee import Committee

    paths = [m for k, m in enumerate(args.members) if k not in set(args.drop_member)]
    dtype = getattr(torch, args.dtype) if args.dtype else None
    members, records = build_committee(paths, device=args.device, dtype=dtype)
    model = committee_mean_potential(
        device=args.device,
        dispersion=False if args.no_dispersion else None,
        eeq_reuse=args.eeq_reuse,
        members=members,
        records=records,
    )
    want = bool(args.spreads or args.flag or args.frames)
    committee = Committee(members, subtracted_dispersion=records) if want else None
    dispersion_added = not args.no_dispersion and records[0] is not None
    mpi_comm = None
    if "MPI" in args.mdi_options.split():
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
    engine = CommitteeEngine(
        model,
        cutoff=float(model.cutoff),
        device=args.device,
        total_charge=args.total_charge,
        committee=committee,
        every=args.every if want else 0,
        spreads=args.spreads,
        flags=_flags(args.flag),
        flagged=args.flagged if args.flag else None,
        frames=args.frames,
        dispersion_added=dispersion_added,
    )
    logger.info(
        f"engine: {len(paths)} members (dropped {sorted(args.drop_member)}), cutoff "
        f"{engine.cutoff:.2f} A, {engine.device}, {engine.dtype}, dispersion "
        f"{'added' if dispersion_added else 'not added'}, spreads every "
        f"{engine.every if want else 0}"
    )
    try:
        engine.run(args.mdi_options, mpi_comm=mpi_comm)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
