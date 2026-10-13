# -*- coding: utf-8 -*-

"""``seamm-al check``: does the committee's spread predict its real error?

For a trained committee and labelled frames, every frame gets its committee spreads
(``rho_*``, as ``seamm-al score``) and its **out-of-fold** errors, the errors of the
members that never trained on it:

* a frame held out by one member (its fold): that member's error;
* a frame no member trained on (a common validation or test set): the mean of the
  members' errors;
* a frame every member trained on is ``trained`` and left out of the statistics.

The errors are the force RMSE over the atoms (``err_forces``, eV/A, the RMS of the
force-vector error), the energy per atom (``err_energy``, eV) and, for cells, the
atomic and molecular pressures (``err_pressure``, ``err_pressure_molecular``, atm).
The label's molecular pressure is computed from its stress and forces with
:func:`~active_learning_step.committee.molecular_pressure`, as in the water campaign.
Alongside, the worst-member errors of Sina's definition (:func:`worst_error`:
``eps_energy`` per sqrt(atom), ``eps_forces``, ``eps_pressure``,
``eps_pressure_molecular``) over all members, which ``--calibrate`` uses.

The committee directory is either one made by ``seamm-al train`` (``manifest.json``
and ``frames.csv``: frame ``i`` is held out by member ``fold[i]``, matched by
configuration name) or one with a ``fold_manifest.json`` per member, listing tags under
``held_out``, ``validation`` and ``test`` (mlff's layout). ``tag_key`` names the info
key holding a frame's tag; by default its configuration name.

Outputs: a CSV with one row per frame, and the Spearman correlations of each spread
with its error, per kind, over all out-of-fold frames and over the held-out ones only.
"""

import csv
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

from .pool import configuration_name, iter_frames

logger = logging.getLogger(__name__)

__all__ = ["PAIRS", "spearman", "read_folds", "check_committee"]

#: The spread and error compared, per kind.
PAIRS = {
    "cluster": [
        ("rho_forces", "err_forces"),
        ("rho_forces_max_atom", "err_forces"),
        ("rho_energy", "err_energy"),
    ],
    "cell": [
        ("rho_forces", "err_forces"),
        ("rho_forces_max_atom", "err_forces"),
        ("rho_energy", "err_energy"),
        ("rho_pressure_molecular", "err_pressure_molecular"),
        ("rho_pressure", "err_pressure"),
    ],
}

#: The worst-member error that calibrates each spread (Smith et al., fig 1).
EPS = {
    "rho_energy": "eps_energy",
    "rho_forces": "eps_forces",
    "rho_forces_max_atom": "eps_forces",
    "rho_pressure": "eps_pressure",
    "rho_pressure_molecular": "eps_pressure_molecular",
}

COLUMNS = (
    "tag",
    "file",
    "index",
    "kind",
    "n_atoms",
    "role",
    "oof_members",
    "rho_energy",
    "rho_forces",
    "rho_forces_max_atom",
    "rho_pressure",
    "rho_pressure_molecular",
    "err_energy",
    "err_forces",
    "err_pressure",
    "err_pressure_molecular",
    "mean_error_pressure_molecular",
    "eps_energy",
    "eps_forces",
    "eps_pressure",
    "eps_pressure_molecular",
)


def _ranks(x):
    """Ranks from 1, ties given their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman's rank correlation and its two-sided p-value.

    The coefficient is the Pearson correlation of the ranks (ties averaged), as
    ``scipy.stats.spearmanr``; the p-value is scipy's when scipy is there, else
    from the t distribution's normal approximation.

    Returns
    -------
    (float, float, int)
        The coefficient, the p-value and the number of pairs (nan with fewer
        than 3 finite pairs).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import spearmanr

        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue), n
    except ImportError:
        pass
    rx, ry = _ranks(x), _ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    t = rho * math.sqrt((n - 2) / (1 - rho**2))
    return rho, float(math.erfc(abs(t) / math.sqrt(2))), n


def member_dirs(directory):
    """The member directories ``m<k>`` of a committee, in order of ``k``."""
    return sorted(
        (p for p in Path(directory).glob("m*") if p.is_dir() and p.name[1:].isdigit()),
        key=lambda p: int(p.name[1:]),
    )


def read_folds(directory, checkpoint="best.pt", drop=()):
    """The members of a committee and which frames each never trained on.

    Parameters
    ----------
    directory : str
        A ``seamm-al train`` directory, or one with ``m<k>/fold_manifest.json``.
    checkpoint : str, optional
        The members' checkpoint file, by default ``best.pt``.
    drop : sequence of int, optional
        Members to leave out (the ``k`` of ``m<k>``). The frames a dropped member
        held out were trained on by all the others, so they become ``trained``.

    Returns
    -------
    (list of str, dict, str, list of int)
        The kept members' checkpoints, in order; for each tag either
        ``("held", [i])`` or ``("unseen", [i, ...])`` with ``i`` the position in
        that list; the layout, ``"train"`` or ``"fold_manifest"``; and the kept
        members' ``k``.
    """
    directory = Path(directory)
    every = member_dirs(directory)
    unknown = sorted(set(drop) - {int(p.name[1:]) for p in every})
    if unknown:
        raise ValueError(f"no member m{unknown[0]} in {directory}")
    mdirs = [p for p in every if int(p.name[1:]) not in set(drop)]
    if len(mdirs) < 2:
        raise ValueError(f"{directory} has fewer than two members (after dropping)")
    paths = [str(p / checkpoint) for p in mdirs]
    for p in paths:
        if not os.path.exists(p):
            raise ValueError(f"no checkpoint {p}")
    members = [int(p.name[1:]) for p in mdirs]
    roles = {}
    if (directory / "manifest.json").exists() and (directory / "frames.csv").exists():
        with open(directory / "frames.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                fold = int(row["fold"])
                if fold in members:
                    roles[row["configuration"]] = ("held", [members.index(fold)])
        return paths, roles, "train", members
    manifests = []
    for p in mdirs:
        path = p / "fold_manifest.json"
        if not path.exists():
            raise ValueError(f"{directory} has neither manifest.json nor {path}")
        manifests.append(json.loads(path.read_text()))
    held, unseen = {}, {}
    for k, m in enumerate(manifests):
        for tag in m.get("held_out", []):
            held.setdefault(str(tag), []).append(k)
        for tag in list(m.get("validation", [])) + list(m.get("test", [])):
            unseen.setdefault(str(tag), []).append(k)
    for tag, ks in unseen.items():
        roles[tag] = ("unseen", ks)
    for tag, ks in held.items():
        roles[tag] = ("held", ks)
    return paths, roles, "fold_manifest", members


def _full_stress(value):
    """A (3, 3) stress from a 9-vector, a Voigt 6-vector or a matrix."""
    s = np.asarray(value, dtype=float).ravel()
    if s.size == 9:
        return s.reshape(3, 3)
    if s.size == 6:
        xx, yy, zz, yz, xz, xy = s
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    raise ValueError(f"cannot read a stress of {s.size} values")


def _labels(atoms, energy_key, forces_key, stress_key):
    """Energy, forces and stress of a labelled frame (from info / arrays / calc)."""

    def get(kind, key):
        if kind == "forces":
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], dtype=float)
        elif key in atoms.info:
            return atoms.info[key]
        if atoms.calc is not None and key == kind:
            getter = "get_potential_energy" if kind == "energy" else f"get_{kind}"
            try:
                return getattr(atoms, getter)()
            except Exception:
                return None
        return None

    energy = get("energy", energy_key)
    forces = get("forces", forces_key)
    stress = get("stress", stress_key)
    return (
        None if energy is None else float(energy),
        forces,
        None if stress is None else _full_stress(stress),
    )


def check_committee(
    directory,
    data,
    out_csv="check.csv",
    tag_key=None,
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    dispersion=False,
    device="cpu",
    dtype=None,
    checkpoint="best.pt",
    calibrate=(),
    summary_json=None,
    drop=(),
    flag_factor=1.5,
    members_csv=None,
):
    """Spreads against out-of-fold errors for every labelled frame.

    Parameters
    ----------
    directory : str
        The committee directory (see the module docstring).
    data : sequence of str
        Labelled extxyz files.
    out_csv : str, optional
        The per-frame CSV.
    tag_key : str, optional
        The info key holding a frame's tag; by default its configuration name.
    energy_key, forces_key, stress_key : str, optional
        Where the labels are (info / arrays); by default ``REF_*``.
    dispersion : bool, optional
        Add the checkpoints' recorded dispersion (for labels that include it); by
        default ``False``, for route-B labels.
    device, dtype : optional
        As in ``seamm-al score``.
    checkpoint : str, optional
        The members' checkpoint, by default ``best.pt``.
    calibrate : sequence of str, optional
        ``NAME:FAIL_LEVEL[:CAPTURE]`` with NAME a spread, e.g.
        ``rho_forces:0.15`` (eV/A) or ``rho_pressure_molecular:500:0.9`` (atm).
    summary_json : str, optional
        Also write the correlations and calibrations here.
    drop : sequence of int, optional
        Members to leave out of the committee (the ``k`` of ``m<k>``).
    flag_factor : float, optional
        Flag a member whose loss or error is more than this times the median of
        the others', by default 1.5.
    members_csv : str, optional
        The member-health table; by default next to ``out_csv``
        (``<stem>_members.csv``).

    Returns
    -------
    (list of dict, dict)
        The per-frame rows and the summary.
    """
    import torch
    from xnn.common.data import collate, structure_to_graph

    from .committee import (
        ATM_PER_EV_A3,
        Committee,
        molecular_pressure,
        pressure,
        structure_from_atoms,
        worst_error,
    )
    from .molecules import masses, molecules
    from .thresholds import calibrate_threshold, split_statistics

    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    paths, roles, layout, names = read_folds(directory, checkpoint, drop)
    com = Committee.from_checkpoints(
        paths, device=device, dtype=dtype, dispersion=None if dispersion else False
    )
    logger.info(f"{com.size} members ({layout} layout), {len(roles)} tags with folds")
    L = com.size

    rows = []
    for path in data:
        fname = str(Path(path).resolve())
        for index, atoms in iter_frames(path):
            tag = (
                configuration_name(atoms)
                if tag_key is None
                else str(atoms.info.get(tag_key, ""))
            )
            role, oof = roles.get(tag, ("trained", []))
            e_ref, f_ref, s_ref = _labels(atoms, energy_key, forces_key, stress_key)
            if f_ref is None:
                raise ValueError(f"frame {index} of {path} has no {forces_key!r}")
            s = structure_from_atoms(atoms)
            periodic = s["cell"] is not None
            n = len(atoms)
            g = collate([structure_to_graph(s, com.cutoff)]).to(device)
            lab = g.pos.dtype  # the labels' tensors, as the water campaign
            mol = mas = None
            if periodic:
                mol = torch.as_tensor(molecules(s), device=device)
                mas = torch.as_tensor(masses(s), dtype=lab, device=device)
            r = com.disagreement(g, mol, mas)
            F = r["forces"].detach().double().cpu()  # (L, N, 3)
            fr = torch.as_tensor(f_ref, dtype=torch.float64)
            per_member_f = ((F - fr) ** 2).sum(-1).mean(-1).sqrt()  # (L,)
            row = {
                "tag": tag,
                "file": fname,
                "index": index,
                "kind": "cell" if periodic else "cluster",
                "n_atoms": n,
                "role": role,
                "oof_members": ";".join(str(k) for k in oof),
                "rho_energy": float(r["rho_energy"][0]),
                "rho_forces": float(r["rho_forces"][0]),
                "rho_forces_max_atom": float(r["rho_forces_max_atom"][0]),
            }
            E = r["energy"].detach().double().cpu()[:, 0]  # (L,)
            if e_ref is not None:
                per_member_e = (E - e_ref).abs() / n
            eps = worst_error(
                r["energy"].detach().double().cpu(),
                F,
                torch.tensor([n]),
                torch.zeros(n, dtype=torch.long),
                torch.tensor([e_ref if e_ref is not None else float("nan")]),
                fr,
            )
            if e_ref is not None:
                row["eps_energy"] = float(eps["eps_energy"][0])
            row["eps_forces"] = float(eps["eps_forces"][0])
            if periodic and r.get("pressure") is not None:
                row["rho_pressure"] = float(r["rho_pressure"][0])
                row["rho_pressure_molecular"] = float(r["rho_pressure_molecular"][0])
                P = r["pressure"].detach().double().cpu()[:, 0]
                Pm = r["pressure_molecular"].detach().double().cpu()[:, 0]
                if s_ref is not None:
                    st = torch.as_tensor(s_ref, dtype=lab, device=device)[None]
                    p_ref = float(pressure(st)[0]) * ATM_PER_EV_A3
                    pm_ref = (
                        float(
                            molecular_pressure(
                                st,
                                g.pos.detach(),
                                torch.as_tensor(f_ref, dtype=lab, device=device),
                                g.cell.detach(),
                                g.batch,
                                mol,
                                g.pbc,
                                mas,
                            )[0]
                        )
                        * ATM_PER_EV_A3
                    )
                    row["eps_pressure"] = float((P - p_ref).abs().max())
                    row["eps_pressure_molecular"] = float((Pm - pm_ref).abs().max())
                    row["mean_error_pressure_molecular"] = float(Pm.mean()) - pm_ref
            member = {"err_forces": per_member_f.numpy()}
            if e_ref is not None:
                member["err_energy"] = per_member_e.numpy()
            P_ = Pm_ = None
            if periodic and r.get("pressure") is not None:
                P_, Pm_ = P.numpy(), Pm.numpy()
                if s_ref is not None:
                    member["err_pressure"] = np.abs(P_ - p_ref)
                    member["err_pressure_molecular"] = np.abs(Pm_ - pm_ref)
            row["_member"] = member
            if L >= 3:
                row["_loo"] = _leave_one_out(E.numpy(), F.numpy(), n, P_, Pm_)
            if oof:
                ks = list(oof)
                row["err_forces"] = float(per_member_f[ks].mean())
                if e_ref is not None:
                    row["err_energy"] = float(per_member_e[ks].mean())
                if periodic and s_ref is not None and r.get("pressure") is not None:
                    row["err_pressure"] = float((P[ks] - p_ref).abs().mean())
                    row["err_pressure_molecular"] = float(
                        (Pm[ks] - pm_ref).abs().mean()
                    )
            rows.append(row)
        logger.info(f"{fname}: {len(rows)} frames checked")

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in COLUMNS})

    summary = {
        "members": paths,
        "member_numbers": names,
        "dropped": sorted(int(k) for k in drop),
        "layout": layout,
        "roles": {},
        "spearman": {},
    }
    health = member_health(directory, names, rows, flag_factor)
    summary["member_health"] = health
    if members_csv is None:
        members_csv = str(Path(out_csv).with_name(Path(out_csv).stem + "_members.csv"))
    _write_health(members_csv, health)
    summary["members_csv"] = members_csv
    for role in ("held", "unseen", "trained"):
        summary["roles"][role] = sum(1 for r in rows if r["role"] == role)
    unmatched = summary["roles"]["trained"]
    if unmatched == len(rows):
        logger.warning(
            "no frame matched a fold: is the tag key right (--tag-key)? "
            f"first tag {rows[0]['tag']!r}"
            if rows
            else "no frames"
        )
    for kind in ("cluster", "cell"):
        these = [r for r in rows if r["kind"] == kind and r["role"] != "trained"]
        if not these:
            continue
        for subset, sel in (
            ("all", these),
            ("held", [r for r in these if r["role"] == "held"]),
        ):
            for rho, err in PAIRS[kind]:
                c, p, n = spearman(
                    [r.get(rho, np.nan) for r in sel], [r.get(err, np.nan) for r in sel]
                )
                summary["spearman"][f"{kind}/{subset}/{rho}~{err}"] = {
                    "rho": c,
                    "p": p,
                    "n": n,
                }

    summary["calibration"] = {}
    for spec in calibrate:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"--calibrate needs NAME:FAIL_LEVEL[:CAPTURE], not {spec!r}"
            )
        name, fail = parts[0], float(parts[1])
        capture = float(parts[2]) if len(parts) == 3 else 0.98
        if name not in EPS:
            raise ValueError(f"cannot calibrate {name!r}; one of {sorted(EPS)}")
        for kind in ("cluster", "cell"):
            these = [
                r
                for r in rows
                if r["kind"] == kind
                and r["role"] != "trained"
                and r.get(name) is not None
                and r.get(EPS[name]) is not None
            ]
            if not these:
                continue
            rho = np.array([r[name] for r in these])
            eps = np.array([r[EPS[name]] for r in these])
            n_fail = int((eps > fail).sum())
            entry = {
                "fail_level": fail,
                "capture": capture,
                "error": EPS[name],
                "n": len(these),
                "n_fail": n_fail,
            }
            if n_fail == 0:
                entry["threshold"] = None
                entry["warning"] = "no frame fails; nothing to calibrate"
            else:
                thr = calibrate_threshold(rho, eps, fail, capture)
                entry["threshold"] = thr
                entry.update(split_statistics(rho, eps, thr))
                if n_fail < 10:
                    entry["warning"] = (
                        f"only {n_fail} failures: the threshold is not to be trusted"
                    )
            summary["calibration"][f"{kind}/{name}"] = entry
    if summary_json is not None:
        Path(summary_json).write_text(json.dumps(summary, indent=1) + "\n")
    return rows, summary


#: The spreads compared with and without each member.
LOO_RHOS = (
    "rho_energy",
    "rho_forces",
    "rho_forces_max_atom",
    "rho_pressure",
    "rho_pressure_molecular",
)


def _spreads(E, F, n, P=None, Pm=None):
    """The committee spreads of one frame from the members' predictions (numpy)."""
    sd_atom = np.sqrt(F.var(axis=0).sum(-1))  # (N,)
    out = {
        "rho_energy": float(E.std() / math.sqrt(n)),
        "rho_forces": float(np.sqrt(np.mean(sd_atom**2))),
        "rho_forces_max_atom": float(sd_atom.max()),
    }
    if P is not None:
        out["rho_pressure"] = float(P.std())
        out["rho_pressure_molecular"] = float(Pm.std())
    return out


def _leave_one_out(E, F, n, P=None, Pm=None):
    """For each member, the ratio of each spread with it to the spread without it."""
    full = _spreads(E, F, n, P, Pm)
    L = len(E)
    ratios = {name: np.full(L, np.nan) for name in full}
    for k in range(L):
        keep = [j for j in range(L) if j != k]
        part = _spreads(
            E[keep],
            F[keep],
            n,
            None if P is None else P[keep],
            None if Pm is None else Pm[keep],
        )
        for name, value in full.items():
            if part[name] > 0:
                ratios[name][k] = value / part[name]
    return ratios


def member_losses(mdir):
    """A member's final and best validation loss, from its history if there is one.

    Reads ``history.json`` (``seamm-al train``'s, with ``val`` a list of metric
    dicts, or a list of numbers as the water campaign's ``train_member.py``
    writes) or, failing that, any ``*history*.json`` in the member directory.

    Returns
    -------
    (float or None, float or None, str or None)
        The final and the best validation loss and the file read.
    """
    mdir = Path(mdir)
    candidates = [mdir / "history.json"] + sorted(mdir.glob("*history*.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            hist = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        val = hist.get("val") if isinstance(hist, dict) else None
        if not val:
            continue
        losses = []
        for v in val:
            if isinstance(v, dict):
                v = v.get("loss")
            if v is not None and np.isfinite(float(v)):
                losses.append(float(v))
        if losses:
            return losses[-1], min(losses), str(path)
    return None, None, None


#: The member-health metrics, each compared with the median of the other members.
HEALTH = (
    "final_val_loss",
    "best_val_loss",
    "held_err_forces",
    "held_err_pressure_molecular",
    "common_err_forces",
    "common_err_pressure_molecular",
)


def member_health(directory, names, rows, factor=1.5):
    """Per member: losses, errors on its held-out and on the common frames, flags.

    Parameters
    ----------
    directory : str
        The committee directory.
    names : sequence of int
        The kept members' ``k``, in the committee's order.
    rows : list of dict
        The frames of :func:`check_committee`, with their per-member errors.
    factor : float, optional
        Flag a metric more than ``factor`` times the median of the other members.

    Returns
    -------
    list of dict
        One entry per member: the metrics of :data:`HEALTH`, the counts, the
        median leave-one-out spread ratios (``loo_<rho>``: the spread with the
        member over the spread without it; well above 1 means the member inflates
        the spreads), and ``flags``.
    """
    dirs = {int(p.name[1:]): p for p in member_dirs(directory)}
    L = len(names)
    out = []
    for i, k in enumerate(names):
        final, best, source = member_losses(dirs[k])
        entry = {
            "member": k,
            "final_val_loss": final,
            "best_val_loss": best,
            "history": source,
        }
        held = [r for r in rows if r["role"] == "held" and r["oof_members"] == str(i)]
        common = [r for r in rows if r["role"] == "unseen"]
        entry["n_held"], entry["n_common"] = len(held), len(common)
        for prefix, sel in (("held", held), ("common", common)):
            for err in ("err_forces", "err_pressure_molecular"):
                values = [r["_member"][err][i] for r in sel if err in r["_member"]]
                entry[f"{prefix}_{err}"] = float(np.mean(values)) if values else None
        for name in LOO_RHOS:
            ratios = [
                r["_loo"][name][i] for r in rows if "_loo" in r and name in r["_loo"]
            ]
            ratios = [x for x in ratios if np.isfinite(x)]
            entry[f"loo_{name}"] = float(np.median(ratios)) if ratios else None
        out.append(entry)
    for entry in out:
        flags = []
        for metric in HEALTH:
            mine = entry[metric]
            others = [
                e[metric] for e in out if e is not entry and e[metric] is not None
            ]
            if mine is None or not others:
                continue
            median = float(np.median(others))
            if median > 0 and mine > factor * median:
                flags.append(f"{metric} {mine:.4g} > {factor:g} x median {median:.4g}")
        entry["flags"] = flags
    if L < 3:
        logger.info("fewer than 3 members: no leave-one-out spreads")
    return out


def _write_health(path, health):
    columns = ["member", *HEALTH, "n_held", "n_common"]
    columns += [f"loo_{name}" for name in LOO_RHOS] + ["flags", "history"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for e in health:
            writer.writerow(
                {
                    c: ("; ".join(e[c]) if c == "flags" else _cell(e.get(c)))
                    for c in columns
                }
            )


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def print_summary(summary, fh=None):
    """Print the correlations and calibrations of :func:`check_committee`."""
    import sys

    fh = sys.stdout if fh is None else fh
    roles = summary["roles"]
    print(
        f"{len(summary['members'])} members; frames: {roles['held']} held out, "
        f"{roles['unseen']} unseen by every member, {roles['trained']} trained on "
        "by every member (left out)",
        file=fh,
    )
    print(f"\n{'Spearman':58s} {'rho':>6s} {'p':>9s} {'n':>5s}", file=fh)
    for key, v in summary["spearman"].items():
        print(f"{key:58s} {v['rho']:+6.2f} {v['p']:9.2g} {v['n']:5d}", file=fh)
    if summary.get("dropped"):
        print(
            f"dropped members: {', '.join(f'm{k}' for k in summary['dropped'])}",
            file=fh,
        )
    health = summary.get("member_health", [])
    if health:
        print(
            "\nmembers (errors: forces eV/A, molecular pressure atm; loo = spread "
            "with / without the member, median over frames)",
            file=fh,
        )
        print(
            f"{'':4s} {'val final':>10s} {'val best':>10s} {'held F':>8s} "
            f"{'held Pm':>8s} {'common F':>9s} {'common Pm':>9s} "
            f"{'loo rhoF':>8s} {'loo rhoPm':>9s}",
            file=fh,
        )

        def f(x, w, fmt):
            return f"{'-':>{w}s}" if x is None else f"{x:{w}{fmt}}"

        for e in health:
            print(
                f"m{e['member']:<3d} {f(e['final_val_loss'], 10, '.4g')} "
                f"{f(e['best_val_loss'], 10, '.4g')} "
                f"{f(e['held_err_forces'], 8, '.4f')} "
                f"{f(e['held_err_pressure_molecular'], 8, '.0f')} "
                f"{f(e['common_err_forces'], 9, '.4f')} "
                f"{f(e['common_err_pressure_molecular'], 9, '.0f')} "
                f"{f(e['loo_rho_forces'], 8, '.3f')} "
                f"{f(e['loo_rho_pressure_molecular'], 9, '.3f')}",
                file=fh,
            )
        for e in health:
            for flag in e["flags"]:
                print(f"  FLAG m{e['member']}: {flag}", file=fh)
    for key, v in summary.get("calibration", {}).items():
        print(
            f"\ncalibration {key}: {v['error']} > {v['fail_level']:g}, "
            f"capture {v['capture']:g}: {v['n_fail']} of {v['n']} fail",
            file=fh,
        )
        if v.get("threshold") is not None:
            print(
                f"  threshold {v['threshold']:.4g}: selects "
                f"{v['fraction_selected']:.1%} ({v['n_above']} above, "
                f"{v['n_below']} below); RMS error above {v['rmse_above']:.4g}, "
                f"below {v['rmse_below']:.4g}",
                file=fh,
            )
        if v.get("warning"):
            print(f"  WARNING: {v['warning']}", file=fh)
