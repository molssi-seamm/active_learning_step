# -*- coding: utf-8 -*-

"""``seamm-al select``: choose frames from ``scores.csv`` within a core-hour budget.

The candidates are put in order and taken until the budget is spent:

* **Ranking** (the default). Within each kind, every frame is ranked on each of the
  kind's ranking columns, largest spread first, and the ranks are summed. Cells use
  ``rho_pressure_molecular`` and ``rho_forces``, clusters ``rho_forces``; both can be
  changed (``--rank cell=rho_forces_max_atom``). Kinds are merged by the rank-sum
  divided by its largest possible value, so the best cell and the best cluster come
  first together.
* **Thresholds** (``--threshold``): only frames with a spread at least the threshold
  are candidates (Sina's rule; with several thresholds any one suffices, or all with
  ``--combine all``). The candidates are then ordered by ranking as above.

Frames are taken in that order. ``--cap COLUMN=N`` allows at most ``N`` frames per
value of a column; ``--exclude FILE`` drops configurations already labelled. A frame
whose cost would take the total past ``--budget`` stops the selection, or with
``--fill`` is skipped while cheaper frames further down still fit.

Costs are core-hours per frame, by kind or by group: ``--cost cell=2800`` or
``--cost cell:EC=2800`` with ``--group-column`` naming the column that holds ``EC``.
Groups are shell-style patterns, and the first match wins.

The selected frames are written as a verbatim text subset of the input files:
the records are copied byte for byte, in selection order, one output file per format.
"""

import csv
import fnmatch
import logging

import numpy as np

from .pool import configuration_name, iter_records, pool_format, record_to_atoms
from .thresholds import select as threshold_mask

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RANK",
    "parse_costs",
    "parse_rank",
    "parse_thresholds",
    "parse_caps",
    "read_scores",
    "read_names",
    "choose",
    "write_records",
]

#: The default ranking columns per kind.
DEFAULT_RANK = {
    "cell": ("rho_pressure_molecular", "rho_forces"),
    "cluster": ("rho_forces",),
}


def _split(spec, what):
    if "=" not in spec:
        raise ValueError(f"{what} must look like KEY=VALUE, not {spec!r}")
    key, value = spec.rsplit("=", 1)
    return key.strip(), value.strip()


def parse_costs(specs):
    """``KIND=COREH`` or ``KIND:GROUP=COREH`` -> list of (kind, group, cost)."""
    out = []
    for spec in specs:
        key, value = _split(spec, "--cost")
        kind, _, group = key.partition(":")
        out.append((kind, group or None, float(value)))
    return out


def parse_rank(specs):
    """``KIND=COL[,COL...]`` -> {kind: (cols)}, over the defaults."""
    out = dict(DEFAULT_RANK)
    for spec in specs:
        kind, value = _split(spec, "--rank")
        out[kind] = tuple(c.strip() for c in value.split(",") if c.strip())
    return out


def parse_thresholds(specs):
    """``NAME=VALUE`` or ``KIND:NAME=VALUE`` -> {kind or None: {name: value}}."""
    out = {}
    for spec in specs:
        key, value = _split(spec, "--threshold")
        kind, _, name = key.rpartition(":")
        out.setdefault(kind or None, {})[name] = float(value)
    return out


def parse_caps(specs):
    """``COLUMN=N`` -> {column: N}."""
    out = {}
    for spec in specs:
        key, value = _split(spec, "--cap")
        out[key] = int(value)
    return out


def read_scores(path):
    """The rows of a scores.csv, as dicts of strings, in file order."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_names(path):
    """Configuration names, one per line; blank lines and ``#`` comments ignored."""
    names = set()
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(line)
    return names


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _cost(row, costs, group_column):
    for kind, group, cost in costs:
        if kind not in (row["kind"], "*"):
            continue
        if group is None:
            continue
        if group_column is None:
            raise ValueError(f"the cost {kind}:{group} needs --group-column")
        if group_column not in row:
            raise ValueError(f"no column {group_column!r} in the scores")
        if fnmatch.fnmatchcase(str(row[group_column]), group):
            return cost
    for kind, group, cost in costs:
        if group is None and kind in (row["kind"], "*"):
            return cost
    return None


def _ranks(rows, columns):
    """Rank-sum of rows over columns (1 = largest), with the per-column ranks."""
    n = len(rows)
    per_column = {}
    total = np.zeros(n)
    for column in columns:
        values = np.array([_float(r.get(column)) for r in rows])
        # largest first; nan last; stable for ties
        key = np.where(np.isnan(values), -np.inf, values)
        order = np.argsort(-key, kind="stable")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(1, n + 1)
        per_column[column] = rank
        total += rank
    return total, per_column


def choose(
    rows,
    costs,
    budget=None,
    group_column=None,
    rank=None,
    thresholds=None,
    combine="any",
    caps=None,
    exclude=None,
    fill=False,
    max_frames=None,
    random_seed=None,
    stratify_columns=(),
):
    """Choose frames from scored rows.

    Parameters
    ----------
    rows : list of dict
        The rows of scores.csv.
    costs : list of (kind, group, cost)
        From :func:`parse_costs`.
    budget : float, optional
        Core-hours available; no limit by default.
    group_column : str, optional
        The column the cost groups match.
    rank : dict, optional
        Ranking columns per kind, by default :data:`DEFAULT_RANK`.
    thresholds : dict, optional
        From :func:`parse_thresholds`; turns on threshold mode.
    combine : str, optional
        ``"any"`` (Sina's rule) or ``"all"`` of the thresholds must be met.
    caps : dict, optional
        Most frames per value of a column.
    exclude : set of str, optional
        Configuration names already labelled.
    fill : bool, optional
        Skip a frame that does not fit the budget and go on, rather than stop.
    max_frames : int, optional
        Most frames to select.
    random_seed : int, optional
        Draw the frames in a random order (seeded) instead of ranking them: round
        0, before any committee exists. Thresholds do not apply.
    stratify_columns : sequence of str, optional
        With ``random_seed``: spread the draw evenly over the values of these
        columns, taking one frame from each stratum in turn (the strata in a
        shuffled order, the frames of each shuffled).

    Returns
    -------
    (list of dict, list of dict)
        The selected rows, in order, with ``order``, ``cost``, ``running_cost``
        and ``reason``; and the rejected rows with their ``reason``.
    """
    if random_seed is None and stratify_columns:
        raise ValueError("--stratify-column applies to a random draw (--random)")
    if random_seed is not None and thresholds:
        raise ValueError("a random draw takes no thresholds")
    rank = DEFAULT_RANK if rank is None else rank
    caps = caps or {}
    exclude = exclude or set()
    rejected = []

    candidates = []
    for row in rows:
        if row.get("configuration") in exclude:
            rejected.append({**row, "reason": "excluded: already labelled"})
        else:
            candidates.append(row)

    flagged_by = {}
    if thresholds:
        kept = []
        for kind in sorted({r["kind"] for r in candidates}):
            these = [r for r in candidates if r["kind"] == kind]
            limits = {**thresholds.get(None, {}), **thresholds.get(kind, {})}
            if not limits:
                rejected.extend(
                    {**r, "reason": f"no threshold for {kind}"} for r in these
                )
                continue
            for name in limits:
                if name not in these[0]:
                    raise ValueError(f"no column {name!r} in the scores")
            rhos = {name: np.array([_float(r[name]) for r in these]) for name in limits}
            if combine == "all":
                mask = np.ones(len(these), dtype=bool)
                for name, value in limits.items():
                    mask &= threshold_mask({name: rhos[name]}, {name: value})
            else:
                mask = threshold_mask(rhos, limits)
            for k, r in enumerate(these):
                hits = [
                    f"{name} {rhos[name][k]:.4g} >= {value:.4g}"
                    for name, value in limits.items()
                    if not np.isnan(rhos[name][k]) and rhos[name][k] >= value
                ]
                if mask[k]:
                    flagged_by[id(r)] = "; ".join(hits)
                    kept.append(r)
                else:
                    rejected.append({**r, "reason": "below the thresholds"})
        candidates = kept

    if random_seed is not None:
        ordered = _random_order(candidates, random_seed, list(stratify_columns))
    else:
        ordered = _rank_order(candidates, rank, flagged_by)

    selected = []
    counts = {column: {} for column in caps}
    running = 0.0
    stopped = None
    for row, text in ordered:
        if stopped is not None:
            rejected.append({**row, "reason": stopped})
            continue
        if max_frames is not None and len(selected) >= max_frames:
            stopped = f"after the {max_frames} frames allowed"
            rejected.append({**row, "reason": stopped})
            continue
        capped = None
        for column, limit in caps.items():
            if column not in row:
                raise ValueError(f"no column {column!r} in the scores for --cap")
            if counts[column].get(row[column], 0) >= limit:
                capped = f"cap: {limit} frames with {column}={row[column]}"
                break
        if capped is not None:
            rejected.append({**row, "reason": capped})
            continue
        cost = _cost(row, costs, group_column)
        if cost is None:
            raise ValueError(
                f"no cost for {row['kind']} {row.get('configuration')!r}: "
                f"add --cost {row['kind']}=COREH"
            )
        if budget is not None and running + cost > budget:
            reason = f"over budget: {running:.0f} + {cost:.0f} > {budget:.0f} core-h"
            rejected.append({**row, "cost": cost, "reason": reason})
            if not fill:
                stopped = "over budget"
            continue
        running += cost
        for column in caps:
            counts[column][row[column]] = counts[column].get(row[column], 0) + 1
        reason = text
        selected.append(
            {
                **row,
                "order": len(selected) + 1,
                "cost": cost,
                "running_cost": running,
                "reason": reason,
            }
        )
    return selected, rejected


def _rank_order(candidates, rank, flagged_by):
    """(row, reason) in order: normalised rank-sum within each kind, merged."""
    ordered = []
    for kind in sorted({r["kind"] for r in candidates}):
        these = [r for r in candidates if r["kind"] == kind]
        columns = rank.get(kind)
        if not columns:
            raise ValueError(
                f"no ranking columns for kind {kind!r}; use --rank {kind}=COL"
            )
        for column in columns:
            if column not in these[0]:
                raise ValueError(f"no column {column!r} in the scores")
        total, per_column = _ranks(these, columns)
        worst = len(these) * len(columns)
        # the position of each frame in its kind's order (ties keep file order)
        position = np.empty(len(these), dtype=int)
        position[np.lexsort((np.arange(len(these)), total))] = np.arange(
            1, len(these) + 1
        )
        for k, r in enumerate(these):
            detail = ", ".join(
                f"{c} #{per_column[c][k]} ({_float(r[c]):.4g})" for c in columns
            )
            text = (
                f"{kind} #{position[k]} of {len(these)} "
                f"(rank-sum {int(total[k])}: {detail})"
            )
            if id(r) in flagged_by:
                text = f"{flagged_by[id(r)]}; {text}"
            info = {
                "rank_in_kind": int(position[k]),
                "n_in_kind": len(these),
                "rank_sum": int(total[k]),
            }
            ordered.append((total[k] / worst, total[k], k, kind, {**r, **info}, text))
    ordered.sort(key=lambda t: (t[0], t[1], t[3], t[2]))
    return [(row, text) for _, _, _, _, row, text in ordered]


def _random_order(candidates, seed, columns):
    """(row, reason) in a seeded random order, round-robin over the strata."""
    rng = np.random.default_rng(seed)
    for column in columns:
        if candidates and column not in candidates[0]:
            raise ValueError(
                f"no column {column!r} in the scores for --stratify-column"
            )
    groups = {}
    for row in candidates:
        label = "|".join(str(row[c]) for c in columns)
        groups.setdefault(label, []).append(row)
    labels = sorted(groups)
    for label in labels:
        g = groups[label]
        groups[label] = [g[j] for j in rng.permutation(len(g))]
    labels = [labels[j] for j in rng.permutation(len(labels))]
    ordered, turn = [], 0
    while any(groups[label] for label in labels):
        turn += 1
        for label in labels:
            if groups[label]:
                row = groups[label].pop(0)
                where = f"stratum {label!r} " if columns else ""
                text = f"random draw (seed {seed}): {where}round {turn}"
                info = {"seed": seed, "stratum": label, "draw": len(ordered) + 1}
                ordered.append(({**row, **info}, text))
    return ordered


def write_rows(path, rows, first):
    """Write rows to a CSV with the ``first`` columns leading."""
    columns = list(first)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_records(selected, prefix):
    """Copy the selected frames' records, byte for byte, into ``prefix.<format>``.

    Parameters
    ----------
    selected : list of dict
        Rows with ``file`` and ``index``, in the order to write.
    prefix : str
        Output path without suffix; ``.extxyz`` and/or ``.sdf`` are added.

    Returns
    -------
    dict of str to str
        The file written per format.

    Raises
    ------
    ValueError
        If a file has fewer frames than a row's index, or a record's configuration
        name differs from the one scored (the pool file changed since scoring).
    """
    wanted = {}
    for row in selected:
        wanted.setdefault(row["file"], set()).add(int(row["index"]))
    records = {}
    for path, indices in wanted.items():
        last = max(indices)
        for index, record in enumerate(iter_records(path)):
            if index in indices:
                records[(path, index)] = record
            if index >= last:
                break
        missing = sorted(i for i in indices if (path, i) not in records)
        if missing:
            raise ValueError(f"{path} has no frames {missing[:5]}")

    out = {}
    handles = {}
    try:
        for row in selected:
            path, index = row["file"], int(row["index"])
            fmt = pool_format(path)
            record = records[(path, index)]
            name = configuration_name(record_to_atoms(record, fmt))
            if name != row.get("configuration", name):
                raise ValueError(
                    f"frame {index} of {path} is {name!r}, but "
                    f"{row['configuration']!r} was scored"
                )
            if fmt not in handles:
                out[fmt] = f"{prefix}.{fmt}"
                handles[fmt] = open(out[fmt], "wb")
            if not record.endswith(b"\n"):
                # only the last record of a file can lack its newline
                record += b"\n"
            handles[fmt].write(record)
    finally:
        for fh in handles.values():
            fh.close()
    return out


def select_main(args):
    """Run ``seamm-al select`` from parsed arguments."""
    rows = []
    for path in args.scores:
        rows.extend(read_scores(path))
    exclude = set()
    for path in args.exclude or []:
        exclude |= read_names(path)
    selected, rejected = choose(
        rows,
        costs=parse_costs(args.cost or []),
        budget=args.budget,
        group_column=args.group_column,
        rank=parse_rank(args.rank or []),
        thresholds=parse_thresholds(args.threshold or []) or None,
        combine=args.combine,
        caps=parse_caps(args.cap or []),
        exclude=exclude,
        fill=args.fill,
        max_frames=args.max_frames,
        random_seed=args.random,
        stratify_columns=args.stratify_column,
    )
    drawn = ("draw", "seed", "stratum") if args.random is not None else ()
    ranked = ("rank_in_kind", "n_in_kind", "rank_sum") if args.random is None else ()
    lead = (
        "order",
        *drawn,
        *ranked,
        "file",
        "index",
        "configuration",
        "kind",
        "n_atoms",
        "formula",
        "cost",
        "running_cost",
        "reason",
    )
    write_rows(args.out, selected, lead)
    if args.rejected:
        write_rows(
            args.rejected,
            rejected,
            (
                "rank_in_kind",
                "n_in_kind",
                "file",
                "index",
                "configuration",
                "kind",
                "cost",
                "reason",
            ),
        )
    files = write_records(selected, args.output) if selected else {}
    total = selected[-1]["running_cost"] if selected else 0.0
    kinds = {}
    for row in selected:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(
        f"Selected {len(selected)} of {len(rows)} frames {kinds}, {total:.1f} core-h"
        + (f" of {args.budget:.1f}" if args.budget is not None else "")
    )
    cut = next((r for r in rejected if r["reason"].startswith("over budget:")), None)
    if cut is not None:
        print(
            f"Budget cut at {cut['kind']} {cut['configuration']!r}, rank "
            f"{cut['rank_in_kind']} of {cut['n_in_kind']} in its kind: {cut['reason']}"
        )
    reasons = {}
    for row in rejected:
        key = row["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print(
            "Not selected: " + ", ".join(f"{v} {k}" for k, v in sorted(reasons.items()))
        )
    print(f"Wrote {args.out}" + "".join(f", {f}" for f in files.values()))
    return selected
