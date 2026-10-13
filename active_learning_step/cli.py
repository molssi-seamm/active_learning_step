# -*- coding: utf-8 -*-

"""The ``seamm-al`` command: query-by-committee active learning from the shell.

    seamm-al score -m m0/best.pt m1/best.pt ... [-o scores.csv] pool.extxyz [...]
    seamm-al train -c recipe.yaml -o committee labelled.extxyz [--cell-share 0.15]
    seamm-al select scores.csv --budget 75000 \\
        --cost cell:EC=2800 --cost cell:FEC=3900 \\
        --group-column liquid --cap temperature=6 --exclude labelled.txt

``score``, ``train``, ``check`` and ``engine`` need torch and xnn: use SEAMM's MLFF
environment (xnn 0.13 or later; training requires it). ``select`` needs neither.
Without installing the package, run ``python -m active_learning_step.cli`` with the
repository on ``PYTHONPATH``.
"""

import argparse
import logging
import sys


def _bool(text):
    value = text.strip().lower()
    if value in ("true", "yes", "1", "on"):
        return True
    if value in ("false", "no", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, not {text!r}")


def parser():
    """The argument parser of ``seamm-al``."""
    p = argparse.ArgumentParser(
        prog="seamm-al",
        description=(
            "Query-by-committee active learning: score candidate pools and select "
            "frames."
        ),
    )
    p.add_argument(
        "--log-level", default="WARNING", help="logging level (default WARNING)"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "engine",
        help="MDI engine serving a committee mean, writing its spreads "
        "(see 'seamm-al engine --help')",
        add_help=False,
    )

    s = sub.add_parser(
        "score", help="committee spreads for every frame of candidate pools"
    )
    s.add_argument("pools", nargs="+", help="pool files: .extxyz, or .sdf from SEAMM")
    s.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=None,
        help="the committee members' checkpoints (best.pt or model directories)",
    )
    s.add_argument(
        "--drop-member",
        type=int,
        action="append",
        default=[],
        metavar="K",
        help="leave out the K-th model of --models, counting from 0 (repeatable)",
    )
    s.add_argument(
        "--no-models",
        action="store_true",
        help="no committee: list the pool's frames for a random draw (select --random)",
    )
    s.add_argument(
        "-o", "--out", default="scores.csv", help="the scores (default scores.csv)"
    )
    s.add_argument(
        "--atoms",
        default=None,
        help="also write every frame with its per-atom force spread to this extxyz",
    )
    s.add_argument(
        "--carry",
        action="append",
        default=[],
        help="an info key to copy into a column (repeatable)",
    )
    s.add_argument(
        "--dispersion",
        type=_bool,
        default=False,
        help="add the recorded dispersion back (default false: route B, as Sina)",
    )
    s.add_argument("--device", default="cpu", help="torch device (default cpu)")
    s.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "float64"),
        help="serve the models in this dtype (default: the weights')",
    )
    s.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="clusters per batch (default 16); cells are scored one at a time",
    )

    r = sub.add_parser("train", help="train the members of a k-fold committee")
    r.add_argument("data", nargs="+", help="labelled data files (.extxyz)")
    r.add_argument("-c", "--config", required=True, help="the xnn YAML config (recipe)")
    r.add_argument("-o", "--out", required=True, help="the committee's directory")
    r.add_argument(
        "-k", "--k", type=int, default=5, help="folds and members (default 5)"
    )
    r.add_argument(
        "--member",
        type=int,
        action="append",
        default=None,
        help="train only this member (repeatable; default all)",
    )
    r.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="seed of the fold permutation (default 0)",
    )
    r.add_argument(
        "--stratify",
        choices=("kind", "none"),
        default="kind",
        help="split clusters and cells separately (default), or plain k-fold",
    )
    r.add_argument(
        "--stratify-column",
        action="append",
        default=[],
        help="also stratify by this info key, e.g. liquid (repeatable)",
    )
    r.add_argument(
        "--init-seed",
        type=int,
        default=1000,
        help="member l uses seed INIT_SEED + l (default 1000)",
    )
    r.add_argument(
        "--cell-share",
        type=float,
        default=None,
        help="cell batches as this share of every epoch (default one pass)",
    )
    r.add_argument(
        "--cell-batch-size",
        type=int,
        default=4,
        help="cells per batch (default 4); clusters use data.batch_size",
    )
    r.add_argument(
        "--steps",
        type=int,
        default=None,
        help="optimiser steps per member, instead of the config's epochs",
    )
    r.add_argument("--device", default=None, help="override the config's device")
    r.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="xnn config override, e.g. optim.lr=0.005",
    )

    c = sub.add_parser("check", help="does the committee's spread predict its error?")
    c.add_argument(
        "committee", help="the committee directory (seamm-al train, or m<k>/)"
    )
    c.add_argument("data", nargs="+", help="labelled extxyz files")
    c.add_argument("-o", "--out", default="check.csv", help="per-frame CSV")
    c.add_argument("--summary", default=None, help="also write the summary as JSON")
    c.add_argument(
        "--tag-key",
        default=None,
        help="info key of the tags in the fold lists (default: configuration name)",
    )
    c.add_argument("--energy-key", default="REF_energy")
    c.add_argument("--forces-key", default="REF_forces")
    c.add_argument("--stress-key", default="REF_stress")
    c.add_argument(
        "--dispersion",
        type=_bool,
        default=False,
        help="add the recorded dispersion (for labels that include it; default false)",
    )
    c.add_argument("--checkpoint", default="best.pt", help="member checkpoint file")
    c.add_argument("--device", default="cpu")
    c.add_argument("--dtype", default=None, choices=("float32", "float64"))
    c.add_argument(
        "--drop-member",
        type=int,
        action="append",
        default=[],
        metavar="K",
        help="leave member m<K> out of the committee (repeatable)",
    )
    c.add_argument(
        "--flag-factor",
        type=float,
        default=1.5,
        help="flag a member worse than this times the others' median (default 1.5)",
    )
    c.add_argument(
        "--members-out",
        default=None,
        help="the member-health CSV (default <out stem>_members.csv)",
    )
    c.add_argument(
        "--calibrate",
        action="append",
        default=[],
        help="NAME:FAIL_LEVEL[:CAPTURE], e.g. rho_forces:0.15 (repeatable)",
    )

    t = sub.add_parser(
        "select", help="choose frames from scores within a core-hour budget"
    )
    t.add_argument("scores", nargs="+", help="scores.csv files from 'seamm-al score'")
    t.add_argument("--budget", type=float, default=None, help="core-hours available")
    t.add_argument(
        "--cost",
        action="append",
        default=[],
        help="core-hours per frame: KIND=COREH or KIND:GROUP=COREH (repeatable)",
    )
    t.add_argument(
        "--group-column",
        default=None,
        help="the column whose values the GROUP of --cost matches (shell patterns)",
    )
    t.add_argument(
        "--rank",
        action="append",
        default=[],
        help="ranking columns of a kind, KIND=COL[,COL...] (default "
        "cell=rho_pressure_molecular,rho_forces and cluster=rho_forces)",
    )
    t.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="select spreads >= a threshold: NAME=VALUE or KIND:NAME=VALUE",
    )
    t.add_argument(
        "--combine",
        choices=("any", "all"),
        default="any",
        help="with several thresholds, any (default, Sina's rule) or all must be met",
    )
    t.add_argument(
        "--cap",
        action="append",
        default=[],
        help="at most N frames per value of a column: COLUMN=N (repeatable)",
    )
    t.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="a file of configuration names already labelled, one per line",
    )
    t.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="SEED",
        help="draw frames at random with this seed instead of ranking (round 0)",
    )
    t.add_argument(
        "--stratify-column",
        action="append",
        default=[],
        help="with --random, spread the draw evenly over this column (repeatable)",
    )
    t.add_argument(
        "--fill",
        action="store_true",
        help="skip frames that do not fit the budget and keep going, rather than stop",
    )
    t.add_argument(
        "--max-frames", type=int, default=None, help="select at most this many frames"
    )
    t.add_argument(
        "-o",
        "--out",
        default="selection.csv",
        help="the selection (default selection.csv)",
    )
    t.add_argument(
        "--output",
        default="selected",
        help="prefix of the selected frames' files (default 'selected')",
    )
    t.add_argument(
        "--rejected", default=None, help="also write the frames not selected, with why"
    )
    return p


def main(argv=None):
    """Run ``seamm-al``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["engine"]:
        # the MDI engine has its own options (``seamm-al engine --help``)
        from .engine import main as engine_main

        return engine_main(argv[1:])
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "score":
        from .score import score_pool

        if bool(args.models) == args.no_models:
            print("seamm-al score: give --models or --no-models", file=sys.stderr)
            return 2
        models = None if args.no_models else list(args.models)
        if args.drop_member:
            if models is None or any(
                not 0 <= k < len(models) for k in args.drop_member
            ):
                print(
                    "seamm-al score: --drop-member K needs 0 <= K < the number of "
                    "--models",
                    file=sys.stderr,
                )
                return 2
            models = [m for k, m in enumerate(models) if k not in set(args.drop_member)]
        n = score_pool(
            args.pools,
            models,
            out_csv=args.out,
            atoms_out=args.atoms,
            carry=args.carry,
            dispersion=args.dispersion,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            all_models=None if args.no_models else list(args.models),
            dropped=args.drop_member,
        )
        print(f"Scored {n} frames -> {args.out}")
    elif args.command == "check":
        from .check import check_committee, print_summary

        try:
            _, summary = check_committee(
                args.committee,
                args.data,
                out_csv=args.out,
                tag_key=args.tag_key,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                stress_key=args.stress_key,
                dispersion=args.dispersion,
                device=args.device,
                dtype=args.dtype,
                checkpoint=args.checkpoint,
                calibrate=args.calibrate,
                summary_json=args.summary,
                drop=args.drop_member,
                flag_factor=args.flag_factor,
                members_csv=args.members_out,
            )
        except ValueError as e:
            print(f"seamm-al check: {e}", file=sys.stderr)
            return 2
        print_summary(summary)
    elif args.command == "train":
        from .train import train_committee

        try:
            train_committee(
                args.config,
                args.data,
                args.out,
                k=args.k,
                members=args.member,
                data_seed=args.data_seed,
                init_seed=args.init_seed,
                cell_share=args.cell_share,
                cell_batch_size=args.cell_batch_size,
                steps=args.steps,
                device=args.device,
                overrides=args.overrides,
                stratify=args.stratify,
                stratify_columns=args.stratify_column,
            )
        except ValueError as e:
            print(f"seamm-al train: {e}", file=sys.stderr)
            return 2
    else:
        from .selection import select_main

        try:
            select_main(args)
        except ValueError as e:
            print(f"seamm-al select: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
