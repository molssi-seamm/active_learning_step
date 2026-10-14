# -*- coding: utf-8 -*-

"""The Active Learning step's run path, end to end.

Flowcharts built from specs are run by ``run_flowchart``, through each sub-step's
``run()``: the parameters, the xnn.ini of the MLFF environment, ``seamm-al`` run
there with this package on its ``PYTHONPATH``, and the analysis of its output.

The "MLFF environment" is this Python: listing a pool (``--no-models``) and
selecting need no torch. Training, checking and scoring with a committee do, so a
stand-in for ``seamm-al`` writes what they would, from the arguments it is given.
"""

import csv
from pathlib import Path
import sys
import textwrap

import pytest

pytest.importorskip("seamm_exec.testing")
from seamm_exec.testing import run_spec  # noqa: E402

SOURCE = Path(__file__).resolve().parents[1]

#: Stands in for ``python -m active_learning_step.cli`` where torch is needed
FAKE_AL = textwrap.dedent("""
    import json
    from pathlib import Path
    import sys

    args = sys.argv[1:]
    command = args[0]
    out = Path(args[args.index("--out") + 1])
    Path("fake.log").open("a").write(" ".join(args) + "\\n")

    if command == "train":
        k = int(args[args.index("--k") + 1])
        data = [a for a in args[1:] if a.endswith(".extxyz")]
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "k": k,
            "n_frames": 10,
            "kinds": {"cluster": 10},
            "data": [{"path": p, "n_frames": 10} for p in data],
        }
        (out / "manifest.json").write_text(json.dumps(manifest))
        for m in range(k):
            mdir = out / f"m{m}"
            mdir.mkdir(exist_ok=True)
            (mdir / "best.pt").write_text("")
            (mdir / "member.json").write_text(
                json.dumps({"n_train": 8, "n_val": 2, "epochs": 3})
            )
            (mdir / "history.json").write_text(
                json.dumps({"val": [1.0 + m, 0.5 + m, 0.4 + m]})
            )
    elif command == "check":
        health = [
            {
                "member": m,
                "final_val_loss": 0.4 + m,
                "best_val_loss": 0.4 + m,
                "held_err_forces": 0.1,
                "held_err_pressure_molecular": None,
                "common_err_forces": None,
                "common_err_pressure_molecular": None,
                "loo_rho_forces": 1.0,
                "loo_rho_pressure_molecular": None,
                "flags": ["final_val_loss"] if m == 4 else [],
                "notes": [],
            }
            for m in range(5)
        ]
        summary = {
            "members": [f"m{m}" for m in range(5)],
            "roles": {"held": 10, "unseen": 0, "trained": 0},
            "spearman": {
                "cluster/held/rho_forces~err_forces": {"rho": 0.8, "p": 0.01, "n": 10}
            },
            "dropped": [],
            "member_health": health,
            "calibration": {
                "cluster/rho_forces": {
                    "error": "err_forces",
                    "fail_level": 0.15,
                    "capture": 0.98,
                    "n": 10,
                    "n_fail": 3,
                    "threshold": 0.25,
                    "fraction_selected": 0.3,
                    "n_above": 3,
                    "n_below": 7,
                    "rmse_above": 0.2,
                    "rmse_below": 0.05,
                }
            },
        }
        Path(args[args.index("--summary") + 1]).write_text(json.dumps(summary))
        out.write_text("frame\\n")
    elif command == "score":
        # the frames, as --no-models lists them, with made-up spreads
        from active_learning_step.score import score_pool

        pools = args[1 : args.index("--out")]
        carry = [args[i + 1] for i, a in enumerate(args) if a == "--carry"]
        score_pool(pools, None, out_csv=str(out), carry=carry)
        rows = list(csv.DictReader(out.open()))
        for i, row in enumerate(rows):
            row["rho_forces"] = f"{0.1 * i:.2f}"
            row["rho_energy"] = f"{0.01 * i:.3f}"
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    """).replace("import json\n", "import csv\nimport json\n", 1)


def _mlff_env(tmp_path, fake=False):
    """A bin directory standing in for the MLFF environment, and its xnn.ini.

    Its ``python`` is this one. With ``fake``, ``seamm-al`` commands that need
    torch go to :data:`FAKE_AL` instead.
    """
    bin_dir = tmp_path / "mlff" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    if fake:
        (bin_dir / "fake_al.py").write_text(FAKE_AL)
        python.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                case " $* " in
                  *" --no-models "*) ;;
                  *" -m active_learning_step.cli "*)
                    shift 2
                    exec "{sys.executable}" "{bin_dir / 'fake_al.py'}" "$@";;
                esac
                exec "{sys.executable}" "$@"
                """))
    else:
        # a wrapper, not a link: a link to a venv's python loses the venv
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    return {
        "xnn": {"code": str(bin_dir / "xnn"), "device": "cpu", "models": "personal:xnn"}
    }


def _write_pool(path, n):
    """An extxyz pool of n water molecules, each with a name; gzipped for .gz."""
    import gzip

    import ase
    from ase.io import write

    frames = []
    for i in range(n):
        atoms = ase.Atoms(
            "OH2",
            positions=[[0, 0, 0], [0.757 + 0.01 * i, 0.586, 0], [-0.757, 0.586, 0]],
        )
        atoms.info["SEAMM/configuration_name"] = f"water {i}"
        frames.append(atoms)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as fh:
            write(fh, frames, format="extxyz")
    else:
        write(str(path), frames, format="extxyz")
    return path


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _records(path):
    from active_learning_step.pool import iter_records

    return list(iter_records(path))


def _text(job, step):
    """The job's output and that of the step's sub-steps."""
    out = (job / "job.out").read_text()
    return out + "".join(p.read_text() for p in sorted(job.glob(f"{step}/*/step.out")))


def test_random_draw_from_files(tmp_path):
    """Round 0: list a pool (one file gzipped) and draw from it at random."""
    inis = _mlff_env(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_pool(inputs / "a.extxyz", 6)
    _write_pool(inputs / "b.extxyz.gz", 4)
    spec = f"""
    title: Random draw
    steps:
    - Active Learning:
        steps:
        - Score Pool:
            committee: none
            pool: files
            pool files: "{inputs}/*.extxyz {inputs}/*.gz"
        - Select Frames: {{method: random, maximum number: 3}}
    """
    job = run_spec(tmp_path, spec, inis=inis, source=SOURCE)
    assert len(_rows(next(job.glob("1/*/scores.csv")))) == 10
    assert len(_records(next(job.glob("1/*/selected.extxyz")))) == 3
    assert "Selected 3 of 10 structures (3 clusters)" in _text(job, 1)


def test_database_pool(tmp_path):
    """Candidates from the database; the selection goes to a variable and a system."""
    inis = _mlff_env(tmp_path)
    spec = """
    title: Database pool
    steps:
    - Water: {}
    - Active Learning:
        steps:
        - Score Pool: {committee: none}
        - Select Frames: {method: random, new system: picked}
    """
    job = run_spec(tmp_path, spec, inis=inis, source=SOURCE)
    scores = _rows(next(job.glob("2/*/scores.csv")))
    assert len(scores) == 1 and scores[0]["configuration_id"] != ""
    text = _text(job, 2)
    assert "'selected_configurations'" in text
    assert "new system 'picked'" in text


def test_train_check_score_select(tmp_path):
    """A round: train, check (calibrating a threshold), score, select above it."""
    inis = _mlff_env(tmp_path, fake=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = _write_pool(inputs / "data.extxyz", 10)
    pool = _write_pool(inputs / "pool.extxyz", 6)
    recipe = inputs / "recipe.yaml"
    recipe.write_text("model: {}\n")
    spec = f"""
    title: A round
    steps:
    - Active Learning:
        steps:
        - Train Committee:
            training data: "{data}"
            recipe: "{recipe}"
            model name: round1
        - Check Committee: {{calibrate: "rho_forces:0.15"}}
        - Score Pool: {{pool: files, pool files: "{pool}"}}
        - Select Frames: {{thresholds: from Check Committee}}
    """
    job = run_spec(tmp_path, spec, inis=inis, source=SOURCE)
    log = "".join(p.read_text() for p in job.glob("1/*/fake.log"))
    assert "train " in log and " --k 5 " in log and "--stratify kind" in log
    # Check Committee takes the training data from the committee's manifest
    assert f"check {job}" in log and str(data) in log.split("check ")[1]
    assert "--calibrate rho_forces:0.15" in log
    # the committee trained above, with its members as the models
    assert "--models " in log and "committee/m0" in log
    # the spreads are 0.0, 0.1, ... 0.5, and the calibrated threshold 0.25
    selection = _rows(next(job.glob("1/*/selection.csv")))
    assert sorted(r["configuration"] for r in selection) == [
        "water 3",
        "water 4",
        "water 5",
    ]
    text = _text(job, 1)
    # saved among the user's xnn models
    saved = tmp_path / "home" / ".seamm.d" / "data" / "Forcefields" / "xnn" / "round1"
    assert sorted(p.name for p in saved.glob("m*/best.pt")) == ["best.pt"] * 5
    assert "xnn:MLFF@round1" in text
    assert "Member 4 is flagged" in text
    assert "Selected 3 of 6 structures (3 clusters)." in text
