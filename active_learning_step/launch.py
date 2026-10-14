# -*- coding: utf-8 -*-

"""Launching the committee-mean MDI engine from SEAMM.

A committee is offered as one model chemistry by the xnn plug-in, which runs it with
the command built here: ``seamm-al engine`` (:mod:`active_learning_step.engine`) in
the MLFF environment, serving the mean of the members and writing their spreads.

The engine runs in the MLFF environment, which has torch and xnn but need not have
this package. A copy of the package's modules, made once per version of the code
in ``~/.seamm.d/active_learning/code/``, is put on that environment's
``PYTHONPATH`` by the command itself, so nothing is installed there.
"""

import logging
import os
from pathlib import Path
import shutil
import tempfile

from .train import source_sha256

logger = logging.getLogger(__name__)

__all__ = ["code_directory", "engine_command", "SPREADS_FILE"]

#: Where the engine writes the spreads, in its working directory
SPREADS_FILE = "committee_spreads.csv"


def code_directory(cache=None):
    """A directory holding a copy of this package, for ``PYTHONPATH``.

    Parameters
    ----------
    cache : str or pathlib.Path, optional
        Where the copies are kept, by default ``~/.seamm.d/active_learning/code``.

    Returns
    -------
    pathlib.Path
        The directory that holds ``active_learning_step/``. It is named after the
        hash of the sources, so a new version of the code gets a new copy and a
        running engine's copy is never changed.
    """
    if cache is None:
        cache = Path("~/.seamm.d/active_learning/code").expanduser()
    cache = Path(cache)
    target = cache / source_sha256()[:16]
    if (target / "active_learning_step" / "__init__.py").exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(dir=cache, prefix=".tmp-"))
    try:
        shutil.copytree(
            package,
            tmp / "active_learning_step",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        try:
            os.rename(tmp, target)
        except OSError:
            # made meanwhile by another job
            if not (target / "active_learning_step" / "__init__.py").exists():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


def engine_command(
    python,
    members,
    mdi,
    *,
    device="cpu",
    charge=0,
    extra_args=(),
    spreads=SPREADS_FILE,
    every=10,
    cache=None,
):
    """The argv of the committee-mean MDI engine.

    Parameters
    ----------
    python : str
        The MLFF environment's Python (``python`` when the environment is active).
    members : [str or pathlib.Path]
        The members' checkpoints, or their directories.
    mdi : str
        The ``-mdi`` options, e.g. ``"-role ENGINE -name XNN -method MPI"``.
    device : str
        The torch device.
    charge : float
        The total charge, for models whose dispersion needs the EEQ charges.
    extra_args : [str]
        More engine options, e.g. ``["--dtype", "float32", "--eeq-reuse"]``.
    spreads : str or None
        The CSV the spreads are written to, every ``every`` steps; ``None`` for
        none.
    every : int
        How often to write the spreads, in calls of the engine (MD steps).
    cache : str or pathlib.Path, optional
        Where the copy of the package is kept (see :func:`code_directory`).

    Returns
    -------
    [str]
        Starting with ``env PYTHONPATH=...``, so that it works as a command, under
        ``conda run`` and in an ``mpirun`` MPMD line alike.
    """
    code = code_directory(cache)
    argv = [
        "env",
        f"PYTHONPATH={code}",
        "PYTHONDONTWRITEBYTECODE=1",
        python,
        "-m",
        "active_learning_step.cli",
        "engine",
        "--members",
        *(str(m) for m in members),
        "--device",
        device,
    ]
    if charge:
        argv += ["--total-charge", f"{float(charge):g}"]
    if spreads:
        argv += ["--spreads", str(spreads), "--every", str(int(every))]
    argv += [str(a) for a in extra_args]
    argv += ["-mdi", mdi]
    return argv
