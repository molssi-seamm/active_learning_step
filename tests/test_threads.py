#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The CPU thread count: requested, else TORCH_NUM_THREADS, else OMP_NUM_THREADS, else
the affinity mask."""

import pytest

from active_learning_step.cli import parser
from active_learning_step.threads import resolve_threads


def test_requested_wins():
    assert resolve_threads(3, {"OMP_NUM_THREADS": "8"}, 16) == (3, "requested")
    with pytest.raises(ValueError):
        resolve_threads(0)


def test_omp_num_threads_as_seamm_and_pbs_set_it():
    assert resolve_threads(None, {"OMP_NUM_THREADS": "6"}, 64) == (6, "OMP_NUM_THREADS")
    # a nested list ("4,2") gives the outer level
    assert resolve_threads(None, {"OMP_NUM_THREADS": "4,2"}, 64) == (
        4,
        "OMP_NUM_THREADS",
    )


def test_torch_num_threads_before_omp_num_threads():
    # mdi_bind.sh beside a GPU: OMP_NUM_THREADS=1 for everything, 4 for torch
    env = {"OMP_NUM_THREADS": "1", "TORCH_NUM_THREADS": "4"}
    assert resolve_threads(None, env, 64) == (4, "TORCH_NUM_THREADS")
    assert resolve_threads(2, env, 64) == (2, "requested")
    # an unusable TORCH_NUM_THREADS falls through to OMP_NUM_THREADS
    env = {"OMP_NUM_THREADS": "3", "TORCH_NUM_THREADS": "0"}
    assert resolve_threads(None, env, 64) == (3, "OMP_NUM_THREADS")


def test_otherwise_the_cpus_this_process_may_use():
    # no scheduler variables are consulted: Slurm's or PBS's are ignored
    env = {"SLURM_CPUS_PER_TASK": "12", "NCPUS": "12", "OMP_NUM_THREADS": ""}
    assert resolve_threads(None, env, 5) == (5, "affinity")
    assert resolve_threads(None, {"OMP_NUM_THREADS": "junk"}, 2) == (2, "affinity")
    n, source = resolve_threads(None, {})
    assert n >= 1 and source == "affinity"


@pytest.mark.parametrize("command", ["score", "check", "train"])
def test_the_commands_take_threads(command):
    sub = [a for a in parser()._subparsers._group_actions][0].choices[command]
    assert "--threads" in sub._option_string_actions
