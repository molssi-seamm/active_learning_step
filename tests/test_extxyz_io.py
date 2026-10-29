# -*- coding: utf-8 -*-

"""ASE's extended XYZ files with names holding '=', spaces, quotes or
backslashes read back as written (mlff's SEAMM job 5433, 2026-10-29)."""

import io

import pytest

from active_learning_step.extxyz_io import quoted_extxyz

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402

NAMES = ["LiDMCct_chel_ctmin_toO=", "x=y", "two words", 'a "quoted" one', "back\\slash"]


def _round_trip(name):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.info["SEAMM/configuration_name"] = name
    atoms.info["total_charge"] = 0.0
    buffer = io.StringIO()
    write(buffer, atoms, format="extxyz")
    return read(io.StringIO(buffer.getvalue()), format="extxyz").info


@pytest.mark.parametrize("name", NAMES)
def test_names_read_back(name):
    with quoted_extxyz():
        info = _round_trip(name)
    assert info["SEAMM/configuration_name"] == name
    assert info["total_charge"] == 0.0


def test_ase_alone_and_restored():
    """Without the context ASE merges the next key into a name ending in '='; the
    context restores ASE's own quoting afterwards."""
    import ase.io.extxyz as extxyz

    original = extxyz.escape
    with quoted_extxyz():
        assert extxyz.escape is not original
    assert extxyz.escape is original
    assert _round_trip("a_toO=")["SEAMM/configuration_name"] != "a_toO="
