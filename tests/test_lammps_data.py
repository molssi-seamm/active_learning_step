#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LAMMPS data files for the engine: sections atom_style atomic cannot read."""

from active_learning_step.cli import main
from active_learning_step.lammps_data import check_data, strip_data

GOOD = """LAMMPS data file via write_data, units = metal

6 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.9995
2 1.008

Atoms # atomic

1 1 1.0 1.0 1.0
2 2 1.9 1.3 1.0
3 2 0.4 1.8 1.0
4 1 5.0 5.0 5.0
5 2 5.9 5.3 5.0
6 2 4.4 5.8 5.0

Velocities

1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 0.0 0.0 0.0
4 0.0 0.0 0.0
5 0.0 0.0 0.0
6 0.0 0.0 0.0
"""

# as water512.data: written with an extra Molecules section (and bonds from a
# molecular style)
BAD = GOOD.replace("2 atom types\n", "2 atom types\n4 bonds\n1 bond types\n") + """
Molecules

1 1
2 1
3 1
4 2
5 2
6 2

Bonds

1 1 1 2
2 1 1 3
3 1 4 5
4 1 4 6
"""


def test_a_readable_file(tmp_path):
    path = tmp_path / "good.data"
    path.write_text(GOOD)
    found, problems = check_data(path)
    assert found == ["Masses", "Atoms", "Velocities"] and problems == []
    assert main(["data", str(path)]) == 0


def test_molecules_and_bonds_are_found_and_stripped(tmp_path, capsys):
    path = tmp_path / "bad.data"
    path.write_text(BAD)
    found, problems = check_data(path)
    assert found == ["Masses", "Atoms", "Velocities", "Molecules", "Bonds"]
    assert any("'Molecules'" in p for p in problems)
    assert any("'Bonds'" in p for p in problems)
    assert any("4 bonds" in p for p in problems)
    assert main(["data", str(path)]) == 1
    out = tmp_path / "stripped.data"
    removed = strip_data(path, out)
    assert len(removed) == 4
    assert check_data(out) == (["Masses", "Atoms", "Velocities"], [])
    # the stripped file is the readable one, byte for byte
    assert out.read_text() == GOOD + "\n"
    assert main(["data", str(path), "--strip", str(tmp_path / "s2.data")]) == 0
    assert "removed section Molecules" in capsys.readouterr().out
    assert bad_untouched(path)


def bad_untouched(path):
    return path.read_text() == BAD
