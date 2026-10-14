# -*- coding: utf-8 -*-

"""LAMMPS data files for the committee engine: what ``atom_style atomic`` can read.

An MLFF engine run with ``fix mdi/qm`` uses ``atom_style atomic``, which reads only
the ``Atoms``, ``Velocities``, ``Masses`` and pair-coefficient sections. A data file
written from a molecular style (or by tools that add their own sections, such as a
``Molecules`` section) stops LAMMPS at ``read_data`` with "Unknown identifier in data
file", after the job has waited in the queue. :func:`check_data` finds such sections
before a job is submitted and :func:`strip_data` writes a copy without them.

A section starts at a line whose first word is not a number; its lines run to the next
such line. Header lines (``1536 atoms``, ``2 atom types``, box bounds) start with a
number. Only the standard library is needed.
"""

from pathlib import Path

__all__ = [
    "ATOMIC_SECTIONS",
    "MOLECULAR_COUNTS",
    "sections",
    "check_data",
    "strip_data",
]

#: Sections ``atom_style atomic`` reads.
ATOMIC_SECTIONS = ("Atoms", "Velocities", "Masses", "Pair Coeffs", "PairIJ Coeffs")

#: Header counts that need a molecular atom style when they are not zero.
MOLECULAR_COUNTS = (
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "bond types",
    "angle types",
    "dihedral types",
    "improper types",
    "extra bond per atom",
    "extra angle per atom",
    "extra dihedral per atom",
    "extra improper per atom",
    "extra special per atom",
)


def _is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    return True


def _name(line):
    """A section's name: the line without its comment, stripped."""
    return line.split("#", 1)[0].strip()


def sections(lines):
    """The sections of a data file: (name, first line index, end index), in order.

    The first line (the title) is never a section.
    """
    out = []
    starts = [
        i
        for i, line in enumerate(lines)
        if i > 0 and line.strip() and not _is_number(line.split()[0])
    ]
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        out.append((_name(lines[i]), i, end))
    return out


def _header(lines):
    """The header lines: after the title, before the first section."""
    found = sections(lines)
    end = found[0][1] if found else len(lines)
    return list(range(1, end))


def check_data(path):
    """What in a data file ``atom_style atomic`` cannot read.

    Parameters
    ----------
    path : str or pathlib.Path
        The LAMMPS data file.

    Returns
    -------
    (list of str, list of str)
        The section names found, and the problems (empty when the file is fine).
    """
    lines = Path(path).read_text().splitlines()
    found = sections(lines)
    problems = [
        f"section {name!r} (line {i + 1}) is not read by atom_style atomic"
        for name, i, _ in found
        if name not in ATOMIC_SECTIONS
    ]
    for i in _header(lines):
        words = _name(lines[i]).split()
        if not words or not _is_number(words[0]):
            continue
        keyword = " ".join(words[1:])
        if keyword in MOLECULAR_COUNTS and float(words[0]) != 0:
            problems.append(
                f"header '{lines[i].strip()}' (line {i + 1}) needs a molecular "
                "atom style"
            )
    if "Atoms" not in [name for name, _, _ in found]:
        problems.append("no Atoms section")
    return [name for name, _, _ in found], problems


def strip_data(path, out):
    """Write a copy of a data file that ``atom_style atomic`` reads.

    Drops every section :data:`ATOMIC_SECTIONS` does not list and the header counts
    of :data:`MOLECULAR_COUNTS`; everything else is copied unchanged.

    Returns
    -------
    list of str
        What was removed.
    """
    path, out = Path(path), Path(out)
    if path.resolve() == out.resolve():
        raise ValueError("write the stripped copy to another file")
    lines = path.read_text().splitlines(keepends=True)
    plain = [line.rstrip("\r\n") for line in lines]
    drop, removed = set(), []
    for name, i, end in sections(plain):
        if name not in ATOMIC_SECTIONS:
            drop.update(range(i, end))
            removed.append(f"section {name} ({end - i} lines)")
    for i in _header(plain):
        words = _name(plain[i]).split()
        if words and _is_number(words[0]) and " ".join(words[1:]) in MOLECULAR_COUNTS:
            drop.add(i)
            removed.append(f"header '{plain[i].strip()}'")
    out.write_text("".join(line for i, line in enumerate(lines) if i not in drop))
    return removed
