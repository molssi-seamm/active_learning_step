# -*- coding: utf-8 -*-

"""Candidate pools: the frames of extxyz and SDF files, as records and as structures.

A pool file is split into **records**, the exact bytes of each frame, so that a
selection can be written as a verbatim subset of the input. Rewriting frames through
ASE would drop keys it does not understand (for example the DDEC6 charges and the
SEAMM header keys), and the labelling flowcharts read these files directly.

* extxyz: a record is the atom-count line, the comment line and that many atom lines.
* SDF: a record runs through its ``$$$$`` line; everything is kept, including the
  blank line that ends each property.

Each record is also turned into an :class:`ase.Atoms` with an ``info`` dictionary
holding the configuration name and the total charge:

* extxyz frames are parsed by ASE, one record at a time, so the frame numbering always
  matches the records;
* SDF records written by SEAMM carry the full-precision coordinates
  (``SEAMM|XYZ|json|``), the cell (``SEAMM|cell|json|``, as a, b, c, alpha, beta,
  gamma) and the net charge (``SEAMM|net charge|int|``); other SDF files fall back to
  the V2000 atom block and the ``M  CHG`` lines.

Nothing here needs torch or xnn.
"""

import io
import json
from pathlib import Path

import numpy as np

__all__ = [
    "pool_format",
    "iter_records",
    "iter_frames",
    "record_to_atoms",
    "configuration_name",
    "total_charge",
]


def pool_format(path):
    """The format of a pool file from its suffix: ``"extxyz"`` or ``"sdf"``.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        For any other suffix.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".extxyz", ".xyz"):
        return "extxyz"
    if suffix in (".sdf", ".sd", ".mol"):
        return "sdf"
    raise ValueError(f"unknown pool format for {path}: use .extxyz or .sdf")


def _xyz_records(fh):
    """Records of an (ext)xyz file, as bytes."""
    while True:
        line = fh.readline()
        if line == b"":
            return
        if line.strip() == b"":
            # blank lines between or after frames are not part of any record
            continue
        try:
            n = int(line.split()[0])
        except ValueError:
            raise ValueError(f"expected an atom count, got {line[:80]!r}")
        lines = [line]
        for _ in range(n + 1):
            nxt = fh.readline()
            if nxt == b"":
                raise ValueError("the file ends inside a frame")
            lines.append(nxt)
        yield b"".join(lines)


def _sdf_records(fh):
    """Records of an SDF file, as bytes, each ending with its ``$$$$`` line."""
    lines = []
    for line in fh:
        lines.append(line)
        if line.rstrip(b"\r\n") == b"$$$$":
            yield b"".join(lines)
            lines = []
    if any(line.strip() for line in lines):
        # a final molecule without the terminator
        yield b"".join(lines)


def iter_records(path):
    """The records of a pool file, in order, as exact bytes.

    Parameters
    ----------
    path : str or pathlib.Path
        An extxyz or SDF file.

    Yields
    ------
    bytes
        One record per frame.
    """
    fmt = pool_format(path)
    with open(path, "rb") as fh:
        if fmt == "extxyz":
            yield from _xyz_records(fh)
        else:
            yield from _sdf_records(fh)


def _sdf_atoms(text):
    """An ase.Atoms from the text of one SDF record."""
    from ase import Atoms
    from ase.geometry import cellpar_to_cell

    lines = text.splitlines()
    title = lines[0].strip() if lines else ""
    counts = lines[3]
    if "V3000" in counts:
        raise ValueError("V3000 SDF records are not supported")
    n_atoms = int(counts[0:3])
    n_bonds = int(counts[3:6])
    symbols, positions = [], []
    for line in lines[4 : 4 + n_atoms]:
        positions.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        symbols.append(line[31:34].strip())
    charge = 0
    end = 4 + n_atoms + n_bonds
    k = end
    while k < len(lines) and not lines[k].startswith("M  END"):
        if lines[k].startswith("M  CHG"):
            fields = lines[k].split()
            charge += sum(int(c) for c in fields[4::2])
        k += 1

    # the data items: "> <key>" (or ">  <key>") then value lines up to a blank line
    props = {}
    k += 1
    while k < len(lines):
        line = lines[k]
        if line.startswith(">") and "<" in line and ">" in line[1:]:
            key = line[line.index("<") + 1 : line.rindex(">")]
            values = []
            k += 1
            while k < len(lines) and lines[k].strip() != "":
                values.append(lines[k])
                k += 1
            props[key] = "\n".join(values)
        k += 1

    if "SEAMM|XYZ|json|" in props:
        positions = json.loads(props["SEAMM|XYZ|json|"])
    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    if "SEAMM|cell|json|" in props:
        atoms.set_cell(cellpar_to_cell(json.loads(props["SEAMM|cell|json|"])))
        atoms.set_pbc(True)
    if "SEAMM|net charge|int|" in props:
        charge = int(props["SEAMM|net charge|int|"])

    info = {"total_charge": float(charge), "SDF/title": title}
    for key, value in props.items():
        info[key] = value
        if key.startswith("SEAMM|"):
            # also by the bare property name, e.g. "temperature"
            info.setdefault(key.split("|")[1], value)
    name = props.get("SEAMM|configuration name|str|")
    if name is None:
        name = title
        if title.startswith("SEAMM=") and "/" in title:
            name = title.split("/", 1)[1].strip()
    info["SEAMM/configuration_name"] = name
    atoms.info.update(info)
    return atoms


def record_to_atoms(record, fmt):
    """Parse one record into an ase.Atoms.

    Parameters
    ----------
    record : bytes
        The record, from :func:`iter_records`.
    fmt : str
        ``"extxyz"`` or ``"sdf"``.

    Returns
    -------
    ase.Atoms
        With ``info`` from the file. SDF records also get ``total_charge``,
        ``SEAMM/configuration_name`` and every data item.
    """
    text = record.decode("utf-8")
    if fmt == "extxyz":
        from ase.io import read

        return read(io.StringIO(text), format="extxyz")
    return _sdf_atoms(text)


def iter_frames(path):
    """The frames of a pool file.

    Parameters
    ----------
    path : str or pathlib.Path
        An extxyz or SDF file.

    Yields
    ------
    (int, ase.Atoms)
        The index of the frame in the file and the structure.
    """
    fmt = pool_format(path)
    for index, record in enumerate(iter_records(path)):
        yield index, record_to_atoms(record, fmt)


def configuration_name(atoms):
    """The configuration name of a frame, or an empty string.

    ``SEAMM/configuration_name`` in the extxyz info (or the SDF record's name).
    """
    name = atoms.info.get("SEAMM/configuration_name", "")
    return "" if name is None else str(name)


def total_charge(atoms):
    """The total charge of a frame: ``total_charge`` (or ``charge``) in its info.

    Returns
    -------
    float or None
        ``None`` when the frame records no charge (the model then assumes neutral).
    """
    for key in ("total_charge", "charge"):
        value = atoms.info.get(key)
        if value is not None and np.ndim(value) == 0:
            return float(value)
    return None
