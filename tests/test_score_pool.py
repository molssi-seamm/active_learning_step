# -*- coding: utf-8 -*-

"""Writing structures from the database as a pool, and copying a selection back."""

import numpy as np
import pytest

molsystem = pytest.importorskip("molsystem")
pytest.importorskip("seamm")


@pytest.fixture()
def system_db():
    db = molsystem.SystemDB(filename="file:seamm_db?mode=memory&cache=shared")
    yield db
    db.close()


def _water_box(system_db):
    system = system_db.create_system(name="box")
    configuration = system.create_configuration(
        name="water box", periodicity=3, coordinate_system="Cartesian"
    )
    configuration.cell.parameters = [5.0, 6.0, 7.0, 90.0, 90.0, 90.0]
    configuration.atoms.append(
        x=[1.0, 1.757, 0.243],
        y=[1.0, 1.586, 1.586],
        z=[1.0, 1.0, 1.0],
        symbol=["O", "H", "H"],
    )
    configuration.charge = -1
    return configuration


def test_write_pool(system_db, tmp_path):
    from active_learning_step.pool import iter_frames
    from active_learning_step.score_pool import write_pool

    configuration = _water_box(system_db)
    path = tmp_path / "pool.extxyz"
    write_pool([configuration], path)
    ((index, atoms),) = list(iter_frames(path))
    assert index == 0
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell.lengths(), [5.0, 6.0, 7.0])
    assert np.allclose(atoms.positions[1], [1.757, 1.586, 1.0])
    assert atoms.info["SEAMM/configuration_name"] == "water box"
    assert atoms.info["total_charge"] == -1
    assert atoms.info["configuration_id"] == configuration.id
    assert atoms.info["system_id"] == configuration.system.id


def test_copy_to_new_system(system_db):
    from active_learning_step.select_frames import copy_to_new_system

    configuration = _water_box(system_db)
    system = copy_to_new_system(system_db, [configuration], "picked")
    assert system.name == "picked"
    assert system.n_configurations == 1
    copy = system.configurations[0]
    assert copy.id != configuration.id
    assert copy.periodicity == 3
    assert copy.charge == -1
    assert copy.atoms.symbols == ["O", "H", "H"]
    assert np.allclose(copy.cell.parameters, configuration.cell.parameters)
    # the system it came from is unchanged
    assert configuration.system.n_configurations == 1


def test_description_keeps_braces_in_a_regexp():
    """A system-name regexp with a {m,n} quantifier is the user's text, not a format
    field (the Loop step raised KeyError: '6,8' on one, ChemAI 5446)."""
    import seamm
    from active_learning_step.score_pool import ScorePool

    step = ScorePool(flowchart=seamm.Flowchart())
    step._id = ("1",)
    P = step.parameters.values_to_dict()
    P["source systems"] = "name regexp"
    P["source system name"] = "^(CBH:)?C{6,8}$"
    assert "^(CBH:)?C{6,8}$" in step.description_text(P)
