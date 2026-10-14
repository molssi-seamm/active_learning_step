#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `active_learning_step` package."""

import pytest  # noqa: F401
import active_learning_step  # noqa: F401

# the step's classes need SEAMM, absent from the MLFF environments
pytest.importorskip("seamm")


def test_construction():
    """Just create an object and test its type."""
    result = active_learning_step.ActiveLearning()
    assert (
        str(type(result))
        == "<class 'active_learning_step.active_learning.ActiveLearning'>"
    )
