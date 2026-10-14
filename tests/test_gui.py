# -*- coding: utf-8 -*-

"""Smoke test of the sub-steps' Tk dialogs: create each and lay it out again for
every choice that changes which parameters apply. Skipped without a display."""

import pytest


@pytest.fixture()
def make_tk_node():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    import Pmw
    import seamm

    Pmw.initialise(root)
    flowchart = seamm.Flowchart(
        namespace="org.molssi.seamm.active_learning", directory="."
    )
    tk_flowchart = seamm.TkFlowchart(
        master=root,
        flowchart=flowchart,
        namespace="org.molssi.seamm.active_learning.tk",
    )

    def make(name):
        node = flowchart.create_node(name)
        flowchart.add_node(node)
        plugin = tk_flowchart.plugin_manager.get(name)
        tk_node = plugin.create_tk_node(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=tk_flowchart.canvas,
            x=100,
            y=100,
        )
        tk_node.create_dialog()
        return tk_node

    yield make
    root.destroy()


def _shown(tk_node, key):
    return tk_node[key].grid_info() != {}


def test_train_committee(make_tk_node):
    node = make_tk_node("Train Committee")
    assert _shown(node, "stratify by")
    node["stratify"].set("no")
    node.reset_dialog()
    assert not _shown(node, "stratify by")


def test_check_committee(make_tk_node):
    node = make_tk_node("Check Committee")
    for key in ("committee", "labelled data", "calibrate", "device", "precision"):
        assert _shown(node, key)


def test_score_pool(make_tk_node):
    node = make_tk_node("Score Pool")
    assert _shown(node, "source systems") and not _shown(node, "pool files")
    node["pool"].set("files")
    node.reset_dialog()
    assert not _shown(node, "source systems") and _shown(node, "pool files")
    assert _shown(node, "device")
    node["committee"].set("none")
    node.reset_dialog()
    for key in ("device", "precision", "leave out members", "per-atom spreads"):
        assert not _shown(node, key)
    # the name field appears only for a by-name choice
    node["pool"].set("structures in the database")
    node["source configurations"].set("name is")
    node.reset_dialog()
    assert _shown(node, "source configuration name")


def test_select_frames(make_tk_node):
    node = make_tk_node("Select Frames")
    assert _shown(node, "ranking") and not _shown(node, "random seed")
    assert not _shown(node, "combine thresholds")
    assert not _shown(node, "fill the budget")
    node["thresholds"].set("from Check Committee")
    node["budget"].set("100")
    node.reset_dialog()
    assert _shown(node, "combine thresholds") and _shown(node, "fill the budget")
    node["method"].set("random")
    node.reset_dialog()
    assert _shown(node, "random seed") and _shown(node, "spread the draw over")
    for key in ("ranking", "thresholds", "combine thresholds"):
        assert not _shown(node, key)


def test_active_learning_subflowchart():
    """The Active Learning step's dialog holds the sub-steps' flowchart."""
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    import Pmw
    import seamm

    Pmw.initialise(root)
    try:
        flowchart = seamm.Flowchart(namespace="org.molssi.seamm", directory=".")
        tk_flowchart = seamm.TkFlowchart(
            master=root, flowchart=flowchart, namespace="org.molssi.seamm.tk"
        )
        node = flowchart.create_node("Active Learning")
        flowchart.add_node(node)
        plugin = tk_flowchart.plugin_manager.get("Active Learning")
        tk_node = plugin.create_tk_node(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=tk_flowchart.canvas,
            x=100,
            y=100,
        )
        tk_node.create_dialog()
        manager = tk_node.tk_subflowchart.plugin_manager
        for step in (
            "Train Committee",
            "Check Committee",
            "Score Pool",
            "Select Frames",
        ):
            assert manager.get(step) is not None
    finally:
        root.destroy()
