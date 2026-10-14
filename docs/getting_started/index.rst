***************
Getting Started
***************

Installation
============
Install the package with pip into the environment where you will use it::

    pip install active-learning-step

``seamm-al select`` and ``seamm-al data`` need nothing more. Scoring, training,
checking and the committee engine need torch and xnn 0.13 or later: run them in the
MLFF environment SEAMM itself uses (see :ref:`user-guide`).

A first selection
=================
Score a pool of candidate structures with a committee of five models, then choose the
most uncertain within a budget of core-hours::

    seamm-al score -m m0/best.pt m1/best.pt m2/best.pt m3/best.pt m4/best.pt \
        -o scores.csv candidates.extxyz
    seamm-al select scores.csv --budget 5000 --cost cluster=4 --cost cell=2800

``selection.csv`` lists the frames chosen, with each one's rank and the running cost,
and ``selected.extxyz`` holds them, copied unchanged from the pool, ready for
labelling.

Before any committee exists, draw the first round at random instead, spread over the
values of a column of the pool::

    seamm-al score --no-models --carry source -o pool.csv candidates.extxyz
    seamm-al select pool.csv --random 1 --stratify-column source \
        --cost cluster=4 --max-frames 50

In a SEAMM flowchart
====================
Add an Active Learning step and, inside it, the sub-steps: for example Score Pool,
taking the candidates from the job's database or from files, then Select Frames with
a budget. The sub-steps that need torch run in the MLFF environment of the xnn
plug-in, set in the installation's ``xnn.ini``; nothing has to be installed there.
Train Committee trains a committee and can save it as an xnn model, which the Model
Chemistry step then offers for molecular dynamics that records where the members
disagree. See :doc:`../user_guide/flowcharts`.

That should be enough to get started. For more detail, see the
:ref:`User Guide <user-guide>`.
