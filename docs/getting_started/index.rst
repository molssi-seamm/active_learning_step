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

That should be enough to get started. For more detail, see the
:ref:`User Guide <user-guide>`.
