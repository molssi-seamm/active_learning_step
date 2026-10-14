*************************************
In a flowchart: Active Learning step
*************************************

The Active Learning step brings query by committee into SEAMM flowcharts. It holds
four sub-steps, used in any order and combination:

=================== ===================================================================
sub-step            what it does
=================== ===================================================================
Train Committee     trains a committee of xnn models, each on all but one fold of the
                    labelled data (``seamm-al train``)
Check Committee     checks that the members' spread predicts their error on frames
                    they never saw, flags members that trained badly and calibrates
                    thresholds of the spreads (``seamm-al check``)
Score Pool          scores candidate structures by how much the members disagree on
                    them (``seamm-al score``)
Select Frames       selects the structures to label, by spread or at random, within
                    a budget (``seamm-al select``)
=================== ===================================================================

A round of active learning might be one flowchart: Train Committee and Check
Committee, then molecular dynamics with the new committee to make candidates (see
`Committees as model chemistries`_), then Score Pool and Select Frames, and the
labelling of the selected structures. The first round, before there is a committee,
is Score Pool with the committee set to ``none`` and Select Frames drawing at random.

What the sub-steps share
========================

Each sub-step can use what an earlier one made in the same job, so the defaults chain
them together:

* the committee (``previous``) is the last one trained, checked or used to score;
* the scores (``previous``) are those of the last Score Pool;
* the thresholds (``from Check Committee``) are those calibrated by the last Check
  Committee.

This holds across Active Learning steps too: a flowchart can train a committee in one,
run molecular dynamics, and score and select in another. Any of them can instead be
given explicitly: a committee's directory (holding the members ``m0``, ``m1``, ...),
or scores files.

Where they run
==============

Training, checking and scoring need torch and xnn. They run in the MLFF environment of
the xnn plug-in, the one named in the installation's ``xnn.ini``, as a task of the job
like any other code, with the job's cores (``--ncores`` limits them). Nothing has to be
installed there: the step copies this package into ``_code/`` in its directory and puts
it on that environment's ``PYTHONPATH``, so the code that runs is the code of the SEAMM
installation. Training needs xnn 0.13 or later (see `Environments
<seamm_al.html#environments>`_). Select Frames needs neither and runs in SEAMM itself.

The device is chosen in each sub-step: by default the one in ``xnn.ini`` for checking
and scoring, and the recipe's for training.

Each sub-step leaves everything ``seamm-al`` wrote in its directory, with the command
in ``step.out`` and its output in ``stdout.txt`` and ``stderr.txt``.

Train Committee
===============

Give the labelled structures, as extended XYZ files with the reference energies and
forces (names or patterns such as ``data/*.extxyz``), and the xnn recipe: the YAML
configuration of the model and its training. The committee is written to
``committee/`` in the step's directory, with a manifest of the folds, seeds, code and
data, and is printed with each member's validation losses.

* **Number of members** is also the number of folds. **Members to train** trains only
  some of them, so a large committee can be trained by several jobs into the same
  directory.
* **Folds** split clusters and periodic cells separately by default, so every fold
  has its share of both; **Also balance by** adds keys of the structures' info, such
  as the temperature.
* **Share of cell batches** and **Cells per batch** control how often periodic cells
  are drawn when they are far fewer, or far more costly, than clusters.
* **Changes to the recipe** override recipe settings, e.g. ``optim.lr=0.005``.
* **Save as the model** copies the members into your xnn models, so that the Model
  Chemistry step offers the committee as ``xnn:MLFF@<name>``.

Check Committee
===============

By default the committee is checked on its own training data: each member is judged
on the fold it never saw, which is what makes the check honest without new labels.
New labelled structures that no member saw can be given instead. The step prints:

* the rank correlation (Spearman) of each spread with the corresponding error, which
  should be clearly positive for the spread to be a useful guide;
* each member's health: its validation losses and errors, and whether it inflates the
  spreads; a member whose final validation loss is well above the others' is
  flagged, and can be left out with **Leave out members** in Score Pool;
* for each ``NAME:ERROR[:CAPTURE]`` in **Calibrate thresholds**, the threshold of the
  spread that catches the fraction CAPTURE (0.98 by default) of the structures whose
  error is above ERROR. For example ``rho_forces:0.15`` calibrates the force spread
  against force errors above 0.15 eV/Å.

Score Pool
==========

The candidates are either structures in the job's database, chosen with the usual
system and configuration selection, or extended XYZ or SDF files (gzipped ones too,
such as the ``trajectory.extxyz.gz`` the LAMMPS step writes). Structures from the
database are written to ``pool.extxyz`` with their system and configuration ids, so
that a selection can be traced back to them.

``scores.csv`` has a row per candidate with its spreads (see `Scoring
<seamm_al.html#scoring>`_); the step prints their median and largest values by kind,
and the candidates with the largest force spreads. **Keep info keys** copies keys of
the candidates' info into the scores, to set costs by or to balance a random draw
over. With the committee set to ``none``, the candidates are only listed, for a
random draw.

Select Frames
=============

By default the candidates the committee disagrees on most are selected: ranked by the
spreads of their kind (the molecular pressure and force spreads for cells, the force
spread for clusters), and, with **Thresholds**, only those whose spread reaches one,
as calibrated by Check Committee or given. **Select** ``random`` draws them at random
instead, optionally spread evenly over columns of the scores.

The **budget** is in core-hours, with the **cost per structure** given by kind (and
group, for structures of one kind that cost differently); without costs each
structure counts one, and the budget is a number of structures. **Already labelled**
files of configuration names keep structures from being selected twice, and **At most
per value** caps how many come from, for example, one trajectory.

The selected structures are copied, byte for byte, from the candidates' files into
``selected.extxyz`` (or ``.sdf``), with ``selection.csv`` listing them and why, and
``rejected.csv`` the others and why not. When the candidates came from the database,
the selected configurations are also put in a variable (``selected_configurations``
by default), which later steps can take as their structures, and, if asked, copied
into a new system.

Committees as model chemistries
===============================

A committee -- a directory of members ``m0/best.pt``, ``m1/best.pt``, ... -- is a model
like any other for the xnn plug-in. Saved with Train Committee's **Save as the model**,
or placed in one of the model directories of ``xnn.ini``, it is offered by the Model
Chemistry step as ``xnn:MLFF@<name>``; its directory can also be given as the Model
Chemistry step's **Checkpoint**, for example ``$committee``, the result of a Train
Committee earlier in the job. A ``committee.json`` in the directory, such as
``{"members": ["m0", "m1", "m3"]}``, chooses the members to use.

Energy, LAMMPS and the other steps that drive a model chemistry then run the
committee as the committee-mean engine (see `Committee molecular dynamics
<seamm_al.html#committee-molecular-dynamics>`_):
the forces and energy are the members' mean, and every ten steps their spreads are
appended to ``committee_spreads.csv`` in the step's directory. The Model Chemistry
step's **Engine options** go to this engine: ``--every 20`` changes how often the
spreads are written, ``--flag rho_forces=0.2`` writes the frames whose spread reaches
the value to ``flagged.extxyz``, and ``--dtype float32`` and ``--eeq-reuse`` work as
for a single model. The engine is about the members' cost together, so expect molecular
dynamics several times slower than with one model on a CPU, and about twice as slow on
a GPU. Without a free GPU the LAMMPS step (2026.10.14.1 or later) runs the engine on
the CPU with the job's cores, which is practical for boxes of a few hundred atoms.
