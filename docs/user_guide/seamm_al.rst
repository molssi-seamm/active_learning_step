***********************************
Command line: ``seamm-al``
***********************************

``seamm-al`` chooses which candidate frames to label next, by query by committee: a
committee of models (typically five, each trained on four fifths of the data) is run
on every candidate, and the frames where the members disagree most are labelled.

Scoring
=======

``seamm-al score`` needs torch and xnn 0.11 or later, so run it in an MLFF
environment::

    seamm-al score -m m0/best.pt m1/best.pt m2/best.pt m3/best.pt m4/best.pt \
        -o scores.csv --carry temperature pool.extxyz

The pools are extxyz files or SDF files written by SEAMM. ``scores.csv`` has one row
per frame:

=========================== ==========================================================
column                      meaning
=========================== ==========================================================
file, index, configuration  where the frame is (index from 0 in its file)
kind                        ``cell`` if periodic, else ``cluster``
n_atoms, formula            the frame's size and Hill formula
total_charge, n_molecules   the charge given to the models; molecules of a cell
rho_energy                  committee standard deviation of the energy / sqrt(N), eV
rho_forces                  RMS over atoms of the per-atom force spread, eV/Å
rho_forces_max_atom         the largest per-atom force spread, eV/Å
rho_pressure                spread of the atomic pressure, atm (cells)
rho_pressure_molecular      spread of the molecular pressure, atm (cells)
energy_mean, pressure_mean  the committee average; also pressure_molecular_mean
=========================== ==========================================================

Molecules are found from a covalent bond graph (alkali and alkaline-earth ions are
never bonded), and the total charge is read from ``total_charge`` in the extxyz info
or from SEAMM's net charge in an SDF. By default the dispersion the checkpoints record
as subtracted is not added back (``--dispersion false``): it is the same in every
member and adds nothing to the spreads. ``--atoms frames.extxyz`` writes every frame
with its per-atom force spread.

The models are served in the dtype of their weights, usually float32, as the water
campaign scored. Float32 reductions on CPUs are not reproducible from run to run: the
same frame scored twice can differ by about 1e-5 relative in the pressure spreads (and
much less in the energy and force spreads), which cannot change a ranking.
``--dtype float64`` gives reproducible numbers at about twice the cost.

Training a committee
====================

``seamm-al train`` trains the ``k`` members of a committee in an MLFF environment::

    seamm-al train -c recipe.yaml -o committee labelled.extxyz [more.extxyz ...] \
        -k 5 --cell-share 0.15 --steps 52950 [--member 2]

The recipe is an ordinary xnn YAML config (model, loss weights, optimiser, epochs and
the dispersion subtracted from the labels); ``seamm-al`` supplies the data, the folds,
the seeds and the batching. The labelled frames are cut into ``k`` folds with the data
seed (``--data-seed``, default 0), splitting clusters and cells separately so that
every fold holds out its share of each (``--stratify kind``, the default; within a
stratum the folds' counts differ by at most one). ``--stratify-column liquid`` adds an
info key to the strata, and ``--stratify none`` is the water campaign's plain k-fold.
Member ``l`` trains on the other folds, validates on
fold ``l`` and is initialised from seed ``1000 + l`` (``--init-seed``). Batches never
mix clusters and cells: clusters use the recipe's ``data.batch_size`` and cells
``--cell-batch-size`` (default 4). ``--cell-share`` makes cell batches that share of
every epoch whatever the selection holds, and ``--steps`` trains every member for the
same number of optimiser steps, so committees trained on different selections differ
only in their data. ``avg_num_neighbors: auto`` in the recipe uses the mean neighbour
count of each member's training set.

The committee's directory holds ``manifest.json`` (the folds, the seeds, the data files
and their SHA-256, a hash of the ordered frames, the recipe, the batching and the code
versions), ``frames.csv`` (every frame's fold) and one directory per member with
``best.pt``, ``last.pt``, ``history.json`` and ``member.json``. Members can be trained
in separate jobs (``--member``): each checks the manifest, a directory holding a
different plan is refused, and a member that has finished is skipped.

Training needs xnn 0.13 or later, which chooses ``best.pt``, drives the plateau
learning rate and logs on dataset-averaged epoch metrics, and tells the batch samplers
the epoch every pass. ``score``, ``check`` and ``engine`` work with older releases on
checkpoints those releases wrote, but a checkpoint trained with xnn 0.13 can only be
read by 0.13 or later (its pickled config), so a committee trained here is scored,
checked and run in an xnn 0.13 environment.
Each member's epochs are recorded in its ``history.json`` through an xnn callback, and
the recipe's ``trainer`` section (loggers, early stopping, checkpoints every n epochs)
applies as in ``xnn train``.

Selecting
=========

``seamm-al select`` needs neither torch nor xnn::

    seamm-al select scores.csv --budget 75000 \
        --cost cell:EC=2800 --cost cell:FEC=3900 --group-column liquid \
        --cap temperature=6 --exclude labelled.txt

By default cells are ranked by the sum of their ranks in ``rho_pressure_molecular``
and ``rho_forces``, clusters by ``rho_forces`` (change with ``--rank``), and frames are
taken in that order until the next would exceed the budget (``--fill`` skips it and
keeps going). ``--threshold rho_forces=0.0106`` limits the candidates to frames whose
spread reaches a threshold. The output is ``selection.csv``, with the reason, the
running cost and the frame's rank within its kind (``rank_in_kind`` of ``n_in_kind``)
for every frame chosen; the command also reports where the budget cut fell, and
``--rejected rejected.csv`` lists every frame not chosen with its rank and why, and the chosen frames copied byte for byte from
the pool files into ``selected.extxyz`` and/or ``selected.sdf``, ready for labelling.

Round 0: a random draw
======================

Before any committee exists, ``score --no-models`` lists a pool (identity and
``--carry`` columns only, no torch or xnn needed) and ``select --random SEED`` draws
from it in a seeded random order, with the same costs, budget, caps, ``--exclude`` and
``--max-frames``, and the same verbatim output::

    seamm-al score --no-models --carry source --carry salt -o pool.csv clusters.extxyz
    seamm-al select pool.csv --random 1 --stratify-column source --stratify-column salt \
        --cost cluster=4 --max-frames 50

``--stratify-column`` spreads the draw evenly over the values of the columns: one frame
from each stratum in turn, the strata in a shuffled order. ``selection.csv`` records
the seed, each frame's stratum and its place in the draw.

Checking a committee
====================

``seamm-al check`` asks whether the committee's spread predicts its real error, on
labelled frames::

    seamm-al check committee wave1_labels.extxyz --tag-key frame \
        --calibrate rho_forces:0.14 --calibrate rho_pressure_molecular:1500:0.9

For each frame it gives the spreads and the **out-of-fold** errors: the error of the one
member that held the frame out, or the mean over the members for a frame none trained
on (a common validation or test set); frames every member trained on are left out. The
errors are the force RMSE, the energy per atom and, for cells, the atomic and molecular
pressures (the label's molecular pressure from its stress and forces). The command
prints, and ``--summary`` writes, the Spearman correlation of each spread with its error
per kind, over all out-of-fold frames and the held-out ones only.

The committee directory is one made by ``seamm-al train`` (frames matched by
configuration name) or one with a ``fold_manifest.json`` per member listing tags under
``held_out``, ``validation`` and ``test``; ``--tag-key`` names the info key of the tags.
Labels are read from ``REF_energy``, ``REF_forces`` and ``REF_stress`` (a 9-vector, a
Voigt 6-vector or a matrix) unless ``--*-key`` says otherwise.

``check`` also judges every member. For each it reports the final and best validation
loss (from the member's ``history.json`` when there is one), its mean error on the
frames it held out and on the common validation and test frames, and the median
**leave-one-out** ratio of each spread: the spread with the member over the spread
without it, so a member that inflates the spreads stands out. A member more than
``--flag-factor`` (default 1.5) times the median of the others on any of these is
flagged — as a note only, except for the final validation loss, which is the gate. A
member that trailed the others for a while (validation losses smoothed over 9 epochs,
at least 5 epochs above the factor) and then recovered is noted, not flagged. The
table is written to ``<out>_members.csv``. ``--drop-member K``
(repeatable) leaves member ``m<K>`` out, in ``check`` and in ``score`` (where ``K``
counts the ``--models`` from 0); the members used and dropped are recorded in the
summary and in ``scores.csv.json``.

``--calibrate NAME:FAIL_LEVEL[:CAPTURE]`` calibrates a threshold as Smith *et al.*
(fig. 1): the largest spread that still catches ``CAPTURE`` (default 0.98) of the
frames whose worst-member error exceeds ``FAIL_LEVEL`` (eV/A for forces, atm for
pressures, eV per sqrt(atom) for energies), with what it separates. Fewer than 10
failures earn a warning: such a threshold is not to be trusted.

Committee molecular dynamics
============================

``seamm-al engine`` serves the committee's mean to an MDI driver such as LAMMPS
(``fix mdi/qm``) as one potential: the mean of the members' short-range energies, with
the dispersion the checkpoints record added once on top, differentiated once for the
forces and stress. It takes the place of ``xnn mdi`` in an MDI pair::

    mpirun --bind-to none \
        -np 1 seamm-al engine --members m0/best.pt m1/best.pt m2/best.pt m3/best.pt \
            m4/best.pt --device cuda:0 --eeq-reuse --spreads spreads.csv --every 10 \
            --flag rho_forces=0.05 --flagged flagged.extxyz \
            -mdi "-role ENGINE -name xnn -method MPI" \
      : -np 1 lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat

With ``--spreads`` the engine also evaluates every member on its own every ``--every``
force evaluations and appends the committee spreads (``rho_energy``, ``rho_forces``,
``rho_forces_max_atom`` and, for periodic systems, ``rho_pressure`` and
``rho_pressure_molecular``) to the CSV, with the committee-mean energy and ``max_dF``,
the largest difference between the forces served and the mean of the members' forces
(rounding only, without dispersion). ``--flag NAME=VALUE`` writes the frames whose
spread reaches a value to ``--flagged`` with their step, spreads and per-atom force
spread, ready for ``seamm-al score`` or labelling; ``--frames`` writes every frame
recorded. ``--drop-member K`` leaves a member out and ``--no-dispersion`` serves the
route-B energy.

On a CPU (8 threads, 64 waters, five MACE members) the mean costs five times one model,
and spreads every 10 steps add about 11 %.
