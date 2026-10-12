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
spread reaches a threshold. The output is ``selection.csv``, with the reason and the
running cost of every frame chosen, and the chosen frames copied byte for byte from
the pool files into ``selected.extxyz`` and/or ``selected.sdf``, ready for labelling.
