=======
History
=======

2026.10.14 -- Initial release: query-by-committee active learning for MLFFs
    * A committee of machine-learned force fields, trained on different folds of the
      data, disagrees where the model is likely wrong; the seamm-al command uses that
      disagreement to choose what to label next. This first release provides the
      library and the command line; the SEAMM flowchart step is a placeholder.
    * seamm-al score: the committee's spreads (energy, forces, the largest per-atom
      force spread, atomic and molecular pressure) for every frame of extxyz or SEAMM
      SDF pools, with the total charge passed to the models. Molecules follow SEAMM's
      own rule (molsystem's bond perception, ions on their own), applied without
      SEAMM in MLFF environments, or a SEAMM structure's own bonds.
    * seamm-al select: rank the frames, or keep those past thresholds, and take them
      within a core-hour budget, with costs per kind or group, caps per column and
      exclusions; or draw at random (optionally stratified) before any committee
      exists. The chosen frames are copied byte for byte from the input files.
    * seamm-al train: train the k members of a committee from an xnn recipe, with
      stratified folds, a fixed share of cell batches and a manifest of the folds,
      seeds, data and code versions. Requires xnn 0.13 or later.
    * seamm-al check: whether the spreads predict the out-of-fold errors (Spearman
      correlations, threshold calibration) and the health of each member (losses,
      errors, leave-one-out spreads), with an option to drop a member.
    * seamm-al engine: the committee mean as one MDI engine for LAMMPS, writing the
      spreads every N steps and flagging frames past a threshold.
    * seamm-al data: check a LAMMPS data file for what atom_style atomic can read.

