=======
History
=======
2026.10.29.1 -- Bugfix: a system-name regexp with braces stopped Score Pool
    * Score Pool with structures selected by a regular expression with a ``{m}`` or
      ``{m,n}`` quantifier, such as ``^(CBH:)?C{6,8}$``, failed before it ran: the
      description took the braces for a format field. The names and expressions are
      now printed as given.

2026.10.29 -- Bugfix: names with '=' in the extended XYZ files
    * The extended XYZ files of scored and selected frames are written with ASE, which
      leaves a value with '=' but no space unquoted and then reads it back merged with
      the next key: a configuration named 'LiDMCct_chel_ctmin_toO=' came back as
      'LiDMCct_chel_ctmin_toO=pbc=F F F'. Such values, and those with backslashes, are
      now quoted, so every name reads back as itself.

2026.10.17 -- Bugfix: a committee's engine beside a GPU uses its CPU threads
    * The CPU threads torch uses now follow ``TORCH_NUM_THREADS`` before
      ``OMP_NUM_THREADS``. Next to a GPU, LAMMPS's launcher keeps ``OMP_NUM_THREADS`` at 1
      and gives torch 4 threads through ``TORCH_NUM_THREADS``, but the committee's
      engine read only ``OMP_NUM_THREADS``, so its work on the CPU, such as the spreads,
      ran on a single thread. The engine's log says which setting it used.

2026.10.14.1 -- The Active Learning step for flowcharts
    * The Active Learning step now holds four sub-steps: Train Committee, Check
      Committee, Score Pool and Select Frames, which do in a flowchart what the
      seamm-al command does. Each can use the committee, scores and thresholds made
      earlier in the job, and their dialogs show only the settings that apply.
    * Training, checking and scoring run in the MLFF environment of the xnn plug-in
      (xnn.ini) as tasks of the job, with nothing to install there; selecting runs in
      SEAMM.
    * Score Pool takes candidates from the job's database, with the standard
      selection of systems and configurations, or from extxyz and SDF files. Select
      Frames puts selected configurations from the database in a variable for later
      steps and can copy them into a new system.
    * Train Committee can save the committee as an xnn model, which the Model
      Chemistry step then offers (with xnn-step 2026.10.14 or later) for molecular
      dynamics with the committee mean, recording the members' spreads; with
      lammps-step 2026.10.14.1 or later this also runs on the CPU.
    * Candidate files may be gzipped, such as the trajectories LAMMPS writes.
    * Requires xnn-step 2026.10.14 or later.

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

