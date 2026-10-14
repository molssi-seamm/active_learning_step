=====
Usage
=====

The command line, ``seamm-al``, is described in the user guide. The library can also
be used directly from Python, for example to score structures with a committee::

    from ase.io import read
    from active_learning_step.committee import (
        Committee, masses, molecules, structure_from_atoms
    )

    committee = Committee.from_checkpoints(["m0/best.pt", "m1/best.pt", "m2/best.pt"])
    frames = [structure_from_atoms(a) for a in read("cells.extxyz", index=":")]
    spreads = committee.score(frames, batch_size=1, molecule=molecules, masses=masses)
    print(spreads["rho_forces"], spreads["rho_pressure_molecular"])

and to choose frames from scores::

    from active_learning_step.selection import choose, parse_costs, read_scores

    rows = read_scores("scores.csv")
    selected, rejected = choose(rows, parse_costs(["cell=2800"]), budget=75000)
