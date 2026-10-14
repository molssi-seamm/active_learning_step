# -*- coding: utf-8 -*-

"""Non-graphical part of the Score Pool step in an Active Learning flowchart"""

import csv
import logging
from pathlib import Path
import importlib.resources
import textwrap

import numpy as np
from tabulate import tabulate

import active_learning_step
import molsystem
import seamm
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

from .check import member_dirs

# In addition to the normal logger, two logger-like printing facilities are
# defined: "job" and "printer". "job" send output to the main job.out file for
# the job, and should be used very sparingly, typically to echo what this step
# will do in the initial summary of the job.
#
# "printer" sends output to the file "step.out" in this steps working
# directory, and is used for all normal output from this step.

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Active Learning")

# Add this module's properties to the standard properties
path = importlib.resources.files("active_learning_step") / "data"
csv_file = path / "properties.csv"
if path.exists():
    molsystem.add_properties_from_file(csv_file)


class ScorePool(active_learning_step.Substep):
    """
    The non-graphical part of a Score Pool step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : ScorePoolParameters
        The control parameters for Score Pool.

    See Also
    --------
    TkScorePool,
    ScorePool, ScorePoolParameters
    """

    def __init__(
        self, flowchart=None, title="Score Pool", extension=None, logger=logger
    ):
        """A substep for Score Pool in a subflowchart for Active Learning.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        logger.debug(f"Creating Score Pool {self}")

        super().__init__(
            flowchart=flowchart,
            title="Score Pool",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "score"
        self._model = None
        self._metadata = active_learning_step.metadata
        self.parameters = active_learning_step.ScorePoolParameters()

    @property
    def header(self):
        """A printable header for this section of output"""
        return "Step {}: {}".format(".".join(str(e) for e in self._id), self.title)

    @property
    def version(self):
        """The semantic version of this module."""
        return active_learning_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return active_learning_step.__git_revision__

    def description_text(self, P=None):
        """Create the text description of what this step will do.

        Parameters
        ----------
        P: dict
            An optional dictionary of the current values of the control
            parameters.

        Returns
        -------
        str
            A description of the current step.
        """
        if not P:
            P = self.parameters.values_to_dict()

        if P["pool"] == "files":
            where = "the structures in {pool files}"
        else:
            where = seamm.standard_parameters.structure_selection_description(P)
            where = "the structures selected from the database. " + where
        if P["committee"] == "none":
            text = f"List {where} for a random draw, without a committee."
        else:
            if P["committee"] == "previous":
                text = "Score " + where
                text += (
                    " with the committee trained or used earlier in this job: how much "
                    "its members disagree on the energy, forces and pressure."
                )
            else:
                text = "Score " + where + " with the committee in {committee}."
            if str(P["leave out members"]).strip() != "":
                text += " Members {leave out members} are left out."

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

    def run(self):
        """Score the candidates.

        Returns
        -------
        seamm.Node
            The next node object in the flowchart.
        """
        next_node = super().run(printer)

        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )
        printer.important(__(self.description_text(P), indent=self.indent))

        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)

        if P["pool"] == "files":
            pools = self.resolve_files(P["pool files"], "pool file")
        else:
            configurations = self.select_configurations(P)
            pools = [directory / "pool.extxyz"]
            write_pool(configurations, pools[0])
            printer.normal(
                __(
                    f"Wrote the {len(configurations)} structures to {pools[0].name}.",
                    indent=self.indent + 4 * " ",
                )
            )
            printer.normal("")

        args = ["score", *(str(p) for p in pools), "--out", "scores.csv"]
        if P["committee"] == "none":
            args.append("--no-models")
        else:
            committee = self.committee_directory(P)
            # each member's checkpoint: xnn reads a directory as a hub model
            members = [m / "best.pt" for m in member_dirs(committee)]
            missing = [str(m) for m in members if not m.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{self.title}: the committee has members without a checkpoint: "
                    + ", ".join(missing)
                )
            args += ["--models", *(str(m) for m in members)]
            for member in str(P["leave out members"]).replace(",", " ").split():
                args += ["--drop-member", member]
            args += ["--dispersion", "true" if P["add dispersion"] else "false"]
            args += ["--device", self.device(P)]
            if P["precision"] != "the model's":
                args += ["--dtype", P["precision"]]
            args += ["--batch-size", str(P["batch size"])]
            if P["per-atom spreads"]:
                args += ["--atoms", "atoms.extxyz"]
        carry = str(P["keep keys"]).split()
        if P["pool"] != "files":
            carry += [key for key in POOL_KEYS if key not in carry]
        for key in carry:
            args += ["--carry", key]

        self.run_seamm_al(args)
        if self.input_only:
            return next_node

        scores = directory / "scores.csv"
        if P["committee"] != "none":
            self.state["committee"] = str(committee)
        self.state["scores"] = [str(scores)]
        self.state["pool"] = "files" if P["pool"] == "files" else "database"

        self.analyze(scores=scores, P=P)

        return next_node

    def analyze(self, indent="", scores=None, P=None, **kwargs):
        """Print a summary of the scores and store the results.

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        scores : pathlib.Path
            The scores file.
        P : dict
            The parameters of the step.
        """
        if scores is None:
            return
        with open(scores, newline="") as fh:
            rows = list(csv.DictReader(fh))

        kinds = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        text = f"Scored {len(rows)} structures: " + ", ".join(
            f"{n} {kind}s" for kind, n in sorted(kinds.items())
        )
        text += f", in {scores.name}."
        printer.normal(__(text, indent=self.indent + 4 * " "))
        printer.normal("")

        data = {"scores": str(scores), "n_frames": len(rows)}
        rho = np.array([_float(row.get("rho_forces")) for row in rows], dtype=float)
        rho = rho[np.isfinite(rho)]
        if rho.size > 0:
            data["max_rho_forces"] = float(rho.max())
            data["median_rho_forces"] = float(np.median(rho))
            self._print_spreads(rows)

        if P is not None:
            self.store_results(data=data)

    def _print_spreads(self, rows):
        """Print the spreads' statistics per kind, and the largest ones."""
        table = {"Kind": [], "Spread": [], "Units": [], "Median": [], "Max": []}
        for kind in sorted({row["kind"] for row in rows}):
            these = [row for row in rows if row["kind"] == kind]
            for name, units in SPREADS:
                values = np.array([_float(row.get(name)) for row in these])
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                table["Kind"].append(kind)
                table["Spread"].append(name)
                table["Units"].append(units)
                table["Median"].append(f"{np.median(values):.4g}")
                table["Max"].append(f"{values.max():.4g}")
        text = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("left", "left", "left", "right", "right"),
            disable_numparse=True,
        )
        printer.normal(textwrap.indent(text, self.indent + 7 * " "))
        printer.normal("")

        top = sorted(rows, key=lambda r: -_float(r.get("rho_forces")))[:10]
        table = {
            "Structure": [_name(r) for r in top],
            "Kind": [r["kind"] for r in top],
            "Atoms": [r["n_atoms"] for r in top],
            "rho_forces": [f"{_float(r.get('rho_forces')):.4g}" for r in top],
            "rho_energy": [f"{_float(r.get('rho_energy')):.4g}" for r in top],
        }
        text = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("left", "left", "right", "right", "right"),
            disable_numparse=True,
        )
        printer.normal(__("The largest force spreads:", indent=self.indent + 4 * " "))
        printer.normal(textwrap.indent(text, self.indent + 7 * " "))
        printer.normal("")


#: The spreads summarised, and their units
SPREADS = (
    ("rho_energy", "eV"),
    ("rho_forces", "eV/Å"),
    ("rho_forces_max_atom", "eV/Å"),
    ("rho_pressure", "atm"),
    ("rho_pressure_molecular", "atm"),
)

#: The keys that tie a scored frame to its configuration in the database
POOL_KEYS = ("system_id", "configuration_id")


def _name(row):
    """A scored frame's name, or its file and index if it has none."""
    if row.get("configuration"):
        return row["configuration"]
    return f"{Path(row['file']).name} #{row['index']}"


def _float(value):
    """A number from a CSV cell, NaN if empty."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_pool(configurations, path):
    """Write SEAMM configurations to an extxyz file for scoring.

    Each frame carries the configuration's name, its total charge, and the ids of
    its system and configuration, so that the selected frames can be found again
    in the database.

    Parameters
    ----------
    configurations : [molsystem._Configuration]
        The structures.
    path : pathlib.Path
        The extxyz file to write.
    """
    from ase import Atoms
    from ase.io import write

    frames = []
    for configuration in configurations:
        periodicity = configuration.periodicity
        if periodicity not in (0, 3):
            raise NotImplementedError(
                f"Configuration {configuration.id} is periodic in {periodicity} "
                "dimensions; only molecules and 3-D cells can be scored."
            )
        atoms = Atoms(
            symbols=list(configuration.atoms.symbols),
            positions=configuration.atoms.get_coordinates(fractionals=False),
        )
        if periodicity == 3:
            atoms.set_cell(configuration.cell.vectors())
            atoms.set_pbc(True)
        name = configuration.name
        if name is None or str(name).strip() == "":
            name = f"configuration {configuration.id}"
        atoms.info["SEAMM/configuration_name"] = str(name)
        atoms.info["total_charge"] = float(configuration.charge or 0)
        atoms.info["system_id"] = int(configuration.system.id)
        atoms.info["configuration_id"] = int(configuration.id)
        frames.append(atoms)
    write(str(path), frames, format="extxyz")
