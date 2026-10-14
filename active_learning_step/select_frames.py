# -*- coding: utf-8 -*-

"""Non-graphical part of the Select Frames step in an Active Learning flowchart"""

import contextlib
import io
import logging
from pathlib import Path
import importlib.resources
from types import SimpleNamespace

import active_learning_step
import molsystem
import seamm
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

from .selection import read_scores, select_main

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


class SelectFrames(active_learning_step.Substep):
    """
    The non-graphical part of a Select Frames step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : SelectFramesParameters
        The control parameters for Select Frames.

    See Also
    --------
    TkSelectFrames,
    SelectFrames, SelectFramesParameters
    """

    def __init__(
        self, flowchart=None, title="Select Frames", extension=None, logger=logger
    ):
        """A substep for Select Frames in a subflowchart for Active Learning.

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
        logger.debug(f"Creating Select Frames {self}")

        super().__init__(
            flowchart=flowchart,
            title="Select Frames",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "select"
        self._model = None
        self._metadata = active_learning_step.metadata
        self.parameters = active_learning_step.SelectFramesParameters()

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

        if P["scores"] == "previous":
            source = "the candidates scored earlier in this job"
        else:
            source = "the candidates scored in {scores}"
        if P["method"] == "random":
            text = f"Draw at random (seed {{random seed}}) from {source}"
            if str(P["spread the draw over"]).strip() != "":
                text += ", spread evenly over {spread the draw over}"
            text += "."
        else:
            text = f"Select from {source} those the committee disagrees on most"
            if P["ranking"] != "default":
                text += ", ranked by {ranking}"
            text += "."
            if P["thresholds"] == "from Check Committee":
                text += (
                    " Only those reaching the thresholds calibrated by Check Committee "
                    "are selected."
                )
            elif str(P["thresholds"]).strip() not in ("", "none"):
                text += " Only those reaching {thresholds} are selected."
        if str(P["budget"]) != "unlimited":
            text += " The budget is {budget} core-hours"
            if str(P["costs"]).strip() != "":
                text += ", with costs {costs}"
            text += "."
        if str(P["maximum number"]) != "no limit":
            text += " At most {maximum number} are selected."
        text += (
            " The selected structures are copied exactly from the candidates' files "
            "to {selected file}.extxyz (or .sdf)."
        )

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

    def run(self):
        """Select the structures to label.

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

        if P["scores"] == "previous":
            if "scores" not in self.state:
                raise RuntimeError(
                    f"{self.title}: there are no scores earlier in this job. Use "
                    "Score Pool first, or give the scores files."
                )
            scores = [Path(p) for p in self.state["scores"]]
        else:
            scores = self.resolve_files(P["scores"], "scores file")

        random = P["method"] == "random"
        thresholds = []
        if not random:
            if P["thresholds"] == "from Check Committee":
                if not self.state.get("thresholds"):
                    raise RuntimeError(
                        f"{self.title}: no thresholds were calibrated earlier in this "
                        "job. Calibrate them in Check Committee, or give them."
                    )
                thresholds = [f"{k}={v}" for k, v in self.state["thresholds"].items()]
            elif str(P["thresholds"]).strip() not in ("", "none"):
                thresholds = str(P["thresholds"]).split()
        prefix = directory / str(P["selected file"]).strip()
        args = SimpleNamespace(
            scores=[str(p) for p in scores],
            budget=None if str(P["budget"]) == "unlimited" else float(P["budget"]),
            # without costs every structure counts one
            cost=str(P["costs"]).split() or ["cluster=1", "cell=1"],
            group_column=str(P["group column"]).strip() or None,
            rank=[] if P["ranking"] == "default" else str(P["ranking"]).split(),
            threshold=thresholds,
            combine=P["combine thresholds"],
            cap=str(P["caps"]).split(),
            exclude=[str(p) for p in self._exclude_files(P)],
            random=int(P["random seed"]) if random else None,
            stratify_column=str(P["spread the draw over"]).split() if random else [],
            fill=bool(P["fill the budget"]),
            max_frames=(
                None
                if str(P["maximum number"]) == "no limit"
                else int(P["maximum number"])
            ),
            out=str(directory / "selection.csv"),
            output=str(prefix),
            rejected=str(directory / "rejected.csv"),
        )
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            selected = select_main(args)
        self.logger.info(report.getvalue())

        files = sorted(directory.glob(prefix.name + ".*"))
        files = [f for f in files if f.suffix in (".extxyz", ".sdf")]
        cost = float(selected[-1]["running_cost"]) if selected else 0.0
        self.state["selected"] = [str(f) for f in files]

        self._print_selection(
            selected, cost, files, args, counted=str(P["costs"]).strip() == ""
        )

        self.database(selected, P)
        self.analyze(selected=selected, files=files, cost=cost, P=P)

        return next_node

    def _print_selection(self, selected, cost, files, args, counted):
        """Print what was selected, what was not and why, and where it went."""
        n = sum(len(read_scores(path)) for path in args.scores)
        kinds = {}
        for row in selected:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        text = f"Selected {len(selected)} of {n} structures"
        if kinds:
            text += " (" + ", ".join(_plural(v, k) for k, v in sorted(kinds.items()))
            text += ")"
        if not counted:
            text += f", costing {cost:.1f}"
            if args.budget is not None:
                text += f" of {args.budget:.1f}"
            text += " core-hours"
        elif args.budget is not None:
            text += f" of the {args.budget:g} allowed"
        text += "."
        reasons = {}
        for row in read_scores(args.rejected):
            key = row["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            text += " Not selected: "
            text += ", ".join(f"{v} {k}" for k, v in sorted(reasons.items())) + "."
        if files:
            text += " They are in " + " and ".join(f.name for f in files) + "."
        printer.normal(__(text, indent=self.indent + 4 * " "))
        printer.normal("")

    def _exclude_files(self, P):
        """The files of names already labelled."""
        if str(P["exclude"]).strip() == "":
            return []
        return self.resolve_files(P["exclude"], "file of labelled names")

    def database(self, selected, P):
        """Put configurations selected from the database in a variable or a system.

        Only for candidates that came from the database, whose scores carry the
        configurations' ids.
        """
        variable = str(P["variable"]).strip()
        name = str(P["new system"]).strip()
        if variable == "" and name == "":
            return
        if not selected or "configuration_id" not in selected[0]:
            # not from the database: the selection is only in the file
            return
        system_db = self.get_variable("_system_db")
        configurations = [
            system_db.get_configuration(int(row["configuration_id"]))
            for row in selected
        ]
        if variable != "":
            self.set_variable(variable, configurations)
            text = (
                f"The {len(configurations)} selected configurations are in the "
                f"variable '{variable}'."
            )
            printer.normal(__(text, indent=self.indent + 4 * " "))
        if name != "":
            system = copy_to_new_system(system_db, configurations, name)
            text = (
                f"The selected configurations were copied into the new system "
                f"'{system.name}'."
            )
            printer.normal(__(text, indent=self.indent + 4 * " "))
        printer.normal("")

    def analyze(self, indent="", selected=None, files=(), cost=0.0, P=None, **kwargs):
        """Store the results.

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        selected : [dict]
            The selected rows.
        files : [pathlib.Path]
            The files of the selected structures.
        cost : float
            Their cost, in core-hours.
        P : dict
            The parameters of the step.
        """
        if selected is None or P is None:
            return
        self.store_results(
            data={
                "n_selected": len(selected),
                "cost": cost,
                "selected_file": " ".join(str(f) for f in files),
            },
        )


def _plural(n, noun):
    """'1 cluster', '2 clusters'."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def copy_to_new_system(system_db, configurations, name):
    """Copy configurations into a new system.

    Parameters
    ----------
    system_db : molsystem.SystemDB
        The database.
    configurations : [molsystem._Configuration]
        The configurations to copy.
    name : str
        The name of the new system.

    Returns
    -------
    molsystem._System
        The new system.
    """
    system = system_db.create_system(name=name, make_current=False)
    for previous in configurations:
        cell_parameters = None
        if previous.periodicity != 0:
            cell_parameters = previous.cell.parameters
        new = system.create_configuration(
            name=previous.name,
            periodicity=previous.periodicity,
            coordinate_system=previous.coordinate_system,
            coordinates=previous.atoms.get_coordinates(),
            symmetry=previous.symmetry_id,
            cell_parameters=cell_parameters,
            atomset=previous.atomset,
            bondset=previous.bondset,
            make_current=False,
        )
        new.charge = previous.charge
        new.spin_multiplicity = previous.spin_multiplicity
    return system
