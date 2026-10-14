# -*- coding: utf-8 -*-

"""Non-graphical part of the Train Committee step in an Active Learning flowchart"""

import json
import logging
from pathlib import Path
import importlib.resources
import shlex
import textwrap

from tabulate import tabulate

import active_learning_step
import molsystem
import seamm
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

from .check import member_dirs, member_losses

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


class TrainCommittee(active_learning_step.Substep):
    """
    The non-graphical part of a Train Committee step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : TrainCommitteeParameters
        The control parameters for Train Committee.

    See Also
    --------
    TkTrainCommittee,
    TrainCommittee, TrainCommitteeParameters
    """

    def __init__(
        self, flowchart=None, title="Train Committee", extension=None, logger=logger
    ):
        """A substep for Train Committee in a subflowchart for Active Learning.

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
        logger.debug(f"Creating Train Committee {self}")

        super().__init__(
            flowchart=flowchart,
            title="Train Committee",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "train"
        self._model = None
        self._metadata = active_learning_step.metadata
        self.parameters = active_learning_step.TrainCommitteeParameters()

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

        members = str(P["members to train"]).strip()
        text = (
            "Train a committee of {number of members} xnn models with the recipe "
            "{recipe} on the labelled structures in {training data}. Each member "
            "trains on all but one of {number of members} folds"
        )
        if P["stratify"] == "no":
            text += ", chosen at random"
        else:
            text += ", with clusters and periodic cells split separately"
            if str(P["stratify by"]).strip() != "":
                text += " and balanced by {stratify by}"
        text += "."
        if members != "all":
            text += " Only members {members to train} are trained in this step."
        if str(P["number of steps"]) != "from the recipe":
            text += " Each member takes {number of steps} optimizer steps."
        if str(P["recipe changes"]).strip() != "":
            text += " The recipe is changed by {recipe changes}."

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

    def run(self):
        """Train the committee.

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

        data = self.resolve_files(P["training data"], "labelled data file")
        recipe = self.resolve_files(P["recipe"], "recipe")
        if len(recipe) != 1:
            raise ValueError(f"{self.title}: give one recipe, not {len(recipe)}")
        out = directory / "committee"
        k = int(P["number of members"])

        args = ["train", *(str(p) for p in data)]
        args += ["--config", str(recipe[0]), "--out", str(out), "--k", str(k)]
        members = str(P["members to train"]).strip()
        if members != "all":
            for member in members.replace(",", " ").split():
                args += ["--member", member]
        if str(P["number of steps"]) != "from the recipe":
            args += ["--steps", str(P["number of steps"])]
        args += ["--stratify", "none" if P["stratify"] == "no" else "kind"]
        for column in str(P["stratify by"]).split():
            args += ["--stratify-column", column]
        if str(P["cell share"]) != "one pass over the data":
            args += ["--cell-share", str(P["cell share"])]
        args += ["--cell-batch-size", str(P["cell batch size"])]
        args += ["--data-seed", str(P["data seed"])]
        args += ["--init-seed", str(P["initial seed"])]
        device = str(P["device"]).strip()
        if device not in ("", "from the recipe"):
            args += ["--device", device]
        for change in shlex.split(str(P["recipe changes"])):
            args += ["--set", change]

        self.run_seamm_al(args)
        if self.input_only:
            return next_node

        self.state["committee"] = str(out)
        self.state.pop("thresholds", None)
        self.analyze(committee=out, P=P)

        return next_node

    def analyze(self, indent="", committee=None, P=None, **kwargs):
        """Print the members' training and store the results.

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        committee : pathlib.Path
            The committee's directory.
        P : dict
            The parameters of the step.
        """
        if committee is None:
            return
        manifest = json.loads((committee / "manifest.json").read_text())

        table = {
            "Member": [],
            "Train": [],
            "Validate": [],
            "Epochs": [],
            "Best val loss": [],
            "Final val loss": [],
        }
        val_loss = []
        for mdir in member_dirs(committee):
            info_file = mdir / "member.json"
            if not info_file.exists():
                continue
            info = json.loads(info_file.read_text())
            final, best, _, _ = member_losses(mdir)
            table["Member"].append(mdir.name)
            table["Train"].append(info.get("n_train"))
            table["Validate"].append(info.get("n_val"))
            table["Epochs"].append(info.get("epochs"))
            table["Best val loss"].append("" if best is None else f"{best:.4g}")
            table["Final val loss"].append("" if final is None else f"{final:.4g}")
            val_loss.append(best)

        kinds = ", ".join(f"{n} {kind}s" for kind, n in manifest["kinds"].items())
        text = (
            f"The committee in {committee} has {manifest['k']} members, trained on "
            f"{manifest['n_frames']} structures ({kinds})."
        )
        printer.normal(__(text, indent=self.indent + 4 * " "))
        printer.normal("")
        if table["Member"]:
            text = tabulate(
                table,
                headers="keys",
                tablefmt="rounded_outline",
                colalign=("center", "right", "right", "right", "right", "right"),
                disable_numparse=True,
            )
            printer.normal(textwrap.indent(text, self.indent + 7 * " "))
            printer.normal("")

        if P is not None:
            self.store_results(
                data={
                    "committee": str(committee),
                    "n_members": manifest["k"],
                    "n_frames": manifest["n_frames"],
                    "val_loss": val_loss,
                },
            )
