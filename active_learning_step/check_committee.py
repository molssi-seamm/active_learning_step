# -*- coding: utf-8 -*-

"""Non-graphical part of the Check Committee step in an Active Learning flowchart"""

import io
import json
import logging
from pathlib import Path
import importlib.resources
import textwrap

import active_learning_step
import molsystem
import seamm
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

from .check import print_summary

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


class CheckCommittee(active_learning_step.Substep):
    """
    The non-graphical part of a Check Committee step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : CheckCommitteeParameters
        The control parameters for Check Committee.

    See Also
    --------
    TkCheckCommittee,
    CheckCommittee, CheckCommitteeParameters
    """

    def __init__(
        self, flowchart=None, title="Check Committee", extension=None, logger=logger
    ):
        """A substep for Check Committee in a subflowchart for Active Learning.

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
        logger.debug(f"Creating Check Committee {self}")

        super().__init__(
            flowchart=flowchart,
            title="Check Committee",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "check"
        self._model = None
        self._metadata = active_learning_step.metadata
        self.parameters = active_learning_step.CheckCommitteeParameters()

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

        if P["committee"] == "previous":
            text = "Check the committee trained or used earlier in this job"
        else:
            text = "Check the committee in {committee}"
        if P["labelled data"] == "the committee's training data":
            text += (
                " on its own training data: each member's errors on the fold it "
                "never saw are compared with the committee's spreads."
            )
        else:
            text += " on the labelled structures in {labelled data}."
        text += (
            " Members whose final validation loss is more than {flag factor} times "
            "the others' median are flagged."
        )
        if str(P["leave out members"]).strip() != "":
            text += " Members {leave out members} are left out."
        if str(P["calibrate"]).strip() != "":
            text += " Thresholds are calibrated for {calibrate}."

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

    def run(self):
        """Check the committee.

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

        committee = self.committee_directory(P)
        if P["labelled data"] == "the committee's training data":
            data = self.training_data(committee)
        else:
            data = self.resolve_files(P["labelled data"], "labelled data file")

        args = ["check", str(committee), *(str(p) for p in data)]
        args += ["--out", "check.csv", "--summary", "summary.json"]
        args += ["--energy-key", P["energy key"], "--forces-key", P["forces key"]]
        args += ["--stress-key", P["stress key"], "--checkpoint", P["checkpoint"]]
        args += ["--flag-factor", str(P["flag factor"])]
        args += ["--dispersion", "true" if P["add dispersion"] else "false"]
        args += ["--device", self.device(P)]
        if P["precision"] != "the model's":
            args += ["--dtype", P["precision"]]
        for member in str(P["leave out members"]).replace(",", " ").split():
            args += ["--drop-member", member]
        for spec in str(P["calibrate"]).split():
            args += ["--calibrate", spec]

        self.run_seamm_al(args)
        if self.input_only:
            return next_node

        self.state["committee"] = str(committee)
        summary = json.loads((directory / "summary.json").read_text())
        thresholds = {}
        for key, entry in summary.get("calibration", {}).items():
            if entry.get("threshold") is not None:
                kind, name = key.split("/", 1)
                thresholds[f"{kind}:{name}"] = entry["threshold"]
        self.state["thresholds"] = thresholds

        self.analyze(summary=summary, thresholds=thresholds, P=P)

        return next_node

    def training_data(self, committee):
        """The labelled data files a committee was trained on, from its manifest."""
        path = committee / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.title}: {committee} has no manifest.json, so its training "
                "data is not known. Give the labelled data files."
            )
        files = []
        for entry in json.loads(path.read_text())["data"]:
            file = Path(entry["path"]).expanduser()
            if not file.is_absolute():
                # relative to where the committee was trained
                for base in (committee.parent, committee):
                    if (base / file).exists():
                        file = base / file
                        break
            if not file.exists():
                raise FileNotFoundError(
                    f"{self.title}: the training data {entry['path']} of {committee} "
                    "is not found. Give the labelled data files."
                )
            files.append(file)
        return files

    def analyze(self, indent="", summary=None, thresholds=None, P=None, **kwargs):
        """Print the check and store the results.

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        summary : dict
            The summary written by ``seamm-al check``.
        thresholds : dict
            The calibrated thresholds, as KIND:NAME -> value.
        P : dict
            The parameters of the step.
        """
        if summary is None:
            return
        fh = io.StringIO()
        print_summary(summary, fh=fh)
        printer.normal(textwrap.indent(fh.getvalue(), self.indent + 4 * " "))

        flagged = [e["member"] for e in summary.get("member_health", []) if e["flags"]]
        if flagged:
            if len(flagged) == 1:
                text = f"Member {flagged[0]} is flagged. Consider leaving it"
            else:
                members = ", ".join(str(k) for k in flagged)
                text = f"Members {members} are flagged. Consider leaving them"
            text += (
                " out of the committee (Leave out members in Score Pool), or "
                "retraining."
            )
            printer.normal(__(text, indent=self.indent + 4 * " "))
            printer.normal("")

        if P is not None:
            self.store_results(
                data={
                    "n_members": len(summary.get("members", [])),
                    "flagged_members": flagged,
                    "spearman": json.dumps(summary.get("spearman", {})),
                    "thresholds": json.dumps(thresholds or {}),
                },
            )
