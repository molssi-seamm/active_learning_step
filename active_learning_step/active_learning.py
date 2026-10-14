# -*- coding: utf-8 -*-

"""Non-graphical part of the Active Learning step in a SEAMM flowchart"""

import logging
import importlib.resources
import sys

import active_learning_step
import molsystem
import seamm
from seamm_util import ureg, Q_  # noqa: F401
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

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


class ActiveLearning(seamm.Node):
    """
    The non-graphical part of an Active Learning step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : ActiveLearningParameters
        The control parameters for Active Learning.

    See Also
    --------
    TkActiveLearning,
    ActiveLearning, ActiveLearningParameters
    """

    def __init__(
        self,
        flowchart=None,
        title="Active Learning",
        namespace="org.molssi.seamm.active_learning",
        extension=None,
        logger=logger,
    ):
        """A step for Active Learning in a SEAMM flowchart.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        namespace : str
            The namespace for the plug-ins of the subflowchart
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        logger.debug(f"Creating Active Learning {self}")
        self.subflowchart = seamm.Flowchart(
            parent=self, name="Active Learning", namespace=namespace
        )  # yapf: disable

        super().__init__(
            flowchart=flowchart,
            title="Active Learning",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._metadata = active_learning_step.metadata

    @property
    def version(self):
        """The semantic version of this module."""
        return active_learning_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return active_learning_step.__git_revision__

    def set_id(self, node_id):
        """Set the id for node to a given tuple"""
        self._id = node_id

        # and set our subnodes
        self.subflowchart.set_ids(self._id)

        return self.next()

    def description_text(self, P=None):
        """Create the text description of what this step will do.
        The dictionary of control values is passed in as P so that
        the code can test values, etc.

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
        self.subflowchart.root_directory = self.flowchart.root_directory

        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        text = self.header + "\n\n"
        while node is not None:
            try:
                text += __(node.description_text(), indent=3 * " ").__str__()
            except Exception as e:
                print(f"Error describing active_learning flowchart: {e} in {node}")
                logger.critical(
                    f"Error describing active_learning flowchart: {e} in {node}"
                )
                raise
            except:  # noqa: E722
                print(
                    "Unexpected error describing active_learning flowchart: "
                    f"{sys.exc_info()[0]} in {str(node)}"
                )
                self.logger.critical(
                    "Unexpected error describing active_learning flowchart: "
                    f"{sys.exc_info()[0]} in {str(node)}"
                )
                raise
            text += "\n"
            node = node.next()

        return text

    def run(self):
        """Run the Active Learning step: each sub-step in turn runs itself.

        Returns
        -------
        seamm.Node
            The next node in the flowchart, after this step.
        """
        next_node = super().run(printer)

        self.subflowchart.root_directory = self.flowchart.root_directory

        node = self.subflowchart.get_node("1").next()
        while node is not None:
            node = node.run()

        return next_node

    def analyze(self, indent="", **kwargs):
        """Do any analysis of the output from this step.

        Also print important results to the local step.out file using
        "printer".

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        """
        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        # Loop over the subnodes, asking them to do their analysis
        while node is not None:
            for value in node.description:
                printer.important(value)
                printer.important(" ")

            node.analyze()

            node = node.next()
