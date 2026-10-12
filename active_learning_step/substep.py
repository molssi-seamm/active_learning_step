# -*- coding: utf-8 -*-

"""Base class for substeps in the Active Learning step."""

import logging

import seamm
import seamm_util.printing as printing

import active_learning_step

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Active Learning")


class Substep(seamm.Node):
    """A base class for substeps in the Active Learning subflowchart.

    The top-level ActiveLearning step iterates over the substeps
    of its subflowchart and calls each substep's `run()` method. Each substep
    is responsible for preparing its own input, invoking Active Learning,
    analyzing its output, and storing its results.

    This base class provides:

    * `is_runable` - whether the top-level loop should call `run()` on this
      substep. Override to return False for substeps that only contribute
      input (e.g., an Initialization step) rather than invoking the code.
    * `input_only` - a flag for "write input but don't run" mode, useful for
      debugging and testing.
    * `options` / `global_options` - delegated to the parent step so that
      command-line options reach substeps uniformly.
    * `header` - a printable header for this section of step.out.
    * `version` / `git_revision` - delegated to the plug-in package so that
      every substep reports the plug-in's version, not seamm.Node's.
    """

    def __init__(
        self,
        flowchart=None,
        title="no title",
        extension=None,
        logger=logger,
        module=None,
    ):
        self._input_only = False

        super().__init__(
            flowchart=flowchart,
            title=title,
            extension=extension,
            logger=logger,
            module=module,
        )

    @property
    def header(self):
        """A printable header for this section of output."""
        return "Step {}: {}".format(".".join(str(e) for e in self._id), self.title)

    @property
    def version(self):
        """The semantic version of the active_learning_step package."""
        return active_learning_step.__version__

    @property
    def git_revision(self):
        """The git revision of the active_learning_step package."""
        return active_learning_step.__git_revision__

    @property
    def global_options(self):
        """The global SEAMM options, delegated to the parent step."""
        return self.parent.global_options

    @property
    def options(self):
        """The Active Learning options, delegated to the parent step."""
        return self.parent.options

    @property
    def input_only(self):
        """Whether to write the input only, not run Active Learning."""
        return self._input_only

    @input_only.setter
    def input_only(self, value):
        self._input_only = value

    @property
    def is_runable(self):
        """Whether this substep should be run by the top-level loop.

        Override to return False for substeps that only contribute input
        (e.g., an Initialization step) rather than invoking
        Active Learning themselves.
        """
        return True
