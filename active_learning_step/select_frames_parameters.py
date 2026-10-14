# -*- coding: utf-8 -*-
"""
Control parameters for the Select Frames step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class SelectFramesParameters(seamm.Parameters):  # noqa: E999
    """
    The control parameters for Select Frames.

    You need to replace the "time" entry in dictionary below these comments with the
    definitions of parameters to control this step. The keys are parameters for the
    current plugin,the values are dictionaries as outlined below.

    Examples
    --------
    ::

        parameters = {
            "time": {
                "default": 100.0,
                "kind": "float",
                "default_units": "ps",
                "enumeration": tuple(),
                "format_string": ".1f",
                "description": "Simulation time:",
                "help_text": ("The time to simulate in the dynamics run.")
            },
        }

    parameters : {str: {str: str}}
        A dictionary containing the parameters for the current step.
        Each key of the dictionary is a dictionary that contains the
        the following keys:

    parameters["default"] :
        The default value of the parameter, used to reset it.

    parameters["kind"] : enum()
        Specifies the kind of a variable. One of  "integer", "float", "string",
        "boolean", or "enum"

        While the "kind" of a variable might be a numeric value, it may still have
        enumerated custom values meaningful to the user. For instance, if the parameter
        is a convergence criterion for an optimizer, custom values like "normal",
        "precise", etc, might be adequate. In addition, any parameter can be set to a
        variable of expression, indicated by having "$" as the first character in the
        field. For example, $OPTIMIZER_CONV.

    parameters["default_units"] : str
        The default units, used for resetting the value.

    parameters["enumeration"]: tuple
        A tuple of enumerated values.

    parameters["format_string"]: str
        A format string for "pretty" output.

    parameters["description"]: str
        A short string used as a prompt in the GUI.

    parameters["help_text"]: str
        A longer string to display as help for the user.

    See Also
    --------
    SelectFrames, TkSelectFrames, SelectFramesParameters, SelectFramesStep
    """

    parameters = {
        "scores": {
            "default": "previous",
            "kind": "string",
            "default_units": "",
            "enumeration": ("previous",),
            "format_string": "",
            "description": "Scores:",
            "help_text": (
                "The scores to select from: 'previous' for those of the last Score "
                "Pool in this job, or scores.csv files (names or patterns separated by "
                "spaces)."
            ),
        },
        "method": {
            "default": "largest spreads",
            "kind": "string",
            "default_units": "",
            "enumeration": ("largest spreads", "random"),
            "format_string": "",
            "description": "Select:",
            "help_text": (
                "Select the candidates the committee disagrees on most, or draw them "
                "at random (the first round, before there is a committee)."
            ),
        },
        "random seed": {
            "default": 0,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Random seed:",
            "help_text": "The seed of the random draw.",
        },
        "spread the draw over": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Spread the draw over:",
            "help_text": (
                "Columns of the scores to spread the random draw evenly over, "
                "separated by spaces, for example the composition or temperature."
            ),
        },
        "ranking": {
            "default": "default",
            "kind": "string",
            "default_units": "",
            "enumeration": ("default",),
            "format_string": "",
            "description": "Rank by:",
            "help_text": (
                "The spreads that rank each kind of structure, as KIND=SPREAD[,SPREAD] "
                "separated by spaces. The default ranks cells by "
                "rho_pressure_molecular and rho_forces, and clusters by rho_forces; "
                "several spreads are combined by the sum of the ranks."
            ),
        },
        "thresholds": {
            "default": "none",
            "kind": "string",
            "default_units": "",
            "enumeration": ("none", "from Check Committee"),
            "format_string": "",
            "description": "Thresholds:",
            "help_text": (
                "Select only candidates whose spread reaches a threshold: none, those "
                "calibrated by the last Check Committee in this job, or NAME=VALUE or "
                "KIND:NAME=VALUE separated by spaces (e.g. cluster:rho_forces=0.1)."
            ),
        },
        "combine thresholds": {
            "default": "any",
            "kind": "string",
            "default_units": "",
            "enumeration": ("any", "all"),
            "format_string": "",
            "description": "Thresholds to meet:",
            "help_text": (
                "With several thresholds, whether any of them (as Smith et al.) or all "
                "of them must be met."
            ),
        },
        "budget": {
            "default": "unlimited",
            "kind": "float",
            "default_units": "",
            "enumeration": ("unlimited",),
            "format_string": ".1f",
            "description": "Budget (core-hours):",
            "help_text": "The core-hours available to label the selected structures.",
        },
        "costs": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Cost per structure:",
            "help_text": (
                "The core-hours to label one structure, as KIND=COST or "
                "KIND:GROUP=COST separated by spaces, e.g. cluster=10 cell=2800. "
                "GROUP is a pattern matched against the group column. Without "
                "costs each structure counts one, and the budget is a number of "
                "structures."
            ),
        },
        "group column": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Group column:",
            "help_text": "The column of the scores that the cost groups match.",
        },
        "fill the budget": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Fill the budget:",
            "help_text": (
                "When a structure does not fit in what is left of the budget, skip it "
                "and keep going with cheaper ones, rather than stop."
            ),
        },
        "maximum number": {
            "default": "no limit",
            "kind": "integer",
            "default_units": "",
            "enumeration": ("no limit",),
            "format_string": "d",
            "description": "Most structures:",
            "help_text": "The most structures to select.",
        },
        "caps": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "At most per value:",
            "help_text": (
                "At most N structures for each value of a column, as COLUMN=N "
                "separated by spaces, e.g. source=3."
            ),
        },
        "exclude": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Already labelled:",
            "help_text": (
                "Files listing the names of structures already labelled, one per "
                "line, which are not selected again."
            ),
        },
        "selected file": {
            "default": "selected",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Selected structures file:",
            "help_text": (
                "The name, without the extension, of the file of the selected "
                "structures, copied exactly from the candidates' files."
            ),
        },
        "variable": {
            "default": "selected_configurations",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Variable for the selection:",
            "help_text": (
                "For candidates from the database: the variable to hold the selected "
                "configurations, for later steps to use as their structures. Leave it "
                "empty for none."
            ),
        },
        "new system": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Copy into a new system:",
            "help_text": (
                "For candidates from the database: also copy the selected "
                "configurations into a new system with this name. Leave it empty for "
                "no copy."
            ),
        },
        "results": {
            "default": {},
            "kind": "dictionary",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "results",
            "help_text": "The results to save to variables or in tables.",
        },
    }

    def __init__(self, defaults={}, data=None):
        """
        Initialize the parameters, by default with the parameters defined above

        Parameters
        ----------
        defaults: dict
            A dictionary of parameters to initialize. The parameters
            above are used first and any given will override/add to them.
        data: dict
            A dictionary of keys and a subdictionary with value and units
            for updating the current, default values.

        Returns
        -------
        None
        """

        logger.debug("SelectFramesParameters.__init__")

        super().__init__(
            defaults={**SelectFramesParameters.parameters, **defaults}, data=data
        )
