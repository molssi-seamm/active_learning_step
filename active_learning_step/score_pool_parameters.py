# -*- coding: utf-8 -*-
"""
Control parameters for the Score Pool step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class ScorePoolParameters(seamm.Parameters):  # noqa: E999
    """
    The control parameters for Score Pool.

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
    ScorePool, TkScorePool, ScorePoolParameters, ScorePoolStep
    """

    parameters = {
        "committee": {
            "default": "previous",
            "kind": "string",
            "default_units": "",
            "enumeration": ("previous", "none"),
            "format_string": "",
            "description": "Committee:",
            "help_text": (
                "The committee: 'previous' for the one trained or used earlier in "
                "this job, or the directory of a committee made by Train Committee "
                "(it holds the members m0, m1, ...). 'none' only lists the candidates, "
                "for a random draw before there is a committee."
            ),
        },
        "pool": {
            "default": "structures in the database",
            "kind": "string",
            "default_units": "",
            "enumeration": ("structures in the database", "files"),
            "format_string": "",
            "description": "Candidates:",
            "help_text": (
                "The candidate structures to score: structures in this job's database, "
                "or extended XYZ or SDF files, for example a trajectory."
            ),
        },
        **seamm.standard_parameters.structure_selection_parameters,
        "pool files": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Files:",
            "help_text": (
                "Extended XYZ (.extxyz, .xyz, optionally gzipped) or SDF files of "
                "candidates: names or patterns separated by spaces, relative to the "
                "job's directory."
            ),
        },
        "keep keys": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Keep info keys:",
            "help_text": (
                "Keys of the candidates' info to copy into the scores, separated by "
                "spaces, for example to set costs or to balance a selection by them."
            ),
        },
        "per-atom spreads": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Write per-atom spreads:",
            "help_text": (
                "Also write every candidate with the spread of the force on each atom "
                "(atoms.extxyz), to see where the committee disagrees."
            ),
        },
        "batch size": {
            "default": 16,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Clusters per batch:",
            "help_text": (
                "The number of clusters scored together. Periodic cells are scored "
                "one at a time."
            ),
        },
        "leave out members": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Leave out members:",
            "help_text": (
                "Members to leave out of the committee, by number (0, 1, ...), "
                "separated by spaces: for example, ones that Check Committee flagged."
            ),
        },
        "add dispersion": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Add the dispersion:",
            "help_text": (
                "For models trained on labels with the dispersion removed, whether to "
                "add it back. It is the same for every member, so it does not change "
                "the spreads."
            ),
        },
        "device": {
            "default": "default",
            "kind": "string",
            "default_units": "",
            "enumeration": ("default", "cpu", "cuda", "mps"),
            "format_string": "",
            "description": "Device:",
            "help_text": (
                "The PyTorch device to run the models on: cpu, cuda (or cuda:N), or "
                "mps. 'default' uses the device in xnn.ini, else the CPU."
            ),
        },
        "precision": {
            "default": "the model's",
            "kind": "string",
            "default_units": "",
            "enumeration": ("the model's", "float32", "float64"),
            "format_string": "",
            "description": "Precision:",
            "help_text": "The floating-point precision to evaluate the models in.",
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

        logger.debug("ScorePoolParameters.__init__")

        super().__init__(
            defaults={**ScorePoolParameters.parameters, **defaults}, data=data
        )
