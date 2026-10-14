# -*- coding: utf-8 -*-
"""
Control parameters for the Check Committee step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class CheckCommitteeParameters(seamm.Parameters):  # noqa: E999
    """
    The control parameters for Check Committee.

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
    CheckCommittee, TkCheckCommittee, CheckCommitteeParameters, CheckCommitteeStep
    """

    parameters = {
        "committee": {
            "default": "previous",
            "kind": "string",
            "default_units": "",
            "enumeration": ("previous",),
            "format_string": "",
            "description": "Committee:",
            "help_text": (
                "The committee: 'previous' for the one trained or used earlier in "
                "this job, or the directory of a committee made by Train Committee "
                "(it holds the members m0, m1, ...)."
            ),
        },
        "labelled data": {
            "default": "the committee's training data",
            "kind": "string",
            "default_units": "",
            "enumeration": ("the committee's training data",),
            "format_string": "",
            "description": "Labelled data:",
            "help_text": (
                "Labelled structures to check on: the committee's own training data, "
                "where each member is checked on the fold it never saw, or extended "
                "XYZ files (names or patterns separated by spaces) of new structures "
                "that no member saw."
            ),
        },
        "energy key": {
            "default": "REF_energy",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Energy key:",
            "help_text": "The key of the reference energy in the extended XYZ files.",
        },
        "forces key": {
            "default": "REF_forces",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Forces key:",
            "help_text": "The key of the reference forces in the extended XYZ files.",
        },
        "stress key": {
            "default": "REF_stress",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Stress key:",
            "help_text": "The key of the reference stress in the extended XYZ files.",
        },
        "checkpoint": {
            "default": "best.pt",
            "kind": "string",
            "default_units": "",
            "enumeration": ("best.pt", "last.pt"),
            "format_string": "",
            "description": "Checkpoint:",
            "help_text": (
                "Which checkpoint of each member to use: the best on its validation "
                "fold, or the last."
            ),
        },
        "flag factor": {
            "default": 1.5,
            "kind": "float",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": ".2f",
            "description": "Flag members worse than:",
            "help_text": (
                "Flag a member whose final validation loss is more than this times "
                "the median of the others."
            ),
        },
        "calibrate": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Calibrate thresholds:",
            "help_text": (
                "Spreads to calibrate a selection threshold for, as "
                "NAME:ERROR[:CAPTURE] separated by spaces. The threshold catches the "
                "fraction CAPTURE (default 0.98) of the frames whose error is above "
                "ERROR, e.g. rho_forces:0.15 for force errors above 0.15 eV/Å. "
                "Select Frames can use the thresholds."
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

        logger.debug("CheckCommitteeParameters.__init__")

        super().__init__(
            defaults={**CheckCommitteeParameters.parameters, **defaults}, data=data
        )
