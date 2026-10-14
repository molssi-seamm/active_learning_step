# -*- coding: utf-8 -*-
"""
Control parameters for the Train Committee step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class TrainCommitteeParameters(seamm.Parameters):  # noqa: E999
    """
    The control parameters for Train Committee.

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
    TrainCommittee, TkTrainCommittee, TrainCommitteeParameters, TrainCommitteeStep
    """

    parameters = {
        "training data": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Labelled data:",
            "help_text": (
                "The labelled structures to train on: extended XYZ files with the "
                "reference energies and forces (for example from the Write Structure "
                "step). Give names or patterns such as data/*.extxyz, separated by "
                "spaces, relative to the job's directory."
            ),
        },
        "recipe": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "xnn recipe:",
            "help_text": (
                "The xnn training configuration (YAML): the model, its "
                "hyperparameters and the training schedule. Every member is trained "
                "with it, on a different part of the data."
            ),
        },
        "recipe changes": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Changes to the recipe:",
            "help_text": (
                "Settings to change in the recipe, as KEY=VALUE separated by spaces, "
                "for example optim.lr=0.005 trainer.epochs=200."
            ),
        },
        "number of members": {
            "default": 5,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Number of members:",
            "help_text": (
                "The number of members, which is also the number of folds: each "
                "member trains on all but one fold, and never sees that fold."
            ),
        },
        "members to train": {
            "default": "all",
            "kind": "string",
            "default_units": "",
            "enumeration": ("all",),
            "format_string": "",
            "description": "Members to train:",
            "help_text": (
                "Which members to train in this step, by number (0, 1, ...) separated "
                "by spaces, or all. Members trained in other jobs into the same "
                "directory join the same committee."
            ),
        },
        "number of steps": {
            "default": "from the recipe",
            "kind": "integer",
            "default_units": "",
            "enumeration": ("from the recipe",),
            "format_string": "d",
            "description": "Optimizer steps:",
            "help_text": (
                "The number of optimizer steps for each member, instead of the "
                "recipe's number of epochs."
            ),
        },
        "stratify": {
            "default": "clusters and cells separately",
            "kind": "string",
            "default_units": "",
            "enumeration": ("clusters and cells separately", "no"),
            "format_string": "",
            "description": "Folds:",
            "help_text": (
                "How the data is split into folds: with clusters and periodic cells "
                "split separately, so each fold has its share of both, or a plain "
                "random split."
            ),
        },
        "stratify by": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Also balance by:",
            "help_text": (
                "Keys in the structures' info to also balance the folds by, separated "
                "by spaces, for example the temperature or the composition."
            ),
        },
        "cell share": {
            "default": "one pass over the data",
            "kind": "float",
            "default_units": "",
            "enumeration": ("one pass over the data",),
            "format_string": ".2f",
            "description": "Share of cell batches:",
            "help_text": (
                "The fraction of each epoch's batches that are periodic cells. By "
                "default each epoch goes once over the clusters and once over the "
                "cells; a set share draws the cells more or less often than that."
            ),
        },
        "cell batch size": {
            "default": 4,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Cells per batch:",
            "help_text": (
                "The number of periodic cells in a batch. Clusters use the recipe's "
                "batch size."
            ),
        },
        "data seed": {
            "default": 0,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Seed for the folds:",
            "help_text": "The random seed that assigns the structures to folds.",
        },
        "initial seed": {
            "default": 1000,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Seed of member 0:",
            "help_text": (
                "The random seed of the first member's weights; member n uses this "
                "plus n."
            ),
        },
        "model name": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Save as the model:",
            "help_text": (
                "Also save the committee among your xnn models with this name, so "
                "that the Model Chemistry step offers it as xnn:MLFF@<name>, for "
                "example to run molecular dynamics with it and record where its "
                "members disagree. Leave empty not to save it."
            ),
        },
        "device": {
            "default": "from the recipe",
            "kind": "string",
            "default_units": "",
            "enumeration": ("from the recipe", "cpu", "cuda", "mps"),
            "format_string": "",
            "description": "Device:",
            "help_text": (
                "The PyTorch device to train on: cpu, cuda (or cuda:N), or mps; by "
                "default the recipe's."
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

        logger.debug("TrainCommitteeParameters.__init__")

        super().__init__(
            defaults={**TrainCommitteeParameters.parameters, **defaults}, data=data
        )
