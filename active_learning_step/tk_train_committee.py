# -*- coding: utf-8 -*-

"""The graphical part of a Train Committee step"""

from .tk_substep import TkSubstep


class TkTrainCommittee(TkSubstep):
    """
    The graphical part of a Train Committee step in a flowchart.

    The dialog shows only the parameters that apply to the current choices.

    See Also
    --------
    TrainCommittee, TrainCommitteeParameters, TkSubstep
    """

    dialog_title = "Train Committee"
    triggers = ("stratify",)

    def visible(self, key):
        """Whether the parameter ``key`` applies to the current choices."""
        if key == "stratify by":
            return self.value("stratify") != "no"
        return True
