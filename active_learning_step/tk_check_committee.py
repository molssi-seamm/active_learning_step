# -*- coding: utf-8 -*-

"""The graphical part of a Check Committee step"""

from .tk_substep import TkSubstep


class TkCheckCommittee(TkSubstep):
    """
    The graphical part of a Check Committee step in a flowchart.

    The dialog shows only the parameters that apply to the current choices.

    See Also
    --------
    CheckCommittee, CheckCommitteeParameters, TkSubstep
    """

    dialog_title = "Check Committee"
    triggers = ("labelled data",)

    def visible(self, key):
        """Whether the parameter ``key`` applies to the current choices."""
        return True
