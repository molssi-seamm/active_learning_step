# -*- coding: utf-8 -*-

"""The graphical part of a Select Frames step"""

from .tk_substep import TkSubstep


class TkSelectFrames(TkSubstep):
    """
    The graphical part of a Select Frames step in a flowchart.

    The dialog shows only the parameters that apply to the current choices.

    See Also
    --------
    SelectFrames, SelectFramesParameters, TkSubstep
    """

    dialog_title = "Select Frames"
    triggers = ("method", "thresholds", "budget")

    def visible(self, key):
        """Whether the parameter ``key`` applies to the current choices."""
        random = self.value("method") == "random"
        if key in ("random seed", "spread the draw over"):
            return random
        if key in ("ranking", "thresholds"):
            return not random
        if key == "combine thresholds":
            return not random and self.value("thresholds") not in ("", "none")
        if key == "fill the budget":
            return self.value("budget") != "unlimited"
        return True
