# -*- coding: utf-8 -*-

"""The graphical part of a Score Pool step"""

from .tk_substep import TkSubstep


class TkScorePool(TkSubstep):
    """
    The graphical part of a Score Pool step in a flowchart.

    The dialog shows only the parameters that apply to the current choices.

    See Also
    --------
    ScorePool, ScorePoolParameters, TkSubstep
    """

    dialog_title = "Score Pool"
    triggers = ("committee", "pool")

    #: The parameters that only apply when there is a committee
    MODEL_KEYS = (
        "leave out members",
        "add dispersion",
        "device",
        "precision",
        "batch size",
        "per-atom spreads",
    )

    def visible(self, key):
        """Whether the parameter ``key`` applies to the current choices."""
        if key == "source systems":
            return self.value("pool") != "files"
        if key == "pool files":
            return self.value("pool") == "files"
        if key in self.MODEL_KEYS:
            return self.value("committee") != "none"
        return True
