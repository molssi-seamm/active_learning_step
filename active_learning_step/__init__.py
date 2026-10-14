# -*- coding: utf-8 -*-

"""
active_learning_step
A SEAMM plug-in for Active Learning
"""

# Bring up the classes so that they appear to be directly in
# the active_learning_step package.
#
# The library (committee, molecules, pool, selection) and the seamm-al command also
# run in MLFF environments that have torch and xnn but not SEAMM, so the step's
# classes are only imported when SEAMM is there.
try:
    import seamm  # noqa: F401
except ImportError:
    pass
else:
    # The base class of the sub-steps, at the package level for them to subclass
    from .substep import Substep  # noqa: F401

    from .active_learning import ActiveLearning  # noqa: F401, E501
    from .active_learning_step import ActiveLearningStep  # noqa: F401, E501
    from .tk_active_learning import TkActiveLearning  # noqa: F401, E501

    from .metadata import metadata  # noqa: F401

    from .train_committee_step import TrainCommitteeStep  # noqa: F401
    from .train_committee import TrainCommittee  # noqa: F401
    from .train_committee_parameters import TrainCommitteeParameters  # noqa: F401
    from .tk_train_committee import TkTrainCommittee  # noqa: F401

    from .check_committee_step import CheckCommitteeStep  # noqa: F401
    from .check_committee import CheckCommittee  # noqa: F401
    from .check_committee_parameters import CheckCommitteeParameters  # noqa: F401
    from .tk_check_committee import TkCheckCommittee  # noqa: F401

    from .score_pool_step import ScorePoolStep  # noqa: F401
    from .score_pool import ScorePool  # noqa: F401
    from .score_pool_parameters import ScorePoolParameters  # noqa: F401
    from .tk_score_pool import TkScorePool  # noqa: F401

    from .select_frames_step import SelectFramesStep  # noqa: F401
    from .select_frames import SelectFrames  # noqa: F401
    from .select_frames_parameters import SelectFramesParameters  # noqa: F401
    from .tk_select_frames import TkSelectFrames  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
