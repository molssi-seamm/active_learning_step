# -*- coding: utf-8 -*-

"""
active_learning_step
A SEAMM plug-in for Active Learning
"""

# Bring up the classes so that they appear to be directly in
# the active_learning_step package.

from .active_learning import ActiveLearning  # noqa: F401, E501
from .active_learning_parameters import ActiveLearningParameters  # noqa: F401, E501
from .active_learning_step import ActiveLearningStep  # noqa: F401, E501
from .tk_active_learning import TkActiveLearning  # noqa: F401, E501

from .metadata import metadata  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
