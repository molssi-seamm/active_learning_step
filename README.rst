=============================
SEAMM Active Learning Plug-in
=============================

.. image:: https://img.shields.io/github/issues-pr-raw/molssi-seamm/active_learning_step
   :target: https://github.com/molssi-seamm/active_learning_step/pulls
   :alt: GitHub pull requests

.. image:: https://github.com/molssi-seamm/active_learning_step/workflows/CI/badge.svg
   :target: https://github.com/molssi-seamm/active_learning_step/actions
   :alt: Build Status

.. image:: https://codecov.io/gh/molssi-seamm/active_learning_step/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/molssi-seamm/active_learning_step
   :alt: Code Coverage

.. image:: https://github.com/molssi-seamm/active_learning_step/workflows/CodeQL/badge.svg
   :target: https://github.com/molssi-seamm/active_learning_step/security/code-scanning
   :alt: Code Quality

.. image:: https://github.com/molssi-seamm/active_learning_step/workflows/Release/badge.svg
   :target: https://molssi-seamm.github.io/active_learning_step/index.html
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/active_learning_step.svg
   :target: https://pypi.python.org/pypi/active_learning_step
   :alt: PyPi VERSION

A SEAMM plug-in for query-by-committee active learning of machine-learned force
fields: a committee of models trained on different folds of the data disagrees where
the model is likely wrong, so only the frames it flags need labelling.

* Free software: BSD-3-Clause
* Documentation: https://molssi-seamm.github.io/active_learning_step/index.html
* Code: https://github.com/molssi-seamm/active_learning_step

Features
--------

* ``active_learning_step.committee``: a committee of xnn potentials and its spreads
  (energy, force, per-atom force, atomic and molecular pressure), for molecules,
  ions and charged systems.
* ``seamm-al score``: the committee's spreads for every frame of extxyz or SDF pools.
* ``seamm-al select``: rank the frames and take them within a core-hour budget, with
  caps per group and exclusions, writing the chosen frames as a verbatim subset of
  the input files.

Acknowledgements
----------------

This package was created with the `molssi-seamm/cookiecutter-seamm-plugin`_ tool, which
is based on the excellent Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`molssi-seamm/cookiecutter-seamm-plugin`: https://github.com/molssi-seamm/cookiecutter-seamm-plugin

Developed by the Molecular Sciences Software Institute (MolSSI_),
which receives funding from the `National Science Foundation`_ under
award CHE-2136142.

.. _MolSSI: https://molssi.org
.. _`National Science Foundation`: https://www.nsf.gov
