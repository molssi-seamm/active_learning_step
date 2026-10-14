Welcome to the documentation for the Active Learning SEAMM plug-in
==================================================================

Active Learning chooses what to label next when training a machine-learned force
field. A committee of models, each trained on a different part of the data, disagrees
where the model is likely to be wrong; the ``seamm-al`` command scores candidate
structures by that disagreement, selects the most useful ones within a computing
budget, trains the committees, checks that their disagreement tracks the real error,
and runs molecular dynamics with the committee to find new candidates. This first
release provides the library and the command line; the SEAMM flowchart step is still
a placeholder.

.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
      :margin: 0 3 0 0

      A simple introduction

      .. button-link:: ./getting_started/index.html
	 :color: primary
	 :expand:

         To the Getting Started Guide

   .. grid-item-card::  User Guide
      :margin: 0 3 0 0

      A complete guide to using this step

      .. button-link:: ./user_guide/index.html
	 :color: primary
	 :expand:

         To the User Guide

   .. grid-item-card::  Developer Guide
      :margin: 0 3 0 0

      Contributing to the code. Fixing bugs, adding functionality

      .. button-link:: ./developer_guide/index.html
	 :color: primary
	 :expand:

         To the Developer Guide

   .. grid-item-card:: API Reference
      :margin: 0 3 0 0

      The API for the Geometry Analysis Step

      .. button-link:: ./api/index.html
	 :color: primary
	 :expand:

	 To the API Reference.


.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   getting_started/index
   user_guide/index
   developer_guide/index
   api/index

More Information
================
.. toctree::
   :maxdepth: 1
   :titlesonly:

   authors
   history

   Main SEAMM documentation <https://molssi-seamm.github.io>
