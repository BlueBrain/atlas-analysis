.. |name| replace:: Atlas-Analysis

Welcome to |name| documentation!
==========================================

Introduction
============


This lib contains the atlas codes used during the hippocampus circuit building which could be
useful for some other brain regions.

The lib already contains some basic functions such as:


* safe_cast_atlas: that allows one to safely cast an atlas into a new dtype
* homogenize_atlas_types: loads different atlases with different dtype safely
* load_coherent_atlases: loads different atlases with different dtype safely and that
  checks the external properties of the nrrd files such as pixel sizes, offsets and shapes.
* extract_labels: extracts particular regions using labels and reset labels if needed
* reset_all_values: resets all labels of an atlas to a given label.
* regroup_atlases: regroups multiple atlases in one atlas and reset labels if needed.
* logical_and: does a logical and for multiple atlases. This is useful to check overlaps between
  different atlases.
* voxel_mask: masks an atlas with an other atlas that will act as a binary mask.
  You have two choices either keeping or removing everything inside the mask. This is useful in case
  of overlapping atlases.
* indices_to_positions: that will give you the position of the center of the voxel corresponding
  to id [a, b, c].
* sample_positions_from_voxeldata: samples randomly N indexes and returns the corresponding
  positions
* change_encoding: changes the encoding of a nrrd file and save the new version.

Some more complex functions will be added in a near future. These functions are: creating a
centerline for concave volumes, creating homogeneous subregions and orientiations.

Installation
============

It can be installed using pip:

.. code-block:: bash

    $ pip install git+https://github.com/BlueBrain/atlas-analysis/

Acknowledgment
==============

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

This project/research received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA).

For license see LICENSE.txt.

Copyright © 2014-2024 Blue Brain Project/EPFL
