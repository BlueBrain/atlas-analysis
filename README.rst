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

A bunch of cli will be available too. And tools for visualization.

Later, we will include also the coordinate query and plane projector used
by Armando at some point. Since this is extracted form the proj42 gerrit, this was highly
prototypic and testing will come with time.