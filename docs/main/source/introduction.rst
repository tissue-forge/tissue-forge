.. _introduction:

Introduction
=============

Biological cells are the prototypical example of active matter.
Cells are actively driven agents that transduct free energy from their environment.
These agents can sense and respond to mechanical, chemical and electrical
environmental stimuli with a range of behaviors, including dynamic changes in
morphology and mechanical properties, chemical uptake and secretion, cell
differentiation, proliferation, death, and migration.
One of the greatest challenges of quantitatively modeling cells is that
descriptions of their dynamics and behaviors typically cannot be derived from first principles.
Rather, their observed behaviors are typically described phenomenologically or empirically.
Thus, those who explore the properties of cells and processes of subcellular,
cellular and tissue dynamics require an extreme degree of flexibility to propose
different kinds of interactions at various scales and test them in virtual experiments.

The need for extreme flexibility of model implementation and simulation presents
a significant challenge to the development of a modeling and simulation environment.
As a simulation environment simplifies the process of writing and simulating models
(*e.g.*, without resorting to hard-coding and building C++ or FORTRAN), the level of
flexibility in writing and simulating a model tends to decrease.
For example, if a user wants to write a standard molecular dynamics
model, there are many excellent choices of simulation engines available, and
these kinds of models can easily be specified by human readable configuration files.
However, when an interaction is not well standardized or formalized
(*e.g.*, those of cells, organelles and biomolecules), the user is almost always left to
hard-coding and building custom software, thus eliminating the value of the simplified
simulation interface.

The goal of the Tissue Forge project is to deliver a modeling and simulation framework
that lets users from all relevant backgrounds interactively create, simulate and
explore models at biologically relevant length scales.
We believe that accessible and interactive modeling and simulation is key to increasing
scientific productivity, much like how modeling environments have revolutionized
many fields of modern engineering.

We thus present Tissue Forge, an interactive modeling and simulation environment
based on an off-lattice formalism that seeks to allow users to create models for a wide range of
biologically relevant problems using any combination of the following modeling methodologies:

* Coarse Grained Molecular Dynamics. Each particle represents whole molecules.
* Dissipative Particle Dynamics (DPD). Each particle represents whole molecules or fluid regions.
* Sub-Cellular Element (SCM). Each particle represents a region of space, is governed by empirically
  derived potentials and can exhibit an active response.
* Reactive Molecular Dynamics. Particles react with other particles and
  form new molecules, and can absorb or emit energy into their environment.
* Diffusion. Proximity-dependent connections allow inter-particle transport
  of particle state variables according to transport laws.
* Event-based modeling. User-defined handlers define what occurs for a
  variety of different *events* that various simulation objects, processes and features can emit,
  such as simulation particles, timers or keyboard input.
* Transport Dissipative Particle Dynamics. Each particle represents whole fluid regions that
  carry cargo that can diffuse during advection.

.. note:: We encourage users to **contact us about what features would best benefit specific problems**.
    Please contact us at `<TissueForge@gmail.com>` or raise an issue on the
    `Tissue Forge repository <https://github.com/tissue-forge/tissue-forge>`_.
