***************************
Tissue Forge Documentation
***************************

Tissue Forge is an interactive, particle-based physics, chemistry and biology
modeling and simulation environment. Tissue Forge provides the ability to create,
simulate and explore models, simulations and virtual experiments of soft condensed
matter physics at multiple scales using a simple, intuitive interface. Tissue Forge
is designed with an emphasis on problems in complex subcellular, cellular and tissue
biophysics. Tissue Forge enables interactive work with simulations on heterogeneous
computing architectures, where models and simulations can be built and interacted
with in real-time during execution of a simulation, and computations can be selectively
offloaded onto available GPUs on-the-fly.

Tissue Forge is a native compiled C++ shared library that's designed to be used for model
and simulation specification in compiled C++ code. Tissue Forge includes extensive C, C++ and
Python APIs and additional support for interactive model and simulation specification in
an IPython console and a Jupyter Notebook. Tissue Forge currently supports installations on
64-bit Windows, Linux and MacOS, and arm64 MacOS.

**Quick Summary**

This documentation provides information on a number of topics related to Tissue Forge.
If you're looking for something specific, refer to the following,

* To get Tissue Forge, refer to :ref:`Getting Tissue Forge <getting>`.

* To get started with Tissue Forge, refer to :ref:`Quickstart <quickstart>`.

* To learn about the physics and philosophy of Tissue Forge, refer to :ref:`Introduction <introduction>`.

* For walkthroughs, examples and other discussions, refer to :ref:`Notes <notes>`.

* For application-specific models and tools, refer to :ref:`Models <models>`.

* To dive into the code, refer to :ref:`Tissue Forge API Reference <api_reference>`.

.. note::

   Tissue Forge supports modeling and simulation in multiple programming languages.
   While many variables, classes and methods are named and behave the same across
   all supported languages, inevitably there are some differences.
   Most examples in this documentation demonstrate usage in Python, and specific
   cases where Tissue Forge behaves differently in a particular language are explicitly
   addressed. In general, assume that a documented example or code snippet in one
   language is the same in another unless stated otherwise. For specific details
   about Tissue Forge in a particular language, refer to
   :ref:`Tissue Forge API Reference <api_reference>`.

**Funding**

Tissue Forge is funded with generous support by NIBIB U24 EB028887.

**Citing**

To use Tissue Forge in research, please cite the following:

   Sego et al. (2023). `"Tissue Forge: Interactive biological and biophysics simulation environment." <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010768>`_ *PLoS Computational Biology*, 19(10), e1010768.

**Content**

.. toctree::
   :maxdepth: 1

   getting
   introduction
   quick_start
   notes
   models/models
   api_reference
   history
   references
   appendices


Indices and tables
##################


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

