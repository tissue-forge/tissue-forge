.. _history:

History
========

Version 0.2.1
-------------
* Added new particle split overloads
* Major performance improvements to mesh I/O
* Added movie gallery to main readme
* Minor corrections to docs typos
* Improvements to build pipelines

  * CI/CD conda package builds split by Python version
  * Added handling arbitrary conda installation location by automated local build
* Bug fixes

  * Added constraints to libxml2 version
  * Fixed particle dihedral getter
  * Fixed minor bug in vertex connectivity update
  * Fixed libmamba dependency bug for linux
  * Fixed vertex solver API docs
  * Fixed missing support for invisible bonds
  * Fixed cluster memory ownership bugs

Version 0.2.0
-------------
* Added support for Apple Silicon
* Added data mapping visualization
* Improvements to build pipelines
* Added support for Python 3.10 and 3.11
* Dropped support for Python 3.7
* Bug fixes

  * Minor bug fix in cell polarity module

Version 0.1.1
-------------
* Improvements to GPU acceleration
* Added flux sub-stepping
* Minor improvements to lists
* Minor improvements to docs
* Improvements to species and flux
* Added split overload for optional direction
* Bug fixes

  * Minor bug fix in factory particle constructor
  * Minor bug fix in lists
  * Fixed multithreading in cell polarity model module
  * Fixed state during particle splitting
  * Minor numpy fix in lattice python module

Version 0.1.0
--------------
* Added vertex solver module
* Added rendered line width control
* Reduced build warnings by more than 50%
* Bug fixes

  * Minor CUDA fixes on Linux
  * Minor fixes to eigen deployment
  * Fixed ParticleHandle getter for frozenX
  * Fixed major I/O memory leak
  * Minor fix to boundary conditions I/O
  * Added missing import from file to C++ API
  * Minor fixes to handle constructors

Version 0.0.2
--------------
* Simplified particle handle construction
* Added guard against bonding a particle to itself
* Improved error handling
* Improved cluster interface
* Improved particle neighbor interface
* Added comparison and string operators
* Reduced pointer usage
* Added basic vector methods
* Added camera lagging control to APIs
* Added convenience property methods
* Added improved bonded interaction renderers and runtime interface
* Impoved docs
* Bug fixes

  * Fixed cluster rendering
  * Added missing equilibrium distance data to potentials
  * Removed returned inactive bonds when retrieving all bonds
  * Fixed IPython support
  * Fixed binding by types in python API
  * Fixed rare bug in boundary conditions

Version 0.0.1
--------------
* Improved error reporting in potential construction
* Improved class handling in python modules
* Added plot example in python
* Added style color data to python interface
* Improved documentation on force object memory management
* Improved documentation on handling particle style
* Improved documentation on exception handling in python functions
* Bug fixes

  * Corrected docs for power potential
  * Fixed inconsistency in potential and force plot calculations
  * Added missing copy constructors on basic types in python
  * Added missing simulator features to python interface
  * Fixed missing API docs
  * Fixed docs API hyperlinks

Version 0.0.0
--------------
First release
