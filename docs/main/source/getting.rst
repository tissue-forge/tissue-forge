.. _getting:

Getting Tissue Forge
=====================

Pre-Built Binaries
-------------------

Binary distributions of Tissue Forge are available via conda from the `tissue-forge` channel,

.. code-block:: bash

    conda install -c conda-forge -c tissue-forge tissue-forge

Pre-built binaries of the latest Tissue Forge developments are also archived at the
`Tissue Forge Azure project <https://dev.azure.com/Tissue-Forge/tissue-forge>`_.
Installing pre-built binaries requires `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Binaries on Linux require the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`.
Packages include a convenience script `install_env` that installs the dependencies
of the Tissue Forge installation on execution. After installing the dependencies
environment, the Tissue Forge installation can be used after executing the following steps
from a terminal with the root of the installation as the current directory.

On Windows

.. code-block:: bash

    call etc/vars
    conda activate %TFENV%

On Linux and MacOS

.. code-block:: bash

    source etc/vars.sh
    conda activate $TFENV

Launching the provided Python examples are then as simple navigating to the ``tissue_forge``
Python module and then executing the following,

.. code-block:: bash

    python examples/cell_sorting.py

Likewise, Tissue Forge can be imported in Python scripts and interactive consoles,

.. code-block:: bash

    import tissue_forge as tf


Installing From Source
-----------------------

Supported installation from source uses Git and Miniconda for building and installing
most dependencies. In addition to requiring `Git <https://git-scm.com/downloads>`_ and
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, installation from source
on Windows requires
`Visual Studio 2019 Build Tools <https://visualstudio.microsoft.com/downloads/>`_,
on Linux requires the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`,
and on MacOS requires Xcode with 10.9 SDK or greater.

To execute the standard installation, open a terminal in a directory to install Tissue Forge
and clone the `Tissue Forge repository <https://github.com/tissue-forge/tissue-forge>`_,

.. code-block:: bash

    git clone --recurse-submodules https://github.com/tissue-forge/tissue-forge

From the directory containing the `tissue-forge` root directory, perform the following.

On Windows

.. code-block:: bat

    call tissue-forge/package/local/install

On Linux and MacOS

.. code-block:: bash

    bash tissue-forge/package/local/install.sh

The standard installation will create the directories `tissue-forge_build` and
`tissue-forge_install` next to the `tissue-forge` root directory, the former containing
the build files, and the latter containing the installed binaries and conda environment.
The source and build directories can be safely deleted after installation.
The conda environment will be installed in the subdirectory `env`.
To activate the conda environment with the Tissue Forge Python module, perform the following.

On Windows

.. code-block:: bat

    call tissue-forge_install/etc/vars
    conda activate %TFENV%

On Linux and MacOS

.. code-block:: bash

    source tissue-forge_install/etc/vars.sh
    conda activate $TFENV

Launching the provided examples are then as simple as the following

.. code-block:: bash

    python tissue-forge/py/examples/cell_sorting.py

Likewise Tissue Forge can be imported in Python scripts and interactive consoles

.. code-block:: python

    import tissue_forge as tf


.. _customizing_the_build:

Customizing the Build
^^^^^^^^^^^^^^^^^^^^^^

Certain aspects of the installation can be readily customized.
The source directory `tissue-forge/package/local` contains subdirectories `linux`, `osx` and
`win` containing scripts `install_vars.sh` and `install_vars.bat` for Linux/MacOS and
Windows, respectively, which declare default installation environment variables.
These environment variables can be customized to specify where to find, build and install
Tissue Forge, as well as the build configuration.
For example, to install Tissue Forge from a source directory ``MYTFSRC``, build Tissue Forge
at path ``MYTFBUILD`` in debug mode and install into directory ``MYTFINSTALL``, perform the
following.

On Windows

.. code-block:: bat

    call %MYTFSRC%/package/local/win/install_vars
    set TFBUILD_CONFIG=Debug
    set TFSRCDIR=%MYTFSRC%
    set TFBUILDDIR=%MYTFBUILD%
    set TFINSTALLDIR=%MYTFINSTALL%
    call %TFSRCDIR%/package/local/win/install_env
    conda activate %TFENV%
    call %TFSRCDIR%/package/local/win/install_all

On Linux

.. code-block:: bash

    source $MYTFSRC/package/local/linux/install_vars.sh
    export TFBUILD_CONFIG=Debug
    export TFSRCDIR=$MYTFSRC
    export TFBUILDDIR=$MYTFBUILD
    export TFINSTALLDIR=$MYTFINSTALL
    bash ${TFSRCDIR}/package/local/linux/install_env.sh
    conda activate $TFENV
    bash ${TFSRCDIR}/package/local/linux/install_all.sh

On MacOS

.. code-block:: bash

    source $MYTFSRC/package/local/osx/install_vars.sh
    export TFBUILD_CONFIG=Debug
    export TFSRCDIR=$MYTFSRC
    export TFBUILDDIR=$MYTFBUILD
    export TFINSTALLDIR=$MYTFINSTALL
    bash ${TFSRCDIR}/package/local/osx/install_env.sh
    conda activate $TFENV
    bash ${TFSRCDIR}/package/local/osx/install_all.sh

Note that Tissue Forge assumes that conda is available by default and if not, is installed in a typical location on Linux and MacOS. 
If Tissue Forge has trouble finding conda, the conda script `conda.sh` can be provided during build 
customization using the environment variables ``TFCONDAENV`` and ``TFENVNEEDSCONDA``. 

.. code-block:: bash

    export TFENVNEEDSCONDA=1
    export TFCONDAENV=${HOME}/mambaforge/etc/profile.d/conda.sh

The default Python version of the installation is 3.8, though Tissue Forge has also been tested
on Windows, Linux and MacOS for Python versions 3.9, 3.10 and 3.11.
To specify a different version of Python, simply add a call to
`update the conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html#updating-or-upgrading-python>`_
in the previous commands before calling `install_all`.


Enabling Interactive Tissue Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissue Forge supports interactive modeling and simulation specification in an
IPython console and Jupyter Notebook. To enable interactive Tissue Forge in an
IPython console, activate the installed environment as previously described and
install the ``ipython`` package from the conda-forge channel,

.. code-block:: bash

    conda install -c conda-forge ipython

To enable interactive Tissue Forge in a Jupyter Notebook, activate the installed
environment as previously described and install the ``notebook``, ``ipywidgets``,
``ipyevents`` and ``ipyfilechooser`` packages from the conda-forge channel,

.. code-block:: bash

    conda install -c conda-forge notebook ipywidgets ipyevents ipyfilechooser


Enabling GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^
Tissue Forge supports GPU acceleration on Windows and Linux using CUDA. To enable GPU
acceleration, simply tell Tissue Forge to build with CUDA support and specify the compute
capability of all available GPUs in the typical way *before* calling `install`.

On Windows

.. code-block:: bat

    set TF_WITHCUDA=1
    set CUDAARCHS=35;50
    call tissue-forge/package/local/install

On Linux

.. code-block:: bash

    export TF_WITHCUDA=1
    export CUDAARCHS=35;50
    bash tissue-forge/package/local/install.sh

.. note::

    Tissue Forge currently supports offloading computations onto CUDA-supporting GPU devices
    of compute capability 3.5 or greater and installed drivers of at least 456.38 on Windows, and
    450.80.02 on Linux.


Setting Up a Development Environment
-------------------------------------

The Tissue Forge codebase includes convenience scripts to quickly set up a
development environment for building models and extensions in C++. The same
environment deployed in `Installing From Source`_ can be used to build a customized
version of Tissue Forge. Set up for setting up a development environment is as simple
as getting the Tissue Forge source code, and installing the pre-configured conda
environment. As such, all requirements described in `Installing From Source`_ are
also applicable for building a custom version of Tissue Forge.

To set up a development environment, clone the
`Tissue Forge repository <https://github.com/tissue-forge/tissue-forge>`_, open a terminal
in the directory containing the `tissue-forge` root directory and perform the following.

On Windows

.. code-block:: bat

    call tissue-forge/package/local/win/install_vars
    call tissue-forge/package/local/win/install_env

On Linux

.. code-block:: bash

    bash tissue-forge/package/local/linux/install_vars.sh
    bash tissue-forge/package/local/linux/install_env.sh

On MacOS

.. code-block:: bash

    bash tissue-forge/package/local/osx/install_vars.sh
    bash tissue-forge/package/local/osx/install_env.sh

The standard configuration will set the build and installation directories to
`tissue-forge_build` and `tissue-forge_install` next to the `tissue-forge` root directory,
respectively, the latter containing the conda environment with the build dependencies.
These locations can be customized in the same way as described in `Customizing the Build`_,
or in your favorite IDE. For configuring `CMake <https://cmake.org/>`_, refer to the
script `install_core` in the subdirectory of `package/local/*` that corresponds to
your platform, which is the script behind the automated installation from source.
This script includes all variables and the compiler(s) that correspond to building a
fully customized version of Tissue Forge.

Tissue Forge supports the `Release`, `Debug` and `RelWithDebInfo` build types. The
computational core of Tissue Forge and C++ front-end can be found throughout the subdirectory
`source`. Bindings for Python language support are generated using
`SWIG <http://swig.org/>`_. To develop the Python interface
(or generate an interface for a new language), refer to the SWIG script `wraps/py/tissue_forge.i`.
To develop the C language interface, refer to the directory `wraps/C`.
