Tissue Forge
============

<p align=center>
<img src="docs/main/source/droplets_3d.gif" width="600">
</p>

Tissue Forge is an interactive, particle-based physics, chemistry and biology
modeling and simulation environment. Tissue Forge provides the ability to create, 
simulate and explore models, simulations and virtual experiments of soft condensed 
matter physics at multiple scales using a simple, intuitive interface. Tissue Forge 
is designed with an emphasis on problems in complex subcellular, cellular and tissue 
biophysics. Tissue Forge enables interactive work with simulations on heterogeneous 
computing architectures, where models and simulations can be built and interacted 
with in real-time during execution of a simulation, and computations can be 
selectively offloaded onto available GPUs on-the-fly. 

Tissue Forge is a native compiled C++ shared library that's designed to be used for model 
and simulation specification in compiled C++ code. Tissue Forge includes extensive C, C++ and 
Python APIs and additional support for interactive model and simulation specification in 
an IPython console and a Jupyter Notebook. 
Tissue Forge currently supports installations on 64-bit Windows, Linux and MacOS, and arm64 MacOS. 

## Build Status ##

### Binaries ###

The latest Tissue Forge developments are archived at the 
[Tissue Forge Azure project](https://dev.azure.com/Tissue-Forge/tissue-forge).

| Platform |                                                                                                                                 Status                                                                                                                                 |
|:--------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Linux   |  [![Build Status](https://dev.azure.com/Tissue-Forge/tissue-forge/_apis/build/status/tissue-forge.develop?branchName=develop&stageName=Local%20build%20for%20Linux)](https://dev.azure.com/Tissue-Forge/tissue-forge/_build/latest?definitionId=4&branchName=develop)  |
|  MacOS   |   [![Build Status](https://dev.azure.com/Tissue-Forge/tissue-forge/_apis/build/status/tissue-forge.develop?branchName=develop&stageName=Local%20build%20for%20Mac)](https://dev.azure.com/Tissue-Forge/tissue-forge/_build/latest?definitionId=4&branchName=develop)   |
| Windows  | [![Build Status](https://dev.azure.com/Tissue-Forge/tissue-forge/_apis/build/status/tissue-forge.develop?branchName=develop&stageName=Local%20build%20for%20Windows)](https://dev.azure.com/Tissue-Forge/tissue-forge/_build/latest?definitionId=4&branchName=develop) |

### Documentation ###

Tissue Forge documentation is available online, 

|          Document          |                                      Link                                       |                                                                                                    Status                                                                                                     |
|:--------------------------:|:-------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Tissue Forge Documentation |      [link](https://tissue-forge-documentation.readthedocs.io/en/latest/)       |            [![Documentation Status](https://readthedocs.org/projects/tissue-forge-documentation/badge/?version=latest)](https://tissue-forge-documentation.readthedocs.io/en/latest/?badge=latest)            |
|   C++ API Documentation    |  [link](https://tissue-forge-cpp-api-documentation.readthedocs.io/en/latest/)   |    [![Documentation Status](https://readthedocs.org/projects/tissue-forge-cpp-api-documentation/badge/?version=latest)](https://tissue-forge-cpp-api-documentation.readthedocs.io/en/latest/?badge=latest)    |
|  Python API Documentation  | [link](https://tissue-forge-python-api-documentation.readthedocs.io/en/latest/) | [![Documentation Status](https://readthedocs.org/projects/tissue-forge-python-api-documentation/badge/?version=latest)](https://tissue-forge-python-api-documentation.readthedocs.io/en/latest/?badge=latest) |
|    C API Documentation     |   [link](https://tissue-forge-c-api-documentation.readthedocs.io/en/latest/)    |      [![Documentation Status](https://readthedocs.org/projects/tissue-forge-c-api-documentation/badge/?version=latest)](https://tissue-forge-c-api-documentation.readthedocs.io/en/latest/?badge=latest)      |

# Gallery #

<p align=center>
<img src="docs/main/source/droplets_3d.gif" width="600">
<br/>
<em>
Tissue Forge simulation of chemical flux during fluid droplet collision. 
Each particle represents a portion of fluid that carries an amount of a diffusive chemical, 
the amount of which varies from zero (blue) to one (red). 
When two droplets carrying different initial chemical amounts collide, 
resulting droplets tend towards homogeneous chemical distributions. 
</em>
</p>

<p align=center>
<img src="docs/main/source/cell_sorting.gif" width="600">
<br/>
<em>
Tissue Forge simulation of cell sorting by differential adhesion. 
Each particle represents a cell, and cells sort by preferential attachment to 
their own type (blue or red). 
This 
<a href="examples/py/cell_sorting.py">example</a> 
is available in Python.
</em>
</p>

<p align=center>
<img src="docs/main/source/models/vertex/solver/capillary_loop.png" width="600">
<br/>
<em>
Tissue Forge vertex model configuration generated from an imported _.obj_ mesh of a local vasculature. 
The _.obj_ was generated in <a href="https://www.blender.org/">Blender</a>. 
<br/>
This 
<a href="wraps/python/models/vertex/solver/examples/capillary_loop.py">example</a> 
is available in Python.
</em>
</p>

<p align=center>
<img src="docs/main/source/colonic_crypt.gif" width="600">
<br/>
<em>
Two-dimensional agent-based model of cell proliferation and differentiation in the colonic crypt. 
Each cell is assigned a state dynamics model of the cell cycle and unique clonal identification 
(visualized as a unique particle color). 
When the cell cycle model of a cell transitions from the M phase to the G1 phase, 
the cell divides and copies its clonal identification to its progeny. 
Cells are removed when they reach the base of the crypt. 
</em>
</p>

<p align=center>
<img src="docs/main/source/cellcrawling.gif" width="600">
<br/>
<em>
Tissue Forge simulation of single cell migration over extracellular matrix fibers using combined 
vertex- and particle-based modeling. 
Integrins (green particles) bind to extracellular matrix fibers (red particles) 
and generate protrusive forces (green lines) 
on the vertices that describe the shape of the cell (red polygon). 
Detailed view shows deformations in the ECM caused by force generation during cell migration.
<br/>
This 
<a href="wraps/python/models/vertex/solver/examples/cell_migration.py">example</a> 
is available in Python.
</em>
</p>

# Installation #

## Pre-Built Binaries ##

Binary distributions of Tissue Forge are available via conda from the `tissue-forge` channel, 

```bash
conda install -c conda-forge -c tissue-forge tissue-forge
```

Pre-built binaries of the latest Tissue Forge developments are also archived at the 
[Tissue Forge Azure project](https://dev.azure.com/Tissue-Forge/tissue-forge). 
Installing pre-built binaries requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 
Binaries on Linux require the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`. 
Packages include a convenience script `install_env` that installs the dependencies 
of the Tissue Forge installation on execution. After installing the dependencies 
environment, the Tissue Forge installation can be used after executing the following steps 
from a terminal with the root of the installation as the current directory. 

On Windows
```bash
call etc/vars
conda activate %TFENV%
```
On Linux and MacOS
```bash
source etc/vars.sh
conda activate $TFENV
```

Launching the provided Python examples are then as simple navigating to the ``tissue_forge`` 
Python module and then executing the following, 

```bash
python examples/cell_sorting.py
```

Likewise, Tissue Forge can be imported in Python scripts and interactive consoles, 

```python
import tissue_forge as tf
```

## From Source ##

Supported installation from source uses Git and Miniconda for building and installing 
most dependencies. In addition to requiring [Git](https://git-scm.com/downloads) and 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html), installation from source 
on Windows requires 
[Visual Studio 2019 Build Tools](https://visualstudio.microsoft.com/downloads/), 
on Linux requires the Mesa packages `libgl1-mesa-dev` and `libegl1-mesa-dev`, 
and on MacOS requires Xcode with 10.9 SDK or greater. 

To execute the standard installation, open a terminal in a directory to install Tissue Forge
and clone this respository,
```bash
git clone --recurse-submodules https://github.com/tissue-forge/tissue-forge
```

From the directory containing the `tissue-forge` root directory, perform the following.

On Windows 
```bash
call tissue-forge/package/local/install
```
On Linux
```bash
bash tissue-forge/package/local/install.sh
```
On MacOS, specify the installed MacOS SDK (*e.g.*, for 10.9)  
```bash
export TFOSX_SYSROOT=10.9
bash tissue-forge/package/local/install.sh
```

The standard installation will create the directories `tissue-forge_build` and 
`tissue-forge_install` next to the `tissue-forge` root directory, the former containing 
the build files, and the latter containing the installed binaries and conda environment. 
The source and build directories can be safely deleted after installation. 
The conda environment will be installed in the subdirectory `env`. 
To activate the conda environment with the Tissue Forge Python module, perform the following. 

On Windows
```bash
call tissue-forge_install/etc/vars
conda activate %TFENV%
```
On Linux and MacOS 
```bash
source tissue-forge_install/etc/vars.sh
conda activate $TFENV
```

Launching the provided Python examples are then as simple as the following

```bash
python tissue-forge/py/examples/cell_sorting.py
```

Likewise Tissue Forge can be imported in Python scripts and interactive consoles

```python
import tissue_forge as tf
```

### Customizing the Build ###

Certain aspects of the installation can be readily customized. 
The source directory `tissue-forge/package/local` contains subdirectories `linux`, `osx` and 
`win` containing scripts `install_vars.sh` and `install_vars.bat` for 
Linux/MacOS and Windows, respectively, which declare default installation 
environment variables. These environment variables can be customized to specify 
where to find, build and install Tissue Forge, as well as the build configuration. 
For example, to install Tissue Forge from a source directory `MYTFSRC`, build Tissue Forge 
at path `MYTFBUILD` in debug mode and install into directory `MYTFINSTALL`, perform the following. 

On Windows
```bash
call %MYTFSRC%/package/local/win/install_vars
set TFBUILD_CONFIG=Debug
set TFSRCDIR=%MYTFSRC%
set TFBUILDDIR=%MYTFBUILD%
set TFINSTALLDIR=%MYTFINSTALL%
call %TFSRCDIR%/package/local/win/install_env
conda activate %TFENV%
call %TFSRCDIR%/package/local/win/install_all
```
On Linux
```bash
source $MYTFSRC/package/local/linux/install_vars.sh
export TFBUILD_CONFIG=Debug
export TFSRCDIR=$MYTFSRC
export TFBUILDDIR=$MYTFBUILD
export TFINSTALLDIR=$MYTFINSTALL
bash ${TFSRCDIR}/package/local/linux/install_env.sh
conda activate $TFENV
bash ${TFSRCDIR}/package/local/linux/install_all.sh
```
On MacOS
```bash
source $MYTFSRC/package/local/osx/install_vars.sh
export TFBUILD_CONFIG=Debug
export TFSRCDIR=$MYTFSRC
export TFBUILDDIR=$MYTFBUILD
export TFINSTALLDIR=$MYTFINSTALL
bash ${TFSRCDIR}/package/local/osx/install_env.sh
conda activate $TFENV
bash ${TFSRCDIR}/package/local/osx/install_all.sh
```

Note that Tissue Forge assumes that conda is installed in a typical location on Linux and MacOS. 
If Tissue Forge has trouble finding the conda script `conda.sh`, then it can be provided during build 
customization using the environment variable `TFCONDAENV`. 

```bash
export TFCONDAENV=${HOME}/mambaforge/etc/profile.d/conda.sh
```

The default Python version of the installation is 3.8, though Tissue Forge has also been tested 
on Windows, Linux and MacOS for Python versions 3.8, 3.9, 3.10, and 3.11. 
To specify a different version of Python, simply add a call to 
[update the conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html#updating-or-upgrading-python) 
in the previous commands before calling `install_all`. 

### Enabling Interactive Tissue Forge ###

Tissue Forge supports interactive modeling and simulation specification in an 
IPython console and Jupyter Notebook. To enable interactive Tissue Forge in an 
IPython console, activate the installed environment as previously described and 
install the `ipython` package from the conda-forge channel, 

```bash
conda install -c conda-forge ipython
```

To enable interactive Tissue Forge in a Jupyter Notebook, activate the installed 
environment as previously described and install the `notebook`, `ipywidgets` and 
`ipyevents` packages from the conda-forge channel, 

```bash
conda install -c conda-forge notebook ipywidgets ipyevents
```

## Citing ##

To use Tissue Forge in research, please cite the following: 

Sego et al. (2023). ["Tissue Forge: Interactive biological and biophysics simulation environment."](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010768) *PLoS Computational Biology*, 19(10), e1010768.
