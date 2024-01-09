/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%module(package="tissue_forge") tissue_forge

%include "TissueForge_include.i"


// config stuff
%ignore TF_MODEL_DIR;
%include "tf_config.h"

%pythoncode %{
    
    __version__ = str(TF_VERSION)
    BUILD_DATE = _tissue_forge.tfBuildDate()
    BUILD_TIME = _tissue_forge.tfBuildTime()

    class version:
        version = __version__
        """Tissue Forge version

        :meta hide-value:
        """

        system_name = TF_SYSTEM_NAME
        """System name

        :meta hide-value:
        """

        system_version = TF_SYSTEM_VERSION
        """System version

        :meta hide-value:
        """

        compiler = TF_COMPILER_ID
        """Package compiler ID

        :meta hide-value:
        """

        compiler_version = TF_COMPILER_VERSION
        """Package compiler version

        :meta hide-value:
        """

        build_date = BUILD_DATE + ', ' + BUILD_TIME
        """Package build date

        :meta hide-value:
        """

        major = TF_VERSION_MAJOR
        """Tissue Forge major version

        :meta hide-value:
        """

        minor = TF_VERSION_MINOR
        """Tissue Forge minor version

        :meta hide-value:
        """

        patch = TF_VERSION_PATCH
        """Tissue Forge patch version

        :meta hide-value:
        """
%}

//                                      Imports

// types namespace
%include "types/tf_types.i"

// Logger
%include "tfLogger.i"

// Error
%include "tfError.i"

// io namespace
%include "io/tf_io.i"

// state namespace
%include "state/tf_state.i"

// event namespace
%include "event/tf_event.i"

// util namespace
%include "tf_util.i"

// mdcore
%include "mdcore/mdcore.i"

// renering namespace
%include "rendering/tf_rendering.i"

// system namespace
%include "tf_system.i"

// cuda namespace
#ifdef TF_WITHCUDA
%include "tf_cuda.i"
#endif

// Simulator
%include "tfSimulator.i"

// Universe
%include "tfUniverse.i"

// bind namespace
%include "tf_bind.i"

// metrics namespace
%include "tf_metrics.i"

// models
%include "models/tf_models.i"

//                                      Post-imports

%pythoncode %{

    has_cuda = tfHasCuda()
    """
    Flag signifying whether CUDA support is installed.

    :meta hide-value:
    """
    
    # From Simulator

    def close():
        """
        Alias of :meth:`tissue_forge.tissue_forge.Simulator.close`
        """
        return Simulator.close()

    def show():
        """
        Alias of :meth:`tissue_forge.tissue_forge.Simulator.show`
        """
        return Simulator.show()

    def irun():
        """
        Alias of :meth:`tissue_forge.tissue_forge.Simulator.irun`
        """
        return Simulator.irun()

    def init(*args, **kwargs):
        """
        Initialize a simulation in Python

        :type args: PyObject
        :param args: positional arguments; first argument is name of simulation (if any)
        :type kwargs: PyObject
        :param kwargs: keyword arguments; currently supported are

                dim: (3-component list of floats) the dimensions of the spatial domain; default is [10., 10., 10.]

                cutoff: (float) simulation cutoff distance; default is 1.

                cells: (3-component list of ints) the discretization of the spatial domain; default is [4, 4, 4]

                threads: (int) number of threads; default is hardware maximum

                flux_steps: (int) number of flux steps per simulation step; default is 1

                integrator: (int) simulation integrator; default is FORWARD_EULER

                dt: (float) time discretization; default is 0.01

                bc: (int or dict) boundary conditions; default is everywhere periodic

                window_size: (2-component list of ints) size of application window; default is [800, 600]

                throw_exc: (bool) whether errors raise exceptions; default is False

                seed: (int) seed for pseudo-random number generator

                load_file: (str) path to saved simulation state to initialize

                logger_level: (int) logger level; default is no logging

                clip_planes: (list of tuple of (FVector3, FVector3)) list of point-normal pairs of clip planes; default is no planes
        """
        return SimulatorPy_init(args, kwargs)

    def run(*args, **kwargs):
        """
        Runs the event loop until all windows close or simulation time expires. 
        Automatically performs universe time propagation.

        :type args: float
        :param args: period to execute, in units of simulation time (default runs infinitely)
        """
        return Simulator.run(*args, **kwargs)

    def throw_exceptions(_throw: bool):
        """Set whether errors result in exceptions"""
        Simulator.throw_exceptions = _throw

    def throwing_exceptions() -> bool:
        """Test whether errors result in exceptions"""
        return Simulator.throw_exceptions

    # From Universe

    step = Universe.step
    """Alias of :meth:`tissue_forge.tissue_forge.Universe.step`"""

    stop = Universe.stop
    """Alias of :meth:`tissue_forge.tissue_forge.Universe.stop`"""

    start = Universe.start
    """Alias of :meth:`tissue_forge.tissue_forge.Universe.start`"""
%}
