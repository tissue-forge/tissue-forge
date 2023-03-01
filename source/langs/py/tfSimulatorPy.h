/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

/**
 * @file tfSimulatorPy.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TFSIMULATORPY_H_
#define _SOURCE_LANGS_PY_TFSIMULATORPY_H_

#include "tf_py.h"

#include <tfSimulator.h>


namespace TissueForge::py {


    struct CAPI_EXPORT SimulatorPy : Simulator {

    public:

        /**
         * gets the global simulator object, throws exception if fail.
         */
        static SimulatorPy *get();

        static PyObject *_run(PyObject *args, PyObject *kwargs);
        
        /**
         * @brief Interactive python version of the run loop. This checks the ipython context and lets 
         * ipython process keyboard input, while we also running the simulator and processing window messages.
         * 
         * @return HRESULT 
         */
        static HRESULT irun();

        static HRESULT _show();

        static void *wait_events(const FloatP_t &timeout=-1);

        /** Set whether errors result in exceptions */
        static HRESULT _throwExceptions(const bool &_throw);

        /** Check whether errors result in exceptions */
        static bool _throwingExceptions();

    };

    CAPI_FUNC(HRESULT) _setIPythonInputHook(PyObject *_ih);

    CAPI_FUNC(HRESULT) _onIPythonNotReady();

    /**
     * @brief Initialize a simulation in Python
     * 
     * @param args positional arguments; first argument is name of simulation (if any)
     * @param kwargs keyword arguments; currently supported are
     * 
     *      dim: (3-component list of floats) the dimensions of the spatial domain; default is [10., 10., 10.]
     * 
     *      cutoff: (float) simulation cutoff distance; default is 1.
     * 
     *      cells: (3-component list of ints) the discretization of the spatial domain; default is [4, 4, 4]
     * 
     *      threads: (int) number of threads; default is hardware maximum
     * 
     *      integrator: (int) simulation integrator; default is FORWARD_EULER
     * 
     *      dt: (float) time discretization; default is 0.01
     * 
     *      bc: (int or dict) boundary conditions; default is everywhere periodic
     * 
     *      window_size: (2-component list of ints) size of application window; default is [800, 600]
     * 
     *      seed: (int) seed for pseudo-random number generator
     * 
     *      load_file: (str) path to saved simulation state to initialize
     * 
     *      logger_level: (int) logger level; default is no logging
     * 
     *      clip_planes: (list of tuple of (Vector3f, Vector3f)) list of point-normal pairs of clip planes; default is no planes
     */
    CAPI_FUNC(PyObject *) SimulatorPy_init(PyObject *args, PyObject *kwargs);

};

#endif // _SOURCE_LANGS_PY_TFSIMULATORPY_H_