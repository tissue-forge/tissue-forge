/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

%{

#include "tfSimulator.h"
#include <langs/py/tfSimulatorPy.h>

%}


%ignore TissueForge::Simulator::getWindow;
%ignore TissueForge::initSimConfigFromFile;
%ignore TissueForge::universe_init;
%ignore TissueForge::modules_init;

%rename(_Simulator) Simulator;
%rename(_SimulatorPy) SimulatorPy;

%include "tfSimulator.h"
%include <langs/py/tfSimulatorPy.h>


%pythoncode %{

    class SimulatorInterface:

        @property
        def threads(self) -> int:
            """
            Number of threads
            """
            return _SimulatorPy.getNumThreads()

        @property
        def cuda_config(self):
            """
            CUDA runtime interface, if any
            """
            return _SimulatorPy.getCUDAConfig() if tfHasCuda() else None

        @staticmethod
        def run(*args, **kwargs):
            """
            Runs the event loop until all windows close or simulation time expires. 
            Automatically performs universe time propogation. 

            :type args: double
            :param args: final time (default runs infinitly)
            """    
            return _SimulatorPy._run(args, kwargs)

        @staticmethod
        def show():
            """
            Shows any windows that were specified in the config. This works just like
            MatPlotLib's ``show`` method. The ``show`` method does not start the
            universe time propagation unlike ``run`` and ``irun``.
            """
            return _SimulatorPy._show()

        @staticmethod
        def close():
            return _SimulatorPy.close()

        @staticmethod
        def destroy():
            return _SimulatorPy.destroy()

        @staticmethod
        def redraw():
            return _SimulatorPy.redraw()

    Simulator = SimulatorInterface()
    
    from enum import Enum as EnumPy

    class EngineIntegratorTypes(EnumPy):
        forward_euler = _tissue_forge._Simulator_EngineIntegrator_FORWARD_EULER
        runge_kutta4 = _tissue_forge._Simulator_EngineIntegrator_RUNGE_KUTTA_4
%}
