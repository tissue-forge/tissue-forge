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

/**
 * @file tf_runtime.h
 * 
 */

#ifndef _INCLUDE_TF_RUNTIME_H_
#define _INCLUDE_TF_RUNTIME_H_

#include <stdio.h>

#include <tfSimulator.h>
#include <tf_bind.h>
#include <tf_util.h>
#include <tfCluster.h>
#include <tfFlux.h>
#include <event/tfParticleEventSingle.h>
#include <event/tfParticleTimeEvent.h>
#include <io/tfIO.h>
#include <rendering/tfClipPlane.h>
#include <rendering/tfKeyEvent.h>
#include <rendering/tfColorMapper.h>
#include <rendering/tfStyle.h>
#include <state/tfSpeciesValue.h>


namespace TissueForge {


    /**
     * @brief Installation version
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) version_str();

    /**
     * @brief System name
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) systemNameStr();

    /**
     * @brief System version
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) systemVersionStr();

    /**
     * @brief Compiler id
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) compilerIdStr();

    /**
     * @brief Compiler version
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) compilerVersionStr();

    /**
     * @brief Build data
     * 
     * @return std::string
     */
    CPPAPI_FUNC(std::string) buildDate();

    /**
     * @brief Build time
     * 
     * @return std::string
     */
    CPPAPI_FUNC(std::string) buildTime();

    /**
     * @brief Flag for whether the installation supports CUDA
     * 
     * @return bool
     */
    CPPAPI_FUNC(bool) hasCuda();

    /**
     * @brief Initialization method that may be a mandatory first 
     * call, depending on the runtime scenario. 
     * 
     */
    CPPAPI_FUNC(HRESULT) initialize(int args);

    /**
     * @brief Closes the main window, while the application / simulation continues to run.
     */
    CPPAPI_FUNC(HRESULT) close();

    /**
     * @brief Shows any windows that were specified in the config. 
     * Does not start the universe time propagation unlike ``run``.
     */
    CPPAPI_FUNC(HRESULT) show();

    /**
     * main simulator init method
     */
    CPPAPI_FUNC(HRESULT) init(const std::vector<std::string> &argv);

    /**
     * main simulator init method
     */
    CPPAPI_FUNC(HRESULT) init(Simulator::Config &conf, const std::vector<std::string> &appArgv=std::vector<std::string>());

    /**
     * @brief Performs a single time step ``dt`` of the universe if no arguments are 
     * given. Optionally runs until ``until``, and can use a different timestep 
     * of ``dt``.
     * 
     * @param until period to execute, in units of simulation time (default executes one time step).
     * @param dt overrides the existing time step, and uses this value for time stepping; currently not supported.
     */
    CPPAPI_FUNC(HRESULT) step(const FloatP_t &until=0, const FloatP_t &dt=0);

    /**
     * @brief Stops the universe time evolution. This essentially freezes the universe, 
     * everything remains the same, except time no longer moves forward.
     */
    CPPAPI_FUNC(HRESULT) stop();

    /**
     * @brief Starts the universe time evolution, and advanced the universe forward by 
     * timesteps in ``dt``. All methods to build and manipulate universe objects 
     * are valid whether the universe time evolution is running or stopped.
     */
    CPPAPI_FUNC(HRESULT) start();

    /**
     * @brief Runs the event loop until all windows close or simulation time expires. 
     * Automatically performs universe time propogation. 
     * 
     * @param et period to execute, in units of simulation time; a negative number runs infinitely
     */
    CPPAPI_FUNC(HRESULT) run(FloatP_t et=-1);

};

#endif // _INCLUDE_TF_RUNTIME_H_