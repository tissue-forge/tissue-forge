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

#ifndef _WRAPS_C_TISSUEFORGE_C_H_
#define _WRAPS_C_TISSUEFORGE_C_H_

#include "tf_port_c.h"

#include "tfCBond.h"
#include "tfCBoundaryConditions.h"
#include "tfCClipPlane.h"
#include "tfCCluster.h"
#include "tfCFlux.h"
#include "tfCForce.h"
#include "tfCLogger.h"
#include "tfCParticle.h"
#include "tfCPotential.h"
#include "tfCSimulator.h"
#include "tfCSpecies.h"
#include "tfCStateVector.h"
#include "tfCStyle.h"
#include "tfCUniverse.h"
#include "tfC_bind.h"
#include "tfC_event.h"
#include "tfC_io.h"
#include "tfC_system.h"
#include "tfC_util.h"

#ifdef TF_WITHCUDA
#   include "tfC_cuda.h"
#endif


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Tissue Forge version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfVersionStr(char **str, unsigned int *numChars);

/**
 * @brief System name
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystemNameStr(char **str, unsigned int *numChars);

/**
 * @brief System version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystemVersionStr(char **str, unsigned int *numChars);

/**
 * @brief Package compiler ID
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCompilerIDStr(char **str, unsigned int *numChars);

/**
 * @brief Package compiler version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCompilerVersionStr(char **str, unsigned int *numChars);

/**
 * @brief Package build date
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBuildDateStr(char **str, unsigned int *numChars);

/**
 * @brief Package build time
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBuildTimeStr(char **str, unsigned int *numChars);

/**
 * @brief Tissue Forge major version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfVersionMajorStr(char **str, unsigned int *numChars);

/**
 * @brief Tissue Forge minor version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfVersionMinorStr(char **str, unsigned int *numChars);

/**
 * @brief Tissue Forge patch version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success  
 */
CAPI_FUNC(HRESULT) tfVersionPatchStr(char **str, unsigned int *numChars);

/**
 * @brief Test whether the installation supports CUDA
 * 
 * @return true if CUDA is supported
 */
CAPI_FUNC(bool) tfHasCUDA();

/**
 * @brief Closes the main window, while the application / simulation continues to run.
 */
CAPI_FUNC(HRESULT) tfClose();

/**
 * @brief Shows any windows that were specified in the config. 
 * Does not start the universe time propagation unlike ``run``.
 */
CAPI_FUNC(HRESULT) tfShow();

/**
 * @brief Main simulator init method
 * 
 * @param argv initializer arguments
 * @param nargs number of arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfInit(char **argv, unsigned int nargs);

/**
 * @brief Main simulator init method
 * 
 * @param conf configuration
 * @param appArgv app arguments
 * @param nargs number of app arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfInitC(struct tfSimulatorConfigHandle *conf, char **appArgv, unsigned int nargs);

/**
 * @brief Integrates the universe for a duration as given by ``until``, or for a single time step 
 * if 0 is passed.
 * 
 * @param until runs the timestep for this length of time.
 * @param dt overrides the existing time step, and uses this value for time stepping; currently not supported.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStep(tfFloatP_t until, tfFloatP_t dt);

/**
 * @brief Stops the universe time evolution. This essentially freezes the universe, 
 * everything remains the same, except time no longer moves forward.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStop();

/**
 * @brief Starts the universe time evolution, and advanced the universe forward by 
 * timesteps in ``dt``. All methods to build and manipulate universe objects 
 * are valid whether the universe time evolution is running or stopped.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStart();

/**
 * @brief Runs the event loop until all windows close or simulation time expires. 
 * Automatically performs universe time propogation. 
 * 
 * @param et final time; a negative number runs infinitely
 */
CAPI_FUNC(HRESULT) tfRun(tfFloatP_t et);


#endif // _WRAPS_C_TISSUEFORGE_C_H_