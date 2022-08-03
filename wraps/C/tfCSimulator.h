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

#ifndef _WRAPS_C_TFCSIMULATOR_H_
#define _WRAPS_C_TFCSIMULATOR_H_

#include "tf_port_c.h"

#include "tfCUniverse.h"

// Handles

struct CAPI_EXPORT tfSimulatorEngineIntegratorHandle {
    int FORWARD_EULER;
    int RUNGE_KUTTA_4;
};

/**
 * @brief Handle to a @ref Simulator::Config instance
 * 
 */
struct CAPI_EXPORT tfSimulatorConfigHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Simulator instance
 * 
 */
struct CAPI_EXPORT tfSimulatorHandle {
    void *tfObj;
};


/////////////////////////////////
// Simulator::EngineIntegrator //
/////////////////////////////////


/**
 * @brief Populate engine integrator enums
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorEngineIntegrator_init(struct tfSimulatorEngineIntegratorHandle *handle);


///////////////////////
// Simulator::Config //
///////////////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_init(struct tfSimulatorConfigHandle *handle);

/**
 * @brief Get the title of a configuration
 * 
 * @param handle populated handle
 * @param title title
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getTitle(struct tfSimulatorConfigHandle *handle, char **title, unsigned int *numChars);

/**
 * @brief Set the title of a configuration
 * 
 * @param handle populated handle
 * @param title title
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_setTitle(struct tfSimulatorConfigHandle *handle, char *title);

/**
 * @brief Get the window size
 * 
 * @param handle populated handle
 * @param x width
 * @param y height
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getWindowSize(struct tfSimulatorConfigHandle *handle, unsigned int *x, unsigned int *y);

/**
 * @brief Set the window size
 * 
 * @param handle populated handle
 * @param x width
 * @param y height
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_setWindowSize(struct tfSimulatorConfigHandle *handle, unsigned int x, unsigned int y);

/**
 * @brief Get the random number generator seed. If none is set, returns NULL.
 * 
 * @param handle populated handle
 * @param seed random number generator seed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getSeed(struct tfSimulatorConfigHandle *handle, unsigned int *seed);

/**
 * @brief Set the random number generator seed.
 * 
 * @param handle populated handle
 * @param seed random number generator seed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_setSeed(struct tfSimulatorConfigHandle *handle, unsigned int seed);

/**
 * @brief Get the windowless flag
 * 
 * @param handle populated handle
 * @param windowless windowless flag
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getWindowless(struct tfSimulatorConfigHandle *handle, bool *windowless);

/**
 * @brief Set the windowless flag
 * 
 * @param handle populated handle
 * @param windowless windowless flag
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_setWindowless(struct tfSimulatorConfigHandle *handle, bool windowless);

/**
 * @brief Get the imported data file path during initialization, if any.
 * 
 * @param handle populated handle
 * @param filePath file path
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getImportDataFilePath(struct tfSimulatorConfigHandle *handle, char **filePath, unsigned int *numChars);

/**
 * @brief Get the current clip planes
 * 
 * @param handle populated handle
 * @param clipPlanes clip planes
 * @param numClipPlanes number of clip planes
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getClipPlanes(struct tfSimulatorConfigHandle *handle, float **clipPlanes, unsigned int *numClipPlanes);

/**
 * @brief Set the clip planes
 * 
 * @param handle populated handle
 * @param clipPlanes clip planes
 * @param numClipPlanes number of clip planes
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_setClipPlanes(struct tfSimulatorConfigHandle *handle, float *clipPlanes, unsigned int numClipPlanes);

/**
 * @brief Get the universe configuration
 * 
 * @param handle populated handle
 * @param confHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_getUniverseConfig(struct tfSimulatorConfigHandle *handle, struct tfUniverseConfigHandle *confHandle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulatorConfig_destroy(struct tfSimulatorConfigHandle *handle);


///////////////
// Simulator //
///////////////

/**
 * @brief Main simulator init method
 * 
 * @param argv initializer arguments
 * @param nargs number of arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_init(char **argv, unsigned int nargs);

/**
 * @brief Main simulator init method
 * 
 * @param conf configuration
 * @param appArgv app arguments
 * @param nargs number of app arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_initC(struct tfSimulatorConfigHandle *conf, char **appArgv, unsigned int nargs);

/**
 * @brief Gets the global simulator object
 * 
 * @param handle handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_get(struct tfSimulatorHandle *handle);

/**
 * @brief Make the instance the global simulator object
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulator_makeCurrent(struct tfSimulatorHandle *handle);

/**
 * @brief Runs the event loop until all windows close or simulation time expires. 
 * Automatically performs universe time propogation. 
 * 
 * @param et final time; a negative number runs infinitely
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_run(tfFloatP_t et);

/**
 * @brief Shows any windows that were specified in the config. 
 * 
 * Does not start the universe time propagation unlike @ref tfSimulator_run().
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_show();

/**
 * @brief Closes the main window, while the application / simulation continues to run.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_close();

/**
 * @brief Destroy the simulation.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_destroy();

/**
 * @brief Issue call to rendering update.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_redraw();

/**
 * @brief Get the number of threads
 * 
 * @param numThreads number of threads
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSimulator_getNumThreads(unsigned int *numThreads);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Test whether running interactively
 * 
 */
CAPI_FUNC(bool) tfIsTerminalInteractiveShell();

/**
 * @brief Set whether running interactively
 */
CAPI_FUNC(HRESULT) tfSetIsTerminalInteractiveShell(bool _interactive);

#endif // _WRAPS_C_TFCSIMULATOR_H_