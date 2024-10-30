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
 * @file tfCUniverse.h
 * 
 */

#ifndef _WRAPS_C_TFCUNIVERSE_H_
#define _WRAPS_C_TFCUNIVERSE_H_

#include "tf_port_c.h"

#include "tfCParticle.h"
#include "tfCBoundaryConditions.h"

// Handles

/**
 * @brief Handle to a @ref UniverseConfig instance
 * 
 */
struct CAPI_EXPORT tfUniverseConfigHandle {
    void *tfObj;
};


//////////////
// Universe //
//////////////


/**
 * @brief Get the dimensions of the universe
 * 
 * @param dim 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getDim(tfFloatP_t **dim);

/**
 * @brief Get whether the universe is running
 * 
 * @param isRunning 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getIsRunning(bool *isRunning);

/**
 * @brief Get the name of the model / script
 * 
 * @param name 
 * @param numChars 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getName(char **name, unsigned int *numChars);

/**
 * @brief Get the virial tensor of the universe
 * 
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getVirial(tfFloatP_t **virial);

/**
 * @brief Get the virial tensor of the universe for a set of particle types
 * 
 * @param phandles array of types
 * @param numTypes number of types
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getVirialT(struct tfParticleTypeHandle **phandles, unsigned int numTypes, tfFloatP_t **virial);

/**
 * @brief Get the virial tensor of a neighborhood
 * 
 * @param origin origin of neighborhood
 * @param radius radius of neighborhood
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getVirialO(tfFloatP_t *origin, tfFloatP_t radius, tfFloatP_t **virial);

/**
 * @brief Get the virial tensor for a set of particle types in a neighborhood
 * 
 * @param phandles array of types
 * @param numTypes number of types
 * @param origin origin of neighborhood
 * @param radius radius of neighborhood
 * @param virial virial tensor
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUniverse_getVirialOT(
    struct tfParticleTypeHandle **phandles, 
    unsigned int numTypes, 
    tfFloatP_t *origin, 
    tfFloatP_t radius, 
    tfFloatP_t **virial
);

/**
 * @brief Get the number of particles in the universe
 * 
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getNumParts(unsigned int *numParts);

/**
 * @brief Get the i'th particle of the universe
 * 
 * @param pidx index of particle
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getParticle(unsigned int pidx, struct tfParticleHandleHandle *handle);

/**
 * @brief Get the center of the universe
 * 
 * @param center 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getCenter(tfFloatP_t **center);

/**
 * @brief Integrates the universe for a duration as given by ``until``, or for a single time step 
 * if 0 is passed.
 * 
 * @param until period to execute, in units of simulation time.
 * @param dt overrides the existing time step, and uses this value for time stepping; currently not supported.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_step(tfFloatP_t until, tfFloatP_t dt);

/**
 * @brief Stops the universe time evolution. This essentially freezes the universe, 
 * everything remains the same, except time no longer moves forward.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_stop();

/**
 * @brief Starts the universe time evolution, and advanced the universe forward by 
 * timesteps in ``dt``. All methods to build and manipulate universe objects 
 * are valid whether the universe time evolution is running or stopped.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_start();

/**
 * @brief Reset the universe
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_reset();

/**
 * @brief Reset all species in all particles
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_resetSpecies();

/**
 * @brief Get the universe temperature. 
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param temperature 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getTemperature(tfFloatP_t *temperature);

/**
 * @brief Set the universe temperature. 
 * 
 * @param temperature temperature; must be greater than zero
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_setTemperature(tfFloatP_t temperature);

/**
 * @brief Get the Boltzmann constant
 * 
 * @param k
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getBoltzmann(tfFloatP_t* k);

/**
 * @brief Set the Boltzmann constant
 * 
 * @param k Boltzmann constant; must be greater than zero
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_setBoltzmann(tfFloatP_t k);

/**
 * @brief Get the current time
 * 
 * @param time 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getTime(tfFloatP_t *time);

/**
 * @brief Get the period of a time step
 * 
 * @param dt 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getDt(tfFloatP_t *dt);

/**
 * @brief Get the boundary conditions
 * 
 * @param bcs boundary conditions 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getBoundaryConditions(struct tfBoundaryConditionsHandle *bcs);

/**
 * @brief Get the current system kinetic energy
 * 
 * @param ke 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getKineticEnergy(tfFloatP_t *ke);

/**
 * @brief Get the current number of registered particle types
 * 
 * @param numTypes 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getNumTypes(int *numTypes);

/**
 * @brief Get the global interaction cutoff distance
 * 
 * @param cutoff 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverse_getCutoff(tfFloatP_t *cutoff);


////////////////////////////
// tfUniverseConfigHandle //
////////////////////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_init(struct tfUniverseConfigHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_destroy(struct tfUniverseConfigHandle *handle);

/**
 * @brief Get the dimensions of the universe
 * 
 * @param handle populated handle
 * @param dim 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getDim(struct tfUniverseConfigHandle *handle, tfFloatP_t **dim);

/**
 * @brief Set the dimensions of the universe
 * 
 * @param handle populated handle
 * @param dim 3-element array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setDim(struct tfUniverseConfigHandle *handle, tfFloatP_t *dim);

/**
 * @brief Get the grid discretization
 * 
 * @param handle populated handle
 * @param cells grid discretization
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getCells(struct tfUniverseConfigHandle *handle, int **cells);

/**
 * @brief Set the grid discretization
 * 
 * @param handle populated handle
 * @param cells grid discretization
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setCells(struct tfUniverseConfigHandle *handle, int *cells);

/**
 * @brief Get the global interaction cutoff distance
 * 
 * @param handle populated handle
 * @param cutoff cutoff distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getCutoff(struct tfUniverseConfigHandle *handle, tfFloatP_t *cutoff);

/**
 * @brief Set the global interaction cutoff distance
 * 
 * @param handle populated handle
 * @param cutoff cutoff distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setCutoff(struct tfUniverseConfigHandle *handle, tfFloatP_t cutoff);

/**
 * @brief Get the universe flags
 * 
 * @param handle populated handle
 * @param flags universe flags
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getFlags(struct tfUniverseConfigHandle *handle, unsigned int *flags);

/**
 * @brief Set the universe flags
 * 
 * @param handle populated handle
 * @param flags universe flags
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setFlags(struct tfUniverseConfigHandle *handle, unsigned int flags);

/**
 * @brief Get the period of a time step
 * 
 * @param handle populated handle
 * @param dt period of a time step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getDt(struct tfUniverseConfigHandle *handle, tfFloatP_t *dt);

/**
 * @brief Set the period of a time step
 * 
 * @param handle populated handle
 * @param dt period of a time step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setDt(struct tfUniverseConfigHandle *handle, tfFloatP_t dt);

/**
 * @brief Get the universe temperature. 
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param handle populated handle
 * @param temperature universe temperature
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getTemperature(struct tfUniverseConfigHandle *handle, tfFloatP_t *temperature);

/**
 * @brief Set the universe temperature.
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param handle populated handle
 * @param temperature universe temperature
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setTemperature(struct tfUniverseConfigHandle *handle, tfFloatP_t temperature);

/**
 * @brief Get the number of threads for parallel execution.
 * 
 * @param handle populated handle
 * @param numThreads number of threads for parallel execution
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getNumThreads(struct tfUniverseConfigHandle *handle, unsigned int *numThreads);

/**
 * @brief Set the number of threads for parallel execution.
 * 
 * @param handle populated handle
 * @param numThreads number of threads for parallel execution
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setNumThreads(struct tfUniverseConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Get the number of flux steps per simulation step.
 * 
 * @param handle populated handle
 * @param numFluxSteps number of flux steps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getNumFluxSteps(struct tfUniverseConfigHandle *handle, unsigned int *numFluxSteps);

/**
 * @brief Set the number of flux steps per simulation step.
 * 
 * @param handle populated handle
 * @param numFluxSteps number of flux steps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setNumFluxSteps(struct tfUniverseConfigHandle *handle, unsigned int numFluxSteps);

/**
 * @brief Get the engine integrator enum.
 * 
 * @param handle populated handle
 * @param integrator engine integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getIntegrator(struct tfUniverseConfigHandle *handle, unsigned int *integrator);

/**
 * @brief Set the engine integrator enum.
 * 
 * @param handle populated handle
 * @param integrator engine integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setIntegrator(struct tfUniverseConfigHandle *handle, unsigned int integrator);

/**
 * @brief Get the boundary condition argument container
 * 
 * @param handle populated handle
 * @param bargsHandle argument container
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_getBoundaryConditions(struct tfUniverseConfigHandle *handle, struct tfBoundaryConditionsArgsContainerHandle *bargsHandle);

/**
 * @brief Set the boundary condition argument container
 * 
 * @param handle populated handle
 * @param bargsHandle argument container
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfUniverseConfig_setBoundaryConditions(struct tfUniverseConfigHandle *handle, struct tfBoundaryConditionsArgsContainerHandle *bargsHandle);

#endif // _WRAPS_C_TFCUNIVERSE_H_