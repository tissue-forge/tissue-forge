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
 * @file tfCForce.h
 * 
 */

#ifndef _WRAPS_C_TFCFORCE_H_
#define _WRAPS_C_TFCFORCE_H_

#include "tf_port_c.h"

typedef void (*tfUserForceFuncTypeHandleFcn)(struct tfCustomForceHandle*, tfFloatP_t*);

// Handles

/**
 * @brief Handle to a @ref FORCE_TYPE instance
 * 
 */
struct CAPI_EXPORT tfFORCE_TYPEHandle {
    unsigned int FORCE_FORCE;
    unsigned int FORCE_BERENDSEN;
    unsigned int FORCE_GAUSSIAN;
    unsigned int FORCE_FRICTION;
    unsigned int FORCE_SUM;
    unsigned int FORCE_CUSTOM;
};

/**
 * @brief Handle to a @ref UserForceFuncType instance
 * 
 */
struct CAPI_EXPORT tfUserForceFuncTypeHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Force instance
 * 
 */
struct CAPI_EXPORT tfForceHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref ForceSum instance
 * 
 */
struct CAPI_EXPORT tfForceSumHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref CustomForce instance
 * 
 */
struct CAPI_EXPORT tfCustomForceHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Berendsen instance
 * 
 */
struct CAPI_EXPORT tfBerendsenHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Gaussian instance
 * 
 */
struct CAPI_EXPORT tfGaussianHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Friction instance
 * 
 */
struct CAPI_EXPORT tfFrictionHandle {
    void *tfObj;
};


////////////////
// FORCE_TYPE //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFORCE_TYPE_init(struct tfFORCE_TYPEHandle *handle);


///////////////////////
// UserForceFuncType //
///////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_EvalFcn_init(struct tfUserForceFuncTypeHandle *handle, tfUserForceFuncTypeHandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_EvalFcn_destroy(struct tfUserForceFuncTypeHandle *handle);


///////////
// Force //
///////////

/**
 * @brief Get the force type
 * 
 * @param handle populated handle
 * @param te type enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_getType(struct tfForceHandle *handle, unsigned int *te);

/**
 * @brief Bind a force to a species. 
 * 
 * When a force is bound to a species, the magnitude of the force is scaled by the concentration of the species. 
 * 
 * @param handle populated handle
 * @param a_type particle type containing the species
 * @param coupling_symbol symbol of the species
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_bind_species(struct tfForceHandle *handle, struct tfParticleTypeHandle *a_type, const char *coupling_symbol);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_toString(struct tfForceHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * The returned type is automatically registered with the engine. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_fromString(struct tfForceHandle *handle, const char *str);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_destroy(struct tfForceHandle *handle);


//////////////
// ForceSum //
//////////////


/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForceSum_checkType(struct tfForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForceSum_toBase(struct tfForceSumHandle *handle, struct tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForceSum_fromBase(struct tfForceHandle *baseHandle, struct tfForceSumHandle *handle);

/**
 * @brief Get the constituent forces
 * 
 * @param handle populated handle
 * @param f1 first constituent force
 * @param f2 second constituent force
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForceSum_getConstituents(struct tfForceSumHandle *handle, struct tfForceHandle *f1, struct tfForceHandle *f2);


/////////////////
// CustomForce //
/////////////////


/**
 * @brief Create a custom force with a force function
 * 
 * @param handle handle to populate
 * @param func function to evaluate the force components
 * @param period force period; infinite if a negative value is passed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_init(struct tfCustomForceHandle *handle, struct tfUserForceFuncTypeHandle *func, tfFloatP_t period);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_checkType(struct tfForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_toBase(struct tfCustomForceHandle *handle, struct tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_fromBase(struct tfForceHandle *baseHandle, struct tfCustomForceHandle *handle);

/**
 * @brief Get force period
 * 
 * @param handle populated handle 
 * @param period force period
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_getPeriod(struct tfCustomForceHandle *handle, tfFloatP_t *period);

/**
 * @brief Set force period
 * 
 * @param handle populated handle 
 * @param period force period
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_setPeriod(struct tfCustomForceHandle *handle, tfFloatP_t period);

/**
 * @brief Set force function
 * 
 * @param handle populated handle 
 * @param fcn force function
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_setFunction(struct tfCustomForceHandle *handle, struct tfUserForceFuncTypeHandle *fcn);

/**
 * @brief Get current force value
 * 
 * @param handle populated handle 
 * @param force force value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_getValue(struct tfCustomForceHandle *handle, tfFloatP_t **force);

/**
 * @brief Set current force value
 * 
 * @param handle populated handle 
 * @param force force value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_setValue(struct tfCustomForceHandle *handle, tfFloatP_t *force);

/**
 * @brief Get time of last force update
 * 
 * @param handle populated handle 
 * @param lastUpdate time of last force update
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCustomForce_getLastUpdate(struct tfCustomForceHandle *handle, tfFloatP_t *lastUpdate);


///////////////
// Berendsen //
///////////////

/**
 * @brief Creates a Berendsen thermostat. 
 * 
 * The thermostat uses the target temperature @f$ T_0 @f$ from the object 
 * to which it is bound. 
 * The Berendsen thermostat effectively re-scales the velocities of an object in 
 * order to make the temperature of that family of objects match a specified 
 * temperature.
 * 
 * The Berendsen thermostat force has the function form: 
 * 
 * @f[
 * 
 *      \frac{\mathbf{p}}{\tau_T} \left(\frac{T_0}{T} - 1 \right),
 * 
 * @f]
 * 
 * where @f$ \mathbf{p} @f$ is the momentum, 
 * @f$ T @f$ is the measured temperature of a family of 
 * particles, @f$ T_0 @f$ is the control temperature, and 
 * @f$ \tau_T @f$ is the coupling constant. The coupling constant is a measure 
 * of the time scale on which the thermostat operates, and has units of 
 * time. Smaller values of @f$ \tau_T @f$ result in a faster acting thermostat, 
 * and larger values result in a slower acting thermostat.
 * 
 * @param handle handle to populate
 * @param tau time constant that determines how rapidly the thermostat effects the system.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_init(struct tfBerendsenHandle *handle, tfFloatP_t tau);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_checkType(struct tfForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_toBase(struct tfBerendsenHandle *handle, struct tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_fromBase(struct tfForceHandle *baseHandle, struct tfBerendsenHandle *handle);

/**
 * @brief Get the time constant
 * 
 * @param handle populated handle
 * @param tau time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_getTimeConstant(struct tfBerendsenHandle *handle, tfFloatP_t *tau);

/**
 * @brief Set the time constant
 * 
 * @param handle populated handle
 * @param tau time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBerendsen_setTimeConstant(struct tfBerendsenHandle *handle, tfFloatP_t tau);


//////////////
// Gaussian //
//////////////

/**
 * @brief Creates a random force. 
 * 
 * A random force has a randomly selected orientation and magnitude. 
 * 
 * Orientation is selected according to a uniform distribution on the unit sphere. 
 * 
 * Magnitude is selected according to a prescribed mean and standard deviation. 
 * 
 * @param handle handle to populate
 * @param std standard deviation of magnitude
 * @param mean mean of magnitude
 * @param duration duration of force. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_init(struct tfGaussianHandle *handle, tfFloatP_t std, tfFloatP_t mean, tfFloatP_t duration);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_checkType(struct tfForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_toBase(struct tfGaussianHandle *handle, struct tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_fromBase(struct tfForceHandle *baseHandle, struct tfGaussianHandle *handle);

/**
 * @brief Get the magnitude standard deviation
 * 
 * @param handle populated handle
 * @param std magnitude standard deviation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_getStd(struct tfGaussianHandle *handle, tfFloatP_t *std);

/**
 * @brief Set the magnitude standard deviation
 * 
 * @param handle populated handle
 * @param std magnitude standard deviation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_setStd(struct tfGaussianHandle *handle, tfFloatP_t std);

/**
 * @brief Get the magnitude mean
 * 
 * @param handle populated handle
 * @param mean mean magnitude
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_getMean(struct tfGaussianHandle *handle, tfFloatP_t *mean);

/**
 * @brief Set the magnitude mean
 * 
 * @param handle populated handle
 * @param mean mean magnitude
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_setMean(struct tfGaussianHandle *handle, tfFloatP_t mean);

/**
 * @brief Get the magnitude duration
 * 
 * @param handle populated handle
 * @param duration magnitude duration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_getDuration(struct tfGaussianHandle *handle, tfFloatP_t *duration);

/**
 * @brief Set the magnitude duration
 * 
 * @param handle populated handle
 * @param duration magnitude duration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfGaussian_setDuration(struct tfGaussianHandle *handle, tfFloatP_t duration);


//////////////
// Friction //
//////////////


/**
 * @brief Creates a friction force. 
 * 
 * A friction force has the form: 
 * 
 * @f[
 * 
 *      - \frac{|| \mathbf{v} ||}{\tau} \mathbf{v} ,
 * 
 * @f]
 * 
 * where @f$ \mathbf{v} @f$ is the velocity of a particle and @f$ \tau @f$ is a time constant. 
 * 
 * @param handle handle to populate
 * @param coeff time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_init(struct tfFrictionHandle *handle, tfFloatP_t coeff);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_checkType(struct tfForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_toBase(struct tfFrictionHandle *handle, struct tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_fromBase(struct tfForceHandle *baseHandle, struct tfFrictionHandle *handle);

/**
 * @brief Get the friction coefficient
 * 
 * @param handle populated handle
 * @param coef friction coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_getCoef(struct tfFrictionHandle *handle, tfFloatP_t *coef);

/**
 * @brief Set the friction coefficient
 * 
 * @param handle populated handle
 * @param coef friction coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFriction_setCoef(struct tfFrictionHandle *handle, tfFloatP_t coef);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Add two forces
 * 
 * @param f1 first force
 * @param f2 second force
 * @param fSum handle to populate with force sum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfForce_add(struct tfForceHandle *f1, struct tfForceHandle *f2, struct tfForceSumHandle *fSum);


#endif // _WRAPS_C_TFCFORCE_H_