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

#ifndef _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_
#define _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_

#include <tf_port_c.h>

#include <tfCParticle.h>
#include <tfCForce.h>
#include <tfCPotential.h>

// Handles

struct CAPI_EXPORT tfCellPolarityPolarContactTypeEnumHandle {
    unsigned int REGULAR;
    unsigned int ISOTROPIC;
    unsigned int ANISOTROPIC;
};

/**
 * @brief Handle to a @ref PersistentForce instance
 * 
 */
struct CAPI_EXPORT tfCellPolarityPersistentForceHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref PolarityArrowData instance
 * 
 */
struct CAPI_EXPORT tfCellPolarityPolarityArrowDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref CellPolarityPotentialContact instance
 * 
 */
struct CAPI_EXPORT tfCellPolarityContactPotentialHandle {
    void *tfObj;
};


//////////////////////
// PolarContactType //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPolarContactType_init(struct tfCellPolarityPolarContactTypeEnumHandle *handle);


/////////////////////
// PersistentForce //
/////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param sensAB proportionality of force to AB vector
 * @param sensPCP proportionality of force to PCP vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityPersistentForce_init(struct tfCellPolarityPersistentForceHandle *handle, float sensAB, float sensPCP);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityPersistentForce_destroy(struct tfCellPolarityPersistentForceHandle *handle);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityPersistentForce_toBase(struct tfCellPolarityPersistentForceHandle *handle, tfForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityPersistentForce_fromBase(tfForceHandle *baseHandle, struct tfCellPolarityPersistentForceHandle *handle);


//////////////////////
// ContactPotential //
//////////////////////


/**
 * @brief Create a polarity state dynamics and anisotropic adhesion potential
 * 
 * @param handle handle to populate
 * @param cutoff cutoff distance
 * @param couplingFlat flat interaction coefficient
 * @param couplingOrtho orthogonal interaction coefficient
 * @param couplingLateral lateral interaction coefficient
 * @param distanceCoeff distance coefficient
 * @param cType contact type (e.g., normal, isotropic or anisotropic)
 * @param mag magnitude of force due to potential
 * @param rate state vector dynamics rate due to potential
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_init(
    struct tfCellPolarityContactPotentialHandle *handle, 
    float cutoff, 
    float couplingFlat, 
    float couplingOrtho, 
    float couplingLateral, 
    float distanceCoeff, 
    unsigned int cType, 
    float mag, 
    float rate, 
    float bendingCoeff
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_destroy(struct tfCellPolarityContactPotentialHandle *handle);

/**
 * @brief Cast to base potential. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_toBase(struct tfCellPolarityContactPotentialHandle *handle, tfPotentialHandle *baseHandle);

/**
 * @brief Cast from base potential. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_fromBase(struct tfPotentialHandle *baseHandle, struct tfCellPolarityContactPotentialHandle *handle);

/**
 * @brief Get the flat interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingFlat flat interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getCouplingFlat(struct tfCellPolarityContactPotentialHandle *handle, float *couplingFlat);

/**
 * @brief Set the flat interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingFlat flat interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setCouplingFlat(struct tfCellPolarityContactPotentialHandle *handle, float couplingFlat);

/**
 * @brief Get the orthogonal interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingOrtho orthogonal interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getCouplingOrtho(struct tfCellPolarityContactPotentialHandle *handle, float *couplingOrtho);

/**
 * @brief Set the orthogonal interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingOrtho orthogonal interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setCouplingOrtho(struct tfCellPolarityContactPotentialHandle *handle, float couplingOrtho);

/**
 * @brief Get the lateral interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingLateral lateral interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getCouplingLateral(struct tfCellPolarityContactPotentialHandle *handle, float *couplingLateral);

/**
 * @brief Set the lateral interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingLateral lateral interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setCouplingLateral(struct tfCellPolarityContactPotentialHandle *handle, float couplingLateral);

/**
 * @brief Get the distance coefficient
 * 
 * @param handle populated handle
 * @param distanceCoeff distance coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getDistanceCoeff(struct tfCellPolarityContactPotentialHandle *handle, float *distanceCoeff);

/**
 * @brief Set the distance coefficient
 * 
 * @param handle populated handle
 * @param distanceCoeff distance coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setDistanceCoeff(struct tfCellPolarityContactPotentialHandle *handle, float distanceCoeff);

/**
 * @brief Get the contact type (e.g., normal, isotropic or anisotropic)
 * 
 * @param handle populated handle
 * @param cType contact type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getCType(struct tfCellPolarityContactPotentialHandle *handle, unsigned int *cType);

/**
 * @brief Set the contact type (e.g., normal, isotropic or anisotropic)
 * 
 * @param handle populated handle
 * @param cType contact type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setCType(struct tfCellPolarityContactPotentialHandle *handle, unsigned int cType);

/**
 * @brief Get the magnitude of force due to potential
 * 
 * @param handle populated handle
 * @param mag magnitude of force due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getMag(struct tfCellPolarityContactPotentialHandle *handle, float *mag);

/**
 * @brief Set the magnitude of force due to potential
 * 
 * @param handle populated handle
 * @param mag magnitude of force due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setMag(struct tfCellPolarityContactPotentialHandle *handle, float mag);

/**
 * @brief Get the state vector dynamics rate due to potential
 * 
 * @param handle populated handle
 * @param rate state vector dynamics rate due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getRate(struct tfCellPolarityContactPotentialHandle *handle, float *rate);

/**
 * @brief Set the state vector dynamics rate due to potential
 * 
 * @param handle populated handle
 * @param rate state vector dynamics rate due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setRate(struct tfCellPolarityContactPotentialHandle *handle, float rate);

/**
 * @brief Get the bending coefficient
 * 
 * @param handle populated handle
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_getBendingCoeff(struct tfCellPolarityContactPotentialHandle *handle, float *bendingCoeff);

/**
 * @brief Set the bending coefficient
 * 
 * @param handle populated handle
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityContactPotential_setBendingCoeff(struct tfCellPolarityContactPotentialHandle *handle, float bendingCoeff);


///////////////////////
// PolarityArrowData //
///////////////////////


/**
 * @brief Get the arrow length
 * 
 * @param handle populated handle
 * @param arrowLength arrow length
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPolarityArrowData_getArrowLength(struct tfCellPolarityPolarityArrowDataHandle *handle, float *arrowLength);

/**
 * @brief Set the arrow length
 * 
 * @param handle populated handle
 * @param arrowLength arrow length
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPolarityArrowData_setArrowLength(struct tfCellPolarityPolarityArrowDataHandle *handle, float arrowLength);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Gets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @param vec polarity vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetVectorAB(int pId, bool current, float **vec);

/**
 * @brief Gets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @param vec polarity vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetVectorPCP(int pId, bool current, float **vec);

/**
 * @brief Sets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; set to true to set the current value
 * @param init initialization flag; set to true to set the initial value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetVectorAB(int pId, float *pVec, bool current, bool init);

/**
 * @brief Sets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; set to true to set the current value
 * @param init initialization flag; set to true to set the initial value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetVectorPCP(int pId, float *pVec, bool current, bool init);

/**
 * @brief Updates all running polarity models
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityUpdate();

/**
 * @brief Registers a particle as polar. 
 * 
 * This must be called before the first integration step.
 * Otherwise, the engine will not know that the particle 
 * is polar and will be ignored. 
 * 
 * @param ph handle of particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityRegisterParticle(struct tfParticleHandleHandle *ph);

/**
 * @brief Unregisters a particle as polar. 
 * 
 * This must be called before destroying a registered particle. 
 * 
 * @param ph handle of particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityUnregister(struct tfParticleHandleHandle *ph);

/**
 * @brief Registers a particle type as polar. 
 * 
 * This must be called on a particle type before any other type-specific operations. 
 * 
 * @param pType particle type
 * @param initMode initialization mode for particles of this type
 * @param initPolarAB initial value of AB polarity vector; only used when initMode="value"
 * @param initPolarPCP initial value of PCP polarity vector; only used when initMode="value"
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityRegisterType(struct tfParticleTypeHandle *pType, const char *initMode, float *initPolarAB, float *initPolarPCP);

/**
 * @brief Gets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param initMode initialization mode
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetInitMode(struct tfParticleTypeHandle *pType, char **initMode, unsigned int *numChars);

/**
 * @brief Sets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param value initialization mode
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetInitMode(struct tfParticleTypeHandle *pType, const char *value);

/**
 * @brief Gets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param vec vector value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetInitPolarAB(struct tfParticleTypeHandle *pType, float **vec);

/**
 * @brief Sets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param value initial AB polar vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetInitPolarAB(struct tfParticleTypeHandle *pType, float *value);

/**
 * @brief Gets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param vec initial PCP polar vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetInitPolarPCP(struct tfParticleTypeHandle *pType, float **vec);

/**
 * @brief Sets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param value initial PCP polar vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetInitPolarPCP(struct tfParticleTypeHandle *pType, float *value);

/**
 * @brief Toggles whether polarity vectors are rendered
 * 
 * @param _draw rendering flag; vectors are rendered when true
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetDrawVectors(bool _draw);

/**
 * @brief Sets rendered polarity vector colors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param colorAB name of AB vector color
 * @param colorPCP name of PCP vector color
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetArrowColors(const char *colorAB, const char *colorPCP);

/**
 * @brief Sets scale of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _scale scale of rendered vectors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetArrowScale(float _scale);

/**
 * @brief Sets length of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _length length of rendered vectors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolaritySetArrowLength(float _length);

/**
 * @brief Gets the rendering info for the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param arrowData rendering info
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetVectorArrowAB(unsigned int pId, struct tfCellPolarityPolarityArrowDataHandle *arrowData);

/**
 * @brief Gets the rendering info for the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param arrowData rendering info
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityGetVectorArrowPCP(unsigned int pId, struct tfCellPolarityPolarityArrowDataHandle *arrowData);

/**
 * @brief Runs the polarity model along with a simulation. 
 * Must be called before doing any operations with this module. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCellPolarityLoad();

#endif // _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_