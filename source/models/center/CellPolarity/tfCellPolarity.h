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

/**
 * Implements model with additional features defined in 
 * Nielsen, Bjarke Frost, et al. "Model to link cell shape and polarity with organogenesis." Iscience 23.2 (2020): 100830.
 */

#ifndef _SOURCE_MODELS_CENTER_TFCELLPOLARITY_H_
#define _SOURCE_MODELS_CENTER_TFCELLPOLARITY_H_

#include <tfParticle.h>
#include <tfAngle.h>
#include <tfBond.h>
#include <tfForce.h>
#include <tfParticleTypeList.h>
#include <tfPotential.h>
#include <rendering/tfArrowRenderer.h>

#include <string>
#include <unordered_map>

namespace TissueForge::models::center::CellPolarity { 

/**
 * @brief Gets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @return FVector3 
 */
CPPAPI_FUNC(FVector3) getVectorAB(const int &pId, const bool &current=true);

/**
 * @brief Gets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @return FVector3 
 */
CPPAPI_FUNC(FVector3) getVectorPCP(const int &pId, const bool &current=true);

/**
 * @brief Sets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; default true
 * @param init initialization flag; default false
 */
CPPAPI_FUNC(void) setVectorAB(const int &pId, const FVector3 &pVec, const bool &current=true, const bool &init=false);

/**
 * @brief Sets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; default true
 * @param init initialization flag; default false
 */
CPPAPI_FUNC(void) setVectorPCP(const int &pId, const FVector3 &pVec, const bool &current=true, const bool &init=false);

/**
 * @brief Updates all running polarity models
 * 
 */
CPPAPI_FUNC(void) update();

/**
 * @brief Registers a particle as polar. 
 * 
 * This must be called before the first integration step.
 * Otherwise, the engine will not know that the particle 
 * is polar and will be ignored. 
 * 
 * @param ph handle of particle
 */
CPPAPI_FUNC(void) registerParticle(ParticleHandle *ph);

/**
 * @brief Unregisters a particle as polar. 
 * 
 * This must be called before destroying a registered particle. 
 * 
 * @param ph handle of particle
 */
CPPAPI_FUNC(void) unregister(ParticleHandle *ph);

/**
 * @brief Registers a particle type as polar. 
 * 
 * This must be called on a particle type before any other type-specific operations. 
 * 
 * @param pType particle type
 * @param initMode initialization mode for particles of this type
 * @param initPolarAB initial value of AB polarity vector; only used when initMode="value"
 * @param initPolarPCP initial value of PCP polarity vector; only used when initMode="value"
 */
CPPAPI_FUNC(void) registerType(
    ParticleType *pType, 
    const std::string &initMode="random", 
    const FVector3 &initPolarAB=FVector3(0.0), 
    const FVector3 &initPolarPCP=FVector3(0.0)
);

/**
 * @brief Gets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @return const std::string 
 */
CPPAPI_FUNC(const std::string) getInitMode(ParticleType *pType);

/**
 * @brief Sets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param value initialization mode
 */
CPPAPI_FUNC(void) setInitMode(ParticleType *pType, const std::string &value);

/**
 * @brief Gets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @return const FVector3 
 */
CPPAPI_FUNC(const FVector3) getInitPolarAB(ParticleType *pType);

/**
 * @brief Sets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param value initial AB polar vector
 */
CPPAPI_FUNC(void) setInitPolarAB(ParticleType *pType, const FVector3 &value);

/**
 * @brief Gets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @return const FVector3 
 */
CPPAPI_FUNC(const FVector3) getInitPolarPCP(ParticleType *pType);

/**
 * @brief Sets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param value initial PCP polar vector
 */
CPPAPI_FUNC(void) setInitPolarPCP(ParticleType *pType, const FVector3 &value);

/**
 * @brief Defines a force due to polarity state
 * 
 */
struct CAPI_EXPORT PersistentForce : Force {
    /** Proportionality of force to AB vector */
    FloatP_t sensAB = 0.0;
    
    /** Proportionality of force to PCP vector */
    FloatP_t sensPCP = 0.0;

    ~PersistentForce();
};

/**
 * @brief Creates a persistent polarity force. 
 * 
 * @param sensAB sensitivity to AB vector
 * @param sensPCP sensitivity to PCP vector
 * @return PersistentForce* 
 */
CPPAPI_FUNC(PersistentForce*) createPersistentForce(const FloatP_t &sensAB=0.0, const FloatP_t &sensPCP=0.0);

typedef enum PolarContactType {
    REGULAR = 0, 
    ISOTROPIC = 1, 
    ANISOTROPIC = 2
} PolarContactType;

struct CAPI_EXPORT PolarityArrowData : rendering::ArrowData {
    FloatP_t arrowLength = 1.0;
};

/**
 * @brief Toggles whether polarity vectors are rendered
 * 
 * @param _draw rendering flag; vectors are rendered when true
 */
CPPAPI_FUNC(void) setDrawVectors(const bool &_draw);

/**
 * @brief Sets rendered polarity vector colors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param colorAB name of AB vector color
 * @param colorPCP name of PCP vector color
 */
CPPAPI_FUNC(void) setArrowColors(const std::string &colorAB, const std::string &colorPCP);

/**
 * @brief Sets scale of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _scale scale of rendered vectors
 */
CPPAPI_FUNC(void) setArrowScale(const FloatP_t &_scale);

/**
 * @brief Sets length of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _length length of rendered vectors
 */
CPPAPI_FUNC(void) setArrowLength(const FloatP_t &_length);

/**
 * @brief Gets the rendering info for the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @return PolarityArrowData* 
 */
CPPAPI_FUNC(PolarityArrowData*) getVectorArrowAB(const int32_t &pId);

/**
 * @brief Gets the rendering info for the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @return PolarityArrowData* 
 */
CPPAPI_FUNC(PolarityArrowData*) getVectorArrowPCP(const int32_t &pId);

/**
 * @brief Runs the polarity model along with a simulation. 
 * Must be called before doing any operations with this module. 
 * 
 */
CPPAPI_FUNC(void) load();

/**
 * @brief Defines polarity state dynamics and anisotropic adhesion
 * 
 */
struct CAPI_EXPORT ContactPotential : Potential {
    /** Flat interaction coefficient */
    FloatP_t couplingFlat;

    /** Orthogonal interaction coefficient */
    FloatP_t couplingOrtho;
    
    /** Lateral interaction coefficient */
    FloatP_t couplingLateral;
    
    /** Distance coefficient */
    FloatP_t distanceCoeff;

    /** Contact type (e.g., normal, isotropic or anisotropic) */
    PolarContactType cType;

    /** Magnitude of force due to potential */
    FloatP_t mag;

    /** State vector dynamics rate due to potential */
    FloatP_t rate;

    /** Bending coefficient */
    FloatP_t bendingCoeff;

    ContactPotential();
};

/**
 * @brief Creates a contact-mediated polarity potential
 * 
 * @param cutoff cutoff distance
 * @param mag magnitude of force
 * @param rate rate of state vector dynamics
 * @param distanceCoeff distance coefficient
 * @param couplingFlat flat coupling coefficient
 * @param couplingOrtho orthogonal coupling coefficient
 * @param couplingLateral lateral coupling coefficient
 * @param contactType type of contact; available are regular, isotropic, anisotropic
 * @param bendingCoeff bending coefficient
 * @return ContactPotential* 
 */
CPPAPI_FUNC(ContactPotential*) createContactPotential(
    const FloatP_t &cutoff, 
    const FloatP_t &mag=1.0, 
    const FloatP_t &rate=1.0,
    const FloatP_t &distanceCoeff=1.0, 
    const FloatP_t &couplingFlat=1.0, 
    const FloatP_t &couplingOrtho=0.0, 
    const FloatP_t &couplingLateral=0.0, 
    std::string contactType="regular", 
    const FloatP_t &bendingCoeff=0.0
);

};


#endif // _SOURCE_MODELS_CENTER_TFCELLPOLARITY_H_