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

#include "tfCCellPolarity.h"

#include <TissueForge_c_private.h>

#include <models/center/CellPolarity/tfCellPolarity.h>


using namespace TissueForge;
namespace CPMod = TissueForge::models::center::CellPolarity;


//////////////////
// Module casts //
//////////////////


namespace TissueForge { 


    CPMod::PersistentForce *castC(struct tfCellPolarityPersistentForceHandle *handle) {
        return castC<CPMod::PersistentForce, tfCellPolarityPersistentForceHandle>(handle);
    }

    CPMod::PolarityArrowData *castC(struct tfCellPolarityPolarityArrowDataHandle *handle) {
        return castC<CPMod::PolarityArrowData, tfCellPolarityPolarityArrowDataHandle>(handle);
    }

    CPMod::ContactPotential *castC(struct tfCellPolarityContactPotentialHandle *handle) {
        return castC<CPMod::ContactPotential, tfCellPolarityContactPotentialHandle>(handle);
    }

}

#define TFC_PERSISTENTFORCEHANDLE_GET(handle, varname) \
    CPMod::PersistentForce *varname = TissueForge::castC<CPMod::PersistentForce, tfCellPolarityPersistentForceHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_POLARITYARROWDATAHANDLE_GET(handle, varname) \
    CPMod::PolarityArrowData *varname = TissueForge::castC<CPMod::PolarityArrowData, tfCellPolarityPolarityArrowDataHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, varname) \
    CPMod::ContactPotential *varname = TissueForge::castC<CPMod::ContactPotential, tfCellPolarityContactPotentialHandle>(handle); \
    TFC_PTRCHECK(varname);


//////////////////////
// PolarContactType //
//////////////////////


HRESULT tfPolarContactType_init(struct tfCellPolarityPolarContactTypeEnumHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->REGULAR = CPMod::PolarContactType::REGULAR;
    handle->ISOTROPIC = CPMod::PolarContactType::ISOTROPIC;
    handle->ANISOTROPIC = CPMod::PolarContactType::ANISOTROPIC;
    return S_OK;
}


/////////////////////
// PersistentForce //
/////////////////////


HRESULT tfCellPolarityPersistentForce_init(struct tfCellPolarityPersistentForceHandle *handle, float sensAB, float sensPCP) {
    TFC_PTRCHECK(handle);
    CPMod::PersistentForce *f = new CPMod::PersistentForce();
    f->sensAB = sensAB;
    f->sensPCP = sensPCP;
    handle->tfObj = (void*)f;
    return S_OK;
}

HRESULT tfCellPolarityPersistentForce_destroy(struct tfCellPolarityPersistentForceHandle *handle) {
    return TissueForge::capi::destroyHandle<CPMod::PersistentForce, struct tfCellPolarityPersistentForceHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfCellPolarityPersistentForce_toBase(struct tfCellPolarityPersistentForceHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_PERSISTENTFORCEHANDLE_GET(handle, pf);
    TFC_PTRCHECK(baseHandle);
    Force *_pf = (Force*)pf;
    baseHandle->tfObj = (void*)_pf;
    return S_OK;
}

HRESULT tfCellPolarityPersistentForce_fromBase(struct tfForceHandle *baseHandle, struct tfCellPolarityPersistentForceHandle *handle) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(baseHandle); TFC_PTRCHECK(baseHandle->tfObj);
    CPMod::PersistentForce *pf = (CPMod::PersistentForce*)baseHandle->tfObj;
    handle->tfObj = (void*)pf;
    return S_OK;
}


/////////////////////////////
// CPMod::ContactPotential //
/////////////////////////////


const char *cTypeMap[] = {
    "regular", 
    "isotropic", 
    "anisotropic"
};


HRESULT tfCellPolarityContactPotential_init(
    struct tfCellPolarityContactPotentialHandle *handle, 
    float cutoff, 
    float couplingFlat, 
    float couplingOrtho, 
    float couplingLateral, 
    float distanceCoeff, 
    unsigned int cType, 
    float mag, 
    float rate, 
    float bendingCoeff) 
{
    TFC_PTRCHECK(handle);
    CPMod::ContactPotential *pc = CPMod::createContactPotential(
        cutoff, mag, rate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, cTypeMap[cType], bendingCoeff
    );
    handle->tfObj = (void*)pc;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_destroy(struct tfCellPolarityContactPotentialHandle *handle) {
    return TissueForge::capi::destroyHandle<CPMod::ContactPotential, tfCellPolarityContactPotentialHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfCellPolarityContactPotential_toBase(struct tfCellPolarityContactPotentialHandle *handle, struct tfPotentialHandle *baseHandle) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(baseHandle);
    Potential *_pc = (Potential*)pc;
    baseHandle->tfObj = (void*)_pc;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_fromBase(struct tfPotentialHandle *baseHandle, struct tfCellPolarityContactPotentialHandle *handle) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(baseHandle); TFC_PTRCHECK(baseHandle->tfObj);
    Potential *_pc = (Potential*)baseHandle->tfObj;
    CPMod::ContactPotential *pc = (CPMod::ContactPotential*)_pc;
    handle->tfObj = (void*)pc;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getCouplingFlat(struct tfCellPolarityContactPotentialHandle *handle, float *couplingFlat) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(couplingFlat);
    *couplingFlat = pc->couplingFlat;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setCouplingFlat(struct tfCellPolarityContactPotentialHandle *handle, float couplingFlat) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->couplingFlat = couplingFlat;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getCouplingOrtho(struct tfCellPolarityContactPotentialHandle *handle, float *couplingOrtho) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(couplingOrtho);
    *couplingOrtho = pc->couplingOrtho;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setCouplingOrtho(struct tfCellPolarityContactPotentialHandle *handle, float couplingOrtho) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->couplingOrtho = couplingOrtho;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getCouplingLateral(struct tfCellPolarityContactPotentialHandle *handle, float *couplingLateral) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(couplingLateral);
    *couplingLateral = pc->couplingLateral;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setCouplingLateral(struct tfCellPolarityContactPotentialHandle *handle, float couplingLateral) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->couplingLateral = couplingLateral;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getDistanceCoeff(struct tfCellPolarityContactPotentialHandle *handle, float *distanceCoeff) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(distanceCoeff);
    *distanceCoeff = pc->distanceCoeff;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setDistanceCoeff(struct tfCellPolarityContactPotentialHandle *handle, float distanceCoeff) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->distanceCoeff = distanceCoeff;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getCType(struct tfCellPolarityContactPotentialHandle *handle, unsigned int *cType) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(cType);
    *cType = (unsigned int)pc->cType;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setCType(struct tfCellPolarityContactPotentialHandle *handle, unsigned int cType) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->cType = (CPMod::PolarContactType)cType;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getMag(struct tfCellPolarityContactPotentialHandle *handle, float *mag) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(mag);
    *mag = pc->mag;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setMag(struct tfCellPolarityContactPotentialHandle *handle, float mag) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->mag = mag;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getRate(struct tfCellPolarityContactPotentialHandle *handle, float *rate) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(rate);
    *rate = pc->rate;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setRate(struct tfCellPolarityContactPotentialHandle *handle, float rate) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->rate = rate;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_getBendingCoeff(struct tfCellPolarityContactPotentialHandle *handle, float *bendingCoeff) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    TFC_PTRCHECK(bendingCoeff);
    *bendingCoeff = pc->bendingCoeff;
    return S_OK;
}

HRESULT tfCellPolarityContactPotential_setBendingCoeff(struct tfCellPolarityContactPotentialHandle *handle, float bendingCoeff) {
    TFC_CELLCONTACTPOTENTIALHANDLE_GET(handle, pc);
    pc->bendingCoeff = bendingCoeff;
    return S_OK;
}


///////////////////////
// PolarityArrowData //
///////////////////////


HRESULT tfPolarityArrowData_getArrowLength(struct tfCellPolarityPolarityArrowDataHandle *handle, float *arrowLength) {
    TFC_POLARITYARROWDATAHANDLE_GET(handle, ad);
    TFC_PTRCHECK(arrowLength);
    *arrowLength = ad->arrowLength;
    return S_OK;
}

HRESULT tfPolarityArrowData_setArrowLength(struct tfCellPolarityPolarityArrowDataHandle *handle, float arrowLength) {
    TFC_POLARITYARROWDATAHANDLE_GET(handle, ad);
    ad->arrowLength = arrowLength;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfCellPolarityGetVectorAB(int pId, bool current, float **vec) {
    TFC_PTRCHECK(vec);
    auto _vec = CPMod::getVectorAB(pId, current);
    TFC_VECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT tfCellPolarityGetVectorPCP(int pId, bool current, float **vec) {
    TFC_PTRCHECK(vec);
    auto _vec = CPMod::getVectorPCP(pId, current);
    TFC_VECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT tfCellPolaritySetVectorAB(int pId, float *pVec, bool current, bool init) {
    TFC_PTRCHECK(pVec);
    CPMod::setVectorAB(pId, fVector3::from(pVec), current, init);
    return S_OK;
}

HRESULT tfCellPolaritySetVectorPCP(int pId, float *pVec, bool current, bool init) {
    TFC_PTRCHECK(pVec);
    CPMod::setVectorPCP(pId, fVector3::from(pVec), current, init);
    return S_OK;
}

HRESULT tfCellPolarityUpdate() {
    CPMod::update();
    return S_OK;
}

HRESULT tfCellPolarityRegisterParticle(struct tfParticleHandleHandle *ph) {
    TFC_PTRCHECK(ph); TFC_PTRCHECK(ph->tfObj);
    CPMod::registerParticle((ParticleHandle*)ph->tfObj);
    return S_OK;
}

HRESULT tfCellPolarityUnregister(struct tfParticleHandleHandle *ph) {
    TFC_PTRCHECK(ph); TFC_PTRCHECK(ph->tfObj);
    CPMod::unregister((ParticleHandle*)ph->tfObj);
    return S_OK;
}

HRESULT tfCellPolarityRegisterType(struct tfParticleTypeHandle *pType, const char *initMode, float *initPolarAB, float *initPolarPCP) {
    TFC_PTRCHECK(pType) TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(initMode);
    TFC_PTRCHECK(initPolarAB);
    TFC_PTRCHECK(initPolarPCP);
    CPMod::registerType((ParticleType*)pType->tfObj, initMode, fVector3::from(initPolarAB), fVector3::from(initPolarPCP));
    return S_OK;
}

HRESULT tfCellPolarityGetInitMode(struct tfParticleTypeHandle *pType, char **initMode, unsigned int *numChars) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    return TissueForge::capi::str2Char(CPMod::getInitMode((ParticleType*)pType->tfObj), initMode, numChars);
}

HRESULT tfCellPolaritySetInitMode(struct tfParticleTypeHandle *pType, const char *value) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(value);
    CPMod::setInitMode((ParticleType*)pType->tfObj, value);
    return S_OK;
}

HRESULT tfCellPolarityGetInitPolarAB(struct tfParticleTypeHandle *pType, float **vec) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(vec);
    auto _vec = CPMod::getInitPolarAB((ParticleType*)pType->tfObj);
    TFC_VECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT tfCellPolaritySetInitPolarAB(struct tfParticleTypeHandle *pType, float *value) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(value);
    CPMod::setInitPolarAB((ParticleType*)pType->tfObj, fVector3::from(value));
    return S_OK;
}

HRESULT tfCellPolarityGetInitPolarPCP(struct tfParticleTypeHandle *pType, float **vec) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(vec);
    auto _vec = CPMod::getInitPolarPCP((ParticleType*)pType->tfObj);
    TFC_VECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT tfCellPolaritySetInitPolarPCP(struct tfParticleTypeHandle *pType, float *value) {
    TFC_PTRCHECK(pType); TFC_PTRCHECK(pType->tfObj);
    TFC_PTRCHECK(value);
    CPMod::setInitPolarPCP((ParticleType*)pType->tfObj, fVector3::from(value));
    return S_OK;
}

HRESULT tfCellPolaritySetDrawVectors(bool _draw) {
    CPMod::setDrawVectors(_draw);
    return S_OK;
}

HRESULT tfCellPolaritySetArrowColors(const char *colorAB, const char *colorPCP) {
    TFC_PTRCHECK(colorAB);
    TFC_PTRCHECK(colorPCP);
    CPMod::setArrowColors(colorAB, colorPCP);
    return S_OK;
}

HRESULT tfCellPolaritySetArrowScale(float _scale) {
    CPMod::setArrowScale(_scale);
    return S_OK;
}

HRESULT tfCellPolaritySetArrowLength(float _length) {
    CPMod::setArrowLength(_length);
    return S_OK;
}

HRESULT tfCellPolarityGetVectorArrowAB(unsigned int pId, struct tfCellPolarityPolarityArrowDataHandle *arrowData) {
    TFC_PTRCHECK(arrowData);
    auto _arrowData = CPMod::getVectorArrowAB(pId);
    TFC_PTRCHECK(_arrowData);
    arrowData->tfObj = (void*)_arrowData;
    return S_OK;
}

HRESULT tfCellPolarityGetVectorArrowPCP(unsigned int pId, struct tfCellPolarityPolarityArrowDataHandle *arrowData) {
    TFC_PTRCHECK(arrowData);
    auto _arrowData = CPMod::getVectorArrowPCP(pId);
    TFC_PTRCHECK(_arrowData);
    arrowData->tfObj = (void*)_arrowData;
    return S_OK;
}

HRESULT tfCellPolarityLoad() {
    CPMod::load();
    return S_OK;
}
