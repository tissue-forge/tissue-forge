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

#include "tfCForce.h"

#include "TissueForge_c_private.h"
#include "tfCParticle.h"

#include <tfForce.h>
#include <tfParticle.h>


using namespace TissueForge;


////////////////////////
// Function factories //
////////////////////////

// UserForceFuncType

static tfUserForceFuncTypeHandleFcn _UserForceFuncType_factory_evalFcn;

FVector3 UserForceFuncType_eval(CustomForce *f) {
    tfCustomForceHandle fhandle {(void*)f};
    FVector3 res;
    _UserForceFuncType_factory_evalFcn(&fhandle, res.data());
    return res;
}

HRESULT UserForceFuncType_factory(struct tfUserForceFuncTypeHandle *handle, tfUserForceFuncTypeHandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    _UserForceFuncType_factory_evalFcn = *fcn;
    UserForceFuncType *eval_fcn = new UserForceFuncType(UserForceFuncType_eval);
    handle->tfObj = (void*)eval_fcn;
    return S_OK;
}


//////////////////
// Module casts //
//////////////////


namespace TissueForge { 


    Force *castC(struct tfForceHandle *handle) {
        return castC<Force, tfForceHandle>(handle);
    }

    ForceSum *castC(struct tfForceSumHandle *handle) {
        return castC<ForceSum, tfForceSumHandle>(handle);
    }

    CustomForce *castC(struct tfCustomForceHandle *handle) {
        return castC<CustomForce, tfCustomForceHandle>(handle);
    }

    Berendsen *castC(struct tfBerendsenHandle *handle) {
        return castC<Berendsen, tfBerendsenHandle>(handle);
    }

    Gaussian *castC(struct tfGaussianHandle *handle) {
        return castC<Gaussian, tfGaussianHandle>(handle);
    }

    Friction *castC(struct tfFrictionHandle *handle) {
        return castC<Friction, tfFrictionHandle>(handle);
}

}

#define TFC_FORCEHANDLE_GET(handle) \
    Force *frc = TissueForge::castC<Force, tfForceHandle>(handle); \
    TFC_PTRCHECK(frc);

#define TFC_FORCESUMHANDLE_GET(handle) \
    ForceSum *frc = TissueForge::castC<ForceSum, tfForceSumHandle>(handle); \
    TFC_PTRCHECK(frc);

#define TFC_CUSTOMFORCEHANDLE_GET(handle) \
    CustomForce *frc = TissueForge::castC<CustomForce, tfCustomForceHandle>(handle); \
    TFC_PTRCHECK(frc);

#define TFC_BERENDSENHANDLE_GET(handle) \
    Berendsen *frc = TissueForge::castC<Berendsen, tfBerendsenHandle>(handle); \
    TFC_PTRCHECK(frc);

#define TFC_GAUSSIANHANDLE_GET(handle) \
    Gaussian *frc = TissueForge::castC<Gaussian, tfGaussianHandle>(handle); \
    TFC_PTRCHECK(frc);

#define TFC_FRICTIONHANDLE_GET(handle) \
    Friction *frc = TissueForge::castC<Friction, tfFrictionHandle>(handle); \
    TFC_PTRCHECK(frc);


////////////////
// FORCE_TYPE //
////////////////


HRESULT tfFORCE_TYPE_init(struct tfFORCE_TYPEHandle *handle) {
    handle->FORCE_FORCE = FORCE_FORCE;
    handle->FORCE_BERENDSEN = FORCE_BERENDSEN;
    handle->FORCE_GAUSSIAN = FORCE_GAUSSIAN;
    handle->FORCE_FRICTION = FORCE_FRICTION;
    handle->FORCE_SUM = FORCE_SUM;
    handle->FORCE_CUSTOM = FORCE_CUSTOM;
    return S_OK;
}


///////////////////////
// UserForceFuncType //
///////////////////////


HRESULT tfForce_EvalFcn_init(struct tfUserForceFuncTypeHandle *handle, tfUserForceFuncTypeHandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return UserForceFuncType_factory(handle, fcn);
}

HRESULT tfForce_EvalFcn_destroy(struct tfUserForceFuncTypeHandle *handle) {
    return TissueForge::capi::destroyHandle<UserForceFuncType, tfUserForceFuncTypeHandle>(handle) ? S_OK : E_FAIL;
}


///////////
// Force //
///////////

HRESULT tfForce_getType(struct tfForceHandle *handle, unsigned int *te) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(te);
    *te = (unsigned int)frc->type;
    return S_OK;
}

HRESULT tfForce_bind_species(struct tfForceHandle *handle, struct tfParticleTypeHandle *a_type, const char *coupling_symbol) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(a_type); TFC_PTRCHECK(a_type->tfObj);
    TFC_PTRCHECK(coupling_symbol);
    ParticleType *_a_type = (ParticleType*)a_type->tfObj;
    return frc->bind_species(_a_type, coupling_symbol);
}

HRESULT tfForce_toString(struct tfForceHandle *handle, char **str, unsigned int *numChars) {
    TFC_FORCEHANDLE_GET(handle);
    return TissueForge::capi::str2Char(frc->toString(), str, numChars);
}

HRESULT tfForce_fromString(struct tfForceHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    Force *frc = Force::fromString(str);
    TFC_PTRCHECK(frc);
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfForce_destroy(struct tfForceHandle *handle) {
    return TissueForge::capi::destroyHandle<Force, tfForceHandle>(handle) ? S_OK : E_FAIL;
}


//////////////
// ForceSum //
//////////////


HRESULT tfForceSum_checkType(struct tfForceHandle *handle, bool *isType) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(isType);
    *isType = frc->type == FORCE_SUM;
    return S_OK;
}

HRESULT tfForceSum_toBase(struct tfForceSumHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_FORCESUMHANDLE_GET(handle);
    TFC_PTRCHECK(baseHandle);
    baseHandle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfForceSum_fromBase(struct tfForceHandle *baseHandle, struct tfForceSumHandle *handle) {
    TFC_FORCEHANDLE_GET(baseHandle);
    TFC_PTRCHECK(handle);
    bool checkType;
    if((tfForceSum_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfForceSum_getConstituents(struct tfForceSumHandle *handle, struct tfForceHandle *f1, struct tfForceHandle *f2) {
    TFC_FORCESUMHANDLE_GET(handle);
    TFC_PTRCHECK(f1);
    TFC_PTRCHECK(f2);
    if(frc->f1) 
        f1->tfObj = (void*)frc->f1;
    if(frc->f2) 
        f2->tfObj = (void*)frc->f2;
    return S_OK;
}


/////////////////
// CustomForce //
/////////////////


HRESULT tfCustomForce_init(struct tfCustomForceHandle *handle, struct tfUserForceFuncTypeHandle *func, tfFloatP_t period) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(func); TFC_PTRCHECK(func->tfObj);
    UserForceFuncType *_func = (UserForceFuncType*)func->tfObj;
    CustomForce *frc = new CustomForce(_func, period);
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfCustomForce_checkType(struct tfForceHandle *handle, bool *isType) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(isType);
    *isType = frc->type == FORCE_CUSTOM;
    return S_OK;
}

HRESULT tfCustomForce_toBase(struct tfCustomForceHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(baseHandle);
    baseHandle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfCustomForce_fromBase(struct tfForceHandle *baseHandle, struct tfCustomForceHandle *handle) {
    TFC_FORCEHANDLE_GET(baseHandle);
    TFC_PTRCHECK(handle);
    bool checkType;
    if((tfCustomForce_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfCustomForce_getPeriod(struct tfCustomForceHandle *handle, tfFloatP_t *period) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(period);
    *period = frc->getPeriod();
    return S_OK;
}

HRESULT tfCustomForce_setPeriod(struct tfCustomForceHandle *handle, tfFloatP_t period) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    frc->setPeriod(period);
    return S_OK;
}

HRESULT tfCustomForce_setFunction(struct tfCustomForceHandle *handle, struct tfUserForceFuncTypeHandle *fcn) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    UserForceFuncType *_fcn = (UserForceFuncType*)fcn->tfObj;
    frc->setValue(_fcn);
    return S_OK;
}

HRESULT tfCustomForce_getValue(struct tfCustomForceHandle *handle, tfFloatP_t **force) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    FVector3 _force = frc->getValue();
    TFC_VECTOR3_COPYFROM(_force, (*force));
    return S_OK;
}

HRESULT tfCustomForce_setValue(struct tfCustomForceHandle *handle, tfFloatP_t *force) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    frc->setValue(FVector3::from(force));
    return S_OK;
}

HRESULT tfCustomForce_getLastUpdate(struct tfCustomForceHandle *handle, tfFloatP_t *lastUpdate) {
    TFC_CUSTOMFORCEHANDLE_GET(handle);
    TFC_PTRCHECK(lastUpdate);
    *lastUpdate = frc->lastUpdate;
    return S_OK;
}


///////////////
// Berendsen //
///////////////


HRESULT tfBerendsen_init(struct tfBerendsenHandle *handle, tfFloatP_t tau) {
    TFC_PTRCHECK(handle);
    Force *frc = Force::berendsen_tstat(tau);
    TFC_PTRCHECK(frc);
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfBerendsen_checkType(struct tfForceHandle *handle, bool *isType) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(isType);
    *isType = frc->type == FORCE_BERENDSEN;
    return S_OK;
}

HRESULT tfBerendsen_toBase(struct tfBerendsenHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_BERENDSENHANDLE_GET(handle);
    TFC_PTRCHECK(baseHandle);
    baseHandle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfBerendsen_fromBase(struct tfForceHandle *baseHandle, struct tfBerendsenHandle *handle) {
    TFC_FORCEHANDLE_GET(baseHandle);
    TFC_PTRCHECK(handle);
    bool checkType;
    if((tfBerendsen_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfBerendsen_getTimeConstant(struct tfBerendsenHandle *handle, tfFloatP_t *tau) {
    TFC_BERENDSENHANDLE_GET(handle);
    TFC_PTRCHECK(tau);
    if(frc->itau == 0.f) 
        return E_FAIL;
    *tau = 1.f / frc->itau;
    return S_OK;
}

HRESULT tfBerendsen_setTimeConstant(struct tfBerendsenHandle *handle, tfFloatP_t tau) {
    TFC_BERENDSENHANDLE_GET(handle);
    if(tau == 0.f) 
        return E_FAIL;
    frc->itau = 1.f / tau;
    return S_OK;
}


//////////////
// Gaussian //
//////////////

HRESULT tfGaussian_init(struct tfGaussianHandle *handle, tfFloatP_t std, tfFloatP_t mean, tfFloatP_t duration) {
    TFC_PTRCHECK(handle);
    Force *frc = Force::random(std, mean, duration);
    TFC_PTRCHECK(frc);
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfGaussian_checkType(struct tfForceHandle *handle, bool *isType) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(isType);
    *isType = frc->type == FORCE_GAUSSIAN;
    return S_OK;
}

HRESULT tfGaussian_toBase(struct tfGaussianHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_GAUSSIANHANDLE_GET(handle);
    TFC_PTRCHECK(baseHandle);
    baseHandle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfGaussian_fromBase(struct tfForceHandle *baseHandle, struct tfGaussianHandle *handle) {
    TFC_FORCEHANDLE_GET(baseHandle);
    TFC_PTRCHECK(handle);
    bool checkType;
    if((tfGaussian_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfGaussian_getStd(struct tfGaussianHandle *handle, tfFloatP_t *std) {
    TFC_GAUSSIANHANDLE_GET(handle);
    TFC_PTRCHECK(std);
    *std = frc->std;
    return S_OK;
}

HRESULT tfGaussian_setStd(struct tfGaussianHandle *handle, tfFloatP_t std) {
    TFC_GAUSSIANHANDLE_GET(handle);
    frc->std = std;
    return S_OK;
}

HRESULT tfGaussian_getMean(struct tfGaussianHandle *handle, tfFloatP_t *mean) {
    TFC_GAUSSIANHANDLE_GET(handle);
    TFC_PTRCHECK(mean);
    *mean = frc->mean;
    return S_OK;
}

HRESULT tfGaussian_setMean(struct tfGaussianHandle *handle, tfFloatP_t mean) {
    TFC_GAUSSIANHANDLE_GET(handle);
    frc->mean = mean;
    return S_OK;
}

HRESULT tfGaussian_getDuration(struct tfGaussianHandle *handle, tfFloatP_t *duration) {
    TFC_GAUSSIANHANDLE_GET(handle);
    TFC_PTRCHECK(duration);
    *duration = frc->durration_steps;
    return S_OK;
}

HRESULT tfGaussian_setDuration(struct tfGaussianHandle *handle, tfFloatP_t duration) {
    TFC_GAUSSIANHANDLE_GET(handle);
    frc->durration_steps = duration;
    return S_OK;
}


//////////////
// Friction //
//////////////


HRESULT tfFriction_init(struct tfFrictionHandle *handle, tfFloatP_t coeff) {
    TFC_PTRCHECK(handle);
    Force *frc = Force::friction(coeff);
    TFC_PTRCHECK(frc);
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfFriction_checkType(struct tfForceHandle *handle, bool *isType) {
    TFC_FORCEHANDLE_GET(handle);
    TFC_PTRCHECK(isType);
    *isType = frc->type == FORCE_FRICTION;
    return S_OK;
}

HRESULT tfFriction_toBase(struct tfFrictionHandle *handle, struct tfForceHandle *baseHandle) {
    TFC_FRICTIONHANDLE_GET(handle);
    TFC_PTRCHECK(baseHandle);
    baseHandle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfFriction_fromBase(struct tfForceHandle *baseHandle, struct tfFrictionHandle *handle) {
    TFC_FORCEHANDLE_GET(baseHandle);
    TFC_PTRCHECK(handle);
    bool checkType;
    if((tfFriction_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->tfObj = (void*)frc;
    return S_OK;
}

HRESULT tfFriction_getCoef(struct tfFrictionHandle *handle, tfFloatP_t *coef) {
    TFC_FRICTIONHANDLE_GET(handle);
    TFC_PTRCHECK(coef);
    *coef = frc->coef;
    return S_OK;
}

HRESULT tfFriction_setCoef(struct tfFrictionHandle *handle, tfFloatP_t coef) {
    TFC_FRICTIONHANDLE_GET(handle);
    frc->coef = coef;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfForce_add(struct tfForceHandle *f1, struct tfForceHandle *f2, struct tfForceSumHandle *fSum) {
    TFC_PTRCHECK(f1); TFC_PTRCHECK(f1->tfObj);
    TFC_PTRCHECK(f2); TFC_PTRCHECK(f2->tfObj);
    TFC_PTRCHECK(fSum);
    Force &_f1 = *(Force*)f1->tfObj;
    Force &_f2 = *(Force*)f2->tfObj;
    Force &_f3 = _f1 + _f2;
    ForceSum *_fSum = (ForceSum*)&_f3;
    fSum->tfObj = (void*)_fSum;
    return S_OK;
}
