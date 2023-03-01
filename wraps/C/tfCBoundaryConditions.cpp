/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tfCBoundaryConditions.h"

#include "TissueForge_c_private.h"

#include <tfBoundaryConditions.h>
#include <tfSpace.h>


using namespace TissueForge;


namespace TissueForge { 


    BoundaryCondition *castC(struct tfBoundaryConditionHandle *handle) {
        return castC<BoundaryCondition, tfBoundaryConditionHandle>(handle);
    }

    BoundaryConditions *castC(struct tfBoundaryConditionsHandle *handle) {
        return castC<BoundaryConditions, tfBoundaryConditionsHandle>(handle);
    }

    BoundaryConditionsArgsContainer *castC(struct tfBoundaryConditionsArgsContainerHandle *handle) {
        return castC<BoundaryConditionsArgsContainer, tfBoundaryConditionsArgsContainerHandle>(handle);
    }

}

#define TFC_BOUNDARYCONDITIONHANDLE_GET(handle, varname) \
    BoundaryCondition *varname = TissueForge::castC<BoundaryCondition, tfBoundaryConditionHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, varname) \
    BoundaryConditions *varname = TissueForge::castC<BoundaryConditions, tfBoundaryConditionsHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, varname) \
    BoundaryConditionsArgsContainer *varname = TissueForge::castC<BoundaryConditionsArgsContainer, tfBoundaryConditionsArgsContainerHandle>(handle); \
    TFC_PTRCHECK(varname);


////////////////////////////////
// BoundaryConditionSpaceKind //
////////////////////////////////


HRESULT tfBoundaryConditionSpaceKind_init(struct tfBoundaryConditionSpaceKindHandle *handle) {
    handle->SPACE_PERIODIC_X = space_periodic_x;
    handle->SPACE_PERIODIC_Y = space_periodic_y;
    handle->SPACE_PERIODIC_Z = space_periodic_z;
    handle->SPACE_PERIODIC_FULL = space_periodic_full;
    handle->SPACE_FREESLIP_X = SPACE_FREESLIP_X;
    handle->SPACE_FREESLIP_Y = SPACE_FREESLIP_Y;
    handle->SPACE_FREESLIP_Z = SPACE_FREESLIP_Z;
    handle->SPACE_FREESLIP_FULL = SPACE_FREESLIP_FULL;
    return S_OK;
}


///////////////////////////
// BoundaryConditionKind //
///////////////////////////


HRESULT tfBoundaryConditionKind_init(struct tfBoundaryConditionKindHandle *handle) {
    handle->BOUNDARY_PERIODIC = BOUNDARY_PERIODIC;
    handle->BOUNDARY_FREESLIP = BOUNDARY_FREESLIP;
    handle->BOUNDARY_RESETTING = BOUNDARY_RESETTING;
    return S_OK;
}


///////////////////////
// BoundaryCondition //
///////////////////////


HRESULT tfBoundaryCondition_getId(struct tfBoundaryConditionHandle *handle, int *bid) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bid);
    *bid = bhandle->id;
    return S_OK;
}

HRESULT tfBoundaryCondition_getVelocity(struct tfBoundaryConditionHandle *handle, tfFloatP_t **velocity) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(velocity);
    TFC_VECTOR3_COPYFROM(bhandle->velocity, (*velocity));
    return S_OK;
}

HRESULT tfBoundaryCondition_setVelocity(struct tfBoundaryConditionHandle *handle, tfFloatP_t *velocity) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(velocity);
    TFC_VECTOR3_COPYTO(velocity, bhandle->velocity);
    return S_OK;
}

HRESULT tfBoundaryCondition_getRestore(struct tfBoundaryConditionHandle *handle, tfFloatP_t *restore) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(restore);
    *restore = bhandle->restore;
    return S_OK;
}

HRESULT tfBoundaryCondition_setRestore(struct tfBoundaryConditionHandle *handle, tfFloatP_t restore) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    bhandle->restore = restore;
    return S_OK;
}

HRESULT tfBoundaryCondition_getNormal(struct tfBoundaryConditionHandle *handle, tfFloatP_t **normal) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(normal);
    TFC_VECTOR3_COPYFROM(bhandle->normal, (*normal));
    return S_OK;
}

HRESULT tfBoundaryCondition_getRadius(struct tfBoundaryConditionHandle *handle, tfFloatP_t *radius) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(radius);
    *radius = bhandle->radius;
    return S_OK;
}

HRESULT tfBoundaryCondition_setRadius(struct tfBoundaryConditionHandle *handle, tfFloatP_t radius) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    bhandle->radius = radius;
    return S_OK;
}

HRESULT tfBoundaryCondition_setPotential(struct tfBoundaryConditionHandle *handle, struct tfParticleTypeHandle *partHandle, struct tfPotentialHandle *potHandle) {
    TFC_BOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(partHandle); TFC_PTRCHECK(partHandle->tfObj);
    TFC_PTRCHECK(potHandle); TFC_PTRCHECK(potHandle->tfObj);
    bhandle->set_potential((ParticleType*)partHandle->tfObj, (Potential*)potHandle->tfObj);
    return S_OK;
}


////////////////////////
// BoundaryConditions //
////////////////////////


HRESULT tfBoundaryConditions_getTop(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->top;
    return S_OK;
}

HRESULT tfBoundaryConditions_getBottom(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->bottom;
    return S_OK;
}

HRESULT tfBoundaryConditions_getLeft(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->left;
    return S_OK;
}

HRESULT tfBoundaryConditions_getRight(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->right;
    return S_OK;
}

HRESULT tfBoundaryConditions_getFront(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->front;
    return S_OK;
}

HRESULT tfBoundaryConditions_getBack(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(bchandle);
    bchandle->tfObj = (void*)&bhandle->back;
    return S_OK;
}

HRESULT tfBoundaryConditions_setPotential(struct tfBoundaryConditionsHandle *handle, struct tfParticleTypeHandle *partHandle, struct tfPotentialHandle *potHandle) {
    TFC_BOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(partHandle); TFC_PTRCHECK(partHandle->tfObj);
    TFC_PTRCHECK(potHandle); TFC_PTRCHECK(potHandle->tfObj);
    bhandle->set_potential((ParticleType*)partHandle->tfObj, (Potential*)potHandle->tfObj);
    return S_OK;
}


/////////////////////////////////////
// BoundaryConditionsArgsContainer //
/////////////////////////////////////


HRESULT tfBoundaryConditionsArgsContainer_init(struct tfBoundaryConditionsArgsContainerHandle *handle) {
    TFC_PTRCHECK(handle);
    BoundaryConditionsArgsContainer *bargs = new BoundaryConditionsArgsContainer();
    handle->tfObj = (void*)bargs;
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_destroy(struct tfBoundaryConditionsArgsContainerHandle *handle) {
    return TissueForge::capi::destroyHandle<BoundaryConditionsArgsContainer, tfBoundaryConditionsArgsContainerHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfBoundaryConditionsArgsContainer_hasValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, bool *has) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(has);
    *has = bhandle->bcValue != NULL;
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_getValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, unsigned int *_bcValue) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(_bcValue);
    TFC_PTRCHECK(bhandle->bcValue);
    *_bcValue = *bhandle->bcValue;
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_setValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, unsigned int _bcValue) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    bhandle->setValueAll(_bcValue);
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_hasValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(has);
    *has = bhandle->bcVals == NULL ? false : bhandle->bcVals->find(name) != bhandle->bcVals->end();
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_getValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int *value) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(value);
    TFC_PTRCHECK(bhandle->bcVals);
    auto itr = bhandle->bcVals->find(name);
    if(itr == bhandle->bcVals->end()) 
        return E_FAIL;
    *value = itr->second;
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_setValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int value) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    bhandle->setValue(name, value);
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_hasVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(has);
    *has = bhandle->bcVels == NULL ? false : bhandle->bcVels->find(name) != bhandle->bcVels->end();
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_getVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t **velocity) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(velocity);
    TFC_PTRCHECK(bhandle->bcVels);
    auto itr = bhandle->bcVels->find(name);
    if(itr == bhandle->bcVels->end()) 
        return E_FAIL;
    FVector3 _velocity = itr->second;
    TFC_VECTOR3_COPYFROM(_velocity, (*velocity));
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_setVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t *velocity) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(velocity);
    bhandle->setVelocity(name, FVector3::from(velocity));
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_hasRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(has);
    *has = bhandle->bcRestores == NULL ? false : bhandle->bcRestores->find(name) != bhandle->bcRestores->end();
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_getRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t *restore) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(restore);
    TFC_PTRCHECK(bhandle->bcRestores);
    auto itr = bhandle->bcRestores->find(name);
    if(itr == bhandle->bcRestores->end()) 
        return E_FAIL;
    *restore = itr->second;
    return S_OK;
}

HRESULT tfBoundaryConditionsArgsContainer_setRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t restore) {
    TFC_BOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(name);
    bhandle->setRestore(name, restore);
    return S_OK;
}
