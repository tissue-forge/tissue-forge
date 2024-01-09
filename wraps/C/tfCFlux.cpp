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

#include "tfCFlux.h"

#include "TissueForge_c_private.h"

#include <tfFlux.h>


using namespace TissueForge;


namespace TissueForge { 


    Flux *castC(struct tfFluxHandle *handle) {
        return castC<Flux, tfFluxHandle>(handle);
    }

    Fluxes *castC(struct tfFluxesHandle *handle) {
        return castC<Fluxes, tfFluxesHandle>(handle);
    }

}

#define TFC_FLUX_GET(handle, varname) \
    Flux *varname = TissueForge::castC<Flux, tfFluxHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_FLUXES_GET(handle, varname) \
    Fluxes *varname = TissueForge::castC<Fluxes, tfFluxesHandle>(handle); \
    TFC_PTRCHECK(varname);


//////////////
// FluxKind //
//////////////


HRESULT tfFluxKindHandle_init(struct tfFluxKindHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->FLUX_FICK = FLUX_FICK;
    handle->FLUX_SECRETE = FLUX_SECRETE;
    handle->FLUX_UPTAKE = FLUX_UPTAKE;
    return S_OK;
}


//////////
// Flux //
//////////


HRESULT tfFlux_getSize(struct tfFluxHandle *handle, unsigned int *size) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(size);
    *size = flx->size;
    return S_OK;
}

HRESULT tfFlux_getKind(struct tfFluxHandle *handle, unsigned int index, unsigned int *kind) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(kind);
    if(index >= flx->size) 
        return E_FAIL;
    *kind = flx->kinds[index];
    return S_OK;
}

HRESULT tfFlux_getTypeIds(struct tfFluxHandle *handle, unsigned int index, unsigned int *typeid_a, unsigned int *typeid_b) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(typeid_a);
    TFC_PTRCHECK(typeid_b);
    if(index >= flx->size) 
        return E_FAIL;
    auto typeids = flx->type_ids[index];
    *typeid_a = typeids.a;
    *typeid_b = typeids.b;
    return S_OK;
}

HRESULT tfFlux_getCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *coef) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(coef);
    if(index >= flx->size) 
        return E_FAIL;
    *coef = flx->coef[index];
    return S_OK;
}

HRESULT tfFlux_setCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t coef) {
    TFC_FLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->coef[index] = coef;
    return S_OK;
}

HRESULT tfFlux_getDecayCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *decay_coef) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(decay_coef);
    if(index >= flx->size) 
        return E_FAIL;
    *decay_coef = flx->decay_coef[index];
    return S_OK;
}

HRESULT tfFlux_setDecayCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t decay_coef) {
    TFC_FLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->decay_coef[index] = decay_coef;
    return S_OK;
}

HRESULT tfFlux_getTarget(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *target) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(target);
    if(index >= flx->size) 
        return E_FAIL;
    *target = flx->target[index];
    return S_OK;
}

HRESULT tfFlux_setTarget(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t target) {
    TFC_FLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->target[index] = target;
    return S_OK;
}

HRESULT tfFlux_getCutoff(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *cutoff) {
    TFC_FLUX_GET(handle, flx);
    TFC_PTRCHECK(cutoff);
    if(index >= flx->size) 
        return E_FAIL;
    *cutoff = flx->cutoff[index];
    return S_OK;
}

HRESULT tfFlux_setCutoff(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t cutoff) {
    TFC_FLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->cutoff[index] = cutoff;
    return S_OK;
}


////////////
// Fluxes //
////////////


HRESULT tfFluxes_getSize(struct tfFluxesHandle *handle, int *size) {
    TFC_FLUXES_GET(handle, flxs);
    TFC_PTRCHECK(size);
    *size = flxs->size;
    return S_OK;
}

HRESULT tfFluxes_getFlux(struct tfFluxesHandle *handle, unsigned int index, struct tfFluxHandle *flux) {
    TFC_FLUXES_GET(handle, flxs);
    TFC_PTRCHECK(flux);
    if(index >= flxs->size) 
        return E_FAIL;
    flux->tfObj = (void*)&flxs->fluxes[index];
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////

HRESULT tfFluxes_fluxFick(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t decay,
    tfFloatP_t cutoff) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(A); TFC_PTRCHECK(A->tfObj);
    TFC_PTRCHECK(B); TFC_PTRCHECK(B->tfObj);
    TFC_PTRCHECK(name);
    Fluxes *flxs = Fluxes::fluxFick((ParticleType*)A->tfObj, (ParticleType*)B->tfObj, name, k, decay, cutoff);
    TFC_PTRCHECK(flxs);
    handle->tfObj = (void*)flxs;
    return S_OK;
}

HRESULT tfFluxes_secrete(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t target, 
    tfFloatP_t decay, 
    tfFloatP_t cutoff) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(A); TFC_PTRCHECK(A->tfObj);
    TFC_PTRCHECK(B); TFC_PTRCHECK(B->tfObj);
    TFC_PTRCHECK(name);
    Fluxes *flxs = Fluxes::secrete((ParticleType*)A->tfObj, (ParticleType*)B->tfObj, name, k, target, decay, cutoff);
    TFC_PTRCHECK(flxs);
    handle->tfObj = (void*)flxs;
    return S_OK;
}

HRESULT tfFluxes_uptake(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t target, 
    tfFloatP_t decay, 
    tfFloatP_t cutoff) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(A); TFC_PTRCHECK(A->tfObj);
    TFC_PTRCHECK(B); TFC_PTRCHECK(B->tfObj);
    TFC_PTRCHECK(name);
    Fluxes *flxs = Fluxes::uptake((ParticleType*)A->tfObj, (ParticleType*)B->tfObj, name, k, target, decay, cutoff);
    TFC_PTRCHECK(flxs);
    handle->tfObj = (void*)flxs;
    return S_OK;
}
