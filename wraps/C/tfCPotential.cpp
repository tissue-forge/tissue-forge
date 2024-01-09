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

#include "tfCPotential.h"

#include "TissueForge_c_private.h"
#include "tfCParticle.h"

#include <tfParticle.h>
#include <tfPotential.h>


using namespace TissueForge;


////////////////////////
// Function factories //
////////////////////////


// PotentialEval_ByParticle

static tfPotentialEval_ByParticleHandleFcn _PotentialEval_ByParticle_factory_evalFcn;

void PotentialEval_ByParticle_eval(Potential *p, Particle *part_i, tfFloatP_t *dx, tfFloatP_t r2, tfFloatP_t *e, tfFloatP_t *f) {
    tfPotentialHandle pHandle {(void*)p};
    tfParticleHandleHandle part_iHandle {(void*)part_i->handle()};
    
    (*_PotentialEval_ByParticle_factory_evalFcn)(&pHandle, &part_iHandle, dx, r2, e, f);
}

HRESULT PotentialEval_ByParticle_factory(struct tfPotentialEval_ByParticleHandle &handle, tfPotentialEval_ByParticleHandleFcn &fcn) {
    _PotentialEval_ByParticle_factory_evalFcn = fcn;
    PotentialEval_ByParticle *eval_fcn = new PotentialEval_ByParticle(PotentialEval_ByParticle_eval);
    handle.tfObj = (void*)eval_fcn;
    return S_OK;
}

// PotentialEval_ByParticles

static tfPotentialEval_ByParticlesHandleFcn _PotentialEval_ByParticles_factory_evalFcn;

void PotentialEval_ByParticles_eval(
    struct Potential *p, 
    struct Particle *part_i, 
    struct Particle *part_j, 
    tfFloatP_t *dx, 
    tfFloatP_t r2, 
    tfFloatP_t *e, 
    tfFloatP_t *f) 
{
    tfPotentialHandle pHandle {(void*)p};
    tfParticleHandleHandle part_iHandle {(void*)part_i->handle()};
    tfParticleHandleHandle part_jHandle {(void*)part_j->handle()};

    (*_PotentialEval_ByParticles_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, dx, r2, e, f);
}

HRESULT PotentialEval_ByParticles_factory(struct tfPotentialEval_ByParticlesHandle &handle, tfPotentialEval_ByParticlesHandleFcn &fcn) {
    _PotentialEval_ByParticles_factory_evalFcn = fcn;
    PotentialEval_ByParticles *eval_fcn = new PotentialEval_ByParticles(PotentialEval_ByParticles_eval);
    handle.tfObj = (void*)eval_fcn;
    return S_OK;
}

// PotentialEval_ByParticles3

static tfPotentialEval_ByParticles3HandleFcn _PotentialEval_ByParticles3_factory_evalFcn;

void PotentialEval_ByParticles3_eval(
    struct Potential *p, 
    struct Particle *part_i, 
    struct Particle *part_j, 
    struct Particle *part_k, 
    tfFloatP_t ctheta, 
    tfFloatP_t *e, 
    tfFloatP_t *fi, 
    tfFloatP_t *fk) 
{
    tfPotentialHandle pHandle {(void*)p};
    tfParticleHandleHandle part_iHandle {(void*)part_i->handle()};
    tfParticleHandleHandle part_jHandle {(void*)part_j->handle()};
    tfParticleHandleHandle part_kHandle {(void*)part_k->handle()};

    (*_PotentialEval_ByParticles3_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, &part_kHandle, ctheta, e, fi, fk);
}

HRESULT PotentialEval_ByParticles3_factory(struct tfPotentialEval_ByParticles3Handle &handle, tfPotentialEval_ByParticles3HandleFcn &fcn) {
    _PotentialEval_ByParticles3_factory_evalFcn = fcn;
    PotentialEval_ByParticles3 *eval_fcn = new PotentialEval_ByParticles3(PotentialEval_ByParticles3_eval);
    handle.tfObj = (void*)eval_fcn;
    return S_OK;
}

// PotentialEval_ByParticles4

static tfPotentialEval_ByParticles4HandleFcn _PotentialEval_ByParticles4_factory_evalFcn;

void PotentialEval_ByParticles4_eval(
    struct Potential *p, 
    struct Particle *part_i, 
    struct Particle *part_j, 
    struct Particle *part_k, 
    struct Particle *part_l, 
    tfFloatP_t cphi, 
    tfFloatP_t *e, 
    tfFloatP_t *fi, 
    tfFloatP_t *fl)
{
    tfPotentialHandle pHandle {(void*)p};
    tfParticleHandleHandle part_iHandle {(void*)part_i->handle()};
    tfParticleHandleHandle part_jHandle {(void*)part_j->handle()};
    tfParticleHandleHandle part_kHandle {(void*)part_k->handle()};
    tfParticleHandleHandle part_lHandle {(void*)part_l->handle()};

    (*_PotentialEval_ByParticles4_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, &part_kHandle, &part_lHandle, cphi, e, fi, fl);
}

HRESULT PotentialEval_ByParticles4_factory(struct tfPotentialEval_ByParticles4Handle &handle, tfPotentialEval_ByParticles4HandleFcn fcn) {
    _PotentialEval_ByParticles4_factory_evalFcn = fcn;
    PotentialEval_ByParticles4 *eval_fcn = new PotentialEval_ByParticles4(PotentialEval_ByParticles4_eval);
    handle.tfObj = (void*)eval_fcn;
    return S_OK;
}

// PotentialClear

static tfPotentialClearHandleFcn _PotentialClear_factory_evalFcn;

void PotentialClear_eval(struct Potential *p) {
    tfPotentialHandle pHandle {(void*)p};
    (*_PotentialClear_factory_evalFcn)(&pHandle);
}

HRESULT PotentialClear_factory(struct tfPotentialClearHandle &handle, tfPotentialClearHandleFcn fcn) {
    _PotentialClear_factory_evalFcn = fcn;
    PotentialClear *eval_fcn = new PotentialClear(PotentialClear_eval);
    handle.tfObj = (void*)eval_fcn;
    return S_OK;
}

//////////////////
// Module casts //
//////////////////


namespace TissueForge { 


    Potential *castC(struct tfPotentialHandle *handle) {
        return castC<Potential, tfPotentialHandle>(handle);
    }

}

#define TFC_POTENTIALHANDLE_GET(handle) \
    Potential *pot = TissueForge::castC<Potential, tfPotentialHandle>(handle); \
    TFC_PTRCHECK(pot);


////////////////////
// PotentialFlags //
////////////////////


HRESULT tfPotentialFlags_init(struct tfPotentialFlagsHandle *handle) {
    handle->POTENTIAL_NONE = POTENTIAL_NONE;
    handle->POTENTIAL_LJ126 = POTENTIAL_LJ126;
    handle->POTENTIAL_EWALD = POTENTIAL_EWALD;
    handle->POTENTIAL_COULOMB = POTENTIAL_COULOMB;
    handle->POTENTIAL_SINGLE = POTENTIAL_SINGLE;
    handle->POTENTIAL_R2 = POTENTIAL_R2;
    handle->POTENTIAL_R = POTENTIAL_R;
    handle->POTENTIAL_ANGLE = POTENTIAL_ANGLE;
    handle->POTENTIAL_HARMONIC = POTENTIAL_HARMONIC;
    handle->POTENTIAL_DIHEDRAL = POTENTIAL_DIHEDRAL;
    handle->POTENTIAL_SWITCH = POTENTIAL_SWITCH;
    handle->POTENTIAL_REACTIVE = POTENTIAL_REACTIVE;
    handle->POTENTIAL_SCALED = POTENTIAL_SCALED;
    handle->POTENTIAL_SHIFTED = POTENTIAL_SHIFTED;
    handle->POTENTIAL_BOUND = POTENTIAL_BOUND;
    handle->POTENTIAL_SUM = POTENTIAL_SUM;
    handle->POTENTIAL_PERIODIC = POTENTIAL_PERIODIC;
    handle->POTENTIAL_COULOMBR = POTENTIAL_COULOMBR;
    return S_OK;
}


///////////////////
// PotentialKind //
///////////////////


HRESULT tfPotentialKind_init(struct tfPotentialKindHandle *handle) {
    handle->POTENTIAL_KIND_POTENTIAL = POTENTIAL_KIND_POTENTIAL;
    handle->POTENTIAL_KIND_DPD = POTENTIAL_KIND_DPD;
    handle->POTENTIAL_KIND_BYPARTICLES = POTENTIAL_KIND_BYPARTICLES;
    handle->POTENTIAL_KIND_COMBINATION = POTENTIAL_KIND_COMBINATION;
    return S_OK;
}


//////////////////////////////
// PotentialEval_ByParticle //
//////////////////////////////


HRESULT tfPotentialEval_ByParticle_init(struct tfPotentialEval_ByParticleHandle *handle, tfPotentialEval_ByParticleHandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return PotentialEval_ByParticle_factory(*handle, *fcn);
}

HRESULT tfPotentialEval_ByParticle_destroy(struct tfPotentialEval_ByParticleHandle *handle) {
    return TissueForge::capi::destroyHandle<PotentialEval_ByParticle, tfPotentialEval_ByParticleHandle>(handle) ? S_OK : E_FAIL;
}

///////////////////////////////
// PotentialEval_ByParticles //
///////////////////////////////


HRESULT tfPotentialEval_ByParticles_init(struct tfPotentialEval_ByParticlesHandle *handle, tfPotentialEval_ByParticlesHandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return PotentialEval_ByParticles_factory(*handle, *fcn);
}

HRESULT tfPotentialEval_ByParticles_destroy(struct tfPotentialEval_ByParticlesHandle *handle) {
    return TissueForge::capi::destroyHandle<PotentialEval_ByParticles, tfPotentialEval_ByParticlesHandle>(handle) ? S_OK : E_FAIL;
}


////////////////////////////////
// PotentialEval_ByParticles3 //
////////////////////////////////


HRESULT tfPotentialEval_ByParticles3_init(struct tfPotentialEval_ByParticles3Handle *handle, tfPotentialEval_ByParticles3HandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return PotentialEval_ByParticles3_factory(*handle, *fcn);
}

HRESULT tfPotentialEval_ByParticles3_destroy(struct tfPotentialEval_ByParticles3Handle *handle) {
    return TissueForge::capi::destroyHandle<PotentialEval_ByParticles3, tfPotentialEval_ByParticles3Handle>(handle) ? S_OK : E_FAIL;
}


////////////////////////////////
// PotentialEval_ByParticles4 //
////////////////////////////////


HRESULT tfPotentialEval_ByParticles4_init(struct tfPotentialEval_ByParticles4Handle *handle, tfPotentialEval_ByParticles4HandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return PotentialEval_ByParticles4_factory(*handle, *fcn);
}

HRESULT tfPotentialEval_ByParticles4_destroy(struct tfPotentialEval_ByParticles4Handle *handle) {
    return TissueForge::capi::destroyHandle<PotentialEval_ByParticles4, tfPotentialEval_ByParticles4Handle>(handle) ? S_OK : E_FAIL;
}


////////////////////
// PotentialClear //
////////////////////


HRESULT tfPotentialClear_init(struct tfPotentialClearHandle *handle, tfPotentialClearHandleFcn *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn);
    return PotentialClear_factory(*handle, *fcn);
}

HRESULT tfPotentialClear_destroy(struct tfPotentialClearHandle *handle) {
    return TissueForge::capi::destroyHandle<PotentialClear, tfPotentialClearHandle>(handle) ? S_OK : E_FAIL;
}


///////////////
// Potential //
///////////////


HRESULT tfPotential_destroy(struct tfPotentialHandle *handle) {
    return TissueForge::capi::destroyHandle<Potential, tfPotentialHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfPotential_getName(struct tfPotentialHandle *handle, char **name, unsigned int *numChars) {
    TFC_POTENTIALHANDLE_GET(handle);
    return TissueForge::capi::str2Char(std::string(pot->name), name, numChars);;
}

HRESULT tfPotential_setName(struct tfPotentialHandle *handle, const char *name) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(name);
    std::string sname(name);
    char *cname = new char[sname.size() + 1];
    std::strcpy(cname, sname.c_str());
    pot->name = cname;
    return S_OK;
}

HRESULT tfPotential_getFlags(struct tfPotentialHandle *handle, unsigned int *flags) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(flags);
    *flags = pot->flags;
    return S_OK;
}

HRESULT tfPotential_setFlags(struct tfPotentialHandle *handle, unsigned int flags) {
    TFC_POTENTIALHANDLE_GET(handle);
    pot->flags = flags;
    return S_OK;
}

HRESULT tfPotential_getKind(struct tfPotentialHandle *handle, unsigned int *kind) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(kind);
    *kind = pot->kind;
    return S_OK;
}

HRESULT tfPotential_evalR(struct tfPotentialHandle *handle, tfFloatP_t r, tfFloatP_t *potE) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(potE);
    *potE = (*pot)(r);
    return S_OK;
}

HRESULT tfPotential_evalR0(struct tfPotentialHandle *handle, tfFloatP_t r, tfFloatP_t r0, tfFloatP_t *potE) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(potE);
    *potE = (*pot)(r, r0);
    return S_OK;
}

HRESULT tfPotential_evalPos(struct tfPotentialHandle *handle, tfFloatP_t *pos, tfFloatP_t *potE) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(potE);
    *potE = (*pot)(std::vector<tfFloatP_t>{pos[0], pos[1], pos[2]});
    return S_OK;
}

HRESULT tfPotential_evalPart(struct tfPotentialHandle *handle, struct tfParticleHandleHandle *partHandle, tfFloatP_t *pos, tfFloatP_t *potE) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(partHandle); TFC_PTRCHECK(partHandle->tfObj);
    TFC_PTRCHECK(potE);
    ParticleHandle *ph = (ParticleHandle*)partHandle->tfObj;
    *potE = (*pot)(ph, FVector3::from(pos));
    return S_OK;
}

HRESULT tfPotential_evalParts2(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    tfFloatP_t *potE) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(potE);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    *potE = (*pot)(pi, pj);
    return S_OK;
}

HRESULT tfPotential_evalParts3(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    struct tfParticleHandleHandle *phk, 
    tfFloatP_t *potE) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(phk); TFC_PTRCHECK(phk->tfObj);
    TFC_PTRCHECK(potE);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    ParticleHandle *pk = (ParticleHandle*)phk->tfObj;
    *potE = (*pot)(pi, pj, pk);
    return S_OK;
}

HRESULT tfPotential_evalParts4(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    struct tfParticleHandleHandle *phk, 
    struct tfParticleHandleHandle *phl, 
    tfFloatP_t *potE) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(phk); TFC_PTRCHECK(phk->tfObj);
    TFC_PTRCHECK(phl); TFC_PTRCHECK(phl->tfObj);
    TFC_PTRCHECK(potE);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    ParticleHandle *pk = (ParticleHandle*)phk->tfObj;
    ParticleHandle *pl = (ParticleHandle*)phl->tfObj;
    *potE = (*pot)(pi, pj, pk, pl);
    return S_OK;
}

HRESULT tfPotential_fevalR(struct tfPotentialHandle *handle, tfFloatP_t r, tfFloatP_t *force) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    *force = pot->force(r);
    return S_OK;
}

HRESULT tfPotential_fevalR0(struct tfPotentialHandle *handle, tfFloatP_t r, tfFloatP_t r0, tfFloatP_t *force) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    *force = pot->force(r, r0);
    return S_OK;
}

HRESULT tfPotential_fevalPos(struct tfPotentialHandle *handle, tfFloatP_t *pos, tfFloatP_t **force) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(force);
    std::vector<tfFloatP_t> _pos {pos[0], pos[1], pos[2]};
    FVector3 f = FVector3::from(pot->force(_pos).data());
    TFC_VECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT tfPotential_fevalPart(struct tfPotentialHandle *handle, struct tfParticleHandleHandle *partHandle, tfFloatP_t *pos, tfFloatP_t **force) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(partHandle); TFC_PTRCHECK(partHandle->tfObj);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(force);
    ParticleHandle *ph = (ParticleHandle*)partHandle->tfObj;
    FVector3 f = FVector3::from(pot->force(ph, FVector3::from(pos)).data());
    TFC_VECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT tfPotential_fevalParts2(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    tfFloatP_t **force) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(force);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    FVector3 f = FVector3::from(pot->force(pi, pj).data());
    TFC_VECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT tfPotential_fevalParts3(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    struct tfParticleHandleHandle *phk, 
    tfFloatP_t **forcei, 
    tfFloatP_t **forcek) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(phk); TFC_PTRCHECK(phk->tfObj);
    TFC_PTRCHECK(forcei);
    TFC_PTRCHECK(forcek);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    ParticleHandle *pk = (ParticleHandle*)phk->tfObj;
    std::vector<tfFloatP_t> fi, fk;
    std::tie(fi, fk) = pot->force(pi, pj, pk);
    FVector3 _fi = FVector3::from(fi.data());
    FVector3 _fk = FVector3::from(fk.data());
    TFC_VECTOR3_COPYFROM(_fi, (*forcei));
    TFC_VECTOR3_COPYFROM(_fk, (*forcek));
    return S_OK;
}

HRESULT tfPotential_fevalParts4(
    struct tfPotentialHandle *handle, 
    struct tfParticleHandleHandle *phi, 
    struct tfParticleHandleHandle *phj, 
    struct tfParticleHandleHandle *phk, 
    struct tfParticleHandleHandle *phl, 
    tfFloatP_t **forcei, 
    tfFloatP_t **forcel) 
{
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(phi); TFC_PTRCHECK(phi->tfObj);
    TFC_PTRCHECK(phj); TFC_PTRCHECK(phj->tfObj);
    TFC_PTRCHECK(phk); TFC_PTRCHECK(phk->tfObj);
    TFC_PTRCHECK(phl); TFC_PTRCHECK(phl->tfObj);
    TFC_PTRCHECK(forcei);
    TFC_PTRCHECK(forcel);
    ParticleHandle *pi = (ParticleHandle*)phi->tfObj;
    ParticleHandle *pj = (ParticleHandle*)phj->tfObj;
    ParticleHandle *pk = (ParticleHandle*)phk->tfObj;
    ParticleHandle *pl = (ParticleHandle*)phl->tfObj;
    std::vector<tfFloatP_t> fi, fl;
    std::tie(fi, fl) = pot->force(pi, pj, pk, pl);
    FVector3 _fi = FVector3::from(fi.data());
    FVector3 _fl = FVector3::from(fl.data());
    TFC_VECTOR3_COPYFROM(_fi, (*forcei));
    TFC_VECTOR3_COPYFROM(_fl, (*forcel));
    return S_OK;
}

HRESULT tfPotential_getConstituents(struct tfPotentialHandle *handle, struct tfPotentialHandle ***chandles, unsigned int *numPots) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(chandles);
    TFC_PTRCHECK(numPots);
    auto constituents = pot->constituents();
    *numPots = constituents.size();
    if(*numPots > 0) {
        tfPotentialHandle **_chandles = (tfPotentialHandle**)malloc(sizeof(tfPotentialHandle*) * *numPots);
        if(!_chandles) 
            return E_OUTOFMEMORY;
        tfPotentialHandle *ph;
        for(unsigned int i = 0; i < *numPots; i++) {
            ph = new tfPotentialHandle();
            ph->tfObj = (void*)constituents[i];
            _chandles[i] = ph;
        }
        *chandles = _chandles;
    }
    return S_OK;
}

HRESULT tfPotential_toString(struct tfPotentialHandle *handle, char **str, unsigned int *numChars) {
    TFC_POTENTIALHANDLE_GET(handle);
    return TissueForge::capi::str2Char(pot->toString(), str, numChars);;
}

HRESULT tfPotential_fromString(struct tfPotentialHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    Potential *pot = Potential::fromString(str);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_setClearFcn(struct tfPotentialHandle *handle, struct tfPotentialClearHandle *fcn) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    pot->clear_func = *(PotentialClear*)fcn->tfObj;
    return S_OK;
}

HRESULT tfPotential_removeClearFcn(struct tfPotentialHandle *handle) {
    TFC_POTENTIALHANDLE_GET(handle);
    if(pot->clear_func) 
        pot->clear_func = NULL;
    return S_OK;
}

HRESULT tfPotential_hasClearFcn(struct tfPotentialHandle *handle, bool *hasClear) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(hasClear);
    *hasClear = pot->clear_func != NULL;
    return S_OK;
}

HRESULT tfPotential_getMin(struct tfPotentialHandle *handle, tfFloatP_t *minR) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(minR);
    *minR = pot->getMin();
    return S_OK;
}

HRESULT tfPotential_getMax(struct tfPotentialHandle *handle, tfFloatP_t *maxR) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(maxR);
    *maxR = pot->getMax();
    return S_OK;
}

HRESULT tfPotential_getBound(struct tfPotentialHandle *handle, bool *bound) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(bound);
    *bound = pot->getBound();
    return S_OK;
}

HRESULT tfPotential_setBound(struct tfPotentialHandle *handle, bool bound) {
    TFC_POTENTIALHANDLE_GET(handle);
    pot->setBound(bound);
    return S_OK;
}

HRESULT tfPotential_getR0(struct tfPotentialHandle *handle, tfFloatP_t *r0) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(r0);
    *r0 = pot->getR0();
    return S_OK;
}

HRESULT tfPotential_setR0(struct tfPotentialHandle *handle, tfFloatP_t r0) {
    TFC_POTENTIALHANDLE_GET(handle);
    pot->setR0(r0);
    return S_OK;
}

HRESULT tfPotential_getRSquare(struct tfPotentialHandle *handle, tfFloatP_t *r2) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(r2);
    *r2 = pot->getRSquare();
    return S_OK;
}

HRESULT tfPotential_getShifted(struct tfPotentialHandle *handle, bool *shifted) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(shifted);
    *shifted = pot->getShifted();
    return S_OK;
}

HRESULT tfPotential_getPeriodic(struct tfPotentialHandle *handle, bool *periodic) {
    TFC_POTENTIALHANDLE_GET(handle);
    TFC_PTRCHECK(periodic);
    *periodic = pot->getPeriodic();
    return S_OK;
}

HRESULT tfPotential_create_lennard_jones_12_6(struct tfPotentialHandle *handle, tfFloatP_t min, tfFloatP_t max, tfFloatP_t A, tfFloatP_t B, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::lennard_jones_12_6(min, max, A, B, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_lennard_jones_12_6_coulomb(struct tfPotentialHandle *handle, tfFloatP_t min, tfFloatP_t max, tfFloatP_t A, tfFloatP_t B, tfFloatP_t q, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::lennard_jones_12_6_coulomb(min, max, A, B, q, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_ewald(struct tfPotentialHandle *handle, tfFloatP_t min, tfFloatP_t max, tfFloatP_t q, tfFloatP_t kappa, tfFloatP_t *tol, unsigned int *periodicOrder) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::ewald(min, max, q, kappa, tol, periodicOrder);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_coulomb(struct tfPotentialHandle *handle, tfFloatP_t q, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol, unsigned int *periodicOrder) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::coulomb(q, min, max, tol, periodicOrder);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_coulombR(struct tfPotentialHandle *handle, tfFloatP_t q, tfFloatP_t kappa, tfFloatP_t min, tfFloatP_t max, unsigned int* modes) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::coulombR(q, kappa, min, max, modes);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_harmonic(struct tfPotentialHandle *handle, tfFloatP_t k, tfFloatP_t r0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::harmonic(k, r0, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_linear(struct tfPotentialHandle *handle, tfFloatP_t k, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::linear(k, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_harmonic_angle(struct tfPotentialHandle *handle, tfFloatP_t k, tfFloatP_t theta0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::harmonic_angle(k, theta0, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_harmonic_dihedral(struct tfPotentialHandle *handle, tfFloatP_t k, tfFloatP_t delta, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::harmonic_dihedral(k, delta, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_cosine_dihedral(struct tfPotentialHandle *handle, tfFloatP_t k, int n, tfFloatP_t delta, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::cosine_dihedral(k, n, delta, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_well(struct tfPotentialHandle *handle, tfFloatP_t k, tfFloatP_t n, tfFloatP_t r0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::well(k, n, r0, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_glj(struct tfPotentialHandle *handle, tfFloatP_t e, tfFloatP_t *m, tfFloatP_t *n, tfFloatP_t *k, tfFloatP_t *r0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol, bool *shifted) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::glj(e, m, n, k, r0, min, max, tol, shifted);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_morse(struct tfPotentialHandle *handle, tfFloatP_t *d, tfFloatP_t *a, tfFloatP_t *r0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::morse(d, a, r0, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_overlapping_sphere(struct tfPotentialHandle *handle, tfFloatP_t *mu, tfFloatP_t *kc, tfFloatP_t *kh, tfFloatP_t *r0, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::overlapping_sphere(mu, kc, kh, r0, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_power(struct tfPotentialHandle *handle, tfFloatP_t *k, tfFloatP_t *r0, tfFloatP_t *alpha, tfFloatP_t *min, tfFloatP_t *max, tfFloatP_t *tol) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::power(k, r0, alpha, min, max, tol);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_dpd(struct tfPotentialHandle *handle, tfFloatP_t *alpha, tfFloatP_t *gamma, tfFloatP_t *sigma, tfFloatP_t *cutoff, bool *shifted) {
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::dpd(alpha, gamma, sigma, cutoff, shifted);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_custom(struct tfPotentialHandle *handle, tfFloatP_t min, tfFloatP_t max, tfFloatP_t (*f)(tfFloatP_t), tfFloatP_t (*fp)(tfFloatP_t), tfFloatP_t (*f6p)(tfFloatP_t),
                            tfFloatP_t *tol, unsigned int *flags) 
{
    TFC_PTRCHECK(handle);
    Potential *pot = Potential::custom(min, max, f, fp, f6p, tol, flags);
    TFC_PTRCHECK(pot);
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_eval_ByParticle(struct tfPotentialHandle *handle, struct tfPotentialEval_ByParticleHandle *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    Potential *pot = new Potential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_bypart = (PotentialEval_ByParticle)fcn->tfObj;
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_eval_ByParticles(struct tfPotentialHandle *handle, struct tfPotentialEval_ByParticlesHandle *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    Potential *pot = new Potential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts = (PotentialEval_ByParticles)fcn->tfObj;
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_eval_ByParticles3(struct tfPotentialHandle *handle, struct tfPotentialEval_ByParticles3Handle *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    Potential *pot = new Potential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts3 = (PotentialEval_ByParticles3)fcn->tfObj;
    handle->tfObj = (void*)pot;
    return S_OK;
}

HRESULT tfPotential_create_eval_ByParticles4(struct tfPotentialHandle *handle, struct tfPotentialEval_ByParticles4Handle *fcn) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fcn); TFC_PTRCHECK(fcn->tfObj);
    Potential *pot = new Potential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts4 = (PotentialEval_ByParticles4)fcn->tfObj;
    handle->tfObj = (void*)pot;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Add two potentials
 * 
 * @param handlei first potential
 * @param handlej second potential
 * @param handleSum resulting potential
 * @return HRESULT 
 */
HRESULT tfPotential_add(struct tfPotentialHandle *handlei, struct tfPotentialHandle *handlej, struct tfPotentialHandle *handleSum) {
    TFC_PTRCHECK(handlei); TFC_PTRCHECK(handlei->tfObj);
    TFC_PTRCHECK(handlej); TFC_PTRCHECK(handlej->tfObj);
    TFC_PTRCHECK(handleSum); TFC_PTRCHECK(handleSum->tfObj);
    Potential *poti = (Potential*)handlei->tfObj;
    Potential *potj = (Potential*)handlej->tfObj;
    Potential *potk = &(*poti + *potj);
    handleSum->tfObj = (void*)(potk);
    return S_OK;
}
