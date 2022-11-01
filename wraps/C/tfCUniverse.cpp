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

#include "tfCUniverse.h"

#include "TissueForge_c_private.h"

#include <tfUniverse.h>


using namespace TissueForge;


namespace TissueForge {


    UniverseConfig *castC(struct tfUniverseConfigHandle *handle) {
        return castC<UniverseConfig, tfUniverseConfigHandle>(handle);
    }

}

#define TFC_UNIVERSECONFIG_GET(handle) \
    UniverseConfig *conf = TissueForge::castC<UniverseConfig, tfUniverseConfigHandle>(handle); \
    TFC_PTRCHECK(conf);

#define TFC_UNIVERSE_STATIC_GET() \
    Universe *univ = Universe::get(); \
    TFC_PTRCHECK(univ);


//////////////
// Universe //
//////////////


HRESULT tfUniverse_getDim(tfFloatP_t **dim) {
    TFC_UNIVERSE_STATIC_GET()
    auto d = univ->dim();
    TFC_VECTOR3_COPYFROM(d, (*dim));
    return S_OK;
}

HRESULT tfUniverse_getIsRunning(bool *isRunning) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(isRunning);
    *isRunning = univ->isRunning;
    return S_OK;
}

HRESULT tfUniverse_getName(char **name, unsigned int *numChars) {
    TFC_UNIVERSE_STATIC_GET()
    return TissueForge::capi::str2Char(univ->name, name, numChars);
}

HRESULT tfUniverse_getVirial(tfFloatP_t **virial) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(virial);
    auto _virial = *univ->virial();
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfUniverse_getVirialT(struct tfParticleTypeHandle **phandles, unsigned int numTypes, tfFloatP_t **virial) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(phandles);
    TFC_PTRCHECK(virial);
    std::vector<ParticleType*> ptypes;
    tfParticleTypeHandle *ph;
    for(unsigned int i = 0; i < numTypes; i++) { 
        ph = phandles[i];
        TFC_PTRCHECK(ph); TFC_PTRCHECK(ph->tfObj);
        ptypes.push_back((ParticleType*)ph->tfObj);
    }
    auto _virial = *univ->virial(NULL, NULL, &ptypes);
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfUniverse_getVirialO(tfFloatP_t *origin, tfFloatP_t radius, tfFloatP_t **virial) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(origin);
    TFC_PTRCHECK(virial);
    FVector3 _origin = FVector3::from(origin);
    tfFloatP_t _radius = radius;
    auto _virial = *univ->virial(&_origin, &_radius);
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfUniverse_getVirialOT(struct tfParticleTypeHandle **phandles, unsigned int numTypes, tfFloatP_t *origin, tfFloatP_t radius, tfFloatP_t **virial) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(phandles);
    TFC_PTRCHECK(origin);
    TFC_PTRCHECK(virial);
    std::vector<ParticleType*> ptypes;
    tfParticleTypeHandle *ph;
    for(unsigned int i = 0; i < numTypes; i++) {
        ph = phandles[i];
        TFC_PTRCHECK(ph); TFC_PTRCHECK(ph->tfObj);
        ptypes.push_back((ParticleType*)ph->tfObj);
    }
    FVector3 _origin = FVector3::from(origin);
    tfFloatP_t _radius = radius;
    auto _virial = *univ->virial(&_origin, &_radius, &ptypes);
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfUniverse_getNumParts(unsigned int *numParts) {
    TFC_PTRCHECK(numParts);
    *numParts = _Engine.s.nr_parts;
    return S_OK;
}

HRESULT tfUniverse_getParticle(unsigned int pidx, struct tfParticleHandleHandle *handle) {
    TFC_PTRCHECK(handle);
    if(pidx >= _Engine.s.nr_parts) 
        return E_FAIL;
    Particle *p = _Engine.s.partlist[pidx];
    TFC_PTRCHECK(p);
    ParticleHandle *ph = new ParticleHandle(p->id, p->typeId);
    handle->tfObj = (void*)ph;
    return S_OK;
}

HRESULT tfUniverse_getCenter(tfFloatP_t **center) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(center);
    auto c = univ->getCenter();
    TFC_VECTOR3_COPYFROM(c, (*center));
    return S_OK;
}

HRESULT tfUniverse_step(tfFloatP_t until, tfFloatP_t dt) {
    TFC_UNIVERSE_STATIC_GET()
    return univ->step(until, dt);
}

HRESULT tfUniverse_stepAsyncStart(tfFloatP_t until, tfFloatP_t dt) {
    TFC_UNIVERSE_STATIC_GET()
    return univ->stepAsyncStart(until, dt);
}

HRESULT tfUniverse_stepAsyncWorking(bool *isWorking) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(isWorking);
    *isWorking = univ->stepAsyncWorking();
    return S_OK;
}

HRESULT tfUniverse_stepAsyncJoin() {
    TFC_UNIVERSE_STATIC_GET()
    return univ->stepAsyncJoin();
}

HRESULT tfUniverse_stop() {
    TFC_UNIVERSE_STATIC_GET()
    return univ->stop();
}

HRESULT tfUniverse_start() {
    TFC_UNIVERSE_STATIC_GET()
    return univ->start();
}

HRESULT tfUniverse_reset() {
    TFC_UNIVERSE_STATIC_GET()
    return univ->reset();
}

HRESULT tfUniverse_resetSpecies() {
    TFC_UNIVERSE_STATIC_GET()
    univ->resetSpecies();
    return S_OK;
}

HRESULT tfUniverse_getTemperature(tfFloatP_t *temperature) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(temperature);
    *temperature = univ->getTemperature();
    return S_OK;
}

HRESULT tfUniverse_getTime(tfFloatP_t *time) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(time);
    *time = univ->getTime();
    return S_OK;
}

HRESULT tfUniverse_getDt(tfFloatP_t *dt) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(dt);
    *dt = univ->getDt();
    return S_OK;
}

HRESULT tfUniverse_getBoundaryConditions(struct tfBoundaryConditionsHandle *bcs) {
    TFC_UNIVERSE_STATIC_GET();
    TFC_PTRCHECK(bcs);
    auto _bcs = univ->getBoundaryConditions();
    TFC_PTRCHECK(_bcs);
    bcs->tfObj = (void*)_bcs;
    return S_OK;
}

HRESULT tfUniverse_getKineticEnergy(tfFloatP_t *ke) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(ke);
    *ke = univ->getKineticEnergy();
    return S_OK;
}

HRESULT tfUniverse_getNumTypes(int *numTypes) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(numTypes);
    *numTypes = univ->getNumTypes();
    return S_OK;
}

HRESULT tfUniverse_getCutoff(tfFloatP_t *cutoff) {
    TFC_UNIVERSE_STATIC_GET()
    TFC_PTRCHECK(cutoff);
    *cutoff = univ->getCutoff();
    return S_OK;
}


////////////////////////////
// tfUniverseConfigHandle //
////////////////////////////


HRESULT tfUniverseConfig_init(struct tfUniverseConfigHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->tfObj = new UniverseConfig();
    return S_OK;
}

HRESULT tfUniverseConfig_destroy(struct tfUniverseConfigHandle *handle) {
    return TissueForge::capi::destroyHandle<UniverseConfig, tfUniverseConfigHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfUniverseConfig_getDim(struct tfUniverseConfigHandle *handle, tfFloatP_t **dim) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(dim);
    TFC_VECTOR3_COPYFROM(conf->dim, (*dim));
    return S_OK;
}

HRESULT tfUniverseConfig_setDim(struct tfUniverseConfigHandle *handle, tfFloatP_t *dim) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(dim);
    TFC_VECTOR3_COPYTO(dim, conf->dim)
    return S_OK;
}

HRESULT tfUniverseConfig_getCells(struct tfUniverseConfigHandle *handle, int **cells) {
    TFC_UNIVERSECONFIG_GET(handle);
    TFC_PTRCHECK(cells);
    TFC_VECTOR3_COPYFROM(conf->spaceGridSize, (*cells));
    return S_OK;
}

HRESULT tfUniverseConfig_setCells(struct tfUniverseConfigHandle *handle, int *cells) {
    TFC_UNIVERSECONFIG_GET(handle);
    TFC_PTRCHECK(cells);
    TFC_VECTOR3_COPYTO(cells, conf->spaceGridSize);
    return S_OK;
}

HRESULT tfUniverseConfig_getCutoff(struct tfUniverseConfigHandle *handle, tfFloatP_t *cutoff) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(cutoff);
    *cutoff = conf->cutoff;
    return S_OK;
}

HRESULT tfUniverseConfig_setCutoff(struct tfUniverseConfigHandle *handle, tfFloatP_t cutoff) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->cutoff = cutoff;
    return S_OK;
}

HRESULT tfUniverseConfig_getFlags(struct tfUniverseConfigHandle *handle, unsigned int *flags) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(flags);
    *flags = conf->flags;
    return S_OK;
}

HRESULT tfUniverseConfig_setFlags(struct tfUniverseConfigHandle *handle, unsigned int flags) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->flags = flags;
    return S_OK;
}

HRESULT tfUniverseConfig_getDt(struct tfUniverseConfigHandle *handle, tfFloatP_t *dt) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(dt);
    *dt = conf->dt;
    return S_OK;
}

HRESULT tfUniverseConfig_setDt(struct tfUniverseConfigHandle *handle, tfFloatP_t dt) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->dt = dt;
    return S_OK;
}

HRESULT tfUniverseConfig_getTemperature(struct tfUniverseConfigHandle *handle, tfFloatP_t *temperature) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(temperature);
    *temperature = conf->temp;
    return S_OK;
}

HRESULT tfUniverseConfig_setTemperature(struct tfUniverseConfigHandle *handle, tfFloatP_t temperature) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->temp = temperature;
    return S_OK;
}

HRESULT tfUniverseConfig_getNumThreads(struct tfUniverseConfigHandle *handle, unsigned int *numThreads) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(numThreads);
    *numThreads = conf->threads;
    return S_OK;
}

HRESULT tfUniverseConfig_setNumThreads(struct tfUniverseConfigHandle *handle, unsigned int numThreads) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->threads = numThreads;
    return S_OK;
}

HRESULT tfUniverseConfig_getIntegrator(struct tfUniverseConfigHandle *handle, unsigned int *integrator) {
    TFC_UNIVERSECONFIG_GET(handle)
    TFC_PTRCHECK(integrator);
    *integrator = (unsigned int)conf->integrator;
    return S_OK;
}

HRESULT tfUniverseConfig_setIntegrator(struct tfUniverseConfigHandle *handle, unsigned int integrator) {
    TFC_UNIVERSECONFIG_GET(handle)
    conf->integrator = (EngineIntegrator)integrator;
    return S_OK;
}

HRESULT tfUniverseConfig_getBoundaryConditions(struct tfUniverseConfigHandle *handle, struct tfBoundaryConditionsArgsContainerHandle *bargsHandle) {
    TFC_UNIVERSECONFIG_GET(handle);
    TFC_PTRCHECK(bargsHandle);
    TFC_PTRCHECK(conf->boundaryConditionsPtr);
    bargsHandle->tfObj = (void*)conf->boundaryConditionsPtr;
    return S_OK;
}

HRESULT tfUniverseConfig_setBoundaryConditions(struct tfUniverseConfigHandle *handle, struct tfBoundaryConditionsArgsContainerHandle *bargsHandle) {
    TFC_UNIVERSECONFIG_GET(handle);
    TFC_PTRCHECK(bargsHandle); TFC_PTRCHECK(bargsHandle->tfObj);
    conf->setBoundaryConditions((BoundaryConditionsArgsContainer*)bargsHandle->tfObj);
    return S_OK;
}
