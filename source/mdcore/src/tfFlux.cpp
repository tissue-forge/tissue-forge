/*******************************************************************************
 * This file is part of mdcore.
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

#include <tf_errs.h>
#include <tfFlux.h>
#include <tfParticle.h>
#include <state/tfSpeciesList.h>
#include <state/tfStateVector.h>
#include "tf_flux_eval.h"
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tfError.h>
#include <tf_util.h>
#include <io/tfFIO.h>
#include <tf_mdcore_io.h>


using namespace TissueForge;


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


static std::string err_type_no_species(const std::string &name) {
    return std::string("particle type ") + name + " does not have any defined species";
}

static std::string err_type_no_species_name(const std::string &type_name, const std::string &species_name) {
    return std::string("particle type ") + type_name + " does not have species " + species_name;
}

static std::string err_simd_size() {
    std::string msg = "currently only ";
    msg += std::to_string(TF_SIMD_SIZE) + " flux species supported, please let the Tissue Forge development team know you want more. ";
    return msg;
}


Fluxes *TissueForge::Fluxes::create(
    FluxKind kind, 
    ParticleType *a, 
    ParticleType *b,
    const std::string& name, 
    FPTYPE k, 
    FPTYPE decay, 
    FPTYPE target,
    FPTYPE cutoff) 
{
    
    if(!a || !b) {
        tf_error(E_FAIL, "Invalid particle types");
        return NULL;
    }
    
    if(!a->species) {
        tf_error(E_FAIL, err_type_no_species(a->name).c_str());
        return NULL;
    }
    
    if(!b->species) {
        tf_error(E_FAIL, err_type_no_species(b->name).c_str());
        return NULL;
    }
    
    int index_a = a->species->index_of(name.c_str());
    int index_b = b->species->index_of(name.c_str());
    
    if(index_a < 0) {
        tf_error(E_FAIL, err_type_no_species_name(a->name, name).c_str());
        return NULL;
    }
    
    if(index_b < 0) {
        tf_error(E_FAIL, err_type_no_species_name(b->name, name).c_str());
        return NULL;
    }

    if(cutoff <= 0) 
        cutoff = _Engine.s.cutoff;
    
    Fluxes *fluxes = engine_getfluxes(&_Engine, a->id, b->id);
    
    if(fluxes == NULL) {
        fluxes = Fluxes::newFluxes(8);
    }
    
    fluxes = Fluxes::addFlux(kind, fluxes, a->id, b->id, index_a, index_b, k, decay, target, cutoff);
    
    if(engine_addfluxes(&_Engine, fluxes, a->id, b->id) != S_OK) {
        error(MDCERR_engine);
        return NULL;
    }
    
    return fluxes;
}

Fluxes *TissueForge::Fluxes::fluxFick(
    ParticleType *A, 
    ParticleType *B, 
    const std::string &name, 
    const FPTYPE &k, 
    const FPTYPE &decay, 
    const FPTYPE &cutoff) 
{
    return Fluxes::create(FLUX_FICK, A, B, name, k, decay, FPTYPE_ZERO, cutoff);
}

Fluxes *TissueForge::Fluxes::flux(
    ParticleType *A, 
    ParticleType *B, 
    const std::string &name, 
    const FPTYPE &k, 
    const FPTYPE &decay, 
    const FPTYPE &cutoff) 
{
    return fluxFick(A, B, name, k, decay, cutoff);
}

Fluxes *TissueForge::Fluxes::secrete(
    ParticleType *A, 
    ParticleType *B, 
    const std::string &name, 
    const FPTYPE &k, 
    const FPTYPE &target, 
    const FPTYPE &decay, 
    const FPTYPE &cutoff) 
{
    return Fluxes::create(FLUX_SECRETE, A, B, name, k, decay, target, cutoff);
}

Fluxes *TissueForge::Fluxes::uptake(
    ParticleType *A, 
    ParticleType *B, 
    const std::string &name, 
    const FPTYPE &k, 
    const FPTYPE &target, 
    const FPTYPE &decay, 
    const FPTYPE &cutoff) 
{
    return Fluxes::create(FLUX_UPTAKE, A, B, name, k, decay, target, cutoff);
}

std::string TissueForge::Fluxes::toString() {
    
    // todo: fix type deduction for io::toString<Fluxes>

    io::IOElement fe = io::IOElement::create();
    io::MetaData metaData;
    if(io::toFile(*this, metaData, fe) != S_OK) 
        return "";
    return io::toStr(fe, metaData);
}

Fluxes *TissueForge::Fluxes::fromString(const std::string &str) {
    return new Fluxes(io::fromString<Fluxes>(str));
}

static void integrate_statevector(state::StateVector *s, FPTYPE dt) {
    for(int i = 0; i < s->size; ++i) {
        s->species_flags[i] = (uint32_t)s->species->item(i)->flags();
        FPTYPE konst = (s->species_flags[i] & state::SpeciesFlags::SPECIES_KONSTANT) ? 0.f : 1.f;
        s->fvec[i] = FPTYPE_FMAX(FPTYPE_ZERO, s->fvec[i] + dt * s->q[i] * konst);
        s->q[i] = 0; // clear flux for next step
    }
}

static void integrate_statevector(state::StateVector *s) {
    return integrate_statevector(s, _Engine.dt_flux);
}

HRESULT TissueForge::Fluxes_integrate(space_cell *c, FPTYPE dt) {
    Particle *p;
    state::StateVector *s;
    
    for(int i = 0; i < c->count; ++i) {
        p = &c->parts[i];
        s = p->state_vector;
        
        if(s) {
            integrate_statevector(s, dt);
        }
    }
    
    return S_OK;
}

HRESULT TissueForge::Fluxes_integrate(int cellId) {
    return Fluxes_integrate(&_Engine.s.cells[cellId]);
}

Fluxes *TissueForge::Fluxes::addFlux(
    FluxKind kind, 
    Fluxes *fluxes,
    int16_t typeId_a, 
    int16_t typeId_b,
    int32_t index_a, 
    int32_t index_b,
    FPTYPE k, 
    FPTYPE decay, 
    FPTYPE target,
    FPTYPE cutoff) 
{
    TF_Log(LOG_TRACE);

    int i = 0;
    if(fluxes->size + 1 < fluxes->fluxes_size * TF_SIMD_SIZE) {
        i = fluxes->fluxes[0].size;
        fluxes->size += 1;
        fluxes->fluxes[0].size += 1;
    }
    else {
        tf_error(E_FAIL, err_simd_size().c_str());
        return NULL;
    }
    
    Flux *flux = &fluxes->fluxes[0];
    
    flux->kinds[i] = kind;
    flux->type_ids[i].a = typeId_a;
    flux->type_ids[i].b = typeId_b;
    flux->indices_a[i] = index_a;
    flux->indices_b[i] = index_b;
    flux->coef[i] = k;
    flux->decay_coef[i] = decay;
    flux->target[i] = target;
    flux->cutoff[i] = cutoff;
    
    return fluxes;
}

Fluxes* TissueForge::Fluxes::newFluxes(int32_t init_size) {
    TF_Log(LOG_TRACE);

    struct Fluxes *obj = NULL;
    
    int32_t blocks = std::ceil((FPTYPE)init_size / TF_SIMD_SIZE);
    
    int total_size = sizeof(Fluxes) + blocks * sizeof(Flux);

    /* allocate the potential */
    if ((obj = (Fluxes *)aligned_Malloc(total_size, 16)) == NULL) {
        error(MDCERR_null);
        return NULL;
    }
    
    ::memset(obj, 0, total_size);
    
    obj->size = 0;
    obj->fluxes_size = blocks;

    return obj;
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const TypeIdPair &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "a", dataElement.a);
        TF_IOTOEASY(fileElement, metaData, "b", dataElement.b);

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TypeIdPair *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_IOFROMEASY(fileElement, metaData, "a", &dataElement->a);
        TF_IOFROMEASY(fileElement, metaData, "b", &dataElement->b);

        return S_OK;
    }

    template <>
    HRESULT toFile(const Flux &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "size", dataElement.size);
        std::vector<int8_t> kinds;
        std::vector<TypeIdPair> type_ids;
        std::vector<int32_t> indices_a;
        std::vector<int32_t> indices_b;
        std::vector<FPTYPE> coef;
        std::vector<FPTYPE> decay_coef;
        std::vector<FPTYPE> target;
        std::vector<FPTYPE> cutoff;

        for(unsigned int i = 0; i < TF_SIMD_SIZE; i++) {
            kinds.push_back(dataElement.kinds[i]);
            type_ids.push_back(dataElement.type_ids[i]);
            indices_a.push_back(dataElement.indices_a[i]);
            indices_b.push_back(dataElement.indices_b[i]);
            coef.push_back(dataElement.coef[i]);
            decay_coef.push_back(dataElement.decay_coef[i]);
            target.push_back(dataElement.target[i]);
            cutoff.push_back(dataElement.cutoff[i]);
        }

        TF_IOTOEASY(fileElement, metaData, "kinds", kinds);
        TF_IOTOEASY(fileElement, metaData, "type_ids", type_ids);
        TF_IOTOEASY(fileElement, metaData, "indices_a", indices_a);
        TF_IOTOEASY(fileElement, metaData, "indices_b", indices_b);
        TF_IOTOEASY(fileElement, metaData, "coef", coef);
        TF_IOTOEASY(fileElement, metaData, "decay_coef", decay_coef);
        TF_IOTOEASY(fileElement, metaData, "target", target);
        TF_IOTOEASY(fileElement, metaData, "cutoff", cutoff);

        fileElement.get()->type = "Flux";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Flux *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_IOFROMEASY(fileElement, metaData, "size", &dataElement->size);
        
        std::vector<int8_t> kinds;
        TF_IOFROMEASY(fileElement, metaData, "kinds", &kinds);

        std::vector<TypeIdPair> type_ids;
        TF_IOFROMEASY(fileElement, metaData, "type_ids", &type_ids);
        
        std::vector<int32_t> indices_a;
        TF_IOFROMEASY(fileElement, metaData, "indices_a", &indices_a);
        
        std::vector<int32_t> indices_b;
        TF_IOFROMEASY(fileElement, metaData, "indices_b", &indices_b);
        
        std::vector<FPTYPE> coef;
        TF_IOFROMEASY(fileElement, metaData, "coef", &coef);
        
        std::vector<FPTYPE> decay_coef;
        TF_IOFROMEASY(fileElement, metaData, "decay_coef", &decay_coef);
        
        std::vector<FPTYPE> target;
        TF_IOFROMEASY(fileElement, metaData, "target", &target);

        for(unsigned int i = 0; i < TF_SIMD_SIZE; i++) {
            
            dataElement->kinds[i] = kinds[i];
            dataElement->type_ids[i] = type_ids[i];
            dataElement->indices_a[i] = indices_a[i];
            dataElement->indices_b[i] = indices_b[i];
            dataElement->coef[i] = coef[i];
            dataElement->decay_coef[i] = decay_coef[i];
            dataElement->target[i] = target[i];

        }

        IOChildMap fec = IOElement::children(fileElement);
        feItr = fec.find("cutoff");
        if(feItr != fec.end()) {
            std::vector<FPTYPE> cutoff;
            TF_IOFROMEASY(fileElement, metaData, "cutoff", &cutoff);
            for(unsigned int i = 0; i < std::min(TF_SIMD_SIZE, (int)cutoff.size()); i++) 
                dataElement->cutoff[i] = cutoff[i];
        }
        
        return S_OK;
    }

    template <>
    HRESULT toFile(const Fluxes &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "fluxes_size", dataElement.fluxes_size);
        
        std::vector<Flux> fluxes;
        for(unsigned int i = 0; i < dataElement.size; i++) 
            fluxes.push_back(dataElement.fluxes[i]);
        TF_IOTOEASY(fileElement, metaData, "fluxes", fluxes);

        fileElement.get()->type = "Fluxes";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Fluxes *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_IOFROMEASY(fileElement, metaData, "fluxes_size", &dataElement->fluxes_size);
        
        std::vector<Flux> fluxes;
        TF_IOFROMEASY(fileElement, metaData, "fluxes", &fluxes);
        Flux flux = fluxes[0];
        for(unsigned int i = 0; i < fluxes.size(); i++) { 
            dataElement = Fluxes::addFlux((FluxKind)flux.kinds[i], 
                                          dataElement, 
                                          flux.type_ids[i].a, flux.type_ids[i].b, 
                                          flux.indices_a[i], flux.indices_b[i], 
                                          flux.coef[i], flux.decay_coef[i], flux.target[i], flux.cutoff[i]);
        }
        
        return S_OK;
    }

};
