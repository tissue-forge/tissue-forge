/*******************************************************************************
 * This file is part of mdcore.
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

#include "tfForce.h"
#include <tfEngine.h>
#include <tfParticle.h>
#include <tfError.h>
#include <tfLogger.h>
#include <io/tfFIO.h>
#include <iostream>
#include <random>
#include <state/tfStateVector.h>
#include <state/tfSpeciesList.h>
#include <tf_util.h>
#include <tf_mdcore_io.h>


using namespace TissueForge;


static Berendsen *berendsen_create(FPTYPE tau);
static Gaussian *random_create(FPTYPE mean, FPTYPE std, FPTYPE durration);
static Friction *friction_create(FPTYPE coef);

static FPTYPE scaling_constant(Particle *part, int stateVectorIndex) {
    if(part->state_vector && stateVectorIndex >= 0) {
        return part->state_vector->fvec[stateVectorIndex];
    }
    else {
        return 1.f;
    }
}

FVector3 CustomForce::getValue() {
    if(userFunc) return (*userFunc)(this);
    return force;
}

void CustomForce::setValue(const FVector3 &f) {
    force = f;
}

void CustomForce::setValue(UserForceFuncType *_userFunc) {
    if(_userFunc) userFunc = _userFunc;
    setValue((*userFunc)(this));
}

FPTYPE CustomForce::getPeriod() {
    return updateInterval;
}

void CustomForce::setPeriod(const FPTYPE &period) {
    updateInterval = period;
}

HRESULT Force::bind_species(ParticleType *a_type, const std::string &coupling_symbol) {
    std::string msg = a_type->name;
    TF_Log(LOG_DEBUG) << msg + coupling_symbol;

    if(a_type->species) {
        int index = a_type->species->index_of(coupling_symbol.c_str());
        if(index < 0) {
            std::string msg = "could not bind force, the particle type ";
            msg += a_type->name;
            msg += " has a chemical species state vector, but it does not have the symbol ";
            msg += coupling_symbol;
            TF_Log(LOG_CRITICAL) << msg;
            return tf_error(E_FAIL, msg.c_str());
        }

        this->stateVectorIndex = index;
    }
    else {
        std::string msg = "could not add force, given a coupling symbol, but the particle type ";
        msg += a_type->name;
        msg += " does not have a chemical species vector";
        TF_Log(LOG_CRITICAL) << msg;
        return tf_error(E_FAIL, msg.c_str());
    }

    return S_OK;
}

Berendsen* Force::berendsen_tstat(const FPTYPE &tau) {
    TF_Log(LOG_DEBUG);

    try {
        return berendsen_create(tau);
    }
    catch (const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

Gaussian* Force::random(const FPTYPE &std, const FPTYPE &mean, const FPTYPE &duration) {
    TF_Log(LOG_DEBUG);

    try {
        return random_create(mean, std, duration);
    }
    catch (const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

Friction* Force::friction(const FPTYPE &coef) {
    TF_Log(LOG_DEBUG);

    try {
        return friction_create(coef);
    }
    catch (const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

void eval_sum_force(struct Force *force, struct Particle *p, FPTYPE *f) {
    ForceSum *sf = (ForceSum*)force;
    
    std::vector<FPTYPE> f1(3, 0.0), f2(3, 0.0);
    (*sf->f1->func)(sf->f1, p, f1.data());
    (*sf->f2->func)(sf->f2, p, f2.data());

    FPTYPE scaling = scaling_constant(p, force->stateVectorIndex);
    for(unsigned int i = 0; i < 3; i++) 
        p->f[i] += scaling * (f1[i] + f2[i]);
}

Force *TissueForge::Force_add(Force *f1, Force *f2) {
    ForceSum *sf = new ForceSum();
    sf->func = (Force_EvalFcn)eval_sum_force;

    sf->f1 = f1;
    sf->f2 = f2;
    
    return (Force*)sf;
}

Force& Force::operator+(const Force& rhs) {
    return *Force_add(this, const_cast<Force*>(&rhs));
}

std::string Force::toString() {
    io::IOElement fe = io::IOElement::create();
    io::MetaData metaData;

    if(io::toFile(this, metaData, fe) != S_OK) 
        return "";

    return io::toStr(fe, metaData);
}

Force *Force::fromString(const std::string &str) {
    return io::fromString<Force*>(str);
}

/**
 * Implements a force:
 *
 * f_b = p / tau * ((T_0 / T) - 1)
 */
static void berendsen_force(Berendsen *t, Particle *p, FPTYPE *f) {
    ParticleType *type = &engine::types[p->typeId];

    if(type->kinetic_energy <= 0 || type->target_energy <= 0) return;

    FPTYPE scale = t->itau * ((type->target_energy / type->kinetic_energy) - 1.0) * scaling_constant(p, t->stateVectorIndex);
    f[0] += scale * p->v[0];
    f[1] += scale * p->v[1];
    f[2] += scale * p->v[2];
}

static void custom_force(CustomForce *cf, Particle *p, FPTYPE *f) {
    FPTYPE scale = scaling_constant(p, cf->stateVectorIndex);
    f[0] += cf->force[0] * scale;
    f[1] += cf->force[1] * scale;
    f[2] += cf->force[2] * scale;
}


/**
 * Implements a friction force:
 *
 * f_f = - || v || / tau * v
 */
static void friction_force(Friction *t, Particle *p, FPTYPE *f) {
    
    FPTYPE scale = -1. * t->coef * p->velocity.length() * scaling_constant(p, t->stateVectorIndex);
    
    f[0] += scale * p->v[0];
    f[1] += scale * p->v[1];
    f[2] += scale * p->v[2];
}

static void gaussian_force(Gaussian *t, Particle *p, FPTYPE *f) {
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        p->persistent_force = randomVector(t->mean, t->std);
    }

    FPTYPE scale = scaling_constant(p, t->stateVectorIndex);
    
    f[0] += scale * p->persistent_force[0];
    f[1] += scale * p->persistent_force[1];
    f[2] += scale * p->persistent_force[2];
}

Berendsen *berendsen_create(FPTYPE tau) {
    auto *obj = new Berendsen();

    obj->type = FORCE_BERENDSEN;
    obj->func = (Force_EvalFcn)berendsen_force;
    obj->itau = 1/tau;

    return obj;
}

Gaussian *random_create(FPTYPE mean, FPTYPE std, FPTYPE durration) {
    auto *obj = new Gaussian();
    
    obj->type = FORCE_GAUSSIAN;
    obj->func = (Force_EvalFcn)gaussian_force;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return obj;
}

Friction *friction_create(FPTYPE coef) {
    auto *obj = new Friction();
    
    obj->type = FORCE_FRICTION;
    obj->func = (Force_EvalFcn)friction_force;
    obj->coef = coef;
    
    return obj;
}

void CustomForce::onTime(FPTYPE time)
{
    if(userFunc && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue((*userFunc)(this));
    }
}

CustomForce::CustomForce() { 
    type = FORCE_CUSTOM;
    func = (Force_EvalFcn)custom_force;
}

CustomForce::CustomForce(const FVector3 &f, const FPTYPE &period) : CustomForce() {
    type = FORCE_CUSTOM;
    updateInterval = period;
    setValue(f);
}

CustomForce::CustomForce(UserForceFuncType *f, const FPTYPE &period) : CustomForce() {
    type = FORCE_CUSTOM;
    updateInterval = period;
    setValue(f);
}


ForceSum *ForceSum::fromForce(Force *f) {
    if(f->type != FORCE_SUM) 
        return 0;
    return (ForceSum*)f;
}

CustomForce *CustomForce::fromForce(Force *f) {
    if(f->type != FORCE_CUSTOM) 
        return 0;
    return (CustomForce*)f;
}

Berendsen *Berendsen::fromForce(Force *f) {
    if(f->type != FORCE_BERENDSEN) 
        return 0;
    return (Berendsen*)f;
}

Gaussian *Gaussian::fromForce(Force *f) {
    if(f->type != FORCE_GAUSSIAN) 
        return 0;
    return (Gaussian*)f;
}

Friction *Friction::fromForce(Force *f) {
    if(f->type != FORCE_FRICTION) 
        return 0;
    return (Friction*)f;
}


namespace TissueForge { namespace io {


    template <>
    HRESULT toFile(const FORCE_TYPE &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        fileElement.get()->type = "FORCE_TYPE";
        fileElement.get()->value = std::to_string((unsigned int)dataElement);

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, FORCE_TYPE *dataElement) {

        unsigned int ui;
        if(fromFile(fileElement, metaData, &ui) != S_OK) 
            return E_FAIL;

        *dataElement = (FORCE_TYPE)ui;

        return S_OK;
    }

    template <>
    HRESULT toFile(const CustomForce &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);
        TF_IOTOEASY(fileElement, metaData, "updateInterval", dataElement.updateInterval);
        TF_IOTOEASY(fileElement, metaData, "lastUpdate", dataElement.lastUpdate);
        TF_IOTOEASY(fileElement, metaData, "force", dataElement.force);

        fileElement.get()->type = "CustomForce";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CustomForce *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
        TF_IOFROMEASY(fileElement, metaData, "updateInterval", &dataElement->updateInterval);
        TF_IOFROMEASY(fileElement, metaData, "lastUpdate", &dataElement->lastUpdate);
        TF_IOFROMEASY(fileElement, metaData, "force", &dataElement->force);
        dataElement->userFunc = NULL;

        return S_OK;
    }

    template <>
    HRESULT toFile(const ForceSum &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);

        if(dataElement.f1 != NULL) 
            TF_IOTOEASY(fileElement, metaData, "Force1", dataElement.f1);
        if(dataElement.f2 != NULL) 
            TF_IOTOEASY(fileElement, metaData, "Force2", dataElement.f2);

        fileElement.get()->type = "SumForce";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ForceSum *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);

        IOChildMap fec = IOElement::children(fileElement);
        
        IOChildMap::const_iterator feItr = fec.find("Force1");
        if(feItr != fec.end()) {
            dataElement->f1 = NULL;
            if(fromFile(feItr->second, metaData, &dataElement->f1) != S_OK) 
                return E_FAIL;
        }

        feItr = fec.find("Force2");
        if(feItr != fec.end()) {
            dataElement->f2 = NULL;
            if(fromFile(feItr->second, metaData, &dataElement->f2) != S_OK) 
                return E_FAIL;
        }

        return S_OK;
    }

    template <>
    HRESULT toFile(const Berendsen &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);
        TF_IOTOEASY(fileElement, metaData, "itau", dataElement.itau);

        fileElement.get()->type = "BerendsenForce";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Berendsen *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
        TF_IOFROMEASY(fileElement, metaData, "itau", &dataElement->itau);
        dataElement->func = (Force_EvalFcn)berendsen_force;

        return S_OK;
    }

    template <>
    HRESULT toFile(const Gaussian &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);
        TF_IOTOEASY(fileElement, metaData, "std", dataElement.std);
        TF_IOTOEASY(fileElement, metaData, "mean", dataElement.mean);
        TF_IOTOEASY(fileElement, metaData, "durration_steps", dataElement.durration_steps);
        
        fileElement.get()->type = "GaussianForce";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Gaussian *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
        TF_IOFROMEASY(fileElement, metaData, "std", &dataElement->std);
        TF_IOFROMEASY(fileElement, metaData, "mean", &dataElement->mean);
        TF_IOFROMEASY(fileElement, metaData, "durration_steps", &dataElement->durration_steps);
        dataElement->func = (Force_EvalFcn)gaussian_force;

        return S_OK;
    }

    template <>
    HRESULT toFile(const Friction &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);
        TF_IOTOEASY(fileElement, metaData, "coef", dataElement.coef);

        fileElement.get()->type = "FrictionForce";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Friction *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
        TF_IOFROMEASY(fileElement, metaData, "coef", &dataElement->coef);
        dataElement->func = (Force_EvalFcn)friction_force;

        return S_OK;
    }

    template <>
    HRESULT toFile(Force *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        if(dataElement->type & FORCE_BERENDSEN) 
            return toFile(*(Berendsen*)dataElement, metaData, fileElement);
        else if(dataElement->type & FORCE_CUSTOM) 
            return toFile(*(CustomForce*)dataElement, metaData, fileElement);
        else if(dataElement->type & FORCE_FRICTION) 
            return toFile(*(Friction*)dataElement, metaData, fileElement);
        else if(dataElement->type & FORCE_GAUSSIAN) 
            return toFile(*(Gaussian*)dataElement, metaData, fileElement);
        else if(dataElement->type & FORCE_SUM) 
            return toFile(*(ForceSum*)dataElement, metaData, fileElement);
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement->type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement->stateVectorIndex);
        
        fileElement.get()->type = "Force";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Force **dataElement) {

        FORCE_TYPE fType;
        TF_IOFROMEASY(fileElement, metaData, "type", &fType);

        if(fType & FORCE_BERENDSEN) {
            Berendsen *f = new Berendsen();
            if(fromFile(fileElement, metaData, f) != S_OK) 
                return E_FAIL;
            *dataElement = f;
            return S_OK;
        }
        else if(fType & FORCE_CUSTOM) {
            CustomForce *f = new CustomForce();
            if(fromFile(fileElement, metaData, f) != S_OK) 
                return E_FAIL;
            *dataElement = f;
            return S_OK;
        }
        else if(fType & FORCE_FRICTION) {
            Friction *f = new Friction();
            if(fromFile(fileElement, metaData, f) != S_OK) 
                return E_FAIL;
            *dataElement = f;
            return S_OK;
        }
        else if(fType & FORCE_GAUSSIAN) {
            Gaussian *f = new Gaussian();
            if(fromFile(fileElement, metaData, f) != S_OK) 
                return E_FAIL;
            *dataElement = f;
            return S_OK;
        }
        else if(fType & FORCE_SUM) {
            ForceSum *f = new ForceSum();
            if(fromFile(fileElement, metaData, f) != S_OK) 
                return E_FAIL;
            *dataElement = f;
            return S_OK;
        }
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::vector<Force*> *dataElement) {

        IOChildMap fec = IOElement::children(fileElement);
        unsigned int numEls = fec.size();
        dataElement->reserve(numEls);
        for(unsigned int i = 0; i < numEls; i++) {
            Force *de = NULL;
            auto itr = fec.find(std::to_string(i));
            if(itr == fec.end()) 
                return E_FAIL;
            if(fromFile(itr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            dataElement->push_back(de);
        }
        return S_OK;
    }

}};

ForceSum *TissueForge::ForceSum_fromStr(const std::string &str) {
    return (ForceSum*)Force::fromString(str);
}

Berendsen *TissueForge::Berendsen_fromStr(const std::string &str)  {
    return (Berendsen*)Force::fromString(str);
}

Gaussian *TissueForge::Gaussian_fromStr(const std::string &str) {
    return (Gaussian*)Force::fromString(str);
}

Friction *TissueForge::Friction_fromStr(const std::string &str) {
    return (Friction*)Force::fromString(str);
}
