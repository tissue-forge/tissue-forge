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

#include "tfParticleEvent.h"
#include <tf_util.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


ParticleHandle *event::particleSelectorUniform(const int16_t &typeId, const int32_t &nr_parts) {
    if(nr_parts == 0) {
        return NULL;
    }
    
    std::uniform_int_distribution<int> distribution(0, nr_parts-1);
    RandomType &randEng = randomEngine();
    
    // index in the type's list of particles
    int tid = distribution(randEng);

    return _Engine.types[typeId].particle(tid)->handle();
}

ParticleHandle *event::particleSelectorLargest(const int16_t &typeId) {
    auto ptype = &_Engine.types[typeId];

    if(ptype->parts.nr_parts == 0) return NULL;

    Particle *pLargest = ptype->particle(0);
    for(int i = 1; i < ptype->parts.nr_parts; ++i) {
        Particle *p = ptype->particle(i);
        if(p->nr_parts > pLargest->nr_parts) pLargest = p;
    }

    return pLargest->handle();
}

ParticleHandle* event::particleEventParticleSelectorUniform(const event::ParticleEvent &event) {
    return particleSelectorUniform(event.targetType->id, event.targetType->parts.nr_parts);
}

ParticleHandle* event::particleEventParticleSelectorLargest(const event::ParticleEvent &event) {
    return particleSelectorLargest(event.targetType->id);
}

event::ParticleEventParticleSelector *event::getParticleEventParticleSelector(event::ParticleEventParticleSelectorEnum selectorEnum) {
    auto x = event::particleEventParticleSelectorMap.find(selectorEnum);
    if (x == event::particleEventParticleSelectorMap.end()) return NULL;
    return &x->second;
}

event::ParticleEventParticleSelector *event::getParticleEventParticleSelectorN(std::string setterName) {
    auto x = event::particleEventParticleSelectorNameMap.find(setterName);
    if (x == event::particleEventParticleSelectorNameMap.end()) return NULL;
    return event::getParticleEventParticleSelector(x->second);
}

event::ParticleEvent::ParticleEvent(
    ParticleType *targetType, 
    event::ParticleEventMethod *invokeMethod, 
    event::ParticleEventMethod *predicateMethod, 
    event::ParticleEventParticleSelector *particleSelector
) : 
    event::EventBase(), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod), 
    targetType(targetType), 
    particleSelector(particleSelector)
{
    if (particleSelector==NULL) setParticleEventParticleSelector(event::ParticleEventParticleSelectorEnum::DEFAULT);
}

HRESULT event::ParticleEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}

HRESULT event::ParticleEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT event::ParticleEvent::eval(const FloatP_t &time) {
    targetParticle = getNextParticle();
    return event::EventBase::eval(time);
}

HRESULT event::ParticleEvent::setParticleEventParticleSelector(event::ParticleEventParticleSelector *particleSelector) {
    this->particleSelector = particleSelector;
    return S_OK;
}

HRESULT event::ParticleEvent::setParticleEventParticleSelector(event::ParticleEventParticleSelectorEnum selectorEnum) {
    auto x = event::particleEventParticleSelectorMap.find(selectorEnum);
    if (x == event::particleEventParticleSelectorMap.end()) return 1;
    return setParticleEventParticleSelector(&x->second);
}

HRESULT event::ParticleEvent::setParticleEventParticleSelector(std::string selectorName) {
    auto *particleSelector = getParticleEventParticleSelectorN(selectorName);
    if(!particleSelector) return E_FAIL;
    return setParticleEventParticleSelector(particleSelector);
}

ParticleHandle *event::ParticleEvent::getNextParticle() {
    return (*particleSelector)(*this);
}

event::ParticleEvent *event::onParticleEvent(
    ParticleType *targetType, 
    event::ParticleEventMethod *invokeMethod, 
    event::ParticleEventMethod *predicateMethod)
{
    event::ParticleEvent *event = new event::ParticleEvent(targetType, invokeMethod, predicateMethod);

    Universe::get()->events->addEvent(event);

    return event;
}
