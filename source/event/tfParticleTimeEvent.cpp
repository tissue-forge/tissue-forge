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

#include "tfParticleTimeEvent.h"

#include <tf_util.h>
#include <tfUniverse.h>
#include <tfLogger.h>


using namespace TissueForge;


FloatP_t event::particleTimeEventSetNextTimeExponential(event::ParticleTimeEvent &event, const FloatP_t &time) {
    return timeEventSetNextTimeExponential(event, time);
}

FloatP_t event::particleTimeEventSetNextTimeDeterministic(event::ParticleTimeEvent &event, const FloatP_t &time) {
    return timeEventSetNextTimeDeterministic(event, time);
}

ParticleHandle* event::particleTimeEventParticleSelectorUniform(const event::ParticleTimeEvent &event) {
    return particleSelectorUniform(event.targetType->id, event.targetType->parts.nr_parts);
}

ParticleHandle* event::particleTimeEventParticleSelectorLargest(const event::ParticleTimeEvent &event) {
    return particleSelectorLargest(event.targetType->id);
}

event::ParticleTimeEventParticleSelector *event::getParticleTimeEventParticleSelector(event::ParticleTimeEventParticleSelectorEnum selectorEnum) {
    auto x = event::particleTimeEventParticleSelectorMap.find(selectorEnum);
    if (x == event::particleTimeEventParticleSelectorMap.end()) return NULL;
    return &x->second;
}

event::ParticleTimeEventParticleSelector *event::getParticleTimeEventParticleSelectorN(std::string setterName) {
    auto x = event::particleTimeEventParticleSelectorNameMap.find(setterName);
    if (x == event::particleTimeEventParticleSelectorNameMap.end()) return NULL;
    return event::getParticleTimeEventParticleSelector(x->second);
}

event::ParticleTimeEventNextTimeSetter* event::getParticleTimeEventNextTimeSetter(event::ParticleTimeEventTimeSetterEnum setterEnum) {
    auto x = event::particleTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == event::particleTimeEventNextTimeSetterMap.end()) return NULL;
    return &x->second;
}

event::ParticleTimeEventNextTimeSetter *event::getParticleTimeEventNextTimeSetterN(std::string setterName) {
    auto x = event::particleTimeEventNextTimeSetterNameMap.find(setterName);
    if (x == event::particleTimeEventNextTimeSetterNameMap.end()) return NULL;
    return event::getParticleTimeEventNextTimeSetter(x->second);
}

event::ParticleTimeEvent::ParticleTimeEvent(ParticleType *targetType, 
                                         const FloatP_t &period, 
                                         event::ParticleTimeEventMethod *invokeMethod, 
                                         event::ParticleTimeEventMethod *predicateMethod, 
                                         event::ParticleTimeEventNextTimeSetter *nextTimeSetter, 
                                         const FloatP_t &start_time, 
                                         const FloatP_t &end_time,
                                         event::ParticleTimeEventParticleSelector *particleSelector) : 
    event::EventBase(), 
    targetType(targetType), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod), 
    period(period), 
    nextTimeSetter(nextTimeSetter),
    next_time(0),
    start_time(start_time),
    end_time(end_time > 0 ? end_time : std::numeric_limits<FloatP_t>::max()), 
    particleSelector(particleSelector)
{
    if (nextTimeSetter == NULL) setParticleTimeEventNextTimeSetter(event::ParticleTimeEventTimeSetterEnum::DEFAULT);
    if (particleSelector == NULL) setParticleTimeEventParticleSelector(event::ParticleTimeEventParticleSelectorEnum::DEFAULT);
}

HRESULT event::ParticleTimeEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);

    return defaultTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
}

HRESULT event::ParticleTimeEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT event::ParticleTimeEvent::eval(const FloatP_t &time) {
    targetParticle = getNextParticle();
    auto result = event::EventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

event::ParticleTimeEvent::operator event::TimeEvent&() const {
    event::TimeEvent *e = new event::TimeEvent(period, NULL, NULL, NULL, start_time, end_time);
    return *e;
}

FloatP_t event::ParticleTimeEvent::getNextTime(const FloatP_t &current_time) {
    return (*nextTimeSetter)(*this, current_time);
}

ParticleHandle *event::ParticleTimeEvent::getNextParticle() {
    return (*particleSelector)(*this);
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventNextTimeSetter(event::ParticleTimeEventNextTimeSetter *nextTimeSetter) {
    this->nextTimeSetter = nextTimeSetter;
    return S_OK;
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventNextTimeSetter(event::ParticleTimeEventTimeSetterEnum setterEnum) {
    auto x = event::particleTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == event::particleTimeEventNextTimeSetterMap.end()) return 1;
    return setParticleTimeEventNextTimeSetter(&x->second);
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventNextTimeSetter(std::string setterName) {
    auto *selector = event::getParticleTimeEventParticleSelectorN(setterName);
    if(!selector) return E_FAIL;
    return setParticleTimeEventParticleSelector(selector);
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventParticleSelector(event::ParticleTimeEventParticleSelector *particleSelector) {
    this->particleSelector = particleSelector;
    return S_OK;
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventParticleSelector(event::ParticleTimeEventParticleSelectorEnum selectorEnum) {
    auto x = event::particleTimeEventParticleSelectorMap.find(selectorEnum);
    if (x == event::particleTimeEventParticleSelectorMap.end()) return 1;
    return setParticleTimeEventParticleSelector(&x->second);
}

HRESULT event::ParticleTimeEvent::setParticleTimeEventParticleSelector(std::string selectorName) {
    auto *selector = event::getParticleTimeEventParticleSelectorN(selectorName);
    if(!selector) return E_FAIL;
    return setParticleTimeEventParticleSelector(selector);
}

event::ParticleTimeEvent *event::onParticleTimeEvent(
    ParticleType *targetType, 
    const FloatP_t &period, 
    event::ParticleTimeEventMethod *invokeMethod, 
    event::ParticleTimeEventMethod *predicateMethod, 
    unsigned int nextTimeSetterEnum, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time, 
    unsigned int particleSelectorEnum)
{
    auto *nextTimeSetter = event::getParticleTimeEventNextTimeSetter((event::ParticleTimeEventTimeSetterEnum)nextTimeSetterEnum);
    if(!nextTimeSetter) return NULL;

    auto *particleSelector = event::getParticleTimeEventParticleSelector((event::ParticleTimeEventParticleSelectorEnum)particleSelectorEnum);
    if(!particleSelector) return NULL;
    
    event::ParticleTimeEvent *event = new event::ParticleTimeEvent(targetType, period, invokeMethod, predicateMethod, nextTimeSetter, start_time, end_time, particleSelector);

    Universe::get()->events->addEvent(event);

    return event;
}

event::ParticleTimeEvent *event::onParticleTimeEventN(
    ParticleType *targetType, 
    const FloatP_t &period, 
    event::ParticleTimeEventMethod *invokeMethod, 
    event::ParticleTimeEventMethod *predicateMethod, 
    const std::string &distribution, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time, 
    const std::string &selector) 
{
    auto x = event::particleTimeEventNextTimeSetterNameMap.find(distribution);
    if (x == event::particleTimeEventNextTimeSetterNameMap.end()) return NULL;
    unsigned int nextTimeSetterEnum = (unsigned) x->second;

    auto y = event::particleEventParticleSelectorNameMap.find(selector);
    if (y == event::particleEventParticleSelectorNameMap.end()) return NULL;
    unsigned int particleSelectorEnum = (unsigned) y->second;

    return event::onParticleTimeEvent(targetType, period, invokeMethod, predicateMethod, nextTimeSetterEnum, start_time, end_time, particleSelectorEnum);
}
