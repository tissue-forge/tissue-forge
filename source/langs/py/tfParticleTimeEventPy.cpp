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

#include "tfParticleTimeEventPy.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


py::ParticleTimeEventPy::ParticleTimeEventPy(
    ParticleType *targetType, 
    const FloatP_t &period, 
    py::ParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
    event::ParticleTimeEventNextTimeSetter *nextTimeSetter, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time,
    event::ParticleTimeEventParticleSelector *particleSelector
) : 
    event::ParticleTimeEvent(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    this->period = period;
    this->next_time = 0;
    this->start_time = start_time;
    this->end_time = end_time > 0 ? end_time : std::numeric_limits<FloatP_t>::max();

    if (nextTimeSetter == NULL) event::ParticleTimeEvent::setParticleTimeEventNextTimeSetter(event::ParticleTimeEventTimeSetterEnum::DEFAULT);
    else event::ParticleTimeEvent::setParticleTimeEventNextTimeSetter(nextTimeSetter);

    if (particleSelector == NULL) event::ParticleTimeEvent::setParticleTimeEventParticleSelector(event::ParticleTimeEventParticleSelectorEnum::DEFAULT);
    else event::ParticleTimeEvent::setParticleTimeEventParticleSelector(particleSelector);
}

py::ParticleTimeEventPy::~ParticleTimeEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT py::ParticleTimeEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) {
        return event::defaultTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
    }
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT py::ParticleTimeEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT py::ParticleTimeEventPy::eval(const FloatP_t &time) {
    targetParticle = getNextParticle();
    auto result = event::EventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

py::ParticleTimeEventPy *py::onParticleTimeEvent(
    ParticleType *targetType, 
    const FloatP_t &period, 
    py::ParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
    const std::string &distribution, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time, 
    const std::string &selector)
{
    TF_Log(LOG_TRACE) << targetType->id;
    
    event::ParticleTimeEventNextTimeSetter *nextTimeSetter = event::getParticleTimeEventNextTimeSetterN(distribution);
    if (!nextTimeSetter) return NULL;

    event::ParticleTimeEventParticleSelector *particleSelector = event::getParticleTimeEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;
    
    py::ParticleTimeEventPy *event = new py::ParticleTimeEventPy(targetType, period, invokeExecutor, predicateExecutor, nextTimeSetter, start_time, end_time, particleSelector);

    Universe::get()->events->addEvent(event);

    return event;
}
