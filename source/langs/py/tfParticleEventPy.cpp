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

#include "tfParticleEventPy.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


py::ParticleEventPy::ParticleEventPy(
    ParticleType *targetType, 
    py::ParticleEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleEventPyPredicatePyExecutor *predicateExecutor, 
    event::ParticleEventParticleSelector *particleSelector
) : 
    event::ParticleEvent(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    if (particleSelector==NULL) event::ParticleEvent::setParticleEventParticleSelector(event::ParticleEventParticleSelectorEnum::DEFAULT);
    else event::ParticleEvent::setParticleEventParticleSelector(particleSelector);
}

py::ParticleEventPy::~ParticleEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT py::ParticleEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) return 1;
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT py::ParticleEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT py::ParticleEventPy::eval(const FloatP_t &time) {
    targetParticle = getNextParticle();
    return event::EventBase::eval(time);
}

py::ParticleEventPy *py::onParticleEvent(
    ParticleType *targetType, 
    py::ParticleEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleEventPyPredicatePyExecutor *predicateExecutor, 
    const std::string &selector)
{
    TF_Log(LOG_TRACE) << targetType->id;

    event::ParticleEventParticleSelector *particleSelector = event::getParticleEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;

    py::ParticleEventPy *event = new py::ParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector);

    Universe::get()->events->addEvent(event);

    return event;
}
