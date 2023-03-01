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

#include "tfParticleEventSinglePy.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


py::ParticleEventSinglePy::ParticleEventSinglePy(
    ParticleType *targetType, 
    py::ParticleEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleEventPyPredicatePyExecutor *predicateExecutor, 
    event::ParticleEventParticleSelector *particleSelector
) : 
    py::ParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector)
{}

HRESULT py::ParticleEventSinglePy::eval(const FloatP_t &time) {
    remove();
    return py::ParticleEventPy::eval(time);
}

py::ParticleEventSinglePy *py::onParticleEventSingle(
    ParticleType *targetType, 
    py::ParticleEventPyInvokePyExecutor *invokeExecutor, 
    py::ParticleEventPyPredicatePyExecutor *predicateExecutor)
{
    TF_Log(LOG_TRACE) << targetType->id;

    py::ParticleEventSinglePy *event = new py::ParticleEventSinglePy(targetType, invokeExecutor, predicateExecutor);

    Universe::get()->events->addEvent(event);

    return event;
}
