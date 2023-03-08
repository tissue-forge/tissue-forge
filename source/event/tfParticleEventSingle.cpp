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

#include "tfParticleEventSingle.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


event::ParticleEventSingle::ParticleEventSingle(
    ParticleType *targetType, 
    event::ParticleEventMethod *invokeMethod, 
    event::ParticleEventMethod *predicateMethod, 
    event::ParticleEventParticleSelector *particleSelector
) : 
    event::ParticleEvent(targetType, invokeMethod, predicateMethod, particleSelector) 
{}

HRESULT event::ParticleEventSingle::eval(const FloatP_t &time) {
    remove();
    return event::ParticleEvent::eval(time);
}

event::ParticleEventSingle *event::onParticleEventSingle(
    ParticleType *targetType, 
    event::ParticleEventMethod *invokeMethod, 
    event::ParticleEventMethod *predicateMethod)
{
    event::ParticleEventSingle *event = new event::ParticleEventSingle(targetType, invokeMethod, predicateMethod);

    Universe::get()->events->addEvent(event);

    return event;
}
