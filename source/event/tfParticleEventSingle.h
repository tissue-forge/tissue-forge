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

#ifndef _SOURCE_EVENT_TFPARTICLEEVENTSINGLE_H_
#define _SOURCE_EVENT_TFPARTICLEEVENTSINGLE_H_

#include "tfParticleEvent.h"


namespace TissueForge::event { 


    // Single particle event
    struct CAPI_EXPORT ParticleEventSingle : ParticleEvent {

        ParticleEventSingle(
            ParticleType *targetType, 
            ParticleEventMethod *invokeMethod, 
            ParticleEventMethod *predicateMethod, 
            ParticleEventParticleSelector *particleSelector=NULL
        );

        virtual HRESULT eval(const FloatP_t &time);

    };

    /**
     * @brief Creates a single particle event using prescribed invoke and predicate functions
     * 
     * @param targetType target particle type
     * @param invokeMethod an invoke function; evaluated when an event occurs
     * @param predicateMethod a predicate function; evaluated to determine if an event occurs
     * @return ParticleEventSingle* 
     */
    CPPAPI_FUNC(ParticleEventSingle*) onParticleEventSingle(
        ParticleType *targetType, 
        ParticleEventMethod *invokeMethod, 
        ParticleEventMethod *predicateMethod
    );

};

#endif // _SOURCE_EVENT_TFPARTICLEEVENTSINGLE_H_