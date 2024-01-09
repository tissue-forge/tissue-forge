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

/**
 * @file tfParticleEvent.h
 * 
 */

#ifndef _SOURCE_EVENT_TFPARTICLEEVENT_H_
#define _SOURCE_EVENT_TFPARTICLEEVENT_H_

#include "tfEvent.h"
#include "tfEventList.h"

#include <unordered_map>


namespace TissueForge {


    struct ParticleHandle;
    struct ParticleType;


    namespace event { 


        // Particle selector function template
        template<typename event_t> using ParticleEventParticleSelectorT = ParticleHandle* (*)(const event_t&);

        // ParticleEvent

        struct ParticleEvent;

        using ParticleEventMethod = EventMethodT<ParticleEvent>;

        using ParticleEventParticleSelector = ParticleEventParticleSelectorT<ParticleEvent>;

        /**
         * @brief Selects a particle according to a uniform random distribution by event target type
         * 
         * @param typeId id of type
         * @param nr_parts number of particles of the type
         * @return ParticleHandle* 
         */
        CAPI_FUNC(ParticleHandle*) particleSelectorUniform(const int16_t &typeId, const int32_t &nr_parts);

        /**
         * @brief Selects largest particle by event target type
         * 
         * @param typeId id of type
         * @return ParticleHandle* 
         */
        CAPI_FUNC(ParticleHandle*) particleSelectorLargest(const int16_t &typeId);

        /**
         * @brief Selects a particle according to a uniform random distribution by event target type
         * 
         * @param event 
         * @return ParticleHandle* 
         */
        CAPI_FUNC(ParticleHandle*) particleEventParticleSelectorUniform(const ParticleEvent &event);

        /**
         * @brief Selects largest particle by event target type
         * 
         * @param event 
         * @return ParticleHandle* 
         */
        CAPI_FUNC(ParticleHandle*) particleEventParticleSelectorLargest(const ParticleEvent &event);

        // keys for selecting a particle selector
        enum class ParticleEventParticleSelectorEnum : unsigned int {
            LARGEST, 
            UNIFORM,
            DEFAULT
        };

        typedef std::unordered_map<ParticleEventParticleSelectorEnum, ParticleEventParticleSelector> ParticleEventParticleSelectorMapType;
        static ParticleEventParticleSelectorMapType particleEventParticleSelectorMap {
            {ParticleEventParticleSelectorEnum::LARGEST, &particleEventParticleSelectorLargest}, 
            {ParticleEventParticleSelectorEnum::UNIFORM, &particleEventParticleSelectorUniform}, 
            {ParticleEventParticleSelectorEnum::DEFAULT, &particleEventParticleSelectorUniform}
        };

        typedef std::unordered_map<std::string, ParticleEventParticleSelectorEnum> ParticleEventParticleSelectorNameMapType;
        static ParticleEventParticleSelectorNameMapType particleEventParticleSelectorNameMap {
            {"largest", ParticleEventParticleSelectorEnum::LARGEST}, 
            {"uniform", ParticleEventParticleSelectorEnum::UNIFORM}, 
            {"default", ParticleEventParticleSelectorEnum::DEFAULT}
        };

        /**
         * @brief Gets the particle selector on an event
         * 
         * @param selectorEnum selector enum
         * @return ParticleEventParticleSelector* 
         */
        CAPI_FUNC(ParticleEventParticleSelector*) getParticleEventParticleSelector(ParticleEventParticleSelectorEnum selectorEnum);

        /**
         * @brief Gets the particle selector on an event
         * 
         * @param setterName name of selector
         * @return ParticleEventParticleSelector* 
         */
        CAPI_FUNC(ParticleEventParticleSelector*) getParticleEventParticleSelectorN(std::string setterName);

        // Particle event
        struct CAPI_EXPORT ParticleEvent : EventBase {

            /**
             * @brief Target particle type of this event
             */
            ParticleType *targetType;

            /**
             * @brief Target particle of an event evaluation
             */
            ParticleHandle *targetParticle;

            ParticleEvent() {}
            ParticleEvent(
                ParticleType *targetType, 
                ParticleEventMethod *invokeMethod, 
                ParticleEventMethod *predicateMethod, 
                ParticleEventParticleSelector *particleSelector=NULL
            );
            virtual ~ParticleEvent() {}

            virtual HRESULT predicate();
            virtual HRESULT invoke();
            virtual HRESULT eval(const FloatP_t &time);

            HRESULT setParticleEventParticleSelector(ParticleEventParticleSelector *particleSelector);
            HRESULT setParticleEventParticleSelector(ParticleEventParticleSelectorEnum selectorEnum);
            HRESULT setParticleEventParticleSelector(std::string selectorName);

        protected:

            ParticleHandle *getNextParticle();

        private:

            ParticleEventMethod *invokeMethod;
            ParticleEventMethod *predicateMethod;
            ParticleEventParticleSelector *particleSelector;

        };

        // Event list for particle events
        using ParticleEventList = EventListT<ParticleEvent>;

        // Module entry points

        /**
         * @brief Creates a particle event using prescribed invoke and predicate functions
         * 
         * @param targetType target particle type
         * @param invokeMethod an invoke function; evaluated when an event occurs
         * @param predicateMethod a predicate function; evaluated to determine if an event occurs
         * @return ParticleEvent* 
         */
        CPPAPI_FUNC(ParticleEvent*) onParticleEvent(
            ParticleType *targetType, 
            ParticleEventMethod *invokeMethod, 
            ParticleEventMethod *predicateMethod
        );

}};

#endif // _SOURCE_EVENT_TFPARTICLEEVENT_H_