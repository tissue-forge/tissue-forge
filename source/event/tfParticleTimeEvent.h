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

/**
 * @file tfParticleTimeEvent.h
 * 
 */

#ifndef _SOURCE_EVENT_TFPARTICLETIMEEVENT_H_
#define _SOURCE_EVENT_TFPARTICLETIMEEVENT_H_

#include "tfTimeEvent.h"
#include "tfParticleEvent.h"


namespace TissueForge {


    struct ParticleType;


    namespace event { 


        struct ParticleEvent;

        struct ParticleTimeEvent;

        using ParticleTimeEventMethod = EventMethodT<ParticleTimeEvent>;

        using ParticleTimeEventNextTimeSetter = FloatP_t (*)(ParticleTimeEvent&, const FloatP_t&);

        /**
         * @brief Sets the next time on an event according to an exponential distribution of the event period
         * 
         * @param event 
         * @param time 
         */
        CPPAPI_FUNC(FloatP_t) particleTimeEventSetNextTimeExponential(ParticleTimeEvent &event, const FloatP_t &time);

        /**
         * @brief Sets the next time on an event according to the period of the event
         * 
         * @param event 
         * @param time 
         */
        CPPAPI_FUNC(FloatP_t) particleTimeEventSetNextTimeDeterministic(ParticleTimeEvent &event, const FloatP_t &time);

        using ParticleTimeEventParticleSelector = ParticleEventParticleSelectorT<ParticleTimeEvent>;

        /**
         * @brief Selects a particle according to a uniform random distribution by event target type
         * 
         * @param event 
         * @return ParticleHandle* 
         */
        CPPAPI_FUNC(ParticleHandle*) particleTimeEventParticleSelectorUniform(const ParticleTimeEvent &event);

        /**
         * @brief Selects largest particle by event target type
         * 
         * @param event 
         * @return ParticleHandle* 
         */
        CPPAPI_FUNC(ParticleHandle*) particleTimeEventParticleSelectorLargest(const ParticleTimeEvent &event);

        // keys for selecting a particle selector
        enum class ParticleTimeEventParticleSelectorEnum : unsigned int {
            LARGEST, 
            UNIFORM,
            DEFAULT
        };

        typedef std::unordered_map<ParticleTimeEventParticleSelectorEnum, ParticleTimeEventParticleSelector> ParticleTimeEventParticleSelectorMapType;
        static ParticleTimeEventParticleSelectorMapType particleTimeEventParticleSelectorMap {
            {ParticleTimeEventParticleSelectorEnum::LARGEST, &particleTimeEventParticleSelectorLargest}, 
            {ParticleTimeEventParticleSelectorEnum::UNIFORM, &particleTimeEventParticleSelectorUniform}, 
            {ParticleTimeEventParticleSelectorEnum::DEFAULT, &particleTimeEventParticleSelectorUniform}
        };

        typedef std::unordered_map<std::string, ParticleTimeEventParticleSelectorEnum> ParticleTimeEventParticleSelectorNameMapType;
        static ParticleTimeEventParticleSelectorNameMapType particleTimeEventParticleSelectorNameMap {
            {"largest", ParticleTimeEventParticleSelectorEnum::LARGEST}, 
            {"uniform", ParticleTimeEventParticleSelectorEnum::UNIFORM}, 
            {"default", ParticleTimeEventParticleSelectorEnum::DEFAULT}
        };

        /**
         * @brief Gets the particle selector on an event
         * 
         * @param selectorEnum selector enum
         * @return ParticleTimeEventParticleSelector* 
         */
        CPPAPI_FUNC(ParticleTimeEventParticleSelector*) getParticleTimeEventParticleSelector(ParticleTimeEventParticleSelectorEnum selectorEnum);

        /**
         * @brief Gets the particle selector on an event
         * 
         * @param setterName name of selector
         * @return ParticleTimeEventParticleSelector* 
         */
        CPPAPI_FUNC(ParticleTimeEventParticleSelector*) getParticleTimeEventParticleSelectorN(std::string setterName);

        /**
         * @brief keys for selecting a next time setter
         * 
         */
        enum class ParticleTimeEventTimeSetterEnum : unsigned int {
            DETERMINISTIC,
            EXPONENTIAL,
            DEFAULT
        };

        typedef std::unordered_map<ParticleTimeEventTimeSetterEnum, ParticleTimeEventNextTimeSetter> ParticleTimeEventNextTimeSetterMapType;
        static ParticleTimeEventNextTimeSetterMapType particleTimeEventNextTimeSetterMap {
            {ParticleTimeEventTimeSetterEnum::DETERMINISTIC, &particleTimeEventSetNextTimeDeterministic},
            {ParticleTimeEventTimeSetterEnum::EXPONENTIAL, &particleTimeEventSetNextTimeExponential},
            {ParticleTimeEventTimeSetterEnum::DEFAULT, &particleTimeEventSetNextTimeDeterministic}
        };

        typedef std::unordered_map<std::string, ParticleTimeEventTimeSetterEnum> ParticleTimeEventNextTimeSetterNameMapType;
        static ParticleTimeEventNextTimeSetterNameMapType particleTimeEventNextTimeSetterNameMap {
            {"deterministic", ParticleTimeEventTimeSetterEnum::DETERMINISTIC},
            {"exponential", ParticleTimeEventTimeSetterEnum::EXPONENTIAL},
            {"default", ParticleTimeEventTimeSetterEnum::DEFAULT}
        };

        /**
         * @brief Gets the next time on an event according to an exponential distribution of the event period
         * 
         * @param setterEnum setter enum
         * @return ParticleTimeEventNextTimeSetter* 
         */
        CPPAPI_FUNC(ParticleTimeEventNextTimeSetter*) getParticleTimeEventNextTimeSetter(ParticleTimeEventTimeSetterEnum setterEnum);

        /**
         * @brief Gets the next time on an event according to an exponential distribution of the event period
         * 
         * @param setterName name of setter
         * @return ParticleTimeEventNextTimeSetter* 
         */
        CPPAPI_FUNC(ParticleTimeEventNextTimeSetter*) getParticleTimeEventNextTimeSetterN(std::string setterName);

        // Time-dependent particle event
        struct CAPI_EXPORT ParticleTimeEvent : EventBase {

            /**
             * @brief Target particle type of this event
             */
            ParticleType *targetType;

            /**
            * @brief Target particle of an event evaluation
            */
            ParticleHandle *targetParticle;

            /**
            * @brief Next time at which an evaluation occurs
            */
            FloatP_t next_time;

            /**
            * @brief Period of event evaluations
            */
            FloatP_t period;

            /**
            * @brief Time at which evaluations begin
            */
            FloatP_t start_time;

            /**
            * @brief Time at which evaluations stop
            */
            FloatP_t end_time;
            
            ParticleTimeEvent() {}
            ParticleTimeEvent(
                ParticleType *targetType, 
                const FloatP_t &period, 
                ParticleTimeEventMethod *invokeMethod, 
                ParticleTimeEventMethod *predicateMethod, 
                ParticleTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                const FloatP_t &start_time=0, 
                const FloatP_t &end_time=-1,
                ParticleTimeEventParticleSelector *particleSelector=NULL
            );
            virtual ~ParticleTimeEvent() {}

            virtual HRESULT predicate();
            virtual HRESULT invoke();
            virtual HRESULT eval(const FloatP_t &time);

            operator TimeEvent&() const;

        protected:

            FloatP_t getNextTime(const FloatP_t &current_time);
            ParticleHandle *getNextParticle();

            HRESULT setParticleTimeEventNextTimeSetter(ParticleTimeEventNextTimeSetter *nextTimeSetter);
            HRESULT setParticleTimeEventNextTimeSetter(ParticleTimeEventTimeSetterEnum setterEnum);
            HRESULT setParticleTimeEventNextTimeSetter(std::string setterName);

            HRESULT setParticleTimeEventParticleSelector(ParticleTimeEventParticleSelector *particleSelector);
            HRESULT setParticleTimeEventParticleSelector(ParticleTimeEventParticleSelectorEnum selectorEnum);
            HRESULT setParticleTimeEventParticleSelector(std::string selectorName);

        private:

            ParticleTimeEventMethod *invokeMethod;
            ParticleTimeEventMethod *predicateMethod;
            ParticleTimeEventNextTimeSetter *nextTimeSetter;
            ParticleTimeEventParticleSelector *particleSelector;

        };

        // Event list for time-dependent particle events
        using ParticleTimeEventList = EventListT<ParticleTimeEvent>;

        // Module entry points

        /**
        * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
        * 
        * @param targetType target particle type
        * @param period period of evaluations
        * @param invokeMethod an invoke function; evaluated when an event occurs
        * @param predicateMethod a predicate function; evaluated to determine if an event occurs
        * @param nextTimeSetterEnum enum of function that sets the next evaulation time
        * @param start_time time at which evaluations begin
        * @param end_time time at which evaluations end
        * @param particleSelectorEnum enum of function that selects the next particle
        * @return ParticleTimeEvent* 
        */
        CPPAPI_FUNC(ParticleTimeEvent*) onParticleTimeEvent(
            ParticleType *targetType, 
            const FloatP_t &period, 
            ParticleTimeEventMethod *invokeMethod, 
            ParticleTimeEventMethod *predicateMethod=NULL, 
            unsigned int nextTimeSetterEnum=0, 
            const FloatP_t &start_time=0, 
            const FloatP_t &end_time=-1, 
            unsigned int particleSelectorEnum=0
        );

        /**
        * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
        * 
        * @param targetType target particle type
        * @param period period of evaluations
        * @param invokeMethod an invoke function; evaluated when an event occurs
        * @param predicateMethod a predicate function; evaluated to determine if an event occurs
        * @param distribution name of function that sets the next evaulation time
        * @param start_time time at which evaluations begin
        * @param end_time time at which evaluations end
        * @param selector name of function that selects the next particle
        * @return ParticleTimeEvent* 
        */
        CPPAPI_FUNC(ParticleTimeEvent*) onParticleTimeEventN(
            ParticleType *targetType, 
            const FloatP_t &period, 
            ParticleTimeEventMethod *invokeMethod, 
            ParticleTimeEventMethod *predicateMethod=NULL, 
            const std::string &distribution="default", 
            const FloatP_t &start_time=0, 
            const FloatP_t &end_time=-1, 
            const std::string &selector="default"
        );

}};

#endif // _SOURCE_EVENT_TFPARTICLETIMEEVENT_H_