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
 * @file tfParticleTimeEventPy.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TFPARTICLETIMEEVENTPY_H_
#define _SOURCE_LANGS_PY_TFPARTICLETIMEEVENTPY_H_

#include "tf_py.h"

#include "tfEventPyExecutor.h"

#include <event/tfParticleTimeEvent.h>


namespace TissueForge {


   namespace py {


        struct ParticleTimeEventPy;

        struct ParticleTimeEventPyPredicatePyExecutor : py::EventPyExecutor<ParticleTimeEventPy> {
            HRESULT _result = 0;
        };

        struct ParticleTimeEventPyInvokePyExecutor : py::EventPyExecutor<ParticleTimeEventPy> {
            HRESULT _result = 0;
        };

        // Time-dependent particle event
        struct CAPI_EXPORT ParticleTimeEventPy : event::ParticleTimeEvent {
            
            ParticleTimeEventPy() {}
            ParticleTimeEventPy(
                ParticleType *targetType, 
                const FloatP_t &period, 
                ParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                ParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                event::ParticleTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                const FloatP_t &start_time=0, 
                const FloatP_t &end_time=-1,
                event::ParticleTimeEventParticleSelector *particleSelector=NULL
            );
            virtual ~ParticleTimeEventPy();

            virtual HRESULT predicate();
            virtual HRESULT invoke();
            virtual HRESULT eval(const FloatP_t &time);

        private:

            ParticleTimeEventPyInvokePyExecutor *invokeExecutor;
            ParticleTimeEventPyPredicatePyExecutor *predicateExecutor;

        };

        /**
        * @brief Creates a time-dependent particle event using prescribed invoke and predicate python function executors
        * 
        * @param targetType target particle type
        * @param period period of evaluations
        * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
        * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
        * @param distribution name of the function that sets the next evaulation time
        * @param start_time time at which evaluations begin
        * @param end_time time at which evaluations end
        * @param selector name of the function that selects the next particle
        * @return ParticleTimeEventPy* 
        */
        CPPAPI_FUNC(ParticleTimeEventPy*) onParticleTimeEvent(
            ParticleType *targetType, 
            const FloatP_t &period, 
            ParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
            ParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
            const std::string &distribution="default", 
            const FloatP_t &start_time=0.0, 
            const FloatP_t &end_time=-1.0, 
            const std::string &selector="default"
        );

}};

#endif // _SOURCE_LANGS_PY_TFPARTICLETIMEEVENTPY_H_