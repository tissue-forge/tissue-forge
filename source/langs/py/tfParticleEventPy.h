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

#ifndef _SOURCE_LANGS_PY_TFPARTICLEEVENTPY_H_
#define _SOURCE_LANGS_PY_TFPARTICLEEVENTPY_H_

#include "tf_py.h"

#include "tfEventPyExecutor.h"

#include <event/tfParticleEvent.h>


namespace TissueForge::py {


    struct ParticleEventPy;

    struct ParticleEventPyPredicatePyExecutor : EventPyExecutor<ParticleEventPy> {
        HRESULT _result = 0;
    };

    struct ParticleEventPyInvokePyExecutor : EventPyExecutor<ParticleEventPy> {
        HRESULT _result = 0;
    };

    struct CAPI_EXPORT ParticleEventPy : event::ParticleEvent {

        ParticleEventPy() {}
        ParticleEventPy(
            ParticleType *targetType, 
            ParticleEventPyInvokePyExecutor *invokeExecutor, 
            ParticleEventPyPredicatePyExecutor *predicateExecutor=NULL, 
            event::ParticleEventParticleSelector *particleSelector=NULL
        );
        virtual ~ParticleEventPy();

        virtual HRESULT predicate();
        virtual HRESULT invoke();
        virtual HRESULT eval(const FloatP_t &time);

    private:
        
        ParticleEventPyInvokePyExecutor *invokeExecutor;
        ParticleEventPyPredicatePyExecutor *predicateExecutor;

    };

    /**
     * @brief Creates a particle event using prescribed invoke and predicate python function executors
     * 
     * @param targetType target particle type
     * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
     * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
     * @param selector name of the function that selects the next particle
     * @return ParticleEventPy* 
     */
    CPPAPI_FUNC(ParticleEventPy*) onParticleEvent(
        ParticleType *targetType, 
        ParticleEventPyInvokePyExecutor *invokeExecutor, 
        ParticleEventPyPredicatePyExecutor *predicateExecutor, 
        const std::string &selector="default"
    );

};

#endif // _SOURCE_LANGS_PY_TFPARTICLEEVENTPY_H_