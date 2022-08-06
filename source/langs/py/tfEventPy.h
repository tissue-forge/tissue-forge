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

/**
 * @file tfEventPy.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TFEVENTPY_H_
#define _SOURCE_LANGS_PY_TFEVENTPY_H_

#include "tf_py.h"

#include <event/tfEvent.h>
#include "tfEventPyExecutor.h"


namespace TissueForge::py {


    struct EventPy;

    struct CAPI_EXPORT EventPyPredicatePyExecutor : EventPyExecutor<EventPy> {
        HRESULT _result = 0;
    };

    struct CAPI_EXPORT EventPyInvokePyExecutor : EventPyExecutor<EventPy> {
        HRESULT _result = 0;
    };

    struct CAPI_EXPORT EventPy : event::EventBase {
        EventPy(EventPyInvokePyExecutor *invokeExecutor, EventPyPredicatePyExecutor *predicateExecutor=NULL);
        ~EventPy();

        HRESULT predicate();
        HRESULT invoke();
        HRESULT eval(const FloatP_t &time);

    private:
        EventPyInvokePyExecutor *invokeExecutor; 
        EventPyPredicatePyExecutor *predicateExecutor;
    };

    /**
     * @brief Creates an event using prescribed invoke and predicate python function executors
     * 
     * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
     * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
     * @return EventPy* 
     */
    CPPAPI_FUNC(EventPy*) onEvent(EventPyInvokePyExecutor *invokeExecutor, EventPyPredicatePyExecutor *predicateExecutor=NULL);

};

#endif // _SOURCE_LANGS_PY_TFEVENTPY_H_