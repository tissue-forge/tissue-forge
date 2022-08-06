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

#include "tfEventPy.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


py::EventPy::EventPy(py::EventPyInvokePyExecutor *invokeExecutor, py::EventPyPredicatePyExecutor *predicateExecutor) : 
    event::EventBase(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{}

py::EventPy::~EventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT py::EventPy::predicate() { 
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) return 1;
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT py::EventPy::invoke() {
    if(!invokeExecutor || !invokeExecutor->hasExecutorPyCallable()) return 0;
    invokeExecutor->invoke(*this);
    return invokeExecutor->_result;
}

HRESULT py::EventPy::eval(const FloatP_t &time) {
    remove();
    return event::EventBase::eval(time);
}


py::EventPy *py::onEvent(py::EventPyInvokePyExecutor *invokeExecutor, py::EventPyPredicatePyExecutor *predicateExecutor) {
    TF_Log(LOG_TRACE);

    py::EventPy *event = new py::EventPy(invokeExecutor, predicateExecutor);
    Universe::get()->events->addEvent(event);
    return event;
}
