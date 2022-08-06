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

#include "tfTimeEventPy.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


py::TimeEventPy::~TimeEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT defaultTimeEventPyPredicateEval(const py::TimeEventPy &e) {
    auto current_time = Universe::getTime();
    HRESULT result = current_time >= e.next_time && current_time >= e.start_time && current_time <= e.end_time;

    return result;
}

HRESULT py::TimeEventPy::predicate() {
    if(!predicateExecutor) return defaultTimeEventPyPredicateEval(*this);
    else if(!predicateExecutor->hasExecutorPyCallable()) return 1;
    return predicateExecutor->invoke(*this);
}

HRESULT py::TimeEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT py::TimeEventPy::eval(const FloatP_t &time) {
    auto result = event::EventBase::eval(time);
    if(result) this->next_time = getNextTime(time);
    return result;
}

FloatP_t py::TimeEventPy::getNextTime(const FloatP_t &current_time) {
    if(!nextTimeSetter || nextTimeSetter == NULL) return current_time + this->period;
    return (*this->nextTimeSetter)(*(event::TimeEvent*)this, current_time);
}

py::TimeEventPy* py::onTimeEvent(
    const FloatP_t &period, 
    py::TimeEventPyInvokePyExecutor *invokeExecutor, 
    py::TimeEventPyPredicatePyExecutor *predicateExecutor, 
    const std::string &distribution, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time) 
{
    TF_Log(LOG_TRACE);

    auto itr = event::timeEventNextTimeSetterNameMap.find(distribution);
    if(itr == event::timeEventNextTimeSetterNameMap.end()) {
        tf_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    event::TimeEventTimeSetterEnum nextTimeSetterEnum = itr->second;

    event::TimeEventNextTimeSetter *nextTimeSetter = event::getTimeEventNextTimeSetter((event::TimeEventTimeSetterEnum)nextTimeSetterEnum);

    auto event = new py::TimeEventPy(period, invokeExecutor, predicateExecutor, nextTimeSetter, start_time, end_time);
    Universe::get()->events->addEvent(event);
    return event;
}
