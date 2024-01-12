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

#include "tfTimeEvent.h"
#include <tf_util.h>
#include <tfLogger.h>
#include <tfEngine.h>
#include <tfUniverse.h>


using namespace TissueForge;


HRESULT event::defaultTimeEventPredicateEval(const FloatP_t &next_time, const FloatP_t &start_time, const FloatP_t &end_time) {
    auto current_time = _Engine.time * _Engine.dt;
    HRESULT result = current_time >= next_time;
    if (start_time > 0) result = result && current_time >= start_time;
    if (end_time > 0) result = result && current_time <= end_time;
    return result;
}

event::TimeEvent::~TimeEvent() {}

FloatP_t event::timeEventSetNextTimeExponential(event::TimeEvent &event, const FloatP_t &time) {
    std::exponential_distribution<> d(1/event.period);
    RandomType &randEng = randomEngine();
    return time + d(randEng);
}

FloatP_t event::timeEventSetNextTimeDeterministic(event::TimeEvent &event, const FloatP_t &time) {
    return time + event.period;
}

event::TimeEventNextTimeSetter* event::getTimeEventNextTimeSetter(event::TimeEventTimeSetterEnum setterEnum) {
    auto x = event::timeEventNextTimeSetterMap.find(setterEnum);
    if (x == event::timeEventNextTimeSetterMap.end()) return NULL;
    return &x->second;
}

event::TimeEventNextTimeSetter* event::getTimeEventNextTimeSetterN(std::string setterName) {
    auto itr = event::timeEventNextTimeSetterNameMap.find(setterName);
    if(itr == event::timeEventNextTimeSetterNameMap.end()) {
        tf_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    return event::getTimeEventNextTimeSetter(itr->second);
}

HRESULT event::TimeEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}

HRESULT event::TimeEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT event::TimeEvent::eval(const FloatP_t &time) {
    auto result = event::EventBase::eval(time);
    if(result) this->next_time = getNextTime(time);
    return result;
}

FloatP_t event::TimeEvent::getNextTime(const FloatP_t &current_time) {
    return (*nextTimeSetter)(*this, current_time);
}

HRESULT event::TimeEvent::setTimeEventNextTimeSetter(event::TimeEventTimeSetterEnum setterEnum) {
    auto x = event::timeEventNextTimeSetterMap.find(setterEnum);
    if (x == event::timeEventNextTimeSetterMap.end()) return 1;
    this->nextTimeSetter = &x->second;
    return S_OK;
}

event::TimeEvent* event::onTimeEvent(const FloatP_t &period, event::TimeEventMethod *invokeMethod, event::TimeEventMethod *predicateMethod, 
    const unsigned int &nextTimeSetterEnum, const FloatP_t &start_time, const FloatP_t &end_time) 
{
    TF_Log(LOG_TRACE);

    event::TimeEventNextTimeSetter *nextTimeSetter = getTimeEventNextTimeSetter((event::TimeEventTimeSetterEnum)nextTimeSetterEnum);
    
    event::TimeEvent *event = new event::TimeEvent(period, invokeMethod, predicateMethod, nextTimeSetter, start_time, end_time);
    
    Universe::get()->events->addEvent(event);

    return event;

}

event::TimeEvent* event::onTimeEventN(
    const FloatP_t &period, 
    event::TimeEventMethod *invokeMethod, 
    event::TimeEventMethod *predicateMethod, 
    const std::string &distribution, 
    const FloatP_t &start_time, 
    const FloatP_t &end_time) 
{
    auto itr = event::timeEventNextTimeSetterNameMap.find(distribution);
    if(itr == event::timeEventNextTimeSetterNameMap.end()) {
        tf_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    event::TimeEventTimeSetterEnum nextTimeSetterEnum = itr->second;

    return event::onTimeEvent(period, invokeMethod, predicateMethod, (unsigned)nextTimeSetterEnum, start_time, end_time);
}
