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

#ifndef _SOURCE_EVENT_TFTIMEEVENT_H_
#define _SOURCE_EVENT_TFTIMEEVENT_H_

#include "tfEventList.h"

#include <limits>
#include <unordered_map>


namespace TissueForge::event { 


    struct TimeEvent;

    using TimeEventMethod = EventMethodT<TimeEvent>;
    using TimeEventNextTimeSetter = FloatP_t (*)(TimeEvent&, const FloatP_t&);

    CAPI_FUNC(HRESULT) defaultTimeEventPredicateEval(const FloatP_t &next_time, const FloatP_t &start_time=-1, const FloatP_t &end_time=-1);

    CAPI_FUNC(FloatP_t) timeEventSetNextTimeExponential(TimeEvent &event, const FloatP_t &time);
    CAPI_FUNC(FloatP_t) timeEventSetNextTimeDeterministic(TimeEvent &event, const FloatP_t &time);

    enum class TimeEventTimeSetterEnum : unsigned int {
        DEFAULT = 0, 
        DETERMINISTIC,
        EXPONENTIAL
    };

    typedef std::unordered_map<TimeEventTimeSetterEnum, TimeEventNextTimeSetter> TimeEventNextTimeSetterMapType;
    static TimeEventNextTimeSetterMapType timeEventNextTimeSetterMap {
        {TimeEventTimeSetterEnum::DETERMINISTIC, &timeEventSetNextTimeDeterministic},
        {TimeEventTimeSetterEnum::EXPONENTIAL, &timeEventSetNextTimeExponential},
        {TimeEventTimeSetterEnum::DEFAULT, &timeEventSetNextTimeDeterministic}
    };

    typedef std::unordered_map<std::string, TimeEventTimeSetterEnum> TimeEventNextTimeSetterNameMapType;
    static TimeEventNextTimeSetterNameMapType timeEventNextTimeSetterNameMap {
        {"deterministic", TimeEventTimeSetterEnum::DETERMINISTIC},
        {"exponential", TimeEventTimeSetterEnum::EXPONENTIAL},
        {"default", TimeEventTimeSetterEnum::DEFAULT}
    };

    CAPI_FUNC(TimeEventNextTimeSetter*) getTimeEventNextTimeSetter(TimeEventTimeSetterEnum setterEnum);
    CAPI_FUNC(TimeEventNextTimeSetter*) getTimeEventNextTimeSetterN(std::string setterName);

    struct CAPI_EXPORT TimeEvent : EventBase {
        /**
         * @brief Next time of evaluation
         */
        FloatP_t next_time;

        /**
         * @brief Period of evaluation
         */
        FloatP_t period;

        /**
         * @brief Start time of evaluations
         */
        FloatP_t start_time;

        /**
         * @brief End time of evaluations
         */
        FloatP_t end_time;

        TimeEvent(
            const FloatP_t &period, 
            TimeEventMethod *invokeMethod, 
            TimeEventMethod *predicateMethod=NULL, 
            TimeEventNextTimeSetter *nextTimeSetter=NULL, 
            const FloatP_t &start_time=0, 
            const FloatP_t &end_time=-1
        ) : 
            EventBase(), 
            period(period), 
            invokeMethod(invokeMethod), 
            predicateMethod(predicateMethod), 
            nextTimeSetter(nextTimeSetter),
            next_time(start_time),
            start_time(start_time),
            end_time(end_time > 0 ? end_time : std::numeric_limits<FloatP_t>::max())
        {
            if (nextTimeSetter == NULL) setTimeEventNextTimeSetter(TimeEventTimeSetterEnum::DEFAULT);
        }
        ~TimeEvent();

        HRESULT predicate();
        HRESULT invoke();
        HRESULT eval(const FloatP_t &time);

    protected:

        FloatP_t getNextTime(const FloatP_t &current_time);
        HRESULT setTimeEventNextTimeSetter(TimeEventTimeSetterEnum setterEnum);

    private:

        TimeEventMethod *invokeMethod;
        TimeEventMethod *predicateMethod;
        TimeEventNextTimeSetter *nextTimeSetter;
    };

    using TimeEventList = EventListT<TimeEvent>;


    // Module entry points

    /**
     * @brief Creates a time-dependent event using prescribed invoke and predicate functions
     * 
     * @param period period of evaluations
     * @param invokeMethod an invoke function; evaluated when an event occurs
     * @param predicateMethod a predicate function; evaluated to determine if an event occurs
     * @param nextTimeSetterEnum enum selecting the function that sets the next evaulation time
     * @param start_time time at which evaluations begin
     * @param end_time time at which evaluations end
     * @return TimeEvent* 
     */
    CAPI_FUNC(TimeEvent*) onTimeEvent(
        const FloatP_t &period, 
        TimeEventMethod *invokeMethod, 
        TimeEventMethod *predicateMethod=NULL, 
        const unsigned int &nextTimeSetterEnum=0, 
        const FloatP_t &start_time=0, 
        const FloatP_t &end_time=-1
    );

    /**
     * @brief Creates a time-dependent event using prescribed invoke and predicate functions
     * 
     * @param period period of evaluations
     * @param invokeMethod an invoke function; evaluated when an event occurs
     * @param predicateMethod a predicate function; evaluated to determine if an event occurs
     * @param distribution name of the function that sets the next evaulation time
     * @param start_time time at which evaluations begin
     * @param end_time time at which evaluations end
     * @return TimeEvent* 
     */
    CAPI_FUNC(TimeEvent*) onTimeEventN(
        const FloatP_t &period, 
        TimeEventMethod *invokeMethod, 
        TimeEventMethod *predicateMethod=NULL, 
        const std::string &distribution="default", 
        const FloatP_t &start_time=0, 
        const FloatP_t &end_time=-1
    );

};

#endif // _SOURCE_EVENT_TFTIMEEVENT_H_