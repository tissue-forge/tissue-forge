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
 * @file tfTimeEventPy.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TFTIMEEVENTPY_H_
#define _SOURCE_LANGS_PY_TFTIMEEVENTPY_H_

#include "tf_py.h"

#include "tfEventPyExecutor.h"

#include <event/tfTimeEvent.h>


namespace TissueForge {


   namespace py {


        struct TimeEventPy;

        struct TimeEventPyPredicatePyExecutor : EventPyExecutor<TimeEventPy> {
            HRESULT _result = 0;
        };

        struct TimeEventPyInvokePyExecutor : EventPyExecutor<TimeEventPy> {
            HRESULT _result = 0;
        };

        struct CAPI_EXPORT TimeEventPy : event::EventBase {
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

            TimeEventPy(
                const FloatP_t &period, 
                TimeEventPyInvokePyExecutor *invokeExecutor, 
                TimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                event::TimeEventNextTimeSetter *nextTimeSetter=NULL, 
                const FloatP_t &start_time=0, 
                const FloatP_t &end_time=-1
            ) : 
                event::EventBase(), 
                period(period), 
                invokeExecutor(invokeExecutor), 
                predicateExecutor(predicateExecutor), 
                nextTimeSetter(nextTimeSetter), 
                next_time(start_time), 
                start_time(start_time), 
                end_time(end_time > 0 ? end_time : std::numeric_limits<FloatP_t>::max())
            {}
            ~TimeEventPy();

            HRESULT predicate();
            HRESULT invoke();
            HRESULT eval(const FloatP_t &time);

        protected:

            FloatP_t getNextTime(const FloatP_t &current_time);

        private:
            TimeEventPyInvokePyExecutor *invokeExecutor;
            TimeEventPyPredicatePyExecutor *predicateExecutor;
            event::TimeEventNextTimeSetter *nextTimeSetter;
        };

        using TimeEventPyList = event::EventListT<TimeEventPy>;

        /**
         * @brief Creates a time-dependent event using prescribed invoke and predicate python function executors
         * 
         * @param period period of evaluations
         * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
         * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
         * @param distribution name of the function that sets the next evaulation time
         * @param start_time time at which evaluations begin
         * @param end_time time at which evaluations end
         * @return TimeEventPy* 
         */
        CPPAPI_FUNC(TimeEventPy*) onTimeEvent(
            const FloatP_t &period, 
            TimeEventPyInvokePyExecutor *invokeExecutor, 
            TimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
            const std::string &distribution="default", 
            const FloatP_t &start_time=0, 
            const FloatP_t &end_time=-1
        );

}};

#endif // _SOURCE_LANGS_PY_TFTIMEEVENTPY_H_