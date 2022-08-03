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

#ifndef _SOURCE_EVENT_TFEVENT_H_
#define _SOURCE_EVENT_TFEVENT_H_

#include <TissueForge_private.h>

#include <forward_list>
#include <iostream>


namespace TissueForge::event { 


    template<typename event_t> using EventMethodT = HRESULT (*)(const event_t&);

    // Flags for providing feedback during predicate and invoke evaluations
    enum class EventFlag : unsigned int {
        REMOVE
    };

    // Base class of all events
    struct CAPI_EXPORT EventBase {

        // Flags set by invoke and predicate to provide feedback
        std::forward_list<EventFlag> flags;

        /**
         * @brief Record of last time fired
         */
        FloatP_t last_fired;

        /**
         * @brief Record of how many times fired
         */
        int times_fired;

        /**
         * Evaluates an event predicate,
         * returns 0 if the event should not fire, 1 if the event should, and a
         * negative value on error.
         * A predicate without a defined predicate method always returns 0
         */
        virtual HRESULT predicate() = 0;

        /**
         * What occurs during an event. 
         * Typically, this invokes an underlying specialized method
         * returns 0 if OK and 1 on error.
         */
        virtual HRESULT invoke() = 0;

        EventBase() : 
            last_fired(0.0), 
            times_fired(0)
        {}
        virtual ~EventBase() {}

        // Tests the predicate and evaluates invoke accordingly
        // Returns 1 if the event was invoked, 0 if not, and a negative value on error
        virtual HRESULT eval(const FloatP_t &time) {
            // check predicate
            if(!predicate()) return 0;

            // invoke
            if(invoke() == 1) return -1;

            // Update internal data
            times_fired += 1;
            last_fired = time;

            return 1;
        }

        /**
         * @brief Designates event for removal
         */
        void remove() { flags.push_front(EventFlag::REMOVE); }

    };

    struct Event;

    using EventMethod = EventMethodT<Event>;

    // Simple event
    struct CAPI_EXPORT Event : EventBase {
        
        Event();

        /**
         * @brief Construct an instance using functions
         * 
         * @param invokeMethod an invoke function
         * @param predicateMethod a predicate function
         */
        Event(EventMethod *invokeMethod, EventMethod *predicateMethod);
        virtual ~Event();

        HRESULT predicate();
        HRESULT invoke();
        HRESULT eval(const FloatP_t &time);

    private:

        EventMethod *invokeMethod;
        EventMethod *predicateMethod;

    };

    /**
     * @brief Creates an event using prescribed invoke and predicate functions
     * 
     * @param invokeMethod an invoke function; evaluated when an event occurs
     * @param predicateMethod a predicate function; evaluated to determine if an event occurs
     * @return Event* 
     */
    CPPAPI_FUNC(Event*) onEvent(EventMethod *invokeMethod, EventMethod *predicateMethod);

};

#endif // _SOURCE_EVENT_TFEVENT_H_