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

#include "tfEventList.h"

#include <tfLogger.h>


using namespace TissueForge;


HRESULT event::event_func_invoke(event::EventBase &event, const FloatP_t &time) {
    return event.eval(time);
}

std::vector<event::EventBase*>::iterator event::EventBaseList::findEventIterator(event::EventBase *event) {
    for(std::vector<event::EventBase*>::iterator itr = events.begin(); itr != events.end(); ++itr)
        if(*itr == event) 
            return itr;
    return events.end();
}

event::EventBaseList::~EventBaseList() {
    events.clear();
    toRemove.clear();
}

void event::EventBaseList::addEvent(event::EventBase *event) { events.push_back(event); }

HRESULT event::EventBaseList::removeEvent(event::EventBase *event) {
    auto itr = findEventIterator(event);
    if (itr == events.end()) return 1;
    delete *itr;
    events.erase(itr);
    return S_OK;
}

HRESULT event::EventBaseList::eval(const FloatP_t &time) {
    for (auto &e : events) {
        auto result = e->eval(time);
        if (result < 0) {
            TF_Log(LOG_DEBUG) << "Event returned error code. Aborting.";
            return result;
        }

        for (auto flag : e->flags) {

            switch (flag)
            {
            case event::EventFlag::REMOVE:
                toRemove.push_back(e);
                break;
            
            default:
                break;
            }

        }
    }

    for (auto e : toRemove)
        removeEvent(e);

    toRemove.clear();

    return S_OK;
}

HRESULT event::eventListEval(event::EventBaseList *eventList, const FloatP_t &time) {
    return eventList->eval(time);
}
