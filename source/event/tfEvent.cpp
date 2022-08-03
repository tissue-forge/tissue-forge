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

#include "tfEvent.h"

#include <tfLogger.h>
#include <tfUniverse.h>


using namespace TissueForge;


event::Event::Event() : 
    event::EventBase(), 
    invokeMethod(NULL), 
    predicateMethod(NULL) 
{}

event::Event::Event(event::EventMethod *invokeMethod, event::EventMethod *predicateMethod) : 
    event::EventBase(), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod) 
{}

event::Event::~Event() {
    if(invokeMethod) {
        delete invokeMethod;
        invokeMethod = 0;
    }
    if(predicateMethod) {
        delete predicateMethod;
        predicateMethod = 0;
    }
}

HRESULT event::Event::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}
HRESULT event::Event::invoke() {
    if(invokeMethod) return (*invokeMethod)(*this);
    return 0;
}

HRESULT event::Event::eval(const FloatP_t &time) {
    remove();
    return event::EventBase::eval(time);
}

event::Event *event::onEvent(event::EventMethod *invokeMethod, event::EventMethod *predicateMethod) {
    TF_Log(LOG_TRACE);

    event::Event *event = new event::Event(invokeMethod, predicateMethod);
    Universe::get()->events->addEvent(event);
    return event;
}
