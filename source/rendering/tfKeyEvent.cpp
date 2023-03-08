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

#include "tfKeyEvent.h"

#include <Magnum/Platform/GlfwApplication.h>

#include <tfLogger.h>
#include <tfError.h>
#include <iostream>
#include <unordered_map>


using namespace TissueForge;


template<typename H, typename T>
using KeyEventCbStorage = std::unordered_map<H, T*>;

static KeyEventCbStorage<size_t, event::KeyEventDelegateType> delegates = {};
static KeyEventCbStorage<size_t, event::KeyEventHandlerType> handlers = {};

HRESULT event::KeyEvent::invoke() {
    TF_Log(LOG_TRACE);

    HRESULT r, result = S_OK;
    for(auto &d : delegates) {
        auto f = d.second;
        r = (*f)(glfw_event);
        if(result == S_OK) 
            result = r;
    }
    for(auto &h : handlers) {
        auto f = h.second;
        r = (*f)(this);
        if(result == S_OK) 
            result = r; 
    }

    // TODO: check result code
    return result;
}

HRESULT event::KeyEvent::invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke) {
    auto event = new event::KeyEvent();
    event->glfw_event = &ke;
    auto result = event->invoke();
    delete event;
    return result;
}

template<typename H, typename T>
H addKeyEventCbStorage(KeyEventCbStorage<H, T> &storage, T *cb) {
    H i = storage.size();
    for(H j = 0; j < storage.size(); j++) {
        if(storage.find(j) == storage.end()) {
            i = j;
            break;
        }
    }

    storage.insert(std::make_pair(i, cb));
    return i;
}

template<typename H, typename T> 
T *getKeyEventCbStorage(const KeyEventCbStorage<H, T> &storage, const H &handle) {
    auto itr = storage.find(handle);
    if(itr == storage.end()) 
        return NULL;
    return itr->second;
}

template<typename H, typename T> 
bool removeKeyEventCbStorage(KeyEventCbStorage<H, T> &storage, const H &handle) {
    auto itr = storage.find(handle);
    if(itr == storage.end()) 
        return false;
    storage.erase(itr);
    return true;
}

event::KeyEventDelegateHandle event::KeyEvent::addDelegate(event::KeyEventDelegateType *_delegate) {
    return addKeyEventCbStorage(delegates, _delegate);
}

event::KeyEventHandlerHandle event::KeyEvent::addHandler(event::KeyEventHandlerType *_handler) {
    return addKeyEventCbStorage(handlers, _handler);
}

event::KeyEventDelegateType *event::KeyEvent::getDelegate(const event::KeyEventDelegateHandle &handle) {
    return getKeyEventCbStorage(delegates, handle);
}

event::KeyEventHandlerType *event::KeyEvent::getHandler(const event::KeyEventHandlerHandle &handle) {
    return getKeyEventCbStorage(handlers, handle);
}

bool event::KeyEvent::removeDelegate(const event::KeyEventDelegateHandle &handle) {
    return removeKeyEventCbStorage(delegates, handle);
}

bool event::KeyEvent::removeHandler(const event::KeyEventHandlerHandle &handle) {
    return removeKeyEventCbStorage(handlers, handle);
}

std::string event::KeyEvent::keyName() {
    if(glfw_event) return glfw_event->keyName();
    return "";
}

bool event::KeyEvent::keyAlt() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Alt);
    return false;
}

bool event::KeyEvent::keyCtrl() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Ctrl);
    return false;
}

bool event::KeyEvent::keyShift() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Shift);
    return false;
}
