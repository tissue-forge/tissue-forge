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

#include "tfKeyEventPy.h"


using namespace TissueForge;


static py::KeyEventPyExecutor *_staticKeyEventPyExecutor = NULL;

static HRESULT keyEventPyExecutorHandler(struct event::KeyEvent *event) {
    if(_staticKeyEventPyExecutor) return _staticKeyEventPyExecutor->invoke(*event);
    return E_FAIL;
}

bool py::KeyEventPyExecutor::hasStaticKeyEventPyExecutor() {
    return _staticKeyEventPyExecutor != NULL;
}

void py::KeyEventPyExecutor::setStaticKeyEventPyExecutor(py::KeyEventPyExecutor *executor) {
    _staticKeyEventPyExecutor = executor;
    
    event::KeyEvent::addHandler(new event::KeyEventHandlerType(keyEventPyExecutorHandler));
}

void py::KeyEventPyExecutor::maybeSetStaticKeyEventPyExecutor(py::KeyEventPyExecutor *executor) {
    if(!hasStaticKeyEventPyExecutor()) _staticKeyEventPyExecutor = executor;
}

py::KeyEventPyExecutor *py::KeyEventPyExecutor::getStaticKeyEventPyExecutor() {
    return _staticKeyEventPyExecutor;
}
