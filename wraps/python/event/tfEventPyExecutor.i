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

%{

#include <langs/py/tfEventPyExecutor.h>

%}


%include <langs/py/tfEventPyExecutor.h>

/**
* Factory for EventPyExecutor specializations. 
* Whatever interface is presented to the user should direct 
* the passed python function callback through `init...`, 
* which will set the executor callback in the C++ layer and store 
* the event callback in the python layer for execution. 
* 
* If a specialization defines a class member '_result', then it 
* will be set to the return value of the callback on execution. 
*/
%define tfEventPyExecutor_extender(wrappedName, eventName)

typedef TissueForge::py::EventPyExecutor<eventName> _ ## wrappedName ## TS;
%template(_ ## wrappedName ## TS) TissueForge::py::EventPyExecutor<eventName>;

%pythoncode %{
    def init ## wrappedName(callback):
        ex = wrappedName()
        
        def executorPyCallable():
            execute ## wrappedName(ex, callback)

        ex.setExecutorPyCallable(executorPyCallable)

        return ex

    def execute ## wrappedName(ex, callback):
        result = callback(ex.getEvent())
        if hasattr(ex, '_result'):
            ex._result = result
%}

%enddef
