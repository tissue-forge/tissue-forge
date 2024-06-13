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

%{

#include <event/tfEvent.h>
#include <event/tfTimeEvent.h>
#include <langs/py/tfEventPy.h>

%}


tfEventPyExecutor_extender(EventPyPredicatePyExecutor, TissueForge::py::EventPy)
tfEventPyExecutor_extender(EventPyInvokePyExecutor, TissueForge::py::EventPy)

%ignore TissueForge::event::onEvent;
%rename(_event_onEvent) TissueForge::py::onEvent;

%rename(_event_EventBase) TissueForge::event::EventBase;
%rename(_event__Event) TissueForge::event::Event;
%rename(_event_Event) TissueForge::py::EventPy;

%include <event/tfEvent.h>
%include <langs/py/tfEventPy.h>

%pythoncode %{
    def _event_on_event(invoke_method, predicate_method=None):
        """
        Creates and registers an event using prescribed invoke and predicate python function executors

        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`event.Event` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`Event` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        """
        invoke_ex = initEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initEventPyPredicatePyExecutor(predicate_method)

        return _event_onEvent(invoke_ex, predicate_ex)

%}
