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

%{

#include <event/tfTimeEvent.h>
#include <langs/py/tfTimeEventPy.h>

%}


tfEventPyExecutor_extender(TimeEventPyPredicatePyExecutor, TissueForge::py::TimeEventPy)
tfEventPyExecutor_extender(TimeEventPyInvokePyExecutor, TissueForge::py::TimeEventPy)

%ignore TissueForge::event::onTimeEvent;
%rename(_event__onTimeEvent) TissueForge::py::onTimeEvent;

%rename(_event__TimeEvent) TissueForge::event::TimeEvent;
%rename(_event_TimeEvent) TissueForge::py::TimeEventPy;

%include <event/tfTimeEvent.h>
%include <langs/py/tfTimeEventPy.h>

%pythoncode %{
    def _event_on_time(period, invoke_method, predicate_method=None, distribution="default", start_time=0.0, end_time=-1.0):
        """
        Like :meth:`on_event`, but for an event that occurs at regular intervals. 

        :type period: float
        :param period: period of event
        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`TimeEvent` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`TimeEvent` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        :type distribution: str
        :param distribution: distribution by which the next event time is selected
        :type start_time: float
        :param start_time: time after which the event can occur
        :type end_time: float
        :param end_time: time before which the event can occur; a negative value is interpreted as until 'forever'
        """
        invoke_ex = initTimeEventPyInvokePyExecutor(invoke_method)

        if predicate_method is not None:
            predicate_ex = initTimeEventPyPredicatePyExecutor(predicate_method)
        else:
            predicate_ex = None

        return _event__onTimeEvent(period, invoke_ex, predicate_ex, distribution, start_time, end_time)


%}
