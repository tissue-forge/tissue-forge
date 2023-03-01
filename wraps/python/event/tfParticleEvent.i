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

#include <event/tfEvent.h>
#include <event/tfParticleEvent.h>
#include <event/tfParticleEventSingle.h>
#include <langs/py/tfParticleEventPy.h>
#include <langs/py/tfParticleEventSinglePy.h>

%}


tfEventPyExecutor_extender(ParticleEventPyInvokePyExecutor, TissueForge::py::ParticleEventPy)
tfEventPyExecutor_extender(ParticleEventPyPredicatePyExecutor, TissueForge::py::ParticleEventPy)

%ignore TissueForge::event::onParticleEvent;
%ignore TissueForge::event::onParticleEventSingle;
%rename(_event__onParticleEvent) TissueForge::py::onParticleEvent;
%rename(_event__onParticleEventSingle) TissueForge::py::onParticleEventSingle;

%rename(_event__ParticleEvent) TissueForge::event::ParticleEvent;
%rename(_event_ParticleEvent) TissueForge::py::ParticleEventPy;

%include <event/tfParticleEvent.h>
%include <event/tfParticleEventSingle.h>
%include <langs/py/tfParticleEventPy.h>
%include <langs/py/tfParticleEventSinglePy.h>

%pythoncode %{
    def _event_on_particle(ptype, invoke_method, predicate_method=None, selector="default", single: bool=False):
        """
        Like :meth:`on_event`, but for a particle of a particular particle type. 

        :type ptype: ParticleType
        :param ptype: the type of particle to select
        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`ParticleEvent` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`ParticleEvent` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        :type selector: str
        :param selector: name of particle selector
        :type single: bool
        :param single: flag to only trigger event once and then return
        """
        invoke_ex = initParticleEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initParticleEventPyPredicatePyExecutor(predicate_method)

        if single:
            return _event__onParticleEventSingle(ptype, invoke_ex, predicate_ex)
        return _event__onParticleEvent(ptype, invoke_ex, predicate_ex, selector)


%}
