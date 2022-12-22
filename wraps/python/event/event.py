# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# ******************************************************************************

from tissue_forge.tissue_forge import _event_EventBase
from tissue_forge.tissue_forge import _event__Event
from tissue_forge.tissue_forge import _event_Event
from tissue_forge.tissue_forge import _event_on_event as on_event
from tissue_forge.tissue_forge import _event_KeyEvent
from tissue_forge.tissue_forge import _event_on_keypress as on_keypress
from tissue_forge.tissue_forge import _event__ParticleEvent
from tissue_forge.tissue_forge import _event_ParticleEvent
from tissue_forge.tissue_forge import _event_on_particle as on_particle
from tissue_forge.tissue_forge import _event__ParticleTimeEvent
from tissue_forge.tissue_forge import _event_ParticleTimeEvent
from tissue_forge.tissue_forge import _event_on_particletime as on_particletime
from tissue_forge.tissue_forge import _event__TimeEvent
from tissue_forge.tissue_forge import _event_TimeEvent
from tissue_forge.tissue_forge import _event_on_time as on_time

class EventBase(_event_EventBase):
    pass

class _Event(_event__Event):
    pass

class Event(_event_Event):
    pass

class KeyEvent(_event_KeyEvent):
    pass

class _ParticleEvent(_event__ParticleEvent):
    pass

class ParticleEvent(_event_ParticleEvent):
    pass

class _ParticleTimeEvent(_event__ParticleTimeEvent):
    pass

class ParticleTimeEvent(_event_ParticleTimeEvent):
    pass

class _TimeEvent(_event__TimeEvent):
    pass

class TimeEvent(_event_TimeEvent):
    pass
