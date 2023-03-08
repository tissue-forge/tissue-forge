# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022, 2023 T.J. Sego
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

from . import tf_config
from .tissue_forge import *
from . import bind
from . import event
from . import io
from . import lattice
from . import metrics
from . import models
from . import rendering
from . import state
from . import system
from .particle_type import ClusterTypeSpec, ParticleTypeSpec

__all__ = ['bind', 'event', 'io', 'lattice', 'metrics', 'models', 'rendering', 'state', 'system']

if has_cuda:
    from . import cuda
    __all__.append('cuda')

if system.is_jupyter_notebook():
    from . import jwidget
    __all__.append('jwidget')
    show = jwidget.show

if system.is_terminal_interactive():
    from .tissue_forge import _onIPythonNotReady

    def _input_hook(context):
        while not context.input_is_ready():
            _onIPythonNotReady()

        return None

    from .tissue_forge import _setIPythonInputHook
    _setIPythonInputHook(_input_hook)
