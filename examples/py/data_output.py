# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
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

import tissue_forge as tf

# potential cutoff distance
cutoff = 1

# new simulator
tf.init(dim=[20., 20., 20.])

pot = tf.Potential.morse(d=1, a=5, r0=0.6, min=0, max=4, shifted=False)


class CellType(tf.ParticleTypeSpec):
    mass = 20
    target_temperature = 0
    radius = 0.5


Cell = CellType.get()


# Callback for time- and particle-dependent events
def fission(e: tf.event.ParticleTimeEvent):
    e.targetParticle.fission()


tf.event.on_particletime(ptype=Cell, period=1, invoke_method=fission, distribution='exponential')

tf.bind.types(pot, Cell, Cell)

Cell([10., 10., 10.])

# run the simulator interactively
tf.run()
