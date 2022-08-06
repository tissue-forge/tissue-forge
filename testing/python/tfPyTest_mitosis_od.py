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

import tissue_forge as tf

# potential cutoff distance
cutoff = 1

# new simulator
tf.init(dim=[20., 20., 20.], windowless=True)


pot = tf.Potential.morse(d=0.1, a=6, min=-1, max=1)


class CellType(tf.ParticleTypeSpec):
    mass = 20
    target_temperature = 0
    radius = 0.5
    dynamics = tf.Overdamped

    @staticmethod
    def on_register(ptype):
        def fission(event: tf.event.ParticleTimeEvent):
            m = event.targetParticle
            d = m.fission()
            m.radius = d.radius = CellType.radius
            m.mass = d.mass = CellType.mass

            print('fission:', len(event.targetType.items()))

        tf.event.on_particletime(ptype=ptype, invoke_method=fission, period=1, distribution='exponential')


Cell = CellType.get()

tf.bind.types(pot, Cell, Cell)

rforce = tf.Force.random(mean=0, std=0.5)

tf.bind.force(rforce, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
