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

import tissue_forge as tf
import numpy as np

cutoff = 1

tf.init(dim=[10., 10., 10.], windowless=True)


class ArgonType(tf.ParticleTypeSpec):
    mass = 39.4
    target_temperature = 100


Argon = ArgonType.get()


# hook up the destroy method on the Argon type to the
# on_time event
def destroy(event: tf.event.ParticleTimeEvent):
    if event.targetParticle:
        print('destroy....')
        event.targetParticle.destroy()
        print('destroy:', len(event.targetType.items()))


tf.event.on_particletime(ptype=Argon, invoke_method=destroy, period=1, distribution='exponential')

pot = tf.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)
tf.bind.types(pot, Argon, Argon)

tstat = tf.Force.berendsen_tstat(10)

tf.bind.force(tstat, Argon)

size = 100

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 3))
velocities = np.random.normal(0, 0.2, size=(size, 3))

for pos, vel in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos, vel)

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
