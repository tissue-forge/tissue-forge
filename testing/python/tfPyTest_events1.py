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

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
tf.init(dim=dim, windowless=True)


class MyCellType(tf.ParticleTypeSpec):

    mass = 39.4
    target_temperature = 50
    radius = 0.2


MyCell = MyCellType.get()

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = tf.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)


# bind the potential with the *TYPES* of the particles
tf.bind.types(pot, MyCell, MyCell)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = tf.Force.berendsen_tstat(10)

# bind it just like any other force
tf.bind.force(tstat, MyCell)


# create a new particle every 0.05 time units. The 'on_particletime' function
# binds the MyCell object and callback 'split' with the event, and is called
# at periodic intervals based on the exponential distribution,
# so the mean time between particle creation is 0.05
def split(event: tf.event.ParticleTimeEvent):
    if event.targetParticle is None:
        event.targetType(tf.Universe.center)
    else:
        p = event.targetParticle.split()
        p.radius = event.targetType.radius
        event.targetParticle.radius = event.targetType.radius
    print('split:', len(event.targetType.items()))


tf.event.on_particletime(ptype=MyCell, invoke_method=split, period=0.05, distribution="exponential")

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
