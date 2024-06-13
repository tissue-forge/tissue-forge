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
import numpy as np

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff, windowless=True)


class AType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'red'}


A = AType.get()


class BType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'blue'}


B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = tf.Potential.morse(d=3, a=5, max=3)
pot_bb = tf.Potential.morse(d=3, a=5, max=3)
pot_ab = tf.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_aa, A, A)
tf.bind.types(pot_bb, B, B)
tf.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=50)

# bind it just like any other force
tf.bind.force(rforce, A)
tf.bind.force(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count, 3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count, 3)) * 15 + 2.5:
    B(p)

# run the simulator
tf.step(20*tf.Universe.dt)


def test_pass():
    pass
