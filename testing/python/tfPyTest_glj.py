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

# potential cutoff distance
cutoff = 8

count = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff, windowless=True)


class BeadType(tf.ParticleTypeSpec):
    mass = 1
    radius = 0.5
    dynamics = tf.Overdamped


Bead = BeadType.get()
pot = tf.Potential.glj(e=1)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=0.01)
tf.bind.force(rforce, Bead)


r = 0.8 * Bead.radius

positions = [tf.Universe.center + [x, 0, 0] for x in np.arange(-count * r + r, count * r + r, 2 * r)]


for p in positions:
    print("position: ", p.as_list())
    Bead(p)

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
