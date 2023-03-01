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

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff, windowless=True)


class BeadType(tf.ParticleTypeSpec):
    mass = 0.4
    radius = 0.2
    dynamics = tf.Overdamped


Bead = BeadType.get()

pot_bb = tf.Potential.coulomb(q=0.1, min=0.1, max=1.0)

# hamonic bond between particles
pot_bond = tf.Potential.harmonic(k=0.4, r0=0.2, max=2)

# angle bond potential
pot_ang = tf.Potential.harmonic_angle(k=0.01, theta0=0.85 * np.pi, tol=0.01)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_bb, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=0.1)

# bind it just like any other force
tf.bind.force(rforce, Bead)

# make a array of positions
xx = np.arange(4., 16, 0.15)

p = None                              # previous bead
bead = Bead([xx[0], 10., 10.0])       # current bead

for i in range(1, xx.size):
    n = Bead([xx[i], 10.0, 10.0])     # create a new bead particle
    tf.Bond.create(pot_bond, bead, n)  # create a bond between prev and current
    if i > 1:
        tf.Angle.create(pot_ang, p, bead, n)  # make an angle bond between prev, cur, next
    p = bead
    bead = n

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
