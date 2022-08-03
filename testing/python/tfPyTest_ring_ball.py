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

tf.init(dim=[20., 20., 20.], cutoff=8, windowless=True)


class BeadType(tf.ParticleTypeSpec):
    mass = 1
    radius = 0.1
    dynamics = tf.Overdamped


Bead = BeadType.get()

# simple harmonic potential to pull particles
pot = tf.Potential.harmonic(k=1, r0=0.1, max=3)

# make a ring of of 50 particles
pts = [p * 5 + tf.Universe.center for p in tf.points(tf.PointsType.Ring.value, 50)]

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
tf.bind.bonds(pot, beads, 1)

# run the model
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
