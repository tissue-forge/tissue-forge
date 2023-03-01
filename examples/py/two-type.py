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
cutoff = 8

count = 3000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff)


class BigType(tf.ParticleTypeSpec):
    mass = 500000
    radius = 3


class SmallType(tf.ParticleTypeSpec):
    mass = 0.1
    radius = 0.2
    target_temperature = 0


Big = BigType.get()
Small = SmallType.get()


pot_bs = tf.Potential.morse(d=1, a=6, min=0.1, max=8, r0=3.2, shifted=False)
pot_ss = tf.Potential.morse(d=1, a=6, min=0.01, max=4, r0=0.2, shifted=False)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_bs, Big, Small)
tf.bind.types(pot_ss, Small, Small)

Big(position=tf.Universe.center, velocity=[0., 0., 0.])

for p in tf.random_points(tf.PointsType.Disk.value, count):
    Small(p * 2.5 * Big.radius + tf.Universe.center + [0, 0, Big.radius + 1])

# run the simulator interactive
tf.run()
