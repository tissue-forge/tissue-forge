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

cutoff = 8
count = 3

# dimensions of universe
dim = [20., 20., 20.]

tf.init(dim=dim, cutoff=cutoff, windowless=True)


class BType(tf.ParticleTypeSpec):
    mass = 1
    dynamics = tf.Overdamped


B = BType.get()

# make a glj potential, this automatically reads the
# particle radius to determine rest distance.
pot = tf.Potential.morse(d=1, a=3, max=3)

tf.bind.types(pot, B, B)

p1 = B(tf.Universe.center + (-2, 0, 0))
p2 = B(tf.Universe.center + (2, 0, 0))
p1.radius = 1
p2.radius = 2

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
