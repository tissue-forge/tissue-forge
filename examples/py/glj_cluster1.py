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

# dimensions of universe
dim = [30., 30., 30.]

tf.init(dim=dim, cutoff=10, dt=0.0005)


class AType(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    mass = 10
    style = {"color": "MediumSeaGreen"}


class BType(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    mass = 10
    style = {"color": "skyblue"}


A = AType.get()
B = BType.get()


class CType(tf.ClusterTypeSpec):
    radius = 3
    types = [A, B]


C = CType.get()

c1 = C(position=tf.Universe.center - (3, 0, 0))
c2 = C(position=tf.Universe.center + (7, 0, 0))

[A() for _ in range(2000)]
[B() for _ in range(2000)]

p1 = tf.Potential.glj(e=7, m=1, max=1)
p2 = tf.Potential.glj(e=7, m=1, max=2)
tf.bind.types(p1, A, A, bound=True)
tf.bind.types(p2, B, B, bound=True)

rforce = tf.Force.random(mean=0, std=10)
tf.bind.force(rforce, A)
tf.bind.force(rforce, B)

tf.run()
