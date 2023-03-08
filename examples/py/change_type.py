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

tf.init(cutoff=3)


class AType(tf.ParticleTypeSpec):
    radius = 0.1
    dynamics = tf.Overdamped
    style = {"color": "MediumSeaGreen"}


class BType(tf.ParticleTypeSpec):
    radius = 0.1
    dynamics = tf.Overdamped
    style = {"color": "skyblue"}


A = AType.get()
B = BType.get()

p = tf.Potential.coulomb(q=0.5, min=0.01, max=3)
q = tf.Potential.coulomb(q=0.5, min=0.01, max=3)
r = tf.Potential.coulomb(q=2.0, min=0.01, max=3)

tf.bind.types(p, A, A)
tf.bind.types(q, B, B)
tf.bind.types(r, A, B)

pos = [x * 10 + tf.Universe.center for x in tf.random_points(tf.PointsType.SolidCube.value, 1000)]

[A(p) for p in pos]

a = A.items()[0]

[p.become(B) for p in a.neighbors(5)]

tf.show()
