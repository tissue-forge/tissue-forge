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

tf.init(dim=dim, cutoff=5, dt=0.001)


class AType(tf.ParticleTypeSpec):
    radius = 0.1
    dynamics = tf.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class CType(tf.ParticleTypeSpec):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
C = CType.get()
C(tf.Universe.center)

pos = [p * (C.radius+A.radius) + tf.Universe.center for p in tf.random_points(tf.PointsType.Sphere.value, 5000)]

[A(p) for p in pos]

pc = tf.Potential.glj(e=30, m=2, max=5)
pa = tf.Potential.coulomb(q=100, min=0.01, max=5)

tf.bind.types(pc, A, C)
tf.bind.types(pa, A, A)

tf.run()
