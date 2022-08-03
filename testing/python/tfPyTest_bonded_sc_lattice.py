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
import numpy as np

tf.init(dt=0.1, dim=[15, 12, 10], windowless=True)

# lattice spacing
a = 0.65


class AType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "seagreen"}
    dynamics = tf.Overdamped


A = AType.get()


class BType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "red"}
    dynamics = tf.Overdamped


B = BType.get()


class FixedType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "blue"}
    frozen = True


Fixed = FixedType.get()

repulse = tf.Potential.coulomb(q=0.08, min=0.01, max=2 * a)

tf.bind.types(repulse, A, A)
tf.bind.types(repulse, A, B)

f = tf.CustomForce(lambda: [0.3, 1 * np.sin(0.4 * tf.Universe.time), 0], 0.01)

tf.bind.force(f, B)

pot = tf.Potential.power(r0=0.5 * a, alpha=2)

uc = tf.lattice.sc(a, A, lambda i, j: tf.Bond.create(pot, i, j, dissociation_energy=100.0))

parts = tf.lattice.create_lattice(uc, [15, 15, 15])

for p in parts[14, :].flatten():
    p[0].become(B)

for p in parts[0, :].flatten():
    p[0].become(Fixed)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
