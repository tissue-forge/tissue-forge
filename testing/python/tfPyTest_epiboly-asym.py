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

# dimensions of universe
dim = [30., 30., 30.]

tf.init(dim=dim,
        cutoff=5,
        dt=0.001, 
        windowless=True)


class AType(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class BType(tf.ParticleTypeSpec):
    radius = 0.2
    dynamics = tf.Overdamped
    mass = 10
    style = {"color": "skyblue"}


class CType(tf.ParticleTypeSpec):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
B = BType.get()
C = CType.get()

pc = tf.Potential.glj(e=10, m=3, max=5)
pa = tf.Potential.glj(e=2, m=4, max=3.0)
pb = tf.Potential.glj(e=1, m=4, max=1)
pab = tf.Potential.harmonic(k=10, r0=0, min=0.01, max=0.55)


# simple harmonic potential to pull particles
h = tf.Potential.harmonic(k=40, r0=0.001, max=5)

tf.bind.types(pc, A, C)
tf.bind.types(pc, B, C)
tf.bind.types(pa, A, A)

r = tf.Force.random(mean=0, std=5)

tf.bind.force(r, A)


c = C(tf.Universe.center)

pos_a = [p * ((1 + 0.125) * C.radius) + tf.Universe.center for p in tf.random_points(tf.PointsType.SolidSphere.value,
                                                                                     3000,
                                                                                     dr=0.25,
                                                                                     phi1=0.60 * np.pi)]

parts, bonds = tf.bind.sphere(h, type=B, n=4, phi=(0.6 * np.pi, np.pi), radius=C.radius + B.radius)

[A(p) for p in pos_a]

# grab a vertical slice of the neighbors of the yolk:
slice_parts = [p for p in c.neighbors() if p.sphericalPosition()[1] > 0]

tf.bind.bonds(pab, slice_parts, cutoff=5 * A.radius, pairs=[(A, B)])

C.style.visible = False
B.style.visible = False


def update(e):
    print(B.items().center_of_mass.as_list())


tf.event.on_time(invoke_method=update, period=0.01)


tf.step(100*tf.Universe.dt)


def test_pass():
    pass
