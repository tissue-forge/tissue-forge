# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
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

tf.init(dim=dim, cutoff=5, dt=0.0005, windowless=True)


class AType(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class BType(tf.ParticleTypeSpec):
    radius = 0.2
    dynamics = tf.Overdamped
    mass = 1
    style = {"color": "skyblue"}


class CType(tf.ParticleTypeSpec):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
B = BType.get()
C = CType.get()

C(tf.Universe.center)

# make a ring of of 50 particles
pts = tf.points(tf.PointsType.Ring.value, 100)
pts = [x * (C.radius+B.radius) + tf.Universe.center - tf.FVector3(0, 0, 1) for x in pts]
[B(p) for p in pts]

pc = tf.Potential.glj(e=30, m=2, max=5)
pa = tf.Potential.glj(e=3, m=2.5, max=3)
pb = tf.Potential.glj(e=1, m=4, max=1)
pab = tf.Potential.glj(e=1, m=2, max=1)
ph = tf.Potential.harmonic(r0=0.001, k=200)

tf.bind.types(pc, A, C)
tf.bind.types(pc, B, C)
tf.bind.types(pa, A, A)
tf.bind.types(pab, A, B)

r = tf.Force.random(mean=0, std=5)

tf.bind.force(r, A)
tf.bind.force(r, B)

tf.bind.bonds(ph, B.items(), 1)


def update(e: tf.event.Event):
    """Callback to report the center of mass of all B-type particles during simulation"""
    print(e.times_fired, B.items().center_of_mass)


# Implement the callback
tf.event.on_time(period=0.01, invoke_method=update)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
