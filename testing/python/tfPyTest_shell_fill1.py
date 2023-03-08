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
cutoff = 10

# number of particles
count = 500

# number of time points we avg things
avg_pts = 3

# dimensions of universe
dim = [50., 50., 100.]

# new simulator
tf.init(dim=dim,
        cutoff=cutoff,
        dt=0.001,
        max_distance=0.2,
        threads=8,
        cells=[5, 5, 5], 
        windowless=True)


class BigType(tf.ParticleTypeSpec):
    mass = 500000
    radius = 20
    frozen = True


class SmallType(tf.ParticleTypeSpec):
    mass = 10
    radius = 0.25
    target_temperature = 0
    dynamics = tf.Overdamped


Big = BigType.get()
Small = SmallType.get()

pot_yc = tf.Potential.glj(e=100, r0=5, m=3, k=500, min=0.1, max=1.5 * Big.radius, tol=0.1)
# pot_cc = tf.Potential.glj(e=0,   r0=2, mx=2, k=10,  min=0.05, max=1 * Big.radius)
pot_cc = tf.Potential.harmonic(r0=0, k=0.1, max=10)

# pot_yc = tf.Potential.glj(e=10, r0=1, mx=3, min=0.1, max=50*Small.radius, tol=0.1)
# pot_cc = tf.Potential.glj(e=100, r0=5, mx=2, min=0.05, max=0.5*Big.radius)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_yc, Big, Small)
tf.bind.types(pot_cc, Small, Small)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=100, duration=0.5)

# bind it just like any other force
tf.bind.force(rforce, Small)

yolk = Big(position=tf.Universe.center)


for p in tf.random_points(tf.PointsType.Sphere.value, count):
    pos = p * (Big.radius + Small.radius) + tf.Universe.center
    Small(position=pos)


# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
