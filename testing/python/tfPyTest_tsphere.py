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


tf.init(dim=[25., 25., 25.], dt=0.0005, cutoff=3, windowless=True)


class GreenType(tf.ParticleTypeSpec):
    mass = 1
    radius = 0.1
    dynamics = tf.Overdamped
    style = {'color': 'mediumseagreen'}


class BigType(tf.ParticleTypeSpec):
    mass = 10
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Green = GreenType.get()
Big = BigType.get()

# simple harmonic potential to pull particles
pot = tf.Potential.harmonic(k=1, r0=0.1, max=3)

# potentials between green and big objects.
pot_yc = tf.Potential.glj(e=1, r0=1, m=3, min=0.01)
pot_cc = tf.Potential.glj(e=0.0001, r0=0.1, m=3, min=0.005, max=2)

# random points on surface of a sphere
pts = [p * (Green.radius + Big.radius) + tf.Universe.center for p in tf.random_points(tf.PointsType.Sphere.value, 1000)]

# make the big particle at the middle
Big(tf.Universe.center)

# constuct a particle for each position, make
# a list of particles
beads = [Green(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
# tf.bind.bonds(pot, beads, 0.7)

rforce = tf.Force.random(mean=0, std=0.01, duration=0.1)

# hook up the potentials
# tf.bind.force(rforce, Green)
tf.bind.types(pot_yc, Big, Green)
tf.bind.types(pot_cc, Green, Green)

tf.bind.bonds(pot, [p for p in tf.Universe.particles if p.position[1] < tf.Universe.center[1]], 1)

# run the model
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
