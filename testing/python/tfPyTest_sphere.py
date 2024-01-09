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
import numpy as np

tf.init(dim=[25., 25., 25.], cutoff=3, windowless=True)


class BlueType(tf.ParticleTypeSpec):
    mass = 1
    radius = 0.1
    dynamics = tf.Overdamped
    style = {'color': 'dodgerblue'}


class BigType(tf.ParticleTypeSpec):
    mass = 1
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Blue = BlueType.get()
Big = BigType.get()

# simple harmonic potential to pull particles
pot = tf.Potential.harmonic(k=1, r0=0.1, max=3)

# make big cell in the middle
Big(tf.Universe.center)

# Big.style.visible = False

# create a uniform mesh of particles and bonds on the surface of a sphere
parts, bonds = tf.bind.sphere(pot, type=Blue, n=5, phi=(0.6 * np.pi, 0.8 * np.pi), radius=Big.radius + Blue.radius)

# run the model
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
