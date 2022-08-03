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

tf.init(dim=[25., 25., 25.], cutoff=3, dt=0.005)


class BlueType(tf.ParticleTypeSpec):
    mass = 10
    radius = 0.05
    dynamics = tf.Overdamped
    style = {'color': 'dodgerblue'}


Blue = BlueType.get()


class BigType(tf.ParticleTypeSpec):
    mass = 10
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Big = BigType.get()

# simple harmonic potential to pull particles
h = tf.Potential.harmonic(k=200, r0=0.001, max=5)

# simple coulomb potential to maintain separation between particles
pb = tf.Potential.coulomb(q=0.01, min=0.01, max=3)

# potential between the small and big particles
pot = tf.Potential.glj(e=1, m=2, max=5)

Big(tf.Universe.center)

Big.style.visible = False

tf.bind.types(pot, Big, Blue)

tf.bind.types(pb, Blue, Blue)

parts, bonds = tf.bind.sphere(h, type=Blue, n=4, phi=[0.55 * np.pi, 1 * np.pi], radius=Big.radius + Blue.radius)

# run the model
tf.show()
