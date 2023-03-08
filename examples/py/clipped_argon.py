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

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
tf.init(dim=dim,
        window_size=[900, 900],
        perfcounter_period=100,
        clip_planes=[([2, 2, 2], [1, 1, 0.5]), ([5, 5, 5], [-1, 1, -1])])

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = tf.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)


# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class ArgonType(tf.ParticleTypeSpec):
    radius = 0.1
    mass = 39.4


Argon = ArgonType.get()


# bind the potential with the *TYPES* of the particles
tf.bind.types(pot, Argon, Argon)

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(13000, 3))

for pos in positions:
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos)

# Create a clip plane on demand
cp = tf.rendering.ClipPlanes.create(tf.Universe.center, tf.FVector3(0., 1., 0.))

# Animate the clip plane


def rotate_clip(e):
    """Rotates the clip plane about the x-axis"""
    cf = 2 * np.pi * tf.Universe.time / 20.0
    cp.setEquation(tf.Universe.center, tf.FVector3(0., np.cos(cf), np.sin(cf)))


tf.event.on_time(period=tf.Universe.dt, invoke_method=rotate_clip)

# Orient the camera to verify animation
tf.system.camera_view_right()

# run the simulator interactive
tf.run()
