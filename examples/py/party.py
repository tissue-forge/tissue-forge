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

"""
This example demonstrates select usage of basic visualization customization
"""

import tissue_forge as tf
from math import sin, cos, pi

tf.init()


class AType(tf.ParticleTypeSpec):
    radius = 0.1


A = AType.get()

# Create a simple oscillator
pot = tf.Potential.harmonic(k=100, r0=0.3)
disp = tf.FVector3(A.radius + 0.07, 0, 0)
p0 = A(tf.Universe.center - disp)
p1 = A(tf.Universe.center + disp)
tf.Bond.create(pot, p0, p1)

# Vary some basic visualization periodically during simulation


def vary_colors(e):
    rate = 2 * pi * tf.Universe.time / 1.
    sf = (sin(rate) + 1) / 2.
    sf2 = (sin(2 * rate) + 1) / 2.
    cf = (cos(rate) + 1) / 2.

    tf.system.set_grid_color(tf.FVector3(sf, 0, sf2))
    tf.system.set_scene_box_color(tf.FVector3(sf2, sf, 0))
    tf.system.set_shininess(1000 * sf + 10)
    tf.system.set_light_direction(tf.FVector3(3 * (2 * sf - 1), 3 * (2 * cf - 1), 2.))
    tf.system.set_light_color(tf.FVector3((sf + 1.) * 0.5, (sf + 1.) * 0.5, (sf + 1.) * 0.5))
    tf.system.set_ambient_color(tf.FVector3(sf, sf, sf))


tf.event.on_time(period=tf.Universe.dt, invoke_method=vary_colors)

# Run the simulator
tf.run()
