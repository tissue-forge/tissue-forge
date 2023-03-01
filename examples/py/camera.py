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

"""
This example demonstrates basic programmatic usage of Tissue Forge camera controls.

*Warning*

Presentation of the simulation defined in this script may cause motion sickness.
"""
import tissue_forge as tf
from math import sin, pi

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

# Vary the camera view
move_ang = 0.025                 # Rate of camera rotation
move_zom = 0.1                   # Rate of camera zoom
cam_per = tf.Universe.dt * 1000  # Camera period
rot_idx = 1                      # Index of current camera axis of rotation

# Initialize camera position
pos_reset = tf.FVector3(0, 0, 0)
rot_reset = tf.fQuaternion(tf.FVector3(0.5495796799659729, 0.09131094068288803, 0.08799689263105392),
                           -0.8257609605789185)
zoom_reset = -2.875530958175659
tf.system.camera_move_to(pos_reset, rot_reset, zoom_reset)


def auto_cam_move(e):
    """Event callback to vary the camera"""
    cf = sin(2 * pi * tf.Universe.time / cam_per)
    angles = tf.FVector3(0.0)
    angles[rot_idx] = move_ang * cf
    zoom = - move_zom * cf
    tf.system.camera_rotate_by_euler_angle(angles=angles)
    tf.system.camera_zoom_by(zoom)


def inc_cam_rotation_axis(e):
    """Event callback to change the camera axis of rotation"""
    global rot_idx
    rot_idx = rot_idx + 1 if rot_idx < 2 else 0


tf.event.on_time(period=tf.Universe.dt, invoke_method=auto_cam_move)
tf.event.on_time(period=cam_per, invoke_method=inc_cam_rotation_axis)

# Run the simulator
tf.run()

# Report the final camera view settings before exiting
print('Camera position:', tf.system.camera_center())
print('Camera rotation:', tf.system.camera_rotation())
print('Camera zoom:', tf.system.camera_zoom())
