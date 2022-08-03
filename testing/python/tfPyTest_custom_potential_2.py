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
from math import acos, cos, pi, sin
from random import random

tf.init(bc={'x': 'noslip', 'y': 'noslip', 'z': 'noslip'}, cutoff=5.0, windowless=True)


class FrozenType(tf.ParticleTypeSpec):
    frozen = True
    style = {'visible': False}


class BondedType(tf.ParticleTypeSpec):
    radius = 0.1


frozen_type, bonded_type = FrozenType.get(), BondedType.get()

# Construct bond potential
rad_a, rad_l = 10.0, 2.0
rad_f = lambda r: rad_a * cos(2 * pi / rad_l * r)
rad_fp = lambda r: - 2 * pi / rad_l * rad_a * sin(2 * pi / rad_l * r)
rad_f6p = lambda r: - (2 * pi / rad_l) ** 6.0 * rad_a * cos(2 * pi / rad_l * r)
pot_rad = tf.Potential.custom(f=rad_f, fp=rad_fp, f6p=rad_f6p, min=0.0, max=2*rad_l)

# Construct angle potential
#   Apporoximating deriviatives, since passed argument is cosine of the angle, which makes differentiation tedious
ang_f = lambda r: 10.0 * cos(6.0 * acos(r))
pot_ang = tf.Potential.custom(min=-0.999, max=0.999, f=ang_f, flags=tf.Potential.Flags.angle.value)

# Create particles
ftp0 = frozen_type(position=tf.Universe.center, velocity=tf.FVector3(0))
ftp1 = frozen_type(position=tf.Universe.center + tf.FVector3(1, 0, 0), velocity=tf.FVector3(0))
btp = bonded_type(position=tf.Universe.center + tf.FVector3((random() - 0.5) * tf.Universe.dim[0] / 4,
                                                              (random() - 0.5) * tf.Universe.dim[1] / 4, 0))
btp.frozen_z = True

# Bind particles
tf.Bond.create(pot_rad, btp, ftp0)
tf.Angle.create(pot_ang, btp, ftp0, ftp1)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
