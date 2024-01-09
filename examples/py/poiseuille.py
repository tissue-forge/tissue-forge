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

tf.init(dt=0.1, dim=[15, 12, 10], cells=[7, 6, 5], cutoff=0.5,
        bc={'x': 'periodic',
            'y': 'periodic',
            'top': {'velocity': [0, 0, 0]},
            'bottom': {'velocity': [0, 0, 0]}})

# lattice spacing
a = 0.15


class AType(tf.ParticleTypeSpec):
    radius = 0.05
    style = {"colormap": {"velocity": "x", "range": (0, 1.5)}}
    dynamics = tf.Newtonian
    mass = 10


A = AType.get()

dpd = tf.Potential.dpd(alpha=10, sigma=1)

tf.bind.types(dpd, A, A)

# driving pressure
pressure = tf.CustomForce([0.1, 0, 0])

tf.bind.force(pressure, A)

uc = tf.lattice.sc(a, A)

parts = tf.lattice.create_lattice(uc, [40, 40, 40])

tf.run()
