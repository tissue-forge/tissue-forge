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

tf.init(dt=0.1, dim=[15, 12, 10],
        bc={'x': 'periodic',
            'y': 'periodic',
            'bottom': 'no_slip',
            'top': {'velocity': [-0.4, 0, 0]}},
        perfcounter_period=100)

# lattice spacing
a = 0.3

tf.Universe.boundary_conditions.left.restore = 0.5


class AType(tf.ParticleTypeSpec):
    radius = 0.2
    style = {"colormap": {"velocity": "x", "range": (-0.4, 0.4)}}
    dynamics = tf.Newtonian
    mass = 10


A = AType.get()

dpd = tf.Potential.dpd(alpha=0.5, gamma=1, sigma=0.1, cutoff=0.5)

tf.bind.types(dpd, A, A)

uc = tf.lattice.sc(a, A)

parts = tf.lattice.create_lattice(uc, [25, 25, 25])

print(tf.Universe.boundary_conditions)

tf.show()
