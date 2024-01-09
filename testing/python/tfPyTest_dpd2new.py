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

tf.init(dt=0.1, 
        dim=[15, 12, 10],
        bc={'x': 'no_slip',
            'y': 'periodic',
            'bottom': 'no_slip',
            'top': {'velocity': [-0.4, 0, 0]}},
        perfcounter_period=100, 
        windowless=True)

# lattice spacing
a = 0.3


class AType(tf.ParticleTypeSpec):
    radius = 0.2
    style = {"color": "seagreen"}
    mass = 10


A = AType.get()

dpd = tf.Potential.dpd(alpha=0.3, gamma=1, sigma=1, cutoff=0.6)
dpd_wall = tf.Potential.dpd(alpha=0.5, gamma=10, sigma=1, cutoff=0.1)
dpd_left = tf.Potential.dpd(alpha=1, gamma=100, sigma=0, cutoff=0.5)

tf.bind.types(dpd, A, A)
tf.bind.boundary_condition(dpd_wall, tf.Universe.boundary_conditions.top, A)
tf.bind.boundary_condition(dpd_left, tf.Universe.boundary_conditions.left, A)


uc = tf.lattice.sc(a, A)

parts = tf.lattice.create_lattice(uc, [25, 25, 25])

print(tf.Universe.boundary_conditions)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
