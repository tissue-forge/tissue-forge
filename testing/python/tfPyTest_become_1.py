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

tf.init(windowless=True)


class AType(tf.ParticleTypeSpec):

    radius = 1

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": (0, 1)}}


A = AType.get()


class BType(tf.ParticleTypeSpec):

    radius = 4

    species = ['S2', 'S3', 'S4']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": (0, 1)}}


B = BType.get()

o = A()

o.species.S2 = 0.5

o.become(B)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
