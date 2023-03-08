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

tf.init(dt=0.1, dim=[15, 6, 6], cells=[9, 3, 3], bc={'x': ('periodic', 'reset')}, cutoff=3)


class AType(tf.ParticleTypeSpec):
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1",
                          "map": "rainbow"}}


A = AType.get()

tf.Fluxes.flux(A, A, "S1", 2)

a1 = A(tf.Universe.center - [0, 1, 0])
a2 = A(tf.Universe.center + [-5, 1, 0], velocity=[0.5, 0, 0])

a1.species.S1 = 1
a2.species.S1 = 0

tf.run()
