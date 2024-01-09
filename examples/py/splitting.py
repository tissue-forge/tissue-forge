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
import numpy as np

# One species to test
species = ['S1']

tf.init(dim=[20, 13, 3])


class AType(tf.ParticleTypeSpec):
    """A big particle type"""
    radius = 1.0


class BType(tf.ParticleTypeSpec):
    """A little particle type"""
    radius = 0.5


class CType(AType):
    """A big particle type that carries a species"""
    species = species
    style = {'colormap': {'species': 'S1', 'map': 'rainbow', 'range': (0, 1)}}


class DType(BType):
    """A little particle type that carries a species"""
    species = species
    style = {'colormap': {'species': 'S1', 'map': 'rainbow', 'range': (0, 1)}}


A = AType.get()
B = BType.get()
C = CType.get()
D = DType.get()

# Set initial concentrations before creating
C.species[species[0]].initial_concentration = 1.0
D.species[species[0]].initial_concentration = 0.0

# Use a direction for all splitting
split_dir = tf.FVector3(0, 1, 0)

# Use variable size coefficients when splitting
child_cfs = np.linspace(0.05, 0.95, 20)

# Create particles along the x-axis
x_poss = np.linspace(AType.radius * 1.5, tf.Universe.dim[0] - AType.radius * 1.5, len(child_cfs))

# Make a position to move around and create particles for splitting
pos = tf.FVector3()
pos[1] = tf.Universe.dim[1] - AType.radius * 2
pos[2] = tf.Universe.center[2]

# Split with child volume coefficient and test center of mass
for i in range(len(child_cfs)):
    pos[0] = x_poss[i]
    p = A(pos)
    com_i = p.position
    c = p.split(split_dir, child_cfs[i])
    com_f = (p.position * p.mass + c.position * c.mass) / (p.mass + c.mass)
    print('COM test:', com_i, com_f, (com_i - com_f).length() / com_i.length())

# Split with volume coefficient and types and test conservation of volume
pos[1] -= AType.radius * 3
for i in range(len(child_cfs)):
    pos[0] = x_poss[i]
    p = A(pos)
    vol_i = p.volume
    c = p.split(split_dir, child_cfs[i], A, B)
    vol_f = p.volume + c.volume
    print('Volume test:', vol_i, vol_f, (vol_f - vol_i) / vol_i)

# Split with species and test conservation of species
pos[1] -= AType.radius * 3
for i in range(len(child_cfs)):
    pos[0] = x_poss[i]
    p = C(pos)
    nspecies_i = p.species.S1.value * p.volume
    c = p.split(split_dir, child_cfs[i])
    nspecies_f = p.species.S1.value * p.volume + c.species.S1.value * c.volume
    print('Species test:', nspecies_i, nspecies_f, (nspecies_f - nspecies_i) / nspecies_i)

# Split with species and species ratios
pos[1] -= AType.radius * 3
for i in range(len(child_cfs)):
    pos[0] = x_poss[i]
    p = C(pos)
    c = p.split(split_dir, child_cfs[i], [1-child_cfs[i]], C, D)

# Show it!
tf.system.camera_view_top()
tf.show()
