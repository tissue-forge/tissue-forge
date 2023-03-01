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
This example demonstrates constructing a lattice from a custom unit cell to simulate
fracture in a two-dimensional elastic sheet.
"""

import math
import tissue_forge as tf

tf.init(dim=[40, 20, 3])


class MtlType(tf.ParticleTypeSpec):
    """Basic material type"""

    radius = 0.1


class BoundaryType(MtlType):
    """Material type with a zero-displacement condition along ``y``"""

    style = {'color': 'orange'}

    @staticmethod
    def apply_boundary(p: tf.ParticleHandle):
        p.frozen_y = True


class LoadedType(MtlType):
    """Material type on which an external load is applied"""

    style = {'color': 'darkgreen'}


mtl_type = MtlType.get()
boundary_type = BoundaryType.get()
loaded_type = LoadedType.get()

n_lattice = [60, 20]
a = 6 * mtl_type.radius
stiffness = 1E2
pot_border = tf.Potential.harmonic(k=stiffness, r0=a/2)
pot_cross = tf.Potential.harmonic(k=stiffness, r0=a/2*math.sqrt(2))
bcb_border = lambda i, j: tf.Bond.create(pot_border, i, j)
bcb_cross = lambda i, j: tf.Bond.create(pot_cross, i, j)

uc_sqcross = tf.lattice.unitcell(N=4,
                                 types=[mtl_type] * 4,
                                 a1=[a, 0, 0],
                                 a2=[0, a, 0],
                                 a3=[0, 0, 1],
                                 dimensions=2,
                                 position=[[0, 0, 0],
                                           [a/2, 0, 0],
                                           [0, a/2, 0],
                                           [a/2, a/2, 0]],
                                 bonds=[
                                     tf.lattice.BondRule(bcb_border, (0, 1), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (0, 2), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (0, 3), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (1, 2), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (1, 3), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (2, 3), (0, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (1, 0), (1, 0, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (1, 2), (1, 0, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (3, 0), (1, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (3, 2), (1, 0, 0)),
                                     tf.lattice.BondRule(bcb_border, (2, 0), (0, 1, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (2, 1), (0, 1, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (3, 0), (0, 1, 0)),
                                     tf.lattice.BondRule(bcb_border, (3, 1), (0, 1, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (3, 0), (1, 1, 0)),
                                     tf.lattice.BondRule(bcb_cross,  (2, 1), (-1, 1, 0))
                                 ])

parts = tf.lattice.create_lattice(uc_sqcross, n_lattice)

p_back, p_front = [], []
for i in range(2):
    p_front.extend([p[i] for p in parts[:, 0, :].flatten().tolist()])
for i in range(4):
    p_back.extend([p[i] for p in parts[:, n_lattice[1]-1, :].flatten().tolist()])

# Apply types

for p in p_front:
    p.become(boundary_type)
    BoundaryType.apply_boundary(p)
for p in p_back:
    p.become(loaded_type)

# Apply fracture criterion to material type

mtl_ids = [p.id for p in mtl_type.parts]
for p in mtl_type.items():
    for b in p.bonds:
        p1id, p2id = b.parts
        if p1id in mtl_ids and p2id in mtl_ids:
            b.dissociation_energy = 1E-2

# Apply force with damping

f_load = tf.CustomForce([0, 2, 0])
f_friction = tf.Force.friction(coef=1000.0)
tf.bind.force(f_load + f_friction, loaded_type)

tf.system.camera_view_top()

tf.show()
