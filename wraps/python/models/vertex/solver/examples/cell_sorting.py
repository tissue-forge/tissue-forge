# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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
Reproduces cell sorting in

    Osborne, James M., et al.
    "Comparing individual-based approaches to modelling
    the self-organization of multicellular tissues."
    PLoS computational biology 13.2 (2017): e1005387.

"""

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
import numpy as np

# From Table 1
osborne_dt = 0.005
osborne_surface_area_val = 1.0
osborne_surface_area_lam = 50.0
osborne_adh_cell_homo = 1.0
osborne_adh_typea_bndr = 10.0
osborne_adh_cell_hetr = 2.0
osborne_adh_typeb_bndr = 20.0
osborne_perimeter_val = 2.0 * np.sqrt(np.pi)
osborne_perimeter_lam = 1.0
merge_dist = 0.1
noise_mag = 0.1

# Derived from Table 1
hex_rad = np.sqrt(osborne_surface_area_val / (3 / 2 * np.sqrt(3)))
edge_tension_lam_typea = osborne_adh_typea_bndr
edge_tension_lam_typeb = osborne_adh_typeb_bndr
adh_types_aa = 2 * (osborne_adh_cell_homo - edge_tension_lam_typea)
adh_types_bb = 2 * (osborne_adh_cell_homo - edge_tension_lam_typeb)
adh_types_ab = 2 * osborne_adh_cell_hetr - (edge_tension_lam_typea + edge_tension_lam_typeb)
split_dist = merge_dist * 2.0

tf.init(dt=osborne_dt / 2,
        dim=[25, 30, 5],
        cells=[5, 6, 1],
        bc={'x': 'noslip', 'y': 'noslip', 'z': 'noslip'})
tfv.init()

# 2D simulation
vtype = tfv.MeshParticleType_get()
vtype.frozen_z = True


# Declare cell types


class CellTypeA(tfv.SurfaceTypeSpec):
    """A cell type"""

    surface_area_val = osborne_surface_area_val
    surface_area_lam = osborne_surface_area_lam

    edge_tension_lam = edge_tension_lam_typea
    edge_tension_order = 1

    perimeter_val = osborne_perimeter_val
    perimeter_lam = osborne_perimeter_lam

    adhesion = {'CellTypeA': adh_types_aa,
                'CellTypeB': adh_types_ab}


class CellTypeB(CellTypeA):
    """Another cell type"""

    edge_tension_lam = edge_tension_lam_typeb

    adhesion = {'CellTypeA': adh_types_ab,
                'CellTypeB': adh_types_bb}


typea = CellTypeA.get()
typeb = CellTypeB.get()
tfv.SurfaceTypeSpec.bind_adhesion([CellTypeA, CellTypeB])

# Implement noise

rforce = tf.Force.random(std=np.sqrt(2 * noise_mag / tf.Universe.dt), mean=0, duration=tf.Universe.dt * 0.9)
tf.bind.force(rforce, vtype)

# Create the cells and randomly assign types

surfs = tfv.create_hex2d_mesh(typea, tf.FVector3(6 * hex_rad, 8 * hex_rad, tf.Universe.center[2]), 20, 20, hex_rad)
surfs_list = []
for si in surfs:
    surfs_list.extend(list(si))
np.random.shuffle(surfs_list)
for i in range(int(len(surfs_list) / 2)):
    surfs_list[i].become(typeb)

# Adjust quality operations

mesh: tfv.Mesh = tfv.MeshSolver.get_mesh()
quality: tfv.Quality = mesh.quality
quality.vertex_merge_distance = merge_dist
quality.edge_split_distance = split_dist

# Run it!

tf.system.camera_view_top()
tf.system.camera_zoom_to(-40)
tf.show()
