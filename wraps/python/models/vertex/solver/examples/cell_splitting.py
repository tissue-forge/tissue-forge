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
Two-dimensional cell splitting. A random cell splits along a random orientation when the "s" key is pressed
"""

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
import numpy as np

tf.init()
tfv.init()


class CellType(tfv.SurfaceTypeSpec):
    """A 2D cell type"""

    surface_area_val = 3 / 8 * np.sqrt(3)
    surface_area_lam = 1.0
    edge_tension_lam = 0.1

    @classmethod
    def hex_rad(cls):
        """Circumradius of the cell when its shape is a hexagon of target area"""
        return np.sqrt(cls.surface_area_val / (3 / 2 * np.sqrt(3)))


stype = CellType.get()

# 2D simulation with visible vertices
mesh_particle: tf.ParticleType = tfv.MeshParticleType_get()
mesh_particle.frozen_z = True
mesh_particle.radius = 0.01
vertex_style: tf.rendering.Style = mesh_particle.style
vertex_style.visible = True
vertex_style.setColor('black')

# Make a cell
stype.n_polygon(6, tf.Universe.center, CellType.hex_rad(), tf.FVector3(1, 0, 0), tf.FVector3(0, 1, 0))


def do_split():
    """Implements splitting along a random orientation in a randomly selected cell"""
    surfaces = stype.instances
    split_s = None
    while split_s is None:
        split_s = surfaces[np.random.randint(0, len(surfaces))]
    ang = np.random.random() * 2 * np.pi
    split_s.split(split_s.centroid, tf.FVector3(np.sin(ang), np.cos(ang), 0))


def handle_keys(e: tf.event.KeyEvent):
    """Implements a keyboard callback to do splitting"""
    if e.key_name == 's':
        do_split()


tf.event.on_keypress(handle_keys)

# Run it!
tf.show()
