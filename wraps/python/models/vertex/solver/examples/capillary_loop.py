# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022, 2023, 2023 T.J. Sego and Tien Comlekoglu
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
Demonstrates importing a Blender mesh of a capillary loop and creating an executable vertex model mesh from it
"""

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
import os
import numpy as np

tf.init(dim=[15, 15, 15])
tfv.init()


class CellType(tfv.SurfaceTypeSpec):
    """A 2D cell type"""
    pass


ctype = CellType.get()

# Import the mesh and adjust to the Universe
mesh_import: tf.io.ThreeDFStructure = tf.io.fromFile3DF(os.path.join(os.path.dirname(__file__), 'CapillaryLoop.obj'))
mesh_import.translateTo(tf.Universe.center)
mesh_import.rotate(tf.FMatrix4.rotationX(np.pi / 2).rotation())

# Create the mesh surfaces and sew them
ctype(face_data=mesh_import.faces, safe_face_data=False)

# Run it!
tf.show()
