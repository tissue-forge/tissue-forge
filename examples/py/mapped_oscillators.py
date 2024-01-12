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

# Harmonic potential parameters
k = 1
r0 = 2
# Displacement at which a bond dissociates
rdis = 1


class PType(tf.ParticleTypeSpec):
    """Free particle of an oscillator"""
    radius = 0.25
    # Visualize force within the range of dissociation
    style = {'colormap': {'force': 'x', 'range': (-k * rdis * 2, k * rdis * 2)}}


class QType(PType):
    """Fixed particle of an oscillator"""
    frozen = True
    style = None


tf.init()

ptype = PType.get()
qtype = QType.get()

pot = tf.Potential.harmonic(k=k, r0=r0, min=0, max=r0 + 2 * rdis)
_pos0 = tf.Universe.center - tf.FVector3(r0, 0, 0)

dx, dy = 0.2, -1
tf.Bond.create(pot,
               qtype(_pos0 + tf.FVector3(0, dy, 0)),
               ptype(_pos0 + tf.FVector3(r0 + dx, dy, 0)),
               dissociation_energy=k * rdis * rdis)

dx, dy = 0.4, -0.5
tf.Bond.create(pot,
               qtype(_pos0 + tf.FVector3(0, dy, 0)),
               ptype(_pos0 + tf.FVector3(r0 + dx, dy, 0)),
               dissociation_energy=k * rdis * rdis)

dx, dy = 0.6, 0
tf.Bond.create(pot,
               qtype(_pos0 + tf.FVector3(0, dy, 0)),
               ptype(_pos0 + tf.FVector3(r0 + dx, dy, 0)),
               dissociation_energy=k * rdis * rdis)

dx, dy = 0.8, 0.5
tf.Bond.create(pot,
               qtype(_pos0 + tf.FVector3(0, dy, 0)),
               ptype(_pos0 + tf.FVector3(r0 + dx, dy, 0)),
               dissociation_energy=k * rdis * rdis)

dx, dy = 0.99, 1
tf.Bond.create(pot,
               qtype(_pos0 + tf.FVector3(0, dy, 0)),
               ptype(_pos0 + tf.FVector3(r0 + dx, dy, 0)),
               dissociation_energy=k * rdis * rdis)

# Visualize bond length with respect to equilibrium length
cmap = tf.rendering.ColorMapper()
cmap.set_map_bond_length_eq()
cmap.min_val = 0
cmap.max_val = rdis
bond_style = tf.rendering.Style(None, True, 0, cmap)
for b in tf.Universe.bonds:
    b.style = bond_style

tf.system.camera_view_top()
tf.system.decorate_scene(False)
tf.system.camera_zoom_to(-10)
tf.system.set_rendering_3d_bonds(True)
tf.run()
