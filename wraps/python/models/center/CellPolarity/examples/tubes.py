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

"""
Reproduces cylindrical symmetry demonstrated in Figure 4 of

    Nissen, Silas Boye, et al.
    "Theoretical tool bridging cell polarities with development of robust morphologies."
    Elife 7 (2018): e38407.
"""

import tissue_forge as tf
from tissue_forge.models.center import cell_polarity
import numpy as np

tf.init(dim=[10., 10., 10.], dt=1.0)


class PolarType(tf.ParticleTypeSpec):
    dynamics = tf.Overdamped
    radius = 0.25


polar_type = PolarType.get()
cell_polarity.load()
cell_polarity.setArrowScale(0.25)
cell_polarity.setArrowLength(polar_type.radius)
cell_polarity.registerType(pType=polar_type)

f_random = tf.Force.random(1E-3, 0)
tf.bind.force(f_random, polar_type)

pot_contact = tf.Potential.morse(d=5E-4, a=5, r0=2*PolarType.radius, min=0, max=2*PolarType.radius, shifted=False)
pot_polar = cell_polarity.createContactPotential(cutoff=2.5*polar_type.radius, mag=2.5E-3, rate=0,
                                                 distanceCoeff=10.0*polar_type.radius, couplingFlat=1.0)
tf.bind.types(pot_contact + pot_polar, polar_type, polar_type)

for pos in tf.random_points(tf.PointsType.SolidSphere.value, 500, 5.0):
    p = polar_type(position=pos + tf.Universe.center, velocity=tf.fVector3(0.0))
    cell_polarity.registerParticle(p)
    # Assign cylindrical polarity orientations
    pos_rel = p.position - tf.Universe.center
    ang = np.arctan2(pos_rel[1], pos_rel[0])
    pvec_ab = tf.fVector3(np.cos(ang), np.sin(ang), 0.0)
    pvec_pcp = tf.fVector3(np.cos(ang+np.pi/2), np.sin(ang+np.pi/2), 0.0)

    cell_polarity.setVectorAB(p.id, pvec_ab, init=True)
    cell_polarity.setVectorPCP(p.id, pvec_pcp, init=True)

tf.show()
