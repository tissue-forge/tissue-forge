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

"""
This example demonstrates making force magnitudes depend on concentrations
"""

import tissue_forge as tf

tf.init(dim=[6.5, 6.5, 6.5], bc=tf.BoundaryTypeFlags.freeslip.value)


class AType(tf.ParticleTypeSpec):
    """A particle type carrying a species"""
    radius = 0.1
    species = ['S1']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}
    dynamics = tf.Overdamped


class BType(AType):
    """A particle type that acts as a constant source"""

    dynamics = tf.Newtonian

    @classmethod
    def get(cls):
        result = super().get()
        result.species.S1.constant = True
        return result


A, B = AType.get(), BType.get()

# Particles are randomly perturbed with increasing S1
force = tf.Force.random(0.1, 1.0)
tf.bind.force(force, A, 'S1')
tf.bind.force(force, B, 'S1')

# S1 diffuses between particles
tf.Fluxes.flux(A, A, "S1", 1)
tf.Fluxes.flux(A, B, "S1", 1)

# Make a lattice of stationary particles
uc = tf.lattice.sc(0.25, A)
parts = tf.lattice.create_lattice(uc, [25, 25, 25])

# Grab a particle to act as a constant source
o = parts[24, 0, 24][0]
o.become(B)
o.species.S1 = 10.0

tf.run()
