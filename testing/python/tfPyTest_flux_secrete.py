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

tf.init(dim=[6.5, 6.5, 6.5], bc=tf.BoundaryTypeFlags.freeslip.value, windowless=True)


class AType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}


class BType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}


A = AType.get()
B = BType.get()

tf.Fluxes.flux(A, A, "S1", 5, 0.005)

uc = tf.lattice.sc(0.25, A)

parts = tf.lattice.create_lattice(uc, [25, 25, 25])

# grap the particle at the top cornder
o = parts[24, 0, 24][0]

print("secreting pos: ", o.position.as_list())

# change type to B, since there is no flux rule between A and B
o.become(B)


def spew(event: tf.event.ParticleTimeEvent):

    print("spew...")

    # reset the value of the species
    # secrete consumes material...
    event.targetParticle.species.S1 = 500
    event.targetParticle.species.S1.secrete(250, distance=1)


tf.event.on_particletime(ptype=B, invoke_method=spew, period=0.3)
tf.step(0.35)


def test_pass():
    pass
