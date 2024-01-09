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

tf.init(dim=[6.5, 6.5, 6.5], bc=tf.BoundaryTypeFlags.freeslip.value, windowless=True)


class AType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 5)}}


class ProducerType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['S1=200', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 5)}}


class ConsumerType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 5)}}


class SourceType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['$S1=5', 'S2', 'S3']  # make S1 a boundary species here.
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 5)}}


class SinkType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['$S1', 'S2', 'S3']  # make S1 a boundary species here.
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 5)}}


A = AType.get()
Producer = ProducerType.get()
Consumer = ConsumerType.get()
Source = SourceType.get()
Sink = SinkType.get()

# define fluxes between objects types
tf.Fluxes.flux(A, A, "S1", 5, 0.001)
tf.Fluxes.secrete(Producer, A, "S1", 5, 0)
tf.Fluxes.uptake(A, Consumer, "S1", 10, 500)
tf.Fluxes.secrete(Source, A, "S1", 5, 0)
tf.Fluxes.flux(Sink, A, "S1", 200, 0)

# make a lattice of objects
uc = tf.lattice.sc(0.25, A)
parts = tf.lattice.create_lattice(uc, [25, 25, 1])

# grap the left part, make it a producer
parts[0, 12, 0][0].become(Producer)

# grab the right part, make it a consumer
parts[24, 12, 0][0].become(Consumer)

# grab the middle, make it a source
parts[12, 12, 0][0].become(Source)

# make the corner parts sinks
parts[0,   0, 0][0].become(Sink)
parts[0,  24, 0][0].become(Sink)
parts[24,  0, 0][0].become(Sink)
parts[24, 24, 0][0].become(Sink)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
