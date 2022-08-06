#*******************************************************************************
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
#*******************************************************************************

import tissue_forge as tf

# dimensions of universe
dim = [30., 30., 30.]

tf.init(dim=dim, cutoff=3, dt=0.001)


class BType(tf.ParticleTypeSpec):
    radius = 0.25
    dynamics = tf.Overdamped
    mass = 15
    style = {"color": "skyblue"}


B = BType.get()


class CType(tf.ClusterTypeSpec):
    radius = 2.3

    types = [B]


C = CType.get()


def split(event: tf.event.ParticleTimeEvent):
    particle: tf.ClusterParticleHandle = event.targetParticle
    ptype: tf.ClusterParticleType = event.targetType

    print("split(" + str(ptype.name) + ")")
    axis = particle.position - yolk.position
    print("axis: " + str(axis))

    particle.split(axis=axis)


tf.event.on_particletime(ptype=C, invoke_method=split, period=0.2, selector="largest")


class YolkType(tf.ParticleTypeSpec):
    radius = 10
    mass = 1000000
    dynamics = tf.Overdamped
    frozen = True
    style = {"color": "gold"}


Yolk = YolkType.get()

total_height = 2 * Yolk.radius + 2 * C.radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2 - C.radius - 1

yolk = Yolk(position=tf.Universe.center + [0., 0., -yshift])

c = C(position=tf.Universe.center + [0., 0., cshift])

[c(B) for _ in range(4000)]

pb = tf.Potential.morse(d=1, a=6, r0=0.5, min=0.01, max=3, shifted=False)

pub = tf.Potential.morse(d=1, a=6, r0=0.5, min=0.01, max=3, shifted=False)

py = tf.Potential.morse(d=0.1, a=6, r0=0.0, min=-5, max=1.0)

rforce = tf.Force.random(mean=0, std=1)

tf.bind.force(rforce, B)
tf.bind.types(pb, C, B, bound=True)
tf.bind.types(pub, C, B, bound=False)
tf.bind.types(py, Yolk, B)

tf.run()
