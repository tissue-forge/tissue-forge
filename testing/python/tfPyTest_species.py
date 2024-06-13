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
import numpy as n

tf.init(windowless=True)

print(tf.state.Species("S1"))

s1 = tf.state.Species("$S1")

s1 = tf.state.Species("const S1")

s1 = tf.state.Species("const $S1")

s1 = tf.state.Species("S1 = 1")

s1 = tf.state.Species("const S1 = 234234.5")


class AType(tf.ParticleTypeSpec):

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S1", "map": "rainbow"}}

    @staticmethod
    def on_register(ptype):
        def update(event: tf.event.ParticleTimeEvent):
            for p in ptype.items():
                p.species.S1 = (1 + n.sin(2. * tf.Universe.time)) / 2

        tf.event.on_particletime(ptype=ptype, invoke_method=update, period=0.01)


A = AType.get()


print("A.species:")
print(A.species)

print("making f")
a = A()

print("f.species")
print(a.species)

print("A.species.S1: ", A.species.S1)

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
