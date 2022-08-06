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

import tissue_forge as tf

# potential cutoff distance
cutoff = 0.5

count = 3000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff, windowless=True)


class YolkType(tf.ParticleTypeSpec):
    mass = 500000
    radius = 3


class CellType(tf.ParticleTypeSpec):
    mass = 5
    radius = 0.2
    target_temperature = 0
    dynamics = tf.Overdamped


Cell = CellType.get()
Yolk = YolkType.get()

pot_bs = tf.Potential.morse(d=1, a=6, min=0, r0=3.0, max=9, shifted=False)
pot_ss = tf.Potential.morse(d=0.1, a=9, min=0, r0=0.3, max=0.6, shifted=False)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_bs, Yolk, Cell)
tf.bind.types(pot_ss, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=0.05)

# bind it just like any other force
tf.bind.force(rforce, Cell)

yolk = Yolk(position=tf.Universe.center, velocity=[0., 0., 0.])

pts = [p * 0.5 * Yolk.radius + tf.Universe.center for p in tf.random_points(tf.PointsType.SolidSphere.value, count)]
pts = [p + [0, 0, 1.3 * Yolk.radius] for p in pts]

for p in pts:
    Cell(p)

# run the simulator interactive
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
