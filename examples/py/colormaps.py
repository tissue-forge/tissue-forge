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

tf.init()


class BeadType(tf.ParticleTypeSpec):
    species = ['S1']
    radius = 3

    style = {"colormap": {"species": "S1", "map": "rainbow"}}

    def __init__(self, pos, value):
        super().__init__(pos)
        self.species.S1 = value


Bead = BeadType.get()

# make a ring of of 50 particles
pts = [x * 4 + tf.Universe.center for x in tf.points(tf.PointsType.Ring.value, 100)]

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

Bead.i = 0


def keypress(e):
    names = tf.system.color_mapper_names()
    name = None

    if e.key_name == "n":
        Bead.i = (Bead.i + 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    elif e.key_name == "p":
        Bead.i = (Bead.i - 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    if name:
        Bead.style.colormap = name


tf.event.on_keypress(keypress)

# run the model
tf.show()
