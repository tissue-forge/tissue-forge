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

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
tf.init(dim=dim, windowless=True, cells=[5, 5, 5], cutoff=1.0)

# create a potential representing a 12-6 Lennard-Jones potential
pot = tf.Potential.lennard_jones_12_6(0.275, 1.0, 9.5075e-06, 6.1545e-03, 1.0e-3)


# create a particle type
class ArgonType(tf.ParticleTypeSpec):
    radius = 0.1
    mass = 39.4


# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot, Argon, Argon)

# random cube
Argon.factory(nr_parts=2500)

# run the simulator
tf.step(100*tf.Universe.dt)


def test_pass():
    pass
