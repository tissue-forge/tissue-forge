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
import numpy as np

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim)


class PType(tf.ParticleTypeSpec):
    radius = 2


P = PType.get()

# Get all available color names
color3_names = tf.color3_names()

# loop over arays for x/y coords
for x in np.arange(0., 20., 2.):
    for y in np.arange(0., 20., 2.):

        # create and register a new particle type based on PType
        # all we need is a unique name to create a type on the fly
        PP = P.newType(f'PP{tf.Universe.num_types}')
        PP.registerType()

        # instantiate that type
        PP = PP.get()
        # set a new style, since the current style is the same as PType
        PP.style = tf.rendering.Style(color3_names[np.random.randint(len(color3_names))])
        PP([x+1.5, y+1.5, 10.])

# run the simulator interactive
tf.run()
