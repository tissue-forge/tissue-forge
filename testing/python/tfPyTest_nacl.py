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

tf.init(windowless=True)


class NaType(tf.ParticleTypeSpec):
    radius = 0.4
    style = {"color": "orange"}


class ClType(tf.ParticleTypeSpec):
    radius = 0.25
    style = {"color": "spablue"}


Na, Cl = NaType.get(), ClType.get()

uc = tf.lattice.bcc(0.9, [Na, Cl])

tf.lattice.create_lattice(uc, [10, 10, 10])


def test_pass():
    pass
