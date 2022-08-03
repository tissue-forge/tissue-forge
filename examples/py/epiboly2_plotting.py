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
cutoff = 3

# number of particles
count = 6000

# number of time points we avg things
avg_pts = 3

# dimensions of universe
dim = [50., 50., 100.]

# new simulator
tf.init(dim=dim,
        cutoff=cutoff,
        dt=0.001,
        max_distance=0.2,
        threads=8,
        cells=[5, 5, 5])

clump_radius = 8


class YolkType(tf.ParticleTypeSpec):
    mass = 500000
    radius = 20
    frozen = True


class CellType(tf.ParticleTypeSpec):
    mass = 10
    radius = 1.2
    target_temperature = 0
    dynamics = tf.Overdamped


Yolk = YolkType.get()
Cell = CellType.get()

total_height = 2 * Yolk.radius + 2 * clump_radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2 - 1.9 * clump_radius

pot_yc = tf.Potential.glj(e=500, r0=1, m=3, k=500, min=0.1, max=2 * Yolk.radius, tol=0.1)
pot_cc = tf.Potential.glj(e=50, r0=1, m=2, min=0.05, max=2.2 * Cell.radius)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_yc, Yolk, Cell)
tf.bind.types(pot_cc, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=100, duration=0.5)

# bind it just like any other force
tf.bind.force(rforce, Cell)

yolk = Yolk(position=tf.Universe.center + [0., 0., -yshift])

for p in tf.random_points(tf.PointsType.SolidSphere.value, count):
    pos = p * clump_radius + tf.Universe.center + [0., 0., cshift]
    Cell(position=pos)

# import sphericalplot as sp
#
# plt = sp.SphericalPlot(Cell.items(), yolk.position)
#
# tf.event.on_time(invoke_method=plt.update, period=0.01)

# run the simulator interactive
tf.run()
