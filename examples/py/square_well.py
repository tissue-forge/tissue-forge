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

# potential cutoff distance
cutoff = 8

receptor_count = 10000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff, cells=[4, 4, 4], threads=8)


class NucleusType(tf.ParticleTypeSpec):
    mass = 500000
    radius = 1


class ReceptorType(tf.ParticleTypeSpec):
    mass = 0.2
    radius = 0.05
    target_temperature = 1
    # dynamics = tf.Overdamped


Nucleus = NucleusType.get()
Receptor = ReceptorType.get()

# locations of initial receptor positions
receptor_pts = [p * 5 + tf.Universe.center for p in tf.random_points(tf.PointsType.SolidSphere.value, receptor_count)]

pot_nr = tf.Potential.well(k=15, n=3, r0=7)
pot_rr = tf.Potential.morse(d=1, a=6, min=0.01, max=1, r0=0.3, shifted=False)

# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_rr, Receptor, Receptor)
tf.bind.types(pot_nr, Nucleus, Receptor)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = tf.Force.random(mean=0, std=3)
vtstat = tf.Force.random(mean=0, std=5)

# bind it just like any other force
tf.bind.force(tstat, Receptor)


n = Nucleus(position=tf.Universe.center, velocity=[0., 0., 0.])

for p in receptor_pts:
    Receptor(p)

# run the simulator interactive
tf.run()
