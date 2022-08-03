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

# dimensions of universe
dim = [30., 30., 30.]

dist = 3.9

offset = 6

tf.init(dim=dim,
        cutoff=7,
        cells=[3, 3, 3],
        integrator=tf.EngineIntegratorTypes.forward_euler.value,
        dt=0.01,
        bc={'z': 'potential', 'x': 'potential', 'y': 'potential'})


class AType(tf.ParticleTypeSpec):
    radius = 1
    dynamics = tf.Newtonian
    mass = 2.5
    style = {"color": "MediumSeaGreen"}


class SphereType(tf.ParticleTypeSpec):
    radius = 3
    frozen = True
    style = {"color": "orange"}


A = AType.get()
Sphere = SphereType.get()

p = tf.Potential.morse(d=100, a=1, min=-3, max=4)

tf.bind.types(p, A, Sphere)

tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.bottom, A)
tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.top, A)
tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.left, A)
tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.right, A)
tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.front, A)
tf.bind.boundary_condition(p, tf.Universe.boundary_conditions.back, A)


# above the sphere
Sphere(tf.Universe.center + [5, 0, 0])
A(tf.Universe.center + [5, 0, Sphere.radius + dist])

# bottom of simulation
A([tf.Universe.center[0], tf.Universe.center[1], dist])

# top of simulation
A([tf.Universe.center[0], tf.Universe.center[1], dim[2] - dist])

# left of simulation
A([dist, tf.Universe.center[1] - offset, tf.Universe.center[2]])

# right of simulation
A([dim[0] - dist, tf.Universe.center[1] + offset, tf.Universe.center[2]])

# front of simulation
A([tf.Universe.center[0], dist, tf.Universe.center[2]])

# back of simulation
A([tf.Universe.center[0], dim[1] - dist, tf.Universe.center[2]])

tf.run()
