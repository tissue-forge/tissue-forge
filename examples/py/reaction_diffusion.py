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
import numpy as np

schnakenberg_string = '''
J1: -> U; g * (U * U * V - U + a);
J2: -> V; g * (- U * U * V + b);

a = 0.1;
b = 0.9;
g = 1.0;
U = 0;
V = 0;
'''

tf.init(dt=0.01,
        dim=[20, 10, 10],
        cells=[6, 3, 3],
        bc={'x': 'free_slip', 'y': 'free_slip', 'z': 'free_slip'})


class AType(tf.ParticleTypeSpec):
    radius = 0.1
    species = ['U', 'V']
    style = {"colormap": {"species": "U", "range": (0.0, 3.0)}}
    reactions = {'schnakenberg': {'antimony_string': schnakenberg_string,
                                  'name_map': [('U', 'U'), ('V', 'V')],
                                  'step_size': tf.Universe.dt
                                  }}


A = AType.get()

sphere_radius = 4.0
offset = tf.FVector3(sphere_radius + A.radius, 0, 0)
for pos in tf.icosphere(5, 0, 2 * np.pi)[0]:
    p = A(position=tf.Universe.center - offset + pos * sphere_radius)
    p.species.U.value = np.random.random()
    p.species.V.value = 0.0
for pos in tf.icosphere(5, 0, 2 * np.pi)[0]:
    p = A(position=tf.Universe.center + offset + pos * sphere_radius)
    p.species.U.value = np.random.random()
    p.species.V.value = 0.0
    p.species.reactions['schnakenberg'].integrator_name = "gillespie"

diff_rel = 20.0
diff_cf = 10.0 / diff_rel
flux_cutoff = 0.28
tf.Fluxes.flux(A, A, "U", diff_cf, cutoff=flux_cutoff)
tf.Fluxes.flux(A, A, "V", diff_cf * diff_rel, cutoff=flux_cutoff)

tf.system.camera_view_front()
tf.system.camera_zoom_to(-25)
tf.system.decorate_scene(False)
tf.show()
