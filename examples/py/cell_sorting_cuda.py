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

"""
Derived from cell_sorting.py, demonstrating runtime-control of GPU acceleration with CUDA.

A callback is implemented such that every time the key "S" is pressed on the keyboard,
Tissue Forge will test and report CPU vs. GPU performance.

Note that this demo will not run for Tissue Forge installations that do not have GPU support enabled.
"""
import tissue_forge as tf
import numpy as np
from time import time_ns

# Test for GPU support
if not tf.has_cuda:
    raise EnvironmentError("This installation of Tissue Forge was not installed with CUDA support.")

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
tf.init(dim=dim, cutoff=cutoff)


class AType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'red'}


A = AType.get()


class BType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'blue'}


B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = tf.Potential.morse(d=3, a=5, min=-0.8, max=2)
pot_bb = tf.Potential.morse(d=3, a=5, min=-0.8, max=2)
pot_ab = tf.Potential.morse(d=0.3, a=5, min=-0.8, max=2)


# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_aa, A, A)
tf.bind.types(pot_bb, B, B)
tf.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=50)

# bind it just like any other force
tf.bind.force(rforce, A)
tf.bind.force(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count, 3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count, 3)) * 15 + 2.5:
    B(p)


# Configure GPU acceleration
cuda_config_engine: tf.cuda.EngineConfig = tf.Simulator.cuda_config.engine
cuda_config_engine.set_threads(numThreads=int(32))


# Implement callback: when "S" is pressed on the keyboard, run some steps on and off the GPU and compare performance
def benchmark(e: tf.event.KeyEvent):
    if e.key_name != "s":
        return

    test_time = 100 * tf.Universe.dt

    print('**************')
    print(' Benchmarking ')
    print('**************')
    print('Sending engine to GPU...', cuda_config_engine.to_device())
    tgpu_i = time_ns()
    tf.step(test_time)
    tgpu_f = time_ns() - tgpu_i
    print('Returning engine from GPU...', cuda_config_engine.from_device())
    print('Execution time (GPU):', tgpu_f / 1E6, 'ms')

    tcpu_i = time_ns()
    tf.step(test_time)
    tcpu_f = time_ns() - tcpu_i
    print('Execution time (CPU):', tcpu_f / 1E6, 'ms')

    print('Measured speedup:', tcpu_f / tgpu_f if tgpu_f > 0 else 0)

    tf.Simulator.redraw()


tf.event.on_keypress(invoke_method=benchmark)


# run the simulator
tf.show()
