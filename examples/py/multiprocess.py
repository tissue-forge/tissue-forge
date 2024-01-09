# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
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
This example demonstrates how Tissue Forge objects can be serialized for use in multiprocessing applications.

Special care must be taken to account for that deserialized Tissue Forge objects are copies of their original
object, and that the Tissue Forge engine is not available in separate processes. As such, calls to methods that
require the engine in a spawned process will fail.
"""
import tissue_forge as tf
from multiprocessing import Pool


def calc_energy_diff(bond: tf.Bond):
    # This call will fail in a spawned process, since a handle requires the engine
    # bh = tf.BondHandle(bond.id)

    # This call is ok in a spawned process, since accessed members are just values
    result = bond.dissociation_energy - bond.potential_energy
    return result


# Protect the entry point using __main__ for safe multiprocessing
if __name__ == '__main__':
    tf.init()
    # Get the default particle type
    ptype = tf.ParticleTypeSpec.get()
    # Construct some bonds
    pot = tf.Potential.harmonic(k=1.0, r0=1.0)
    [tf.Bond.create(potential=pot, i=ptype(), j=ptype(), dissociation_energy=10.0) for _ in range(1000)]
    # Do a step
    tf.step()
    # Do calculations in 8 processes
    with Pool(8) as p:
        diff = p.map(calc_energy_diff, [bh.get() for bh in tf.Universe.bonds])
