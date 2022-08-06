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

"""
This example demonstrates basic usage of boundary 'reset' conditions.

In 'reset' conditions, the initial concentration of a species is restored for a particle that crosses a
restoring boundary.
"""

import tissue_forge as tf

# Initialize a domain like a tunnel, with flow along the x-direction
tf.init(dim=[20, 10, 10],
        cells=[5, 5, 5],
        cutoff=5,
        bc={'x': ('periodic', 'reset'), 'y': 'free_slip', 'z': 'free_slip'})

# Make a clip plane through the middle of the domain to better visualize transport
tf.rendering.ClipPlanes.create(tf.Universe.center, tf.FVector3(0, 1, 0))


class CarrierType(tf.ParticleTypeSpec):
    """A particle type to carry stuff"""

    radius = 0.5
    mass = 0.1
    species = ['S1']
    style = {'colormap': {'species': 'S1', range: (0, 1)}}


class SinkType(tf.ParticleTypeSpec):
    """A particle type to absorb stuff"""

    frozen = True
    radius = 1.0
    species = ['S1']
    style = {'colormap': {'species': 'S1', range: (0, 1)}}


carrier_type, sink_type = CarrierType.get(), SinkType.get()

# Carrier type begins carrying stuff
carrier_type.species.S1.initial_concentration = 1.0
# Sink type absorbs stuff by acting as a void
sink_type.species.S1.constant = True

# Carrier type like a fluid
dpd = tf.Potential.dpd(alpha=1, gamma=1, sigma=0.1, cutoff=3 * CarrierType.radius)
tf.bind.types(dpd, carrier_type, carrier_type)

# Sink type like a barrier
rep = cp = tf.Potential.harmonic(k=1000,
                                 r0=carrier_type.radius + 1.05 * sink_type.radius,
                                 min=carrier_type.radius + sink_type.radius,
                                 max=carrier_type.radius + 1.05 * sink_type.radius,
                                 tol=0.001)
tf.bind.types(rep, carrier_type, sink_type)

# Flow to the right
force = tf.CustomForce([0.01, 0, 0])
tf.bind.force(force, carrier_type)

# Diffusive transport and high flux into sink type
tf.Fluxes.flux(carrier_type, carrier_type, "S1", 0.001)
tf.Fluxes.flux(carrier_type, sink_type, "S1", 0.5)

# Put a sink at the center and carrier types randomly, though not in the sink
st = sink_type(tf.Universe.center)
[carrier_type() for _ in range(2000)]
to_destroy = []
for p in carrier_type.items():
    if p.relativePosition(tf.Universe.center).length() < (sink_type.radius + carrier_type.radius) * 1.1:
        to_destroy.append(p)
[p.destroy() for p in to_destroy]

# Run it!
tf.run()
