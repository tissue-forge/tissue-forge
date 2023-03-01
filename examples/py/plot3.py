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
from matplotlib import pyplot as plt
from enum import Enum


class PotType(Enum):
    """Available potentials"""
    Harmonic = 1
    Power = 2
    Morse = 3
    MorseShifted = 4


pot_type = PotType.Harmonic     # Potential to plot
s = 1                           # Sum of imaginary particle radii
num_pts = 1000                  # Number of distances to plot

if pot_type == PotType.Harmonic:
    pot_1 = tf.Potential.harmonic(k=1, r0=0.5, min=0, max=2)
    pot_2 = tf.Potential.harmonic(k=1, r0=1.5, min=0, max=2)
    r_min = 0
    r_max = 2
elif pot_type == PotType.Power:
    pot_1 = tf.Potential.power(k=1, r0=0.5, alpha=3, min=0, max=2)
    pot_2 = tf.Potential.power(k=1, r0=1.5, alpha=3, min=0, max=2)
    r_min = 0
    r_max = 2
elif pot_type == PotType.Morse:
    pot_1 = tf.Potential.morse(d=1, a=6, r0=0.75, min=0, max=2, tol=0.0001, shifted=False)
    pot_2 = tf.Potential.morse(d=1, a=3, r0=1.00, min=0, max=2, tol=0.0001, shifted=False)
    r_min = 0
    r_max = 2
elif pot_type == PotType.MorseShifted:
    pot_1 = tf.Potential.morse(d=1, a=6, r0=0.25, min=-1.0, max=1.0, tol=0.0001)
    pot_2 = tf.Potential.morse(d=1, a=3, r0=0,    min=-1.0, max=1.0, tol=0.0001)
    r_min = 0
    r_max = 2
else:
    raise ValueError
pot = pot_1 + pot_2

r = [i / num_pts * (r_max - r_min) + r_min for i in range(1, num_pts)]

_, ax = plt.subplots(ncols=2)

ax[0].plot(r, [pot_1(rv, s) for rv in r], label='pot_1')
ax[0].plot(r, [pot_2(rv, s) for rv in r], label='pot_2')
ax[0].plot(r, [pot(rv, s) for rv in r],   label='pot_1 + pot_2')
ax[0].set_title('Potential')

ax[1].plot(r, [pot_1.force(rv, s) for rv in r], label='pot_1')
ax[1].plot(r, [pot_2.force(rv, s) for rv in r], label='pot_2')
ax[1].plot(r, [pot.force(rv, s) for rv in r],   label='pot_1 + pot_2')
ax[1].set_title('Force')

if pot_type in [PotType.Morse, PotType.MorseShifted]:
    print('Got Morse')
    ax[0].set_ylim([-2, 5])
    ax[1].set_ylim([-10, 2])

plt.legend()
plt.show()
