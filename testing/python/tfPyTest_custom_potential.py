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
Demonstrates how to simulate 1D particles in a custom pseudo-Gaussian potential well
"""
import tissue_forge as tf
from math import exp, factorial
import sys

tf.init(cutoff=5, bc={'x': 'noslip'}, windowless=True)


class WellType(tf.ParticleTypeSpec):
    frozen = True
    style = {'visible': False}


class SmallType(tf.ParticleTypeSpec):
    radius = 0.1


well_type, small_type = WellType.get(), SmallType.get()
small_type.frozen_y = True

# Build functions for custom potential
lam, mu, s = -0.5, 1.0, 3


def He(r, n):
    """nth Hermite polynomial evaluated at r"""
    if n == 0:
        return 1.0
    elif n == 1:
        return r
    return r * He(r, n-1) - (n-1) * He(r, n-2)


def dgdr(r, n):
    """Utility function for simplifying potential calculations"""
    r = max(r, sys.float_info.min)
    result = 0.0
    for k in range(1, s+1):
        if 2*k - n >= 0:
            result += factorial(2*k) / factorial(2*k - n) * (lam + k) * mu ** k / factorial(k) * r ** (2*k)
    return result / r ** n


def f_n(r: float, n: int):
    """nth derivative of potential function evaluated at r"""
    u_n = lambda r, n: (-1) ** n * He(r, n) * lam * exp(-mu * r ** 2.0)
    w_n = 0.0
    for j in range(0, n+1):
        w_n += factorial(n) / factorial(j) / factorial(n-j) * dgdr(r, j) * u_n(r, n-j)
    return 10.0 * (u_n(r, n) + w_n / lam)


pot_c = tf.Potential.custom(min=0, max=5, f=lambda r: f_n(r, 0), fp=lambda r: f_n(r, 1), f6p=lambda r: f_n(r, 6))
pot_c.name = "Pseudo-Gaussian"
# pot_c.plot(min=0, max=5, potential=True, force=False)
tf.bind.types(p=pot_c, a=well_type, b=small_type)

# Create particles
well_type(position=tf.Universe.center, velocity=tf.FVector3(0))
for i in range(20):
    small_type(position=tf.FVector3((i+1)/21 * tf.Universe.dim[0], tf.Universe.center[1], tf.Universe.center[2]),
               velocity=tf.FVector3(0))

tf.step(100*tf.Universe.dt)


def test_pass():
    pass
