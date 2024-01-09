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

import tissue_forge as tf


def plot_glj(r0, e=1, rr0=1, m=2, n=3):
    p = tf.Potential.glj(e=e, r0=rr0, m=m, n=n, min=0.01, max=10)

    p.plot(s=r0, min=0.01, max=10, potential=True, force=True, ymin=-1E5, ymax=1E2)


def plot_glj2(r0, e=30, m=4, n=2, tol=1E-4):

    p = tf.Potential.glj(e=e, m=m, n=n, max=10, tol=tol)

    p.plot(s=r0, min=0.5, max=2, potential=False, force=True)


def plot_glj3(r0, e=1, rr0=1, m=3, k=0):

    p = tf.Potential.glj(e=e, r0=rr0, m=m, k=k, min=0.1, max=1.5 * 20, tol=0.1)

    p.plot(s=r0, min=19, max=21, potential=False, force=True)


radius = 0.01

plot_glj(20)
plot_glj2(20)
plot_glj3(20)
