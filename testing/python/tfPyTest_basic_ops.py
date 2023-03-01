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

tf.init()

# Make some test particle types
print('Test particle types:')


class AType(tf.ParticleTypeSpec):
    pass


class BType(tf.ParticleTypeSpec):
    pass


A = AType.get()
print('\tA:', A)

B = BType.get()
print('\tB:', B)


class CType(tf.ClusterTypeSpec):

    types = [A]


class DType(tf.ClusterTypeSpec):

    types = [B]


C = CType.get()
print('\tC:', C)

D = DType.get()
print('\tD:', D)

# Make some test particles
print('Test particles:')

a = A()
print('\ta :', a)

b = B()
print('\tb :', b)

c = C()
print('\tc :', c)

d = D()
print('\td :', d)

c0 = c(A)
print('\tc0:', c0)

d0 = d(B)
print('\td0:', d0)



# Make some test bonds
print('Test bonds:')

pot_bond = tf.Potential.linear(k=1)
bond_ab = tf.Bond.create(pot_bond, a, b)
print('\tbond    :', bond_ab)
angle_abc0 = tf.Angle.create(pot_bond, a, b, c0)
print('\tangle   :', angle_abc0)
dihedral_abc0d0 = tf.Dihedral.create(pot_bond, a, b, c0, d0)
print('\tdihedral:', dihedral_abc0d0)


def check_fnc(op, res):
    if op != res:
        raise RuntimeError('Failed!', op, res)


# Container-like operations

print('Container-like operations')

check_fnc(len(A.parts), 2)
check_fnc(len(B.parts), 2)
check_fnc(len(c.parts), 1)
check_fnc(len(d.parts), 1)
check_fnc(a in A, True)
check_fnc(b in A, False)
check_fnc(b in B, True)
check_fnc(A in C, True)
check_fnc(B in C, False)
check_fnc(A in D, False)
check_fnc(B in D, True)
check_fnc(c0 in c, True)
check_fnc(c0 in d, False)
check_fnc(d0 in c, False)
check_fnc(d0 in d, True)

check_fnc(a in bond_ab, True)
check_fnc(b in bond_ab, True)
check_fnc(c in bond_ab, False)
check_fnc(d in bond_ab, False)
check_fnc(c0 in bond_ab, False)
check_fnc(d0 in bond_ab, False)

check_fnc(a in angle_abc0, True)
check_fnc(b in angle_abc0, True)
check_fnc(c in angle_abc0, False)
check_fnc(d in angle_abc0, False)
check_fnc(c0 in angle_abc0, True)
check_fnc(d0 in angle_abc0, False)

check_fnc(a in dihedral_abc0d0, True)
check_fnc(b in dihedral_abc0d0, True)
check_fnc(c in dihedral_abc0d0, False)
check_fnc(d in dihedral_abc0d0, False)
check_fnc(c0 in dihedral_abc0d0, True)
check_fnc(d0 in dihedral_abc0d0, True)

check_fnc(len([p for p in A]), 2)
check_fnc(len([p for p in B]), 2)
check_fnc(len([p for p in c]), 1)
check_fnc(len([p for p in d]), 1)
check_fnc(len([p for p in tf.Universe.particles]), 6)
check_fnc(len([p for p in bond_ab]), 2)
check_fnc(len([p for p in angle_abc0]), 3)
check_fnc(len([p for p in dihedral_abc0d0]), 4)

print('\tPassed!')


# Comparison operations

print('Comparison operations')

aa = tf.ParticleHandle(a.id)    # Make another handle to the first particle
print(f'\tAnother handle to particle {a.id}:', aa)
check_fnc(a is aa, False)
check_fnc(a == aa, True)
check_fnc(a < b, True)
check_fnc(a > c, False)
check_fnc(A < B, True)
check_fnc(A > C, False)

print('\tPassed!')


def test_pass():
    pass
