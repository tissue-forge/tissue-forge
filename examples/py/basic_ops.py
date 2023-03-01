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


class AType(tf.ParticleTypeSpec):
    pass


class BType(tf.ParticleTypeSpec):
    pass


A = AType.get()
B = BType.get()


class CType(tf.ClusterTypeSpec):

    types = [A]


class DType(tf.ClusterTypeSpec):

    types = [B]


C = CType.get()
D = DType.get()

print('Test particle types:')
print('\tA:', A)
print('\tB:', B)
print('\tC:', C)
print('\tD:', D)

# Make some test particles

a = A()
b = B()
c = C()
d = D()
c0 = c(A)
d0 = d(B)

print('Test particles:')
print('\ta :', a)
print('\tb :', b)
print('\tc :', c)
print('\td :', d)
print('\tc0:', c0)
print('\td0:', d0)

# Make some test bonds

pot_bond = tf.Potential.linear(k=1)
bond_ab = tf.Bond.create(pot_bond, a, b)
angle_abc0 = tf.Angle.create(pot_bond, a, b, c0)
dihedral_abc0d0 = tf.Dihedral.create(pot_bond, a, b, c0, d0)
print('Test bonds:')
print('\tbond    :', bond_ab)
print('\tangle   :', angle_abc0)
print('\tdihedral:', dihedral_abc0d0)

# Container-like operations

print('Container-like operations')
print('\tNo. A particles   :', len(A.parts))
print('\tNo. B particles   :', len(B.parts))
print('\tNo. c particles   :', len(c.parts))
print('\tNo. d particles   :', len(d.parts))
print('\ta in A particles? :', a in A)
print('\tb in A particles? :', b in A)
print('\tb in B particles? :', b in B)
print('\tA in C?           :', A in C)
print('\tB in C?           :', B in C)
print('\tA in D?           :', A in D)
print('\tB in D?           :', B in D)
print('\tc0 in c particles?:', c0 in c)
print('\tc0 in d particles?:', c0 in d)
print('\td0 in c particles?:', d0 in c)
print('\td0 in d particles?:', d0 in d)

print('\ta in bond? :', a in bond_ab)
print('\tb in bond? :', b in bond_ab)
print('\tc in bond? :', c in bond_ab)
print('\td in bond? :', d in bond_ab)
print('\tc0 in bond?:', c0 in bond_ab)
print('\td0 in bond?:', d0 in bond_ab)

print('\ta in angle? :', a in angle_abc0)
print('\tb in angle? :', b in angle_abc0)
print('\tc in angle? :', c in angle_abc0)
print('\td in angle? :', d in angle_abc0)
print('\tc0 in angle?:', c0 in angle_abc0)
print('\td0 in angle?:', d0 in angle_abc0)

print('\ta in dihedral? :', a in dihedral_abc0d0)
print('\tb in dihedral? :', b in dihedral_abc0d0)
print('\tc in dihedral? :', c in dihedral_abc0d0)
print('\td in dihedral? :', d in dihedral_abc0d0)
print('\tc0 in dihedral?:', c0 in dihedral_abc0d0)
print('\td0 in dihedral?:', d0 in dihedral_abc0d0)

print(f'\tParticles in A ({A})...')
for p in A:
    print('\t\tA has', p)
print(f'\tParticles in B ({B})...')
for p in B:
    print('\t\tB has', p)
print(f'\tParticles in c ({c})...')
for p in c:
    print('\t\tc has', p)
print(f'\tParticles in d ({d})...')
for p in d:
    print('\t\td has', p)
print(f'\tParticles in Universe...')
for p in tf.Universe.particles:
    print('\t\tUniverse has', p)
print(f'\tParticles in bond ({bond_ab})')
for p in bond_ab:
    print('\t\tbond has', p)
print(f'\tParticles in angle ({angle_abc0})')
for p in angle_abc0:
    print('\t\tangle has', p)
print(f'\tParticles in dihedral ({dihedral_abc0d0})')
for p in dihedral_abc0d0:
    print('\t\tdihedral has', p)

# Comparison operations

print('Comparison operations')
aa = tf.ParticleHandle(a.id)    # Make another handle to the first particle
print(f'\tAnother handle to particle {a.id}:', aa)
print('\tHandles are the same?:', a is aa)
print('\tHandles are equal?   :', a == aa)
print('\ta < b?:', a < b)
print('\ta > c?:', a > c)
print('\tA < B?:', A < B)
print('\tA > C?:', A > C)
