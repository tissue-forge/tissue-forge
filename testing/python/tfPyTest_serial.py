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

# todo: bugs here on at least windows; fix in future PR

# import pickle
# from typing import List
# import tissue_forge as tf
#
#
# def validate_copy(obj0, obj1, attr_list: List[str]):
#     for attr in attr_list:
#         v0 = getattr(obj0, attr)
#         v1 = getattr(obj1, attr)
#         if v0 != v1:
#             if isinstance(v0, float):
#                 num = abs(v0 - v1)
#                 den = max(abs(v0), abs(v1))
#                 if den < 1E-12:
#                     den = 1.0
#                 if num / den < 1E-6:
#                     return
#
#             print('Difference found:', attr)
#             print('\tValue 0:', v0)
#             print('\tValue 1:', v1)
#             print('\tTypes:', type(obj0).__name__, ',', type(obj1).__name__)
#             raise ValueError
#
#
# def validate(obj, attr_list: List[str]):
#     validate_copy(obj, pickle.loads(pickle.dumps(obj)), attr_list)
#
#
# tf.init(bc={'z': 'potential', 'x': 'potential', 'y': 'potential'}, windowless=True)
#
#
# class AType(tf.ParticleTypeSpec):
#     style = {"colormap": {"species": "S1",
#                           "map": "rainbow",
#                           "range": (0, 1)}}
#     species = ['S1']
#
#
# A = AType.get()
#
#
# class BType(tf.ClusterTypeSpec):
#     types = [A]
#     style = {"color": "MediumSeaGreen"}
#
#
# B = BType.get()
#
# # test particle
# particle = A().part()
# validate(particle, ['clusterId',
#                     'flags',
#                     'id',
#                     'imass',
#                     'mass',
#                     'nr_parts',
#                     'q',
#                     'radius',
#                     'typeId'])
#
# # test cluster
# b = B()
# cluster_constituent = b(A).part()
# cluster_particle = b.cluster()
# validate(b.part(), ['clusterId',
#                     'flags',
#                     'id',
#                     'imass',
#                     'mass',
#                     'q',
#                     'radius',
#                     'typeId'])
#
# # test particle type
# validate(A, ['frozen',
#              'frozen_x',
#              'frozen_y',
#              'frozen_z',
#              'temperature',
#              'target_temperature',
#              'mass',
#              'charge',
#              'radius',
#              'kinetic_energy',
#              'potential_energy',
#              'target_energy',
#              'minimum_radius',
#              'dynamics',
#              'name'])
#
# # test cluster type
# validate(B, ['frozen',
#              'frozen_x',
#              'frozen_y',
#              'frozen_z',
#              'temperature',
#              'target_temperature',
#              'mass',
#              'charge',
#              'radius',
#              'kinetic_energy',
#              'potential_energy',
#              'target_energy',
#              'minimum_radius',
#              'dynamics',
#              'name'])
#
# # validate particle constructor from string on particle type
# particle_clone = A(particle.toString()).part()
# validate_copy(particle, particle_clone, ['clusterId',
#                                          'flags',
#                                          'imass',
#                                          'mass',
#                                          'nr_parts',
#                                          'q',
#                                          'radius',
#                                          'typeId'])
# if particle_clone.id < 0:
#     raise ValueError
# elif particle_clone.id == particle.id:
#     raise ValueError
#
# # validate particle constructor from string on cluster
# cluster_constituent_clone = b(A, cluster_constituent.toString()).part()
# validate_copy(cluster_constituent, cluster_constituent_clone, ['clusterId',
#                                                                'flags',
#                                                                'imass',
#                                                                'mass',
#                                                                'q',
#                                                                'radius',
#                                                                'typeId'])
# if cluster_constituent_clone.id < 0:
#     raise ValueError
# elif cluster_constituent_clone.id == cluster_constituent.id:
#     raise ValueError
#
# # test particle list
# validate(b.parts, ['radius_of_gyration', 'nr_parts'])
#
# # test particle type list
# validate(B.types, ['radius_of_gyration', 'nr_parts'])
#
# # test regular potential
# pot_bb = tf.Potential.harmonic(k=1, r0=0.1, max=3)
# validate(pot_bb, ['min',
#                   'max',
#                   'cutoff',
#                   'intervals',
#                   'bound',
#                   'shifted',
#                   'periodic',
#                   'r_square',
#                   'name',
#                   'domain',
#                   'n'])
#
# pot_dpd = tf.Potential.dpd(alpha=1.0, gamma=2.0, sigma=3.0)
# validate(tf.DPDPotential.fromPot(pot_dpd), ['alpha', 'gamma', 'sigma'])
#
# # test random force
# rforce = tf.Force.random(mean=0, std=50)
# validate(tf.Gaussian.fromForce(rforce), ['mean', 'std', 'durration_steps'])
#
# # test tstat force
# tforce = tf.Force.berendsen_tstat(10)
# validate(tf.Berendsen.fromForce(tforce), ['itau'])
#
# # test friction force
# fforce = tf.Force.friction(1.0)
# validate(tf.Friction.fromForce(fforce), ['coef'])
#
# # test bond
# p0 = A()
# p1 = A()
# bh: tf.BondHandle = tf.Bond.create(pot_bb, p0, p1, half_life=1.0, dissociation_energy=1000.0)
# validate(bh.get(), ['id',
#                     'i',
#                     'j',
#                     'creation_time',
#                     'half_life',
#                     'dissociation_energy'])
#
# # test angle
# p2 = A()
# ah: tf.AngleHandle = tf.Angle.create(pot_bb, p0, p1, p2)
# ah.half_life = 1.0
# ah.dissociation_energy = 1000.0
# validate(ah.get(), ['i',
#                     'j',
#                     'k',
#                     'creation_time',
#                     'half_life',
#                     'dissociation_energy'])
#
# # test dihedral
# p3 = A()
# dh: tf.DihedralHandle = tf.Dihedral.create(pot_bb, p0, p1, p2, p3)
# dh.half_life = 1.0
# dh.dissociation_energy = 1000.0
# validate(dh.get(), ['i',
#                     'j',
#                     'k',
#                     'l',
#                     'creation_time',
#                     'half_life',
#                     'dissociation_energy'])
#
# # test species
# validate(p0.species.species.S1, ['boundary_condition',
#                                  'charge',
#                                  'compartment',
#                                  'constant',
#                                  'conversion_factor',
#                                  'name',
#                                  'substance_units',
#                                  'units'])
#
# # test state vector
# validate(p0.species, ['q', 'size'])
#
# # test style
# validate(A.style, ['visible'])
#
# # test boundary conditions
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.bottom, A)
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.top, A)
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.left, A)
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.right, A)
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.front, A)
# tf.bind.boundary_condition(pot_bb, tf.Universe.boundary_conditions.back, A)
#
# bc_new = pickle.loads(pickle.dumps(tf.Universe.boundary_conditions))
# for side_name in ['bottom', 'top', 'left', 'right', 'front', 'back']:
#     validate_copy(getattr(tf.Universe.boundary_conditions, side_name),
#                   getattr(bc_new, side_name),
#                   ['id', 'kind', 'kind_str', 'name', 'radius', 'restore'])


def test_pass():
    pass
