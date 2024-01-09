# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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
Mixed-method model of cell migration over an ECM substrate
"""

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
import numpy as np

tf.init(dim=[9.0, 6.0, 6.0], cells=[9, 6, 6], bc={'x': 'noslip', 'y': 'noslip', 'z': 'noslip'})
tfv.init()

# 2D simulation
vtype: tf.ParticleType = tfv.MeshParticleType_get()
vtype.frozen_z = True


class CellType(tfv.SurfaceTypeSpec):
    """A 2D cell"""

    surface_area_lam = 1.0
    surface_area_val = 1.0

    edge_tension_lam = 5E-1
    edge_tension_order = 2

    @classmethod
    def hex_rad(cls):
        """Circumradius of the cell when its shape is a hexagon of target area"""
        return np.sqrt(cls.surface_area_val / (3 / 2 * np.sqrt(3)))


class ECMPart(tf.ParticleTypeSpec):
    """A segment of an ECM fiber"""

    radius = 0.01
    dynamics = tf.Overdamped
    mass = 1E2


class IntegPart(tf.ParticleTypeSpec):
    """An integrin on the surface of a cell"""

    radius = 0.01
    dynamics = tf.Overdamped
    mass = ECMPart.mass / 10
    style = {'color': 'green'}


type_cell = CellType.get()
type_ecmp = ECMPart.get()
type_integ = IntegPart.get()


class ECMFiber(tf.ClusterTypeSpec):
    """An ECM fiber"""

    types = [type_ecmp]


type_ecmf = ECMFiber.get()

# Fiber model

pot_fiber_intra_tens = tf.Potential.harmonic(
    k=type_ecmp.mass * 1E0,
    r0=ECMPart.radius,
    min=0,
    max=3 * ECMPart.radius
)
"""Tensile interaction potential between segments of an ECM fiber"""
pot_fiber_intra_tors = tf.Potential.harmonic_angle(
    k=1E-4,
    theta0=np.pi,
    min=np.pi/2,
    max=np.pi
)
"""Torsional interaction potential between segments of an ECM fiber"""
pot_fiber_inter = tf.Potential.morse(
    d=1E-3,
    a=12,
    r0=2*ECMPart.radius,
    min=0,
    max=3 * ECMPart.radius,
    shifted=False
)
"""Interaction potential between ECM fibers"""
tf.bind.types(pot_fiber_inter, type_ecmp, type_ecmp, bound=False)


def fiber_check(fiber_pos: tf.FVector3, fiber_dir: tf.FVector3, nparts: int, dom_buff: float) -> bool:
    """
    Check whether a fiber would be within the universe

    :param fiber_pos: position of initial fiber segment
    :param fiber_dir: vector from one segment to a neighbor
    :param nparts: number of segments
    :param dom_buff: buffer around universe
    :return: True if ok
    """

    fiber_pos_end = fiber_pos + fiber_dir * nparts
    is_ok = True
    for i in range(2):
        is_ok = is_ok and dom_buff < fiber_pos_end[i] < tf.Universe.dim[i] - dom_buff
    return is_ok


def fiber_make(position_init: tf.FVector3, direction: tf.FVector3, nparts: int, path_width: float, path_period: float):
    """
    Make an ECM fiber along a curved path

    :param position_init: position of initial fiber segment
    :param direction: vector from one segment to a neighbor
    :param nparts: number of segments
    :param path_width: width of path
    :param path_period: period of path center, which is a sine wave
    """

    # Generate positions of fiber segments along the path
    positions = []
    _direction = direction.normalized() * type_ecmp.radius
    making = False
    for i in range(nparts):
        pos = position_init + _direction * i - tf.FVector3(0, 0, 0.5 * ECMPart.radius * np.random.random())
        y0 = tf.Universe.center[1] + path_width / 2 * np.sin(2 * np.pi * pos[0] / path_period) - path_width / 2
        y1 = tf.Universe.center[1] + path_width / 2 * np.sin(2 * np.pi * pos[0] / path_period) + path_width / 2
        if y0 < pos[1] < y1:
            positions.append(pos)
            if not making:
                making = True
        elif making:
            break
    if len(positions) < 3:
        return

    # Create the fiber and its segments
    fiber = type_ecmf(positions[0])
    parts = []
    for i in range(len(positions)):
        parts.append(fiber(type_ecmp, positions[i]))
        if len(parts) > 1:
            tf.Bond.create(pot_fiber_intra_tens, parts[-2], parts[-1])
    for i in range(1, len(parts) - 1):
        tf.Angle.create(pot_fiber_intra_tors, parts[i-1], parts[i], parts[i+1])

    # Ensure fiber is correctly understood by Tissue Forge; this step will be unnecessary in a future release
    fiber.position = fiber.centroid


# Actomyosin model

act_len_f = 1.5 * CellType.hex_rad() - ECMPart.radius
"""Maximum length of a bonded interaction between an integrin and a vertex of the cell"""
act_dist_min = 0.1 * CellType.hex_rad()
"""Minimum distance from the leading edge where an integrin can be created"""
act_dist_max = 0.2 * CellType.hex_rad()
"""Maximum distance from the leading edge where an integrin can be created"""
act_pot_intra = tf.Potential.linear(
    k=-IntegPart.mass * 1E-3,
    min=0,
    max=act_len_f
)
"""Interaction potential between an integrin and a vertex of the cell"""
act_pot_ecm = tf.Potential.harmonic(
    k=IntegPart.mass,
    r0=(ECMPart.radius + IntegPart.radius) * 0.5,
    min=0,
    max=CellType.hex_rad()
)
"""Interaction potential between an integrin and the ECM"""


def integ_make(cell: tfv.SurfaceHandle):
    """Add an integrin"""

    # Find the leading edge
    verts_by_pos = {i: v.position[0] for i, v in enumerate(cell.vertices)}
    pos_x = list(verts_by_pos.values())
    pos_x.sort(reverse=True)
    pos_x0 = pos_x[0]
    pos_x1 = pos_x[1]
    vid0 = None
    vid1 = None
    for vid, vpos in verts_by_pos.items():
        if vpos == pos_x0:
            vid0 = vid
        elif vpos == pos_x1:
            vid1 = vid
    if vid0 is None or vid1 is None:
        return
    v0: tfv.VertexHandle = cell.vertices[vid0]
    v1: tfv.VertexHandle = cell.vertices[vid1]
    pos0 = v0.position
    pos1 = v1.position

    # Generate an integrin at an initial, randomly selected location
    pos_integ = (pos0 + pos1) * 0.5 + (pos1 - pos0) * 0.5 * (2 * np.random.random() - 1)
    pos_integ[0] = pos1[0] - (act_dist_min + (act_dist_max - act_dist_min) * np.random.random())
    p_integ = type_integ(pos_integ)
    p_integ.frozen_z = True

    # Find the nearest ECM segment and move the integrin to it
    ptypes = tf.ParticleTypeList()
    ptypes.insert(type_ecmp)
    neighbors = p_integ.neighbors(distance=0.1 * CellType.hex_rad(), types=ptypes)
    if len(neighbors) == 0:
        p_integ.destroy()
        return
    n_ecm = None
    dist_ecm = 1E6
    for n in neighbors:
        dist = n.relativePosition(p_integ.position).dot()
        if dist < dist_ecm:
            n_ecm = n
            dist_ecm = dist
    if n_ecm is None:
        p_integ.destroy()
        return
    p_integ.position = tf.FVector3(n_ecm.position[0], n_ecm.position[1], p_integ.position[2])

    # Create the integrin bonded interactions between the leading edge and ECM
    tf.Bond.create(act_pot_intra, p_integ, v0.particle())
    tf.Bond.create(act_pot_intra, p_integ, v1.particle())
    tf.Bond.create(act_pot_ecm, p_integ, n_ecm)


def integ_rem(p: tf.ParticleHandle):
    """Remove an integrin"""

    p.destroy()


def integ_do_model(cell: tfv.SurfaceHandle, integ_num_max: int):
    """
    Do the integin model.

    An integrin is bonded to a segment of ECM fiber and at least one of the vertices of the leading edge of the cell
    when the integrin is created.

    A bonded interaction between an integrin and a vertex of the leading edge is destroyed when its length
    exceeds a threshold.

    An integrin is destroyed when it has no bonded interactions with the leading edge,
    or when it is no longer in the area occupied by the cell.

    :param cell: the cell
    :param integ_num_max: maximum number of integrins
    """

    # Remove all integrin-cell interactions that exceed the specified maximum length,
    # and all integrins with no interactions with the cell
    act_len_f2 = act_len_f * act_len_f
    to_remove = []
    for p in type_integ:
        p: tf.ParticleHandle
        if not cell.contains(p.position):
            to_remove.append(p)
            continue
        to_remove_b = []
        p_bonds = p.bonds
        for b in p_bonds:
            b: tf.BondHandle
            pi, pj = b[0], b[1]
            po = pi if pj.id == p.id else pj
            if po.relativePosition(p.position).dot() > act_len_f2:
                to_remove_b.append(b)
        if len(to_remove_b) + 1 == len(p_bonds):
            to_remove.append(p)
        [b.destroy() for b in to_remove_b]
    [integ_rem(p) for p in to_remove]
    # Try to make as many integrins as the cell should have
    [integ_make(cell) for _ in range(len(type_integ.items()), integ_num_max)]


# Generate the substrate

ecm_pt = tf.FVector2(1, 2)
"""Starting point of rectangle where ECM fibers can be placed"""
ecm_size = tf.FVector2(7, 2)
"""Size of rectangle where ECM fibers can be placed"""
ecm_domain_buff = 1E-1
"""Distance into domains where ECM fibers cannot be placed"""
for _ in range(500):
    fiber_position = tf.FVector3(
        ecm_pt[0] + ecm_domain_buff + (ecm_size[0] - 2 * ecm_domain_buff) * np.random.random(),
        ecm_pt[1] + ecm_domain_buff + (ecm_size[1] - 2 * ecm_domain_buff) * np.random.random(),
        tf.Universe.center[2] - ECMPart.radius
    )
    fiber_n = np.random.randint(10, 100)
    fiber_direction = tf.FVector3()
    acceptable = False
    while not acceptable:
        rnd = np.random.random() * 2 - 1
        fiber_direction = tf.FVector3(rnd, np.sqrt(1 - rnd * rnd), 0) * type_ecmp.radius
        acceptable = fiber_check(fiber_position, fiber_direction, fiber_n, ecm_domain_buff)
    fiber_make(fiber_position, fiber_direction, fiber_n, 1.0, 2.0)

# Place a cell

_cell = type_cell.n_polygon(6,
                            tf.FVector3(3 * CellType.hex_rad(), tf.Universe.center[1], tf.Universe.center[2]),
                            CellType.hex_rad(), tf.FVector3(1, 0, 0), tf.FVector3(0, 1, 0))
"""Cell of the simulation"""

# Go!

tf.event.on_time(period=tf.Universe.dt, invoke_method=lambda e: integ_do_model(_cell, 100))

tf.system.camera_view_bottom()
tf.system.camera_zoom_to(-10)

tf.show()
