.. _cell_polarity:

.. py:currentmodule:: tissue_forge.models.center.cell_polarity

Cell Polarity
^^^^^^^^^^^^^^

The cell polarity model module implements an adaptation of cell polarity center
model described in :cite:`Nielsen:2020`. Each polarized cellular particle has
two state vectors that describe their apicobasal (AB) and planar cell
polarity (PCP). Anisotropic adhesion between particles can then be modeled in terms
of the state vectors of each pair of interacting particles. The module also implements
state vector dynamics in terms of neighborhood particle interactions and force
generation due to particle polarity. For documentation of the module API, see the
:ref:`Cell Polarity API Documentation <api_cell_polarity>`.

Each :math:`i\mathrm{th}` polarized cellular particle has an AB vector :math:`\mathbf{p}_{i}`
and PCP vector :math:`\mathbf{q}_{i}`, where each vector has a magnitude of one. State vector
dynamics and anisotropic adhesion for each :math:`i\mathrm{th}` particle are governed by a
potential :math:`V_{i}`, which is a summation of pair-potentials :math:`V_{ij}` with
each :math:`j\mathrm{th}` interacting particle,

.. math::

    V_{i} = \sum_{j} V_{ij}.

A corresponding force :math:`\mathbf{f}_{i}` and rate equations for the state vectors
naturally follow,

.. math::

    \mathbf{f}_{i} &= \gamma_{fp} \mathbf{p}_{i} + \gamma_{fq} \mathbf{q}_{i} - \gamma_{f} \frac{\partial V_{i}}{\partial \mathbf{r}_{i}}, \\
    \frac{\partial \mathbf{p}_{i}}{\partial t} &= - \gamma_{p} \frac{\partial V_{i}}{\partial \mathbf{p}_{i}}, \\
    \frac{\partial \mathbf{q}_{i}}{\partial t} &= - \gamma_{q} \frac{\partial V_{i}}{\partial \mathbf{q}_{i}},

where :math:`\gamma_{fp}`, :math:`\gamma_{fq}`, :math:`\gamma_{f}`, :math:`\gamma_{p}`
and :math:`\gamma_{q}` are parameters and :math:`\mathbf{f}_{i}` includes persistent
motion along polarity vectors.

The pair-potential describes apicobasal :math:`A_{ij}`, orthgonal :math:`H_{ij}` and
lateral :math:`L_{ij}` interactions between pairs of particles separated by a
relative position :math:`\mathbf{r}_{ij}` from the :math:`i\mathrm{th}` to
:math:`j\mathrm{th}` particle,

.. math::

    V_{ij} = - \left(a A_{ij} + h H_{ij} + l L_{ij} \right) e ^ { - \frac{\lVert \mathbf{r}_{ij} \rVert}{ \beta } },

where :math:`a`, :math:`h`, :math:`l` and :math:`\beta` are model parameters.
The apicobasal interaction :math:`A_{ij}` defines interactions based on the AB
vector of interacting particles and a shape parameter :math:`\alpha`,

.. math::

    A_{ij} = \left(\hat{\mathbf{r}}_{ij} \times \tilde{\mathbf{p}}_{i} \right) \cdot \left(\hat{\mathbf{r}}_{ij} \times \tilde{\mathbf{p}}_{j} \right),

where :math:`\hat{\mathbf{r}}_{ij}` is the normalized relative displacement
(`i.e.`, :math:`\mathbf{r}_{ij} = \lVert \mathbf{r}_{ij} \rVert \hat{\mathbf{r}}_{ij}`)
and shape effects occur for particles that isotropically or anisotropically wedge, while
particles that do not wedge tend to maintain a flat sheet,

.. math::

    \tilde{\mathbf{p}}_{i} =
    \begin{cases}
        \mathbf{p}_{i}  & \mbox{no wedging} \\
        \frac{\mathbf{p}_{i} - \alpha \hat{\mathbf{r}}_{ij}}{\lVert \mathbf{p}_{i} - \alpha \hat{\mathbf{r}}_{ij} \rVert}       & \mbox{isotropic wedging} \\
        \frac{\mathbf{p}_{i} - \alpha \mathbf{q}_{i}}{\lVert \mathbf{p}_{i} - \alpha \mathbf{q}_{i} \rVert}       & \mbox{anisotropic wedging}
    \end{cases}

The orthogonal interaction :math:`H_{ij}` makes AB and PCP vectors tend to
remain orthogonal,

.. math::

    H_{ij} = \left(\mathbf{p}_{i} \times \mathbf{q}_{i} \right) \cdot \left( \mathbf{p}_{j} \times \mathbf{q}_{j} \right).

The lateral interaction :math:`L_{ij}` defines interactions based on the PCP
vector of interacting particles such that particles tend to align laterally,

.. math::

    L_{ij} = \left(\hat{\mathbf{r}}_{ij} \times \mathbf{q}_{i} \right) \cdot \left(\hat{\mathbf{r}}_{ij} \times \mathbf{q}_{j} \right).

In Python, all functionality of the cell polarity model module can be accessed from the
:py:module:`cell_polarity` (``models::center::CellPolarity`` namespace in C++), ::

    from tissue_forge.models.center import cell_polarity

In C++, the module can be included when building from source with

.. code-block:: cpp

    #include <models/center/CellPolarity/tfCellPolarity.h>

Before using any functionality of the module, the module method :meth:`load`
must be called, ::

    cell_polarity.load()

Declarations of model processes are defined on the basis of particle type.
Each particle type that is polar must first be registered with the module, ::

    import tissue_forge as tf

    class MyTubeType(tf.ParticleTypeSpec):
        pass

    tube_type = MyTubeType.get()
    cell_polarity.registerType(pType=tube_type)

By default, state vectors of newly created particles are initialized randomly.
The initial state of each newly created particle can instead be declared during
registration of their particle type, ::

    class MySheetType(tf.ParticleType):
        pass

    sheet_type = MySheetType.get()
    cell_polarity.registerType(pType=sheet_type, initMode="value",
                               initPolarAB=tf.FVector3(0, 0, 1),
                               initPolarPCP=tf.FVector3(1, 0, 0))

When a polarized particle is created, it must also be registered with the cell
polarity module before continuing with a simulation. Likewise, before destroying a
polarized particle, the particle must be unregistered, ::

    p = sheet_type()
    cell_polarity.registerParticle(p)
    cell_polarity.unregisterParticle(p)
    p.destroy()

Polarity vectors can be accessed during simulation with special handling of setting
a state vector for a newly created particle, ::

    p = sheet_type()
    # Set initial AB vector using "init"
    cell_polarity.setVectorAB(p.id, tf.FVector3(1, 0, 0), init=True)
    # Get initial PCP vector
    pvec_pcp = cell_polarity.getVectorPCP(p.id)
    tf.step()
    # Overwrite PCP vector after first step
    cell_polarity.setVectorPCP(p.id, tf.FVector3(0, 1, 0))
    # Get AB vector after first step
    pvec_ab = cell_polarity.getVectorAB(p.id)

Cell polarity model processes can be added to a simulation like other processes in
Tissue Forge. The cell polarity model module defines a :ref:`potential <potentials>` for
specifying state vector dynamics and anisotropic adhesion, and a :ref:`force <forces>`
for specifying persistent motion, and each can be :ref:`bound <binding>` to particle
types in the typical way.

.. note::

    The cell polarity model potential only defines attraction. As such, it is most often
    useful when used in combination with another potential that defines a repulsive interaction.

A potential can be created and bound to pairs of particle types, ::

    pot_sheet = cell_polarity.createContactPotential(cutoff=2.5 * sheet_type.radius,
                                                     mag=2.0,
                                                     rate=0.4,
                                                     distanceCoeff=5.0 * sheet_type.radius,
                                                     couplingFlat=1.0)
    pot_tube = cell_polarity.createContactPotential(cutoff=3.0 * tube_type.radius,
                                                    mag=1.0,
                                                    rate=0.2,
                                                    distanceCoeff=10.0 * tube_type.radius,
                                                    couplingFlat=0.8,
                                                    couplingOrtho=0.1,
                                                    couplingLateral=0.1,
                                                    contactType="isotropic",
                                                    bendingCoeff=0.5)
    tf.bind.types(pot_sheet, sheet_type, sheet_type)
    tf.bind.types(pot_tube, tube_type, tube_type)

Likewise, a force can be created and bound to a particle type, ::

    force_polar = cell_polarity.createPersistentForce(sensAB=0.1, sensPCP=0.2)
    tf.bind.force(force_polar, sheet_type)

By default, Tissue Forge renders the state vectors of each polarized particle, where
AB vectors are shown as blue arrows, and PCP vectors are shown as green arrows.
The length and overall size of rendered arrows are also set to default values.
All of these details can be customized, including disabling of vector visualization,
on demand, ::

    # Rescale size of arrows to 25% of default
    cell_polarity.setArrowScale(0.25)
    # Set arrow length to the radius of the particles
    cell_polarity.setArrowLength(sheet_type.radius)
    # Set arrow colors
    cell_polarity.setArrowColors(colorAB="red", colorPCP="white")
    tf.step()
    # Disable vector visualization
    cell_polarity.setDrawVectors(False)

.. note::

    All cell polarity model data is automatically imported and exported during file operations,
    with the exception of rendering data. When importing a simulation state that includes the
    cell polarity model, all commands associated with rendering state vectors must be reissued
    after import and load to regenerate the same visualization.

.. note::

    The cell polarity model module currently does not support :ref:`GPU acceleration <cuda>`.
