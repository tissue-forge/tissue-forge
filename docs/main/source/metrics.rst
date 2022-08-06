.. _metrics:

.. py:currentmodule:: tissue_forge

Metrics and Derived Quantities
-------------------------------

Tissue Forge provides numerous methods to compute a range of derived
quantities. Some quantities are top-level metrics that depend on the entire
simulation volume, and others are localized to individual or groups of objects.

Pressure and Virial Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a system of :math:`N` particles in a volume :math:`V`, the surface tension
can be computed from the diagonal components of the pressure tensor
:math:`P_{\alpha,\alpha}(\alpha=x,y,z)`. The :math:`P_{xx}` components are

.. math::

   P_{\alpha,\beta} = \rho k T + \
       \frac{1}{V} \
       \left( \
       \sum^{N-1}_{i=1} \
       \sum^{N}_{j>i} \
       (\mathbf{r}_{ij})_{\alpha} \
       (\mathbf{f}_{ij})_{\beta} \
       \right),

where :math:`\rho` is the particle density, :math:`k` is the Boltzmann constant,
:math:`T` is the temperature, and :math:`\mathbf{r}_{ij}` and
:math:`\mathbf{f}_{ij}` are the relative position of, and force between,
the :math:`i\mathrm{th}` and :math:`j\mathrm{th}` particles, respectively.
Here :math:`\mathbf{f}_{ij}` is *only* due to inter-particle interactions
and excludes external forces. The pressure tensor is a measure of how
much *internal* force exists in the specified set of particles.

.. _virial:

A term in the definition of the pressure tensor is known as the `virial`
tensor, which is defined as

.. math::
    V_{\alpha,\beta} = \sum^{N-1}_{i=1} \
        \sum^{N}_{j>i} \
        (\mathbf{r}_{ij})_{\alpha} \
        (\mathbf{f}_{ij})_{\beta}.

The virial tensor represents half of the the product of the stress due to the net
force between pairs of particles and the distance between them. Since the volume
of a group of particles is not well defined, Tissue Forge provides the flexibility of
using different volume metrics for computing the virial tensor and corresponding
pressure tensor.

The pressure tensor for the entire simulation domain, or for a specific region,
can be calculated using the method :meth:`virial <Universe.virial>` on the
:ref:`universe <universe>`. ::

    import tissue_forge as tf
    ...
    virial_universe = tf.Universe.virial()
    virial_region = tf.Universe.virial(origin=[5.0, 6.0, 7.0], radius=2.0)

The viritual tensor about a particle can also be computed with the particle method
:meth:`virial <Universe.virial>` within a specific distance. ::

    class MyParticleType(tf.ParticleTypeSpec):
        pass
    my_particle_type = MyParticleType.get()
    my_particle = my_particle_type()
    virial = my_particle.virial(radius=3.0)

Centroid
^^^^^^^^^

The centroid of a system of :math:`N` particles :math:`\mathbf{C}`,
where each :math:`i\mathrm{th}` particle has position :math:`\mathbf{r}_i`,
is defined as

.. math::

   \mathbf{C} = \frac{1}{N} \sum_{i=1}^N \mathbf{r}_i,

The centroid of a cluster is avilable using the property
:attr:`centroid <ClusterParticleHandle.centroid>`.

Radius of Gyration
^^^^^^^^^^^^^^^^^^^

The radius of gyration is a measure of the dimensions of a group
(:py:attr:`Cluster`) of particles. The radius of gyration of a group of
:math:`N` particles is defined as

.. math:: 
   R_\mathrm{g}^2 \ \stackrel{\mathrm{def}}{=}\ 
   \frac{1}{N} \sum_{k=1}^{N} \left( \mathbf{r}_k - \mathbf{C}
   \right)^2 ,

where :math:`\mathbf{C}` is the centroid of the particles.
The radius of gyration for a cluster is available using the property
:attr:`radius_of_gyration <ClusterParticleHandle.radius_of_gyration>`.

Center of Mass
^^^^^^^^^^^^^^^

The center of mass of a system of :math:`N` particles :math:`\mathbf{R}`,
where each :math:`i\mathrm{th}` particle has mass :math:`m_i` and position
:math:`\mathbf{r}_i`, satisfies the condition

.. math::

   \sum_{i=1}^N m_i(\mathbf{r}_i - \mathbf{R}) = \mathbf{0} .

:math:`\mathbf{R}` is then defined as

.. math::

   \mathbf{R} = \frac{1}{M} \sum_{i=1}^N m_i \mathbf{r}_i,

where :math:`M` is the sum of the masses of all of the particles.
The center of mass of a cluster is available using the property
:attr:`center_of_mass <ClusterParticleHandle.center_of_mass>`.

Moment of Inertia
^^^^^^^^^^^^^^^^^^

For a system of :math:`N` particles, the moment of inertia tensor \mathbf{I}
is a symmetric tensor defined as

.. math::
   \mathbf{I} =
   \begin{bmatrix}
   I_{11} & I_{12} & I_{13} \\
   I_{21} & I_{22} & I_{23} \\
   I_{31} & I_{32} & I_{33}
   \end{bmatrix}

Its diagonal elements are defined as

.. math::

   \begin{align*}
   I_{xx} &\stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (y_{k}^{2}+z_{k}^{2}), \\
   I_{yy} &\stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (x_{k}^{2}+z_{k}^{2}), \\
   I_{zz} &\stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (x_{k}^{2}+y_{k}^{2})
   \end{align*} ,

and its off-diagonal elements are defined as

.. math::
   \begin{align*}
   I_{xy} &= I_{yx} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} x_{k} y_{k}, \\
   I_{xz} &= I_{zx} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} x_{k} z_{k}, \\
   I_{yz} &= I_{zy} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} y_{k} z_{k}
   \end{align*} .

Here :math:`m_{k}` is the mass of the :math:`k\mathrm{th}` particle, and
:math:`x_{k}`, :math:`y_{k}` and :math:`z_{k}` are its relative coordinates
with respect to the centroid of the cluster along the first, second and
third dimensions, respectively.
The moment of inertia tensor of a cluster is available using the property
:attr:`moment_of_inertia <ClusterParticleHandle.moment_of_inertia>`.
