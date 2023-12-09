.. _cleavage:

.. py:currentmodule:: tissue_forge

Splitting
----------

Tissue Forge supports modeling processes associated with a
:ref:`particle <creating_particles_and_types>` dividing into two particles,
called *splitting*. In the simplest case, a particle can spawn a new
particle in a mass- and volume-preserving split operation under the
assumption that the spawned (child) and spawning (parent) particles are
identical. During particle splitting, the parent and child particles are randomly
placed exactly in contact at the initial position of the parent particle, and
both have the same velocity as the parent before the split. The split operation on
a particle occurs when calling the :py:attr:`Particle` method
:py:meth:`split <ParticleHandle.split>`, which returns the child particle. ::

    import tissue_forge as tf
    class SplittingType(tf.ParticleTypeSpec):
        pass

    splitting_type = SplittingType.get()
    parent_particle = splitting_type()
    child_particle = parent_particle.split()

The :py:attr:`Particle` :py:meth:`split <ParticleHandle.split>` method can 
customize a number of details about particle splitting by passing certain 
arguments, 
including the direction of the split (random by default), 
relative size of the resulting particles (equal by default), 
allocation of any carried :ref:`species <species-label>` (equal by total species by default), 
and :ref:`type <particle_types>` of the resulting particles (equal to the split particle by default). 
For example, passing only a float tells Tissue Forge the ratio 
of the volume of the newly created particle to the volume of the particle 
before the split, ::

    # Split the particle again and produce a new particle with 
    # a volume equal to 25% of the volume of the parent particle before 
    # before the split
    smaller_particle = parent_particle.split(0.25)

In general, the following always hold true during particle splitting, 

* The two resulting particles are in contact. 
* The total mass, volume, and any species amounts of the two resulting particles are equal to those of the split particle just before the split. 
* The center of mass of the two resulting particles are the same as that of the split particle just before the split. 

For details on using various particle splitting features, refer to the
:ref:`Tissue Forge API Reference <api_reference>`.

Splitting Clusters
^^^^^^^^^^^^^^^^^^^

:ref:`Clusters <clusters-label>` introduce details of morphology and
constituent particles to the process of splitting. Tissue Forge provides support
for specifying a number of details concerning how, where and when a cluster
divides. The split operation on a cluster also occurs when calling
:py:meth:`split <ClusterParticleHandle.split>`, though the corresponding cluster
method supports a variable number of arguments that define the details of the split.
In general, cluster splitting occurs according to a *cleavage plane* that intersects
the cluster, where the constituent particles of the parent cluster before the split
are allocated to the parent and child clusters on either side of the intersecting plane.

In the simplest case, a cluster can be divided by randomly selecting a cleavage
plane at the center of mass of the cluster. Such a case is implemented by
calling :py:meth:`split <ClusterParticleHandle.split>` without arguments, as with a
particle, ::

    class MyClusterType(tf.ClusterParticleTypeSpec):
        types = [splitting_type]

    my_cluster_type = MyClusterType.get()
    my_cluster = my_cluster_type()
    my_cluster_d1 = my_cluster.split()

:py:meth:`split <ClusterParticleHandle.split>` accepts optional keyword arguments
``normal`` and ``point`` to define a cleavage plane. If only a normal vector is given,
:py:meth:`split <ClusterParticleHandle.split>` uses the center of mass of the cluster
as the point. For example, to split a cluster along the `x` axis, ::

    my_cluster_d2 = my_cluster.split(normal=[1., 0., 0.])

or to specify the full normal/point form, ::

    my_cluster_d3 = my_cluster.split(normal=[x, y, z], point=[px, py, pz])

:py:meth:`split <ClusterParticleHandle.split>` also supports splitting a cluster along
an *axis* at the center of mass of the cluster, where a random cleavage plane is generated
that contains the axis. This case can be implemented by using the optional keyword argument
``axis``. ::

    my_cluster_d4 = my_cluster.split(axis=[x, y, z])

:py:meth:`split <ClusterParticleHandle.split>` can also split the cluster by randomly
selecting half of the particles in a cluster and assigning them to a child cluster by using the
``random`` argument, ::

    my_cluster_d5 = my_cluster.split(random=True)
