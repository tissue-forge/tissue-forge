.. _cuda:

.. py:currentmodule:: tissue_forge

GPU Acceleration
=================

Tissue Forge supports modular, runtime-configurable GPU acceleration of a simulation using CUDA.
Computational features of Tissue Forge that support GPU-acceleration can be configured, offloaded to
a GPU, brought back to the CPU and reconfigured at any time during a simulation.
For Tissue Forge installations with enabled GPU acceleration, no computations are performed on a GPU by default.
Rather, GPU-supporting features of Tissue Forge must be explicitly configured and offloaded to a GPU
using their corresponding interactive interface.
This modular, configurable approach allows fine-grain control of computations to achieve maximum performance
for a given set of hardware *and* a particular simulation.

For example, suppose a simulation begins with a few hundred particles. Such a simulation would likely not
benefit from GPU acceleration (or even run slower on a GPU). However, suppose that over the course of the
simulation, hundreds of thousands of :ref:`particles are created <creating_particles_and_types>`.
At some point, this simulation will run faster on a GPU. Tissue Forge easily handles such a situation by
allowing the computations of the particle interactions to be offloaded to a GPU mid-execution of the
simulation (and brought back to the CPU, should the particle number significantly decrease).

Deployment on a GPU is best accomplished when running Tissue Forge in
:ref:`windowless mode <running_a_sim_windowless>`, since real-time rendering of interactive
Tissue Forge simulations also utilizes available GPUs.

.. note::

    Tissue Forge currently supports acceleration using a single GPU.
    Future releases will support deploying computations on multiple GPUs by computational feature.

Tissue Forge includes a flag :py:attr:`has_cuda` to check whether GPU acceleration is supported by the
installation (``hasCuda`` in C++), ::

    import tissue_forge as tf

    print(tf.has_cuda)  # True if GPU acceleration is installed; False otherwise

GPU-Accelerated Simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:attr:`Simulator` provides access to runtime control of GPU-accelerated simulation features.
Each GPU-accelerated simulation feature has its own runtime control interface for configuring and
deploying on a GPU. GPU runtime control of simulation modules can be accessed directly from
:py:attr:`Simulator`, ::

    cuda_config_sim: tf.cuda.SimulatorConfig = tf.Simulator.cuda_config

The returned :py:class:`cuda.SimulatorCUDAConfig` (``cuda::SimulatorConfig`` in C++) provides
convenient access to all current GPU-accelerated simulation features.

GPU-Accelerated Engine
^^^^^^^^^^^^^^^^^^^^^^^

Engine GPU acceleration is a GPU-accelerated simulation feature that offloads nonbonded potential
interactions, fluxes, particle sorting and space partitioning onto a GPU.
All runtime controls of engine GPU acceleration are available on :py:class:`cuda.EngineConfig`
(``cuda::EngineConfig`` in C++), which is an attribute with name ``engine``
on :py:class:`cuda.SimulatorConfig`, ::

    cuda_config_engine = tf.Simulator.cuda_config.engine  # Get engine cuda runtime interface

Engine GPU acceleration can be enabled, disabled and customized during simulation according to hardware
capabilities and simulation state, ::

    cuda_config_engine.set_blocks(numBlocks=64)               # Set number of blocks
    cuda_config_engine.set_threads(numThreads=32)             # Set number of threads per block
    cuda_config_engine.to_device()                            # Send engine to GPU
    # Simulation code here...
    if cuda_config_engine.on_device():                        # Ensure engine is on GPU
        cuda_config_engine.from_device()                      # Bring engine back from GPU

Setting a number of blocks specifies the maximum number of CUDA thread blocks that can be deployed
during a simulation step, which work on various engine tasks (*e.g.*, calculating interactions among
particles in a subspace of the simulation space).
Setting a number of threads per block specifies the number of threads launched per block to work on each
engine task.

Many Tissue Forge operations automatically update data when running on a GPU.
However, some operations (*e.g.*, :ref:`binding <binding>` a :py:attr:`Potential`)
requires manual refreshing of engine data for changes to be reflected when running on a GPU.
Engine GPU acceleration runtime control provides methods to explicitly tell Tissue Forge to
refresh data on a GPU at various levels of granularity, ::

    cuda_config_engine.refresh_potentials()           # Capture changes to potentials
    cuda_config_engine.refresh_fluxes()               # Capture changes to fluxes
    cuda_config_engine.refresh_boundary_conditions()  # Capture changes to boundary conditions
    cuda_config_engine.refresh()                      # Capture all changes

Refer to the :ref:`Tissue Forge API Reference <api_reference>` for which operations automatically update
engine data on a GPU.

.. note::

    It's not always clear what changes are automatically detected by Tissue Forge
    when running on a GPU. When in doubt, refresh the data! Performing a refresh comes with
    additional computational cost but must be performed only after all changes to simulation data
    have been made, and before the next simulation step is called.

GPU-Accelerated Bonds
^^^^^^^^^^^^^^^^^^^^^^
Bond GPU acceleration is a GPU-accelerated simulation feature that offloads
:ref:`bonded interactions <bonded_interactions>` onto a GPU.
All runtime controls of bond GPU acceleration are available on :py:class:`cuda.BondConfig`
(``cuda::BondConfig`` in C++), which is an attribute with name ``bonds``
on :py:class:`cuda.SimulatorConfig`, ::

    cuda_config_bonds = tf.Simulator.cuda_config.bonds    # Get bond cuda runtime interface

The bond GPU acceleration runtime control interface is very similar to that of engine GPU acceleration.
Bond GPU acceleration can be enabled, disabled and customized at any point in simulation, ::

    cuda_config_bonds.set_blocks(numBlocks=64)                # Set number of blocks
    cuda_config_bonds.set_threads(numThreads=32)              # Set number of threads per block
    cuda_config_bonds.to_device()                             # Send bonds to GPU
    # Simulation code here...
    if cuda_config_bonds.on_device():                         # Ensure bonds are on GPU
        cuda_config_bonds.from_device()                       # Bring bonds back from GPU

Setting a number of blocks specifies the maximum number of CUDA thread blocks that can be deployed
during a simulation step, which calculate pairwise forces due to each bond.
Setting a number of threads per block specifies the number of threads launched per block to work
force calculations.

Adding and destroying bonds both automatically update data while running on a GPU.
However, changes to bond properties (*e.g.*, half life) and bond potential
require manual refreshing of bond data for changes to be reflected when running on a GPU.
Bond GPU acceleration runtime control provides methods to explicitly tell Tissue Forge to
refresh data on a GPU at various levels of granularity, ::

    cuda_config_bonds.refresh_bond(bond)    # Capture changes to a bond
    cuda_config_bonds.refresh_bonds(bonds)  # Capture changes to multiple bonds
    cuda_config_bonds.refresh()             # Capture all changes

Angle GPU acceleration is a similar GPU-accelerated simulation feature that offloads
angle interactions onto a GPU.
The angle GPU acceleration runtime control interface is practically identical to that
of bond GPU acceleration (*e.g.*, ``refresh_angles`` for angle GPU acceleration is analogous
to ``refresh_bonds`` for bond GPU acceleration).
The angle GPU acceleration runtime control interface is accessible on :py:class:`cuda.AngleConfig`
(``cuda::AngleConfig`` in C++), which is available as an attribute with name ``angles``
on :py:class:`cuda.SimulatorConfig`, ::

    cuda_config_angles = tf.Simulator.cuda_config.angles  # Get angle cuda runtime interface

Refer to the :ref:`Tissue Forge API Reference <api_reference>` for which operations automatically
update bond and angle data on a GPU.
