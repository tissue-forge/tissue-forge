.. _file_io:

.. py:currentmodule:: tissue_forge

I/O Operations
---------------

Tissue Forge supports a number of operations associated with importing and exporting simulation and
simulation data. At any time during simulation, data can be archived during simulation for later
import, execution and analysis, or exported to common 3D model or `JSON <https://www.json.org/>`_
file formats for easy sharing of model objects and browsable three-dimensional simulation results
among colleagues, research groups and the broader scientific community. In general, I/O operations
are defined in the :py:mod:`io` module (``io`` namespace in C++). For detailed information on classes
and methods available in the :py:mod:`io` module, refer to the :ref:`Tissue Forge API Reference <api_io>`.

Loading and Saving a Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissue Forge supports saving the state of a simulation to file at any time during simulation. Almost
all simulation data can be written to file in JSON format using
function :meth:`toFile <io.toFile>`, ::

    import tissue_forge as tf
    from os import path

    tf.init()

    # Simulation code here...

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to export
    tf.io.toFile(fp)                                                         # Export to file

Data files exported using :meth:`toFile <io.toFile>` can be imported and used to initialize a simulation
in approximately the same state as when the exported file was generated. The path to a file containing
exported data can be passed directly to the keyword ``load_file`` of :func:`init` when initializing
Tissue Forge, ::

    import tissue_forge as tf
    from os import path

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to import
    tf.init(load_file=fp)

Initialization from an imported simulation state is not limited to the data defined in the
imported file. Rather, Tissue Forge begins by initializing from whatever data is defined in the imported
file, and then resumes all other initializations in the typical way. For example, exported data includes
select simulator details like the timestep used by the simulator. However, passing an explicit timestep
to Tissue Forge during initialization will use the specified timestep, and not the one defined in the imported
file, ::

    import tissue_forge as tf
    from os import path

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to import
    tf.init(load_file=fp, dt=0.005)

Furthermore, all Tissue Forge functionality concerning creating objects and interactions between them
are fully available after importing a simulation state, while (almost) all objects and processes
defined in the imported file are also available after initialization, ::

    # Get an instance of particle type named "ExportedType" from imported data
    Exported = tf.ParticleType_FindFromName('ExportedType')
    # Print the number of imported ExportedType particles
    print('Number of imported particles:', len(Exported.items()))
    # Create a few more ExportedType particles
    [Exported() for _ in range(10)]

The versatility of Tissue Forge's approach to importing simulation data comes with the tradeoff of
that not all simulation data is conserved during import. Certain data used to identify
objects like particles and particle types are not necessarily
the same between an original simulation state and its state after import. For example, suppose that a
certain particle has an ``id`` attribute value of ``10`` at export. After import, the attribute value
for ``id`` is not guaranteed to again be ``10``. However, Tissue Forge provides mappings of simulation
state data from values in the original state at export to values in the current
simulation state after import, ::

    # Get the id of the particle that had an id of 20 at export
    id_part_20 = tf.io.mapImportParticleId(20)

.. note::

    All data import maps are available between initialization and the first simulation step,
    after which they are purged.

Not all features of Tissue Forge are (or even can be) written to file during export.
While rendering details of particles, particle types and all bond types are exported,
non-critical simulation details like camera view are not exported.
More importantly, features that rely on custom functions and callbacks
(*e.g.*, :ref:`custom potentials <potentials>`, :ref:`custom forces <forces>` and
:ref:`events <events>`) cannot be exported.
Whenever necessary, such features must be created and loaded into Tissue Forge in the same
way after import to reproduce the complete simulation state.
For a complete list of information exported by Tissue Forge feature, see :ref:`Appendix A <appendix_a>`.

3D Model Formats
^^^^^^^^^^^^^^^^^

Tissue Forge makes sharing 3D results simple. At any time during simulation execution, the state of
the simulation can be exported to a 3D model format as a mesh, ::

    fp_3df = path.join(path.dirname(path.abspath(__file__)), 'fileexport.stl')  # Path to export stl
    tf.io.toFile3DF(format="stl", filePath=fp_3df, pRefinements=2)              # Export stl mesh

Tissue Forge integrates the Open Asset Import Library (`Assimp <http://assimp.org/>`_) for working
with 3D model formats, and so
`all formats supported by Assimp <https://assimp-docs.readthedocs.io/en/latest/about/introduction.html>`_
are also supported by Tissue Forge.

Tissue Forge can also import mesh data in a 3D file and make it available for constructing
a simulation. The :py:mod:`io` module method :meth:`fromFile3DF <io.fromFile3DF>` returns
a structure of mesh data as imported from a 3D file, ::

    fp_mesh = path.join(path.dirname(path.abspath(__file__)), 'mesh.obj')  # Path of mesh to import
    io_struct = tf.io.fromFile3DF(fp_mesh)                                 # Import mesh
    # Print import summary
    print(io_struct.num_meshes, 'meshes')
    print(io_struct.num_faces, 'faces')
    print(io_struct.num_edges, 'edges')
    print(io_struct.num_nodes, 'nodes')
    print('Mesh centroid:', io_struct.centroid)

The :py:class:`Structure3DF` instance returned by :meth:`fromFile3DF <io.fromFile3DF>` contains
all vertices, edges, faces and meshes imported from the 3D file, and provides a few useful methods
for using the mesh data in a simulation (`e.g.`, building a simulation from a mesh designed in Blender), ::

    import math

    # Translate mesh centroid to center of universe
    io_struct.translateTo(tf.Universe.center)
    # Rotate 90 degrees about X
    io_struct.rotate(tf.FMatrix4.rotationX(math.pi/2).rotation())
    # Double the size about the centroid
    io_struct.scale(2.0)

For example, particles can readily be constructed at each vertex of a mesh by simply iterating
over all vertices of the mesh, ::

    class VertexType(tf.ParticleTypeSpec):
        """A type for particles built from mesh data"""
        pass

    Vertex = VertexType.get()
    # Create particles from mesh vertices
    for v in io_struct.vertices:
        Vertex(v.position)

Serializing Tissue Forge Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissue Forge supports serialization of most objects using JSON strings for sharing individual model
objects. Any object that can be serialized has the method ``toString``, and its class has the static
method ``fromString``. ``toString`` returns a JSON-formatted string of the state of the object,
which can be exported for sharing, ::

    # A Tissue Forge simulation written by Modeler A.
    import tissue_forge as tf
    from os import path
    tf.init()

    class ParticleTypeA(tf.ParticleTypeSpec):
        """Awesome Tissue Forge particle"""

    A = ParticleTypeA.get()
    # Export the type to share with a friend
    fp = path.join(path.dirname(path.abspath(__file__)), 'ptypea.json')
    with open(fp, 'w') as f:
        f.write(A.toString())

The generated string can later be used by the ``fromString`` method of the class that generated the
string to recreate the object, ::

    # A Tissue Forge simulation written by Modeler B.
    import tissue_forge as tf
    from os import path
    tf.init()

    # Import a type shared by a friend
    fp = path.join(path.dirname(path.abspath(__file__)), 'ptypea.json')
    with open(fp, 'r') as f:
        A = tf.ParticleType.fromString(f.read())

Tissue Forge provides built-in support in Python for pickling all objects that can be serialized.
All objects that support pickling can be seamlessly integrated into multithreading applications, ::

    from multiprocessing import Pool

    def energy_diff(bond):
        """Calculates the difference of the potential and dissociation energies of a bond"""
        return bond.dissociation_energy - bond.potential_energy

    # Calculate all bond energy differences in parallel
    with Pool(8) as p:
        energy_diffs = p.map(energy_diff, [bh.get() for bh in tf.Universe.bonds])

All objects that can be pickled have the method ``__reduce__`` marked in the
documentation of their class in the :doc:`Tissue Forge Python API Reference <docs_api_py:index>`.

.. note:: Special care must be taken to account for that deserialized Tissue Forge objects are copies of
    their original object, and that the Tissue Forge engine is not available in separate processes. As such,
    calls to methods that require the engine in a spawned Python process will fail.
