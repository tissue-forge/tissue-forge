.. _api_io:

File I/O
---------

.. currentmodule:: tissue_forge.io

.. autofunction:: fromFile3DF

.. autofunction:: toFile3DF

.. autofunction:: toFile

.. autofunction:: toString

.. autofunction:: mapImportParticleId

.. autofunction:: mapImportParticleTypeId


.. autoclass:: ThreeDFRenderData

    .. autoproperty:: color


.. autoclass:: ThreeDFStructure

    .. autoproperty:: centroid

        Centroid of all constituent data

    .. autoproperty:: vertices

        Constituent vertices

    .. autoproperty:: edges

        Constituent edges

    .. autoproperty:: faces

        Constituent faces

    .. autoproperty:: meshes

        Constituent meshes

    .. autoproperty:: num_vertices

        Number of constituent vertices

    .. autoproperty:: num_edges

        Number of constituent edges

    .. autoproperty:: num_faces

        Number of constituent faces

    .. autoproperty:: num_meshes

        Number of constituent meshes

    .. autoproperty:: vRadiusDef

    .. automethod:: fromFile

    .. automethod:: toFile

    .. automethod:: flush

    .. automethod:: extend

    .. automethod:: clear

    .. automethod:: has

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: translate

    .. automethod:: translateTo

    .. automethod:: rotateAt

    .. automethod:: rotate

    .. automethod:: scaleFrom

    .. automethod:: scale


.. autoclass:: ThreeDFMeshData

    .. autoproperty:: structure

    .. autoproperty:: id

    .. autoproperty:: name

    .. autoproperty:: renderData

    .. autoproperty:: vertices

        Constituent vertices

    .. autoproperty:: edges

        Constituent edges

    .. autoproperty:: faces

        Constituent faces

    .. autoproperty:: num_vertices

        Number of constituent vertices

    .. autoproperty:: num_edges

        Number of constituent edges

    .. autoproperty:: num_faces

        Number of constituent faces

    .. autoproperty:: centroid

        Centroid of all constituent data

    .. automethod:: has

    .. automethod:: _in

    .. automethod:: is_in

    .. automethod:: translate

    .. automethod:: translateTo

    .. automethod:: rotateAt

    .. automethod:: rotate

    .. automethod:: scaleFrom

    .. automethod:: scale


.. autoclass:: ThreeDFFaceData

    .. autoproperty:: structure

    .. autoproperty:: normal

    .. autoproperty:: id

    .. autoproperty:: vertices

        Constituent vertices

    .. autoproperty:: edges

        Constituent edges

    .. autoproperty:: meshes

        Parent meshes

    .. autoproperty:: num_vertices

        Number of constituent vertices

    .. autoproperty:: num_edges

        Number of constituent edges

    .. autoproperty:: num_meshes

        Number of parent meshes

    .. automethod:: has

    .. automethod:: _in

    .. automethod:: is_in


.. autoclass:: ThreeDFEdgeData

    .. autoproperty:: structure

    .. autoproperty:: id

    .. autoproperty:: vertices

        Constituent vertices

    .. autoproperty:: faces

        Constituent faces

    .. autoproperty:: meshes

        Parent meshes

    .. autoproperty:: num_vertices

        Number of constituent vertices

    .. autoproperty:: num_faces

        Number of parent faces

    .. autoproperty:: num_meshes

        Number of parent meshes

    .. automethod:: has

    .. automethod:: _in

    .. automethod:: is_in


.. autoclass:: ThreeDFVertexData

    .. autoproperty:: structure

    .. autoproperty:: position

    .. autoproperty:: id

    .. autoproperty:: edges

        Parent edges

    .. autoproperty:: faces

        Parent faces

    .. autoproperty:: meshes

        Parent meshes

    .. autoproperty:: num_edges

        Number of parent edges

    .. autoproperty:: num_faces

        Number of parent faces

    .. autoproperty:: num_meshes

        Number of parent meshes

    .. automethod:: _in

    .. automethod:: is_in
