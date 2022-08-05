.. _rendering:

.. py:currentmodule:: tissue_forge

Rendering and System Interaction
--------------------------------

Tissue Forge provides a number of methods to interact with the rendering
engine and host CPU via the :py:mod:`system` module (``system`` namespace in C++).
Basic information about a Tissue Forge installation can be retrieved on demand,
including information about the CPU, software compilation and available graphics
hardware, ::

    import tissue_forge as tf
    tf.init()

    print('CPU info:', tf.system.cpu_info())
    print('Compilation info:', tf.system.compile_flags())
    print('OpenGL info:', tf.system.gl_info())

The :py:mod:`system` module provides rendering methods for customizing basic
visualization during simulation. Basic visualization customization combines
with specifications made using :ref:`Style <style>` objects, ::

    # Disable scene decorations
    tf.system.decorate_scene(False)
    # Reduce shininess by 50%
    tf.system.set_shininess(0.5 * tf.system.get_shininess())
    # Move camera location
    tf.system.set_light_direction(tf.FVector3(1, 1, 1))

Screenshots can be taken of the current view and saved to file. Temporary modifications
to select features can be applied when generating a screenshot, like disabling scene
decorations and altering the background color of the scene, ::

    # Save a .png screenshot of the current scene as it's displayed
    tf.system.screenshot('mysim.png')
    # Save a .jpg screenshot without decorations and a white background
    tf.system.screenshot('mysim.jpg', decorate=False, bgcolor=[1, 1, 1])

Tissue Forge currently supports generating screenshots and saving with the following file formats,

* Windows Bitmap (.bmp)
* JPEG (.jpg/.jpeg/.jpe)
* Radiance HDR (.hdr)
* PNG (.png)
* Truevision TGA (.tga)

Controlling the Camera
^^^^^^^^^^^^^^^^^^^^^^^

The view of a simulation can be retrieved or set at any time during simulation,
including in :ref:`custom keyboard commands <events_input_driven>`, ::

    # camera view parameters for storing and reloading
    view_center, view_rotation, view_zoom = None, None, None
    # key "s" stores a view; key "e" restores a stored view
    def do_key_actions(event):
        if event.key_name == "s":
            global view_center, view_rotation, view_zoom
            view_center = tf.system.camera_center()
            view_rotation = tf.system.camera_rotation()
            view_zoom = tf.system.camera_zoom()
        elif event.key_name == "e" and view_center is not None:
            tf.system.camera_move_to(view_center, view_rotation, view_zoom)

    tf.on_keypress(do_key_actions)

Tissue Forge also provides commands to perform precise adjustments to the camera view
in terms of rotations, zoom and camera position, ::

    from math import pi
    # Move to an isometric view (camera position, view center, upward axis)
    tf.system.camera_move_to(tf.FVector3(10, 10, 10), tf.FVector3(0, 0, 0), tf.FVector3(0, 0, 1))
    # Zoom in
    tf.system.camera_zoom_by(10)
    # Rotate about the z-axis
    tf.system.camera_rotate_by_euler_angle(tf.FVector3(0, 0, pi / 2))
    # Show a preview
    tf.show()
    # Reset the camera after closing and then run
    tf.system.camera_reset()
    tf.run()

Creating and Controlling Clip Planes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Large and densly populated simulations can generate regions of space that are difficult to
inspect during simulation. For such cases, Tissue Forge supports introducing clip planes
to the visualization of simulation data. A clip plane divides the simulation domain by an imaginary
plane, one side of which is visualized, and the other side of which is not visualized.
Tissue Forge supports up to 8 clip planes at any given time in simulation.

A Tissue Forge simulation can be initialized in Python with one or more clip planes using the keyword
argument ``clip_planes`` in the :func:`init` function. Clip planes in Python are specified in a list of
tuples (in C++, a string with the same syntax is passed), where each tuple specifies a clip plane.
Each tuple contains two elements: a three-element list specifying a point on the clip plane, and
a three-element list specifying the components of the normal vector of the plane, ::

    import tissue_forge as tf
    # Initialize with a clip plane at the center along the y-z plane
    tf.init(dim=[10, 10, 10], clip_planes=[([5, 5, 5], [1, 0, 0])])

Existing clip planes can be retrieved using the :py:class:`rendering.ClipPlanes`
interface, which provides :py:class:`rendering.ClipPlane` objects for interacting
with clip planes during a simulation, ::

    # See how many clip planes we currently have
    print('Number of clip planes:', tf.rendering.ClipPlanes.len())  # Prints "1", from init
    # Get the clip plane created during initialization
    clip_plane0 = tf.rendering.ClipPlanes.item(0)  # Returned object is a tf.rendering.ClipPlane

The :py:class:`rendering.ClipPlanes` interface also provides the ability to create new clip planes
at any time during a simulation, ::

    # Create a second clip plane at the center along the x-z plane
    clip_plane1 = tf.ClipPlanes.create(tf.Universe.center, tf.FVector3(0, 1, 0))

A :py:class:`rendering.ClipPlane` instance provides a live interface to its clip plane in the Tissue Forge
rendering engine, so that clip planes can be manipulated or destroyed at any time in simulation after
their creation, ::

    # Move the first clip plane to the origin and cut diagonally across the domain
    clip_plane0.setEquation(tf.FVector(0, 0, 0), tf.FVector3(1, 1, 1))
    # Remove the second clip plane
    clip_plane1.destroy()
    tf.run()

.. note:: Destroying a :py:class:`rendering.ClipPlane` can have downstream effects on the validity of
    other :py:class:`rendering.ClipPlane` instances. When a :py:class:`rendering.ClipPlane` instance is
    created, it refers to a clip plane by index from a list of clip planes in the rendering engine.
    If a clip plane is removed from the middle of the list of clip planes, then all instances
    after it in the list are shifted downward (like popping from a Python list). As such, all
    :py:class:`rendering.ClipPlane` instances that refer to downshifted clip planes have invalid reference
    indices. Invalid references can be repaired by decrementing their attribute
    :attr:`index <rendering.ClipPlane.index>`, though a more reliable approach is to always refer to clip
    planes using the :py:class:`rendering.ClipPlanes` static method :meth:`item <rendering.ClipPlanes.item>`
    (*e.g.*, ``tf.rendering.ClipPlanes.item(1).destroy()``).
