Widget Renderer Documentation
-----------------------------

Overview
^^^^^^^^
The `widgetRenderer` in Tissue Forge provides a real-time user interface to control and visualize aspects of the ongoing simulation. This feature allows for dynamic updates of simulation metrics (e.g., time, particle counts) and enables interactive elements such as buttons and input fields for altering simulation parameters during runtime.

Incorporating **events** and **callback functions** into the `widgetRenderer` enhances the interactivity of the simulation. Users can tie actions to simulation steps, key presses, or other event triggers to update UI elements or adjust simulation parameters in real time.

Key Concepts
^^^^^^^^^^^^

1. **Callback Functions**: Custom functions that are executed in response to a specific event (such as a button press).
2. **Event System**: Tissue Forge provides built-in events that allow custom behaviors to be triggered by particle interactions, or user inputs.

Examples
^^^^^^^^

Adding Widgets
--------------
You can add various widgets, such as buttons, input fields, and display outputs, to the simulation interface. These widgets allow users to interact with the simulation in real time by modifying simulation parameters or visualizing ongoing metrics.

- **Displaying Simulation Metrics**

  To visualize key metrics like simulation time or particle count:

  .. code-block:: python

      tf.system.show_widget_time()
      tf.system.show_widget_particle_number()
      tf.system.show_widget_bond_number()

These methods enable dynamic tracking of the simulation's state directly within the UI.

Custom Widget Outputs
^^^^^^^^^^^^^^^^^^^^^

You can add custom widgets to display dynamic values, such as the current standard deviation of a random force in the simulation. 

First, create a random force applied to particles in the simulation:

.. code-block:: python

    # Create a random force with mean 0 and standard deviation 50
    rforce = tf.Force.random(mean=0, std=50)

Next, add a widget to display the current standard deviation (`std`) of the random force:

.. code-block:: python

    # Add a widget to display the noise level
    idx_noise = tf.system.add_widget_output_float(rforce.std, 'Noise')

.. note::
   The `add_widget_output_float` function requires a float value. Ensure that the variable passed (`rforce.std` in this case) is of the correct type.

To ensure that the displayed noise level updates during the simulation, a **callback function** is necessary.

Callback: Update Displayed Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can register a callback function to update the widget display as the simulation progresses:

.. code-block:: python

    def update_display_noise():
        tf.system.set_widget_output_float(idx_noise, rforce.std)

Register this callback to be called at each simulation step:

.. code-block:: python

    # Register the callback to update the noise display on each simulation step
    tf.event.on_time(period=0, invoke_method=lambda e: update_display_noise())

This ensures that the widget displaying the noise level (`std` of `rforce`) is updated automatically during the simulation.

Button and Input Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Buttons and input fields allow users to interactively modify the simulation parameters. For example:

- **Buttons for Noise Control**

  .. code-block:: python

      def increase_noise():
          rforce.std += 1
          update_display_noise()

      def decrease_noise():
          rforce.std = max(1E-12, rforce.std - 1)
          update_display_noise()

      # Add buttons to increase or decrease noise
      tf.system.add_widget_button(increase_noise, '+Noise')
      tf.system.add_widget_button(decrease_noise, '-Noise')

  These buttons modify the noise level of the random force in the simulation and update the widget accordingly.

- **Input Fields for Particle Increment**

  To allow users to specify how many particles to add or remove during the simulation, you can add input fields:

  .. code-block:: python

      # Initial particle increment value
      part_incr = 10

      def set_part_incr(val: int):
          global part_incr
          if val > 0:
              part_incr = val

      # Add an input field for setting particle increment
      tf.system.add_widget_input_int(set_part_incr, part_incr, 'Part incr.')

  This input field lets the user set the number of particles to add or remove. The `set_part_incr` function updates the `part_incr` variable based on user input.

Registering Callbacks to Simulation Events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Callbacks can be tied to simulation events to make the simulation responsive to changes. For example, you might want to update certain widgets every time the simulation steps.

Summary
^^^^^^^
The `widgetRenderer` in Tissue Forge provides a flexible interface for building interactive simulations. By integrating **events** and **callbacks**, you can create dynamic, real-time control and visualization tools that react to both user input and simulation states. This allows for a seamless exploration of complex simulations, where users can interact with the system while the simulation is running.
.. note::
   For more detailed explanations of individual methods and their parameters, refer to the API reference for `widgetRenderer`.

   
.. _widget_renderer:

.. py:currentmodule:: tissue_forge
