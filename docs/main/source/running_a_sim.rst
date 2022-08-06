.. _running_a_sim:

.. py:currentmodule:: tissue_forge

Running a Simulation
---------------------

In most cases, running a simulation is as simple as initializing the simulator
with the :func:`init` function (:func:`tfSimulator_initC` in C),
building the physical model, and running
the simulation by calling the :func:`run` function (or :func:`irun` for
interactive mode in Python). However, Tissue Forge provides many options for
controlling the simulation during execution.

A running simulation has three key components: interacting with the operating
system, handling events (both model-specific events and user input), and integrating the
model in time (time stepping). Whenever the :func:`run` (or :func:`irun` in Python)
is invoked, it automatically start time stepping the simulation. However, Tissue Forge
provides additional functions to finely control displaying, time stepping and stopping
a simulation. For example, calling the function :func:`show`
(:func:`tfShow` in C) only displays the application window and starts the
internals of Tissue Forge without performing any time stepping.
The :func:`start`, :func:`step`, :func:`stop` functions
(:meth:`tfStart`, :meth:`tfStep`, :meth:`tfStop` in C)
start the :ref:`universe <universe>` time evolution,
perform a single time step, and stop the time evolution, respectively.
If the universe is stopped, the :meth:`start` function can be
called to continue where the universe was stopped. All functions to build and manipulate
the universe are available either with the universe stopped or running.

When passing no arguments to :func:`run` (when passing a negative value to
:func:`tfRun` in C), the main window opens, and the simulation runs and
only returns when the window closes. When an argument is passed, the value is understood
as the simulation time at which the simulation should stop, at which point the window closes.
With :func:`irun` in IPython, the main window opens, and the simulation runs and
responds to user interactions with it and its objects in real time while it runs.

For convenience, all simulation control functions are aliased as top-level functions in Python, ::

    import tissue_forge as tf   # import the package
    tf.init()                   # initialize the simulator
    # create the model here
    ...
    tf.irun()                   # run in interactive mode (only for ipython console)
    tf.run()                    # display the window and run
    tf.close()                  # close the main window
    tf.show()                   # display the window
    tf.step()                   # time steps the simulation
    tf.stop()                   # stops the simulation

In C++, the same functions are available through the ``TissueForge`` namespace in ``TissueForge.h``,

.. code-block:: cpp

    #include <TissueForge.h>

    using tf = TissueForge;

    tf::Simulator::Config config;
    tf::init(config);               // initialize Tissue Forge
    // create the model here
    ...
    tf::run();                      // display the window and run
    tf::close();                    // close the main window
    tf::show();                     // display the window
    tf::step();                     // time steps the simulation
    tf::stop();                     // stops the simulation

In C, the same functions are available in ``wraps/C/TissueForge_c.h``,

.. code-block:: c

    #include <TissueForge_c.h>

    struct tfSimulatorConfigHandle config;
    tfSimulatorConfig_init(&config);
    tfInitC(&config, NULL, 0);              // initialize Tissue Forge
    // create the model here
    ...
    tfRun();                                // display the window and run
    tfClose();                              // close the main window
    tfShow();                               // display the window
    tfStep();                               // time steps the simulation
    tfStop();                               // stops the simulation

.. _running_a_sim_windowless:

Running Windowless
^^^^^^^^^^^^^^^^^^^

Many applications like massively-parallel execution of lots of simulations
require running Tissue Forge without real-time visualization and interactivity, where
Tissue Forge can execute simulations tens to hundreds of times faster.
Tissue Forge supports such an execution mode, called `Windowless`, in which case
all Tissue Forge functionality is the same, except that Tissue Forge does no rendering
except when instructed to do so in the instructions of a scripted simulation.

Tissue Forge can be informed that a simulation should be executed in Windowless mode
during initialization with the keyword argument ``windowless``, ::

    tf.init(windowless=True)

In C++, the same can be accomplished using ``Simulator::Config``,
and in C with function ``tfSimulatorConfig_setWindowless``.

Execution of a simulation occurs through the function :func:`step` (rather than
:func:`run`), where each call executes one simulation step, ::

    num_steps = int(1E6)  # Number of steps to execute
    for step_num in range(num_steps):
        tf.step()

Reproducible Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^

Some features of Tissue Forge are stochastic (*e.g.*, random :ref:`forces <forces>`).
Tissue Forge uses a pseudo-random number generator to implement stochasticity.
By default, Tissue Forge generates a different stream of random numbers on each
execution of a simulation. However, in cases where results from a simulation with
stochasticity need to be reproduced (*e.g.*, when :ref:`sharing results <file_io>`),
Tissue Forge can use the same stream of random numbers when given the seed of the
pseudo-random number generator. Tissue Forge accepts specification of the seed during
initialization with the keyword argument ``seed``, as well as at any time during
simulation, ::

    tf.init(seed=1)                 # Set the seed during initialization...
    tf.set_seed(tf.get_seed() + 1)  # ... or after initialization.
