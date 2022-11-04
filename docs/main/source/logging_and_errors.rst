.. _logging_and_errors:

.. py:currentmodule:: tissue_forge

Logging
--------

Tissue Forge has a detailed logging system. Many internal methods will log
extensive details to either the log (typically stderr) or a user
specified file path. The logging system can be configured to log events
at various levels of detail. All methods of the Logger are static,
and are available immediately upon loading the Tissue Forge package.

To display logging at the lowest level (``TRACE``), where every logging message is
displayed, is as simple as, ::

   import tissue_forge as tf
   tf.Logger.setLevel(tf.Logger.TRACE)

Enabling logging to terminal and disabling are also single commands, ::

   tf.Logger.enableConsoleLogging(tf.Logger.DEBUG)
   ...
   tf.Logger.disableConsoleLogging()

Messages can also be added to the log by logging level. ::

  tf.Logger.log(tf.Logger.FATAL, "A fatal message. This is the highest priority.")
  tf.Logger.log(tf.Logger.CRITICAL, "A critical message")
  tf.Logger.log(tf.Logger.ERROR, "An error message")
  tf.Logger.log(tf.Logger.WARNING, "A warning message")
  tf.Logger.log(tf.Logger.NOTICE, "A notice message")
  tf.Logger.log(tf.Logger.INFORMATION, "An informational message")
  tf.Logger.log(tf.Logger.DEBUG, "A debugging message.")
  tf.Logger.log(tf.Logger.TRACE,  "A tracing message. This is the lowest priority.")


Error Handling
---------------

To support interactive execution, many errors in Tissue Forge do not
necessarily terminate execution of a simulation.
Rather, an error that does not corrupt simulation data
(*e.g.*, bad specification of a particle location)
allow a simulation to proceed but posts an :py:class:`Error`, and also
logs the error at the ``ERROR`` level. Errors that have been posted can
be retrieved in order of their posting with :py:func:`err_get_all`, or the
current first posted error can be retrieved and removed from the list of
errors with :py:func:`err_pos_first`. The function :py:func:`err_occured`
returns ``True`` when any error is currently posted, and :py:func:`err_clear`
clears all posted errors. ::

    # Create a bond with potential ``pot`` between particles ``p0`` and ``p1``
    b = tf.Bond.create(pot, p0, p1)
    # The particles are accessible at indices ``0`` and ``1``; an error results at other indices
    p2 = b[2]
    # Do error checking and maybe terminate if certain errors occurred
    while tf.err_occurred():
        error: tf.Error = tf.err_pop_first()
        print(error)
        # Exit or continue, depending on the error code
        if error.err < 0:
            exit(error.err)

Many errors include useful information about what caused the error, and
useful suggestions about how to prevent the error from occuring. However,
error logging and reporting is an ongoing effort, and so in some cases mitigating
an error may not be obvious.

.. note::

    In cases where error reporting can be improved, users are welcome to create
    an issue on the
    `Tissue Forge repository <https://github.com/tissue-forge/tissue-forge>`_ and
    report the error circumstances and error information provided by Tissue Forge.
    Users are also welcome to submit pull requests that improve existing error
    reporting and add new reporting.
