Logging
--------

.. currentmodule:: tissue_forge

Tissue Forge has a detailed logging system. Many internal methods will log
extensive details to either the clog (typically stderr) or a user
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

.. autoclass:: Logger

   .. autoproperty:: CURRENT

   .. autoproperty:: FATAL

   .. autoproperty:: CRITICAL

   .. autoproperty:: ERROR

   .. autoproperty:: WARNING

   .. autoproperty:: NOTICE

   .. autoproperty:: INFORMATION

   .. autoproperty:: DEBUG

   .. autoproperty:: TRACE

   .. automethod:: setLevel

   .. automethod:: getLevel

   .. automethod:: disableLogging

   .. automethod:: enableConsoleLogging

   .. automethod:: disableConsoleLogging

   .. automethod:: enableFileLogging

   .. automethod:: disableFileLogging

   .. automethod:: getCurrentLevelAsString

   .. automethod:: getFileName

   .. automethod:: levelToString

   .. automethod:: stringToLevel

   .. automethod:: log
