# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# ******************************************************************************

import _tissue_forge


try:
    import IPython
    import datetime
    import sys
    import time
    import signal
    from timeit import default_timer as clock

    # Frame per second : 60
    # Should probably be an IPython option
    tissue_forge_fps = 60


    ip = IPython.get_ipython()

    outfile = open("pylog.txt",  "a")


    def inputhook(context):
        """Run the event loop to process window events

        This keeps processing pending events until stdin is ready.  After
        processing all pending events, a call to time.sleep is inserted.  This is
        needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
        though for best performance.
        """

        try:
            t = clock()

            # Make sure the default window is set after a window has been closed

            while not context.input_is_ready():


                outfile.write(datetime.datetime.now().__str__() + "\n")
                outfile.flush()

                _tissue_forge.pollEvents()

                continue

                # We need to sleep at this point to keep the idle CPU load
                # low.  However, if sleep to long, GUI response is poor.  As
                # a compromise, we watch how often GUI events are being processed
                # and switch between a short and long sleep time.  Here are some
                # stats useful in helping to tune this.
                # time    CPU load
                # 0.001   13%
                # 0.005   3%
                # 0.01    1.5%
                # 0.05    0.5%
                used_time = clock() - t
                if used_time > 10.0:
                    # print 'Sleep for 1 s'  # dbg
                    time.sleep(1.0)
                elif used_time > 0.1:
                    # Few GUI events coming in, so we can sleep longer
                    # print 'Sleep for 0.05 s'  # dbg
                    time.sleep(0.05)
                else:
                    # Many GUI events coming in, so sleep only very little
                    time.sleep(0.001)

        except KeyboardInterrupt:
            pass

        outfile.write("user input ready, returning, " + datetime.datetime.now().__str__())



    def registerInputHook():
        """
        Registers the Tissue Forge input hook with the ipython pt_inputhooks
        class.

        The ipython TerminalInteractiveShell.enable_gui('name') method
        looks in the registered input hooks in pt_inputhooks, and if it
        finds one, it activtes that hook.

        To acrtivate the gui mode, call:

        ip = IPython.get_ipython()
        ip.
        """
        import IPython.terminal.pt_inputhooks as pt_inputhooks
        pt_inputhooks.register("tissue_forge", inputhook)


    def enableGui():

        import IPython
        ip = IPython.get_ipython()
        registerInputHook()
        _tissue_forge.initializeGraphics()
        ip.enable_gui("tissue_forge")

    def createTestWindow():
        enableGui()
        _tissue_forge.createTestWindow()

    def destroyTestWindow():
        _tissue_forge.destroyTestWindow()



    ### Module Initialization ###

    registerInputHook()

except:
    pass
