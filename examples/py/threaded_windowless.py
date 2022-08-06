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

import tissue_forge as tf
import threading

tf.Logger.setLevel(tf.Logger.INFORMATION)

tf.init(windowless=True, window_size=[1024, 1024])

print(tf.system.gl_info())


class NaType(tf.ParticleTypeSpec):
    radius = 0.4
    style = {"color": "orange"}


class ClType(tf.ParticleTypeSpec):
    radius = 0.25
    style = {"color": "spablue"}


uc = tf.lattice.bcc(0.9, [NaType.get(), ClType.get()])

tf.lattice.create_lattice(uc, [10, 10, 10])


def threaded_steps(steps):

    print('thread start')

    print("thread, calling context_has_current()")
    tf.system.context_has_current()

    print("thread, calling context_make_current())")
    tf.system.context_make_current()

    tf.step()

    tf.system.screenshot('threaded.jpg')

    print("thread calling release")
    tf.system.context_release()

    print("thread done")


print("main writing main.jpg")

tf.system.screenshot('main.jpg')

print("main calling context_release()")
tf.system.context_release()

thread = threading.Thread(target=threaded_steps, args=(1,))

thread.start()

thread.join()

print("main thread context_has_current: ", tf.system.context_has_current())

tf.system.context_make_current()

tf.step()

tf.system.screenshot('main2.jpg', decorate=False, bgcolor=[1, 1, 1])

print("all done")
