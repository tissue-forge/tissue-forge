import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import logging
import time
import re


pnumber_data = [len(tf.Universe.particles)]
time_data = [0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()

class WidgetUpdateHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        if "Particle number changed to" in record.msg:
            match = re.search(r"\((.*?),(.*?)\)", record.msg)
            if match:
                new_pnumber = float(match.group(1))
                timestamp = (float(match.group(2)))
                time_data.append(timestamp)
                pnumber_data.append(new_pnumber)
                fig.canvas.draw()

handler = WidgetUpdateHandler(widgets.IntSlider)
logger.addHandler(handler)

fig, ax = plt.subplots()
scat = ax.scatter(time_data, pnumber_data, label='Particles')
ax.set_xlabel('Time')
ax.set_ylabel('Particles')
ax.set_title('Particle Number Variation')


def update(frame):
    scat.set_offsets(np.c_[time_data, pnumber_data])
    ax.set_xlim(0, max(time_data) +5)
    ax.set_ylim(min(pnumber_data) - 5, max(pnumber_data) + 5)
    return (scat,)


is_updating_pnumber = False

def _update_logging_number(_ptype: tf.ParticleType, _number: int):
    global is_updating_pnumber
    current = len(_ptype.parts)
    delta_particles = _number - current
    if delta_particles > 0:
        _ptype.factory(nr_parts=delta_particles)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)
        updated_current = len(tf.Universe.particles)
        logger.info(f"Particle number changed to: {(updated_current), (elapsed_time)}")
        animation.FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100, blit=True)

        is_updating_pnumber = False
    elif delta_particles < 0:
        for i in reversed(range(min(-delta_particles, current))):
            _ptype[np.random.randint(0, i+1)].destroy()
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)
        updated_current = len(tf.Universe.particles)
        logger.info(f"Particle number changed to: {(updated_current), (elapsed_time)}")
        animation.FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100, blit=True)

        is_updating_pnumber = False