import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import time

pnumber_data = [1]

start_time = time.time()
fig, ax = plt.subplots()
ax.set_ylabel('Particles')
ax.set_title('Particle Number')
colors = ['peachpuff']
_current_tick_label = ['Particle Type']
bplot = ax.boxplot(pnumber_data, patch_artist=True, tick_labels=_current_tick_label) 
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

def _update_ptype_number(_ptype: tf.ParticleType, _number: int):
    current = len(_ptype.parts)
    delta_particles = _number - current
    if delta_particles > 0:
        _ptype.factory(nr_parts=delta_particles)
    elif delta_particles < 0:
        for i in reversed(range(min(-delta_particles, current))):
            _ptype[np.random.randint(0, i+1)].destroy()


def _dist_event_update(event):
    global pnumber_data
    current_pnumber = len(tf.Universe.particles)
    if current_pnumber != pnumber_data[-1]:
        pnumber_data.append(current_pnumber)
        ax.clear() 
        ax.set_ylabel('Particles')
        ax.set_title('Particle Number')
        _current_tick_label = ['Particle Type']
        colors = ['peachpuff']
        bplot = ax.boxplot(pnumber_data, patch_artist=True) 
        for patch, color in zip(bplot['boxes'], colors, tick_label=_current_tick_label):
            patch.set_facecolor(color)
        fig.canvas.draw()
    return 0