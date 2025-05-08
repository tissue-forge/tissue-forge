import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import logging
import time
import re

sideLength = 10
dim = [sideLength, sideLength, sideLength]

# new simulator
tf.init(dim=dim)

class ArgonType(tf.ParticleTypeSpec):
    radius = 0.25
    mass = 39.4
    style = {'color': 'blue'}

pot = tf.Potential.lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, 1.0e-3)
pnumber_data = [50]
time_data = [0]

# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()
tf.bind.types(pot, Argon, Argon)
for p in np.random.random((50, 3)) * sideLength:
    Argon(p)


particle_slider = widgets.IntSlider(
    min=0,
    max=500,
    value=50,
    step=1,
    description='Particle #',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',)


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

handler = WidgetUpdateHandler(particle_slider)
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
    return scat

update(0)

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100, blit=True)

is_updating_pnumber = False

def particleNumber(change):
    global is_updating_pnumber
    change.new
    current = len(tf.Universe.particles)
    deltaParticle = change.new - current 
    if deltaParticle >0  and is_updating_pnumber is False:
        is_updating_pnumber = True
        positions = np.random.uniform(low=0, high=sideLength, size=(deltaParticle, 3))
        for pos in positions:
            Argon(pos)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)
        updated_current = len(tf.Universe.particles)
        logger.info(f"Particle number changed to: {(updated_current), (elapsed_time)}")
        is_updating_pnumber = False
        
    elif deltaParticle <0 and is_updating_pnumber is False:
        is_updating_pnumber = True
        for i in range(-deltaParticle):
            index=np.random.randint(0, len(tf.Universe.particles))
            tf.Universe.particles[index].destroy()
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)
        updated_current = len(tf.Universe.particles)
        logger.info(f"Particle number changed to: {(updated_current), (elapsed_time)}")
        is_updating_pnumber = False

particle_slider.observe(particleNumber, names='value')

display(particle_slider)

tf.show()