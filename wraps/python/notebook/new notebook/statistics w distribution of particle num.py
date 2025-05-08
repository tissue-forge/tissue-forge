import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import time

sideLength = 10
dim = [sideLength, sideLength, sideLength]

# new simulator
tf.init(dim=dim)

class ArgonType(tf.ParticleTypeSpec):
    radius = 0.25
    mass = 39.4
    style = {'color': 'blue'}

pot = tf.Potential.lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, 1.0e-3)
pnumber_data = [1]

# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()
tf.bind.types(pot, Argon, Argon)
for p in np.random.random((1, 3)) * sideLength:
    Argon(p)


particle_slider = widgets.IntSlider(
    min=0,
    max=50,
    value=1,
    step=1,
    description='Particle #',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',)


start_time = time.time()


fig, ax = plt.subplots()
ax.set_ylabel('Particles')
ax.set_title('Particle Number')
tick_labels = ['Argon']
colors = ['peachpuff']
bplot = ax.boxplot(pnumber_data, patch_artist=True, tick_labels=tick_labels) 
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


def particleNumber(change):
    current = len(tf.Universe.particles)
    deltaParticle = change.new - current 
    if deltaParticle >0:
        positions = np.random.uniform(low=0, high=sideLength, size=(deltaParticle, 3))
        for pos in positions:
            Argon(pos)
        
    elif deltaParticle <0:
        for i in range(-deltaParticle):
            index=np.random.randint(0, len(tf.Universe.particles))
            tf.Universe.particles[index].destroy()    

def event_update(event):
    global pnumber_data
    current_pnumber = len(tf.Universe.particles)
    if current_pnumber != pnumber_data[-1]:
        pnumber_data.append(current_pnumber)
        ax.clear() 
        ax.set_ylabel('Particles')
        ax.set_title('Particle Number')
        tick_labels = ['Argon']
        colors = ['peachpuff']
        bplot = ax.boxplot(pnumber_data, patch_artist=True, tick_labels=tick_labels) 
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        fig.canvas.draw()

    return 0

particle_slider.observe(particleNumber, names='value')
tf.event.on_time(invoke_method=event_update, period=0.1, start_time=0.01)

display(particle_slider)

tf.show()
plt.show()