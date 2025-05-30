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
time_data = [0]

# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()
tf.bind.types(pot, Argon, Argon)
for p in np.random.random((1, 3)) * sideLength:
    Argon(p)


particle_slider = widgets.IntSlider(
    min=0,
    max=500,
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
ax.hist(pnumber_data, bins=10)
ax.set_xlabel('Particles')
ax.set_ylabel('Frequency')
ax.set_title('Particle Number Histogram')

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
    print(current_pnumber)
    print(pnumber_data)
    print(tf.Universe.time)
    if current_pnumber != pnumber_data[-1]:
        pnumber_data.append(current_pnumber)
        ax.clear() 
        ax.set_xlabel('Particles')
        ax.set_ylabel('Frequency')
        ax.set_title('Particle Number Histogram')
        ax.hist(pnumber_data, bins=10) 
        fig.canvas.draw()
    return 0

particle_slider.observe(particleNumber, names='value')
tf.event.on_time(invoke_method=event_update, period=0.1, start_time=0.01)

display(particle_slider)

tf.show()
plt.show()