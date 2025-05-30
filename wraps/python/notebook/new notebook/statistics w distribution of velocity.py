import tissue_forge as tf
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import ipywidgets as widgets
from IPython.display import display

tf.init(dt=0.1, dim=[15, 12, 10])

a = 0.65

class AType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "seagreen"}
    dynamics = tf.Overdamped

A = AType.get()


class BType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "red"}
    dynamics = tf.Overdamped

B = BType.get()


class FixedType(tf.ParticleTypeSpec):
    radius = 0.3
    style = {"color": "blue"}
    frozen = True

Fixed = FixedType.get()

repulse = tf.Potential.coulomb(q=0.08, min=0.01, max=2 * a)

tf.bind.types(repulse, A, A)
tf.bind.types(repulse, A, B)

f = tf.CustomForce(lambda: [0.3, 1 * np.sin(0.4 * tf.Universe.time), 0], 0.01)

tf.bind.force(f, B)

pot = tf.Potential.power(r0=0.5 * a, alpha=2, max=10 * a)

uc = tf.lattice.sc(a, A, lambda i, j: tf.Bond.create(pot, i, j, dissociation_energy=100.0))

parts = tf.lattice.create_lattice(uc, [5, 5, 5])

for p in parts[4, :].flatten():
    p[0].become(B)

for p in parts[0, :].flatten():
    p[0].become(Fixed)

output = widgets.Output()


velocity_data=[[], [], []]

fig, ax = plt.subplots()
ax.set_ylabel('Velocity')
ax.set_title('Particle Velocity')
tick_labels = ['Veloxity X', 'Veloxity Y', 'Veloxity Z']
colors = ['peachpuff', 'orange', 'tomato']
bplot = ax.boxplot(velocity_data, patch_artist=True, tick_labels=tick_labels)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

def event_update(event):
    global velocity_data

    velocity_data = [[], [], []]

    for particle in tf.Universe.particles:
        if particle in A:
            velocity_data[0].append(particle.velocity[0])
            velocity_data[1].append(particle.velocity[1])
            velocity_data[2].append(particle.velocity[2])
        if particle in B:
            velocity_data[0].append(particle.velocity[0])
            velocity_data[1].append(particle.velocity[1])
            velocity_data[2].append(particle.velocity[2])
        if particle in Fixed:
            velocity_data[0].append(particle.velocity[0])
            velocity_data[1].append(particle.velocity[1])
            velocity_data[2].append(particle.velocity[2])

        ax.clear()
        ax.set_ylabel('Velocity')
        ax.set_title('Particle Velocity')
        tick_labels = ['Veloxity X', 'Veloxity Y', 'Veloxity Z']
        colors = ['peachpuff', 'orange', 'tomato']
        bplot = ax.boxplot(velocity_data, patch_artist=True, tick_labels=tick_labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        fig.canvas.draw()
        
    return 0

tf.event.on_time(invoke_method=event_update, period=0.1, start_time=0.01)

display (output)
tf.show()
plt.show()