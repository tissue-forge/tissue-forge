import tissue_forge as tf
import numpy as np
import ipywidgets as widgets
from background_widgets import default_color_picker

def colorConverter(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def dimensions_and_particles(pType, sideLength, radius, mass):
    radius_value = widgets.FloatText( 
        value = radius,
        min=0.000001,
        max=0.3,
        step =0.05,
        description='Radius',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',)

    mass_value = widgets.FloatText( 
        value = mass,
        min=0.00001,
        max=300,
        description='Mass',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',)

    def radiusValue(change):
        pType.radius=change.new 
        for i in range (len(tf.Universe.particles)):
            tf.Universe.particles[i].radius=change.new

    def massValue(change):
        pType.mass= change.new
        for i in range (len(tf.Universe.particles)):
            tf.Universe.particles[i].mass=change.new

    radius_value.observe(radiusValue, names='value')
    mass_value.observe(massValue, names='value')

    particleNumber_text = widgets.IntText( 
        min=0,
        max=10000,
        step=10,
        description='Particle #',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',)

    particleNumber_slider = widgets.IntSlider(
        min=0,
        max=10000,
        step=10,
        description='Particle #',
        disabled=False,
        continuous_update=True,
        msg_throttle =10,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',)
    
    #linking the widgets such that a = b
    mylink = widgets.jslink((particleNumber_text, 'value'), (particleNumber_slider, 'value'))

    #function to add or subtract particles so the number of particles matches the value inputted in widget a or b
    def particleNumber(x):
        x.new
        current = len(tf.Universe.particles)
        deltaParticle = x.new - current 
        if deltaParticle >0:
            positions = np.random.uniform(low=0, high=sideLength, size=(deltaParticle, 3))
            #make a call to tissueforge to find sideLength and then input it into high value
            for pos in positions:
                pType(pos)
        elif deltaParticle <0:
            for i in range(-deltaParticle):
                if len(tf.Universe.particles)>0:
                    index=np.random.randint(0, len(tf.Universe.particles))
                    tf.Universe.particles[index].destroy()
    particleNumber_text.observe(particleNumber, names='value')
    return radius_value, mass_value, particleNumber_text, particleNumber_slider

def set_pColor(pType):
    color_picker= default_color_picker()
    def setColor(_change):
        if len(tf.Universe.particles) > 0:
            pType.style.setColor(_change.new)
        if len(tf.Universe.particles) < 1:
            print("Error: No particles in the simulation")
    color_picker.observe(setColor, names ='value') 
    return color_picker