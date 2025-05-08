import logging
import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import numpy as np

sideLength = 10
dim = [sideLength, sideLength, sideLength]

# new simulator
tf.init(dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
pot = tf.Potential.lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, 1.0e-3)
tf.Logger.setLevel(tf.Logger.TRACE)

class ArgonType(tf.ParticleTypeSpec):
    radius = 0.25
    mass = 39.4
    style = {'color': 'blue'}

# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()
tf.bind.types(pot, Argon, Argon)
for p in np.random.random((50, 3)) * sideLength:
    Argon(p)

radius_text = widgets.FloatText(
    value=0.25,
    min=0.000001,
    max=0.3,
    step=0.05,
    description='Radius',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.3f',
)

radius_slider = widgets.FloatSlider(
    value=0.25,
    min=0.000001,
    max=0.3,
    step=0.05,
    description='Radius',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.3f',
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WidgetUpdateHandler(logging.Handler):
    def __init__(self, radius_text, radius_slider):
        super().__init__()

    def emit(self, record):
        if "Radius changed to" in record.msg:
            new_radius = float(record.msg.split(": ")[-1])
            radius_text.value = new_radius
            radius_slider.value = new_radius

handler = WidgetUpdateHandler(radius_text, radius_slider)
logger.addHandler(handler)


def update_radius(new_radius):
    old_radius = Argon.radius
    Argon.radius = new_radius
    if abs(old_radius - new_radius) > 0.01:
      for i in range(len(tf.Universe.particles)):
          tf.Universe.particles[i].radius = new_radius
      logger.info(f"Radius changed to: {new_radius}")

def radius_changed(change):
  new_radius = change.new
  if new_radius >= 0.000001:
      update_radius(new_radius)


radius_slider.observe(radius_changed, names='value')
radius_text.observe(radius_changed, names='value')

display(radius_text, radius_slider)
tf.show()