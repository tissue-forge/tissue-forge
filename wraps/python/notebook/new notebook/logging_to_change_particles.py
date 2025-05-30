import logging
import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WidgetUpdateHandler(logging.Handler):
    def __init__(self, _radiius_text: widgets.IntText, _radiius_slider: widgets.IntSlider):
        super().__init__()

    def emit(self, record):
        if "Radius changed to" in record.msg:
            new_radius = float(record.msg.split(": ")[-1])
            self._radiius_text.value = new_radius
            self._radiius_slider.value = new_radius

handler = WidgetUpdateHandler(widgets.IntText, widgets.IntSlider)
logger.addHandler(handler)

def _update_radius(_ptype: tf.ParticleType, new_radius):
    _old_radius = _ptype.radius
    _ptype.radius = new_radius
    if abs(_old_radius - new_radius) > 0.01:
      for i in range(len(tf.Universe.particles)):
          tf.Universe.particles[i].radius = new_radius
      logger.info(f"Radius changed to: {new_radius}")

def _radius_changed(change):
  new_radius = change.new
  if new_radius >= 0.000001:
      _update_radius(tf.ParticleType, new_radius)

