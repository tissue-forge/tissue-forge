import logging
import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import re

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

def default_fvector_text(value=None):
    return widgets.FloatText(
            value= value,
            min=0,
            max=1,
            step= 0.050,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '25%'})

def default_fvector_slider(value=None):
    return widgets.FloatSlider(
                value = value,
                min=0,
                max=1,
                step= 0.001,
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
                layout={'width': '25%'})

fVector1 = default_fvector_text(value = 0.35)
fVector2 = default_fvector_text(value = 0.35)
fVector3 = default_fvector_text(value = 0.35)

fVectora = default_fvector_slider(value = 0.35)
fVectorb = default_fvector_slider(value = 0.35)
fVectorc = default_fvector_slider(value = 0.35)

def default_color_picker(concise=False, description='Pick a color', value='gray', disabled=False, continuous_update=True):
    return widgets.ColorPicker(
                concise=concise,
                description=description,
                value=value,
                disabled=disabled,
                continuous_update=continuous_update)

colorpicker = default_color_picker()

def colorConverter(hex_code):
    hex_code = hex_code.lstrip('#')
    
    fvectord = round(int(hex_code[0:2], 16) / 255.0, 3)
    fvectore = round(int(hex_code[2:4], 16) / 255.0, 3)
    fvectorf = round(int(hex_code[4:6], 16) / 255.0, 3)
    
    return (fvectord, fvectore, fvectorf)

def reverseColorConverter(fvector):
    r = int(fvector[0] * 255)
    g = int(fvector[1] * 255)
    b = int(fvector[2] * 255)
    
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


style = {'description_width': 'initial'}
hbox1 = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])
hbox2 = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVectora, fVectorb, fVectorc])
display(hbox1)
display(hbox2)
display(colorpicker)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

out = widgets.Output(layout={'border': '.25px solid black'})
out
display(out)

class WidgetUpdateHandler(logging.Handler):
    def __init__(self, fVector1,fVector2,fVector3, fVectora, fVectorb, fVectorc):
        super().__init__()

    def emit(self, record):
        if "fVector text changed to:" in record.msg:
            match = re.search(r"\((.*?),(.*?),(.*?)\)", record.msg)
            if match:
                new_fvector1 = round(float(match.group(1)), 3)
                new_fvector2 = round(float(match.group(2)), 3)
                new_fvector3 = round(float(match.group(3)), 3)

                fVectora.value = new_fvector1
                fVectorb.value = new_fvector2
                fVectorc.value = new_fvector3
                colorpicker.value = reverseColorConverter((new_fvector1, new_fvector2, new_fvector3))
        elif "fVector slider changed to:" in record.msg:
            match = re.search(r"\((.*?),(.*?),(.*?)\)", record.msg)
            if match:
                new_fvectora = round(float(match.group(1)), 3)
                new_fvectorb = round(float(match.group(2)), 3)
                new_fvectorc = round(float(match.group(3)), 3)

                fVector1.value = new_fvectora
                fVector2.value = new_fvectorb
                fVector3.value = new_fvectorc
                colorpicker.value = reverseColorConverter((new_fvectora, new_fvectorb, new_fvectorc))
    
        elif "ColorPicker changed to:"in record.msg:
            match = re.search(r"\((.*?),(.*?),(.*?)\)", record.msg)
            if match:
                new_fvectord = round(float(match.group(1)), 3)
                new_fvectore = round(float(match.group(2)), 3)
                new_fvectorf = round(float(match.group(3)), 3)
                

                fVector1.value = new_fvectord
                fVector2.value = new_fvectore
                fVector3.value = new_fvectorf

                fVectora.value = new_fvectord
                fVectorb.value = new_fvectore
                fVectorc.value = new_fvectorf

handler = WidgetUpdateHandler(fVector1,fVector2,fVector3, fVectora, fVectorb, fVectorc)
logger.addHandler(handler)


is_updating_color = False

def fVector_update(change):
    global is_updating_color
    source_widget = change['owner']
    new_fvector = change['new']

    if source_widget is colorpicker and is_updating_color is False:
        is_updating_color = True
        with out: 
            out.clear_output()
            colorpicker_fVectors= colorConverter(new_fvector)
            logger.info(f"ColorPicker changed to: {colorpicker_fVectors}")
        is_updating_color = False


    if source_widget is fVector1 and is_updating_color is False:
        is_updating_color = True
        with out: 
            out.clear_output()
            logger.info(f"fVector text changed to: {round(new_fvector, 3), round(fVector2.value, 3), round(fVector3.value, 3)}")
        is_updating_color = False
    if source_widget is fVector2 and is_updating_color is False:
        is_updating_color = True
        with out: 
            out.clear_output()
            logger.info(f"fVector text changed to: {round(fVector1.value, 3), round(new_fvector, 3), round(fVector3.value, 3)}")
        is_updating_color = False
    if source_widget is fVector3 and is_updating_color is False:
        is_updating_color = True
        with out:
            out.clear_output()
            logger.info(f"fVector text changed to: {round(fVector1.value, 3), round(fVector2.value, 3), round(new_fvector, 3)}")
        is_updating_color = False


    if source_widget is fVectora and is_updating_color is False:
        is_updating_color = True
        with out: 
            out.clear_output()
            logger.info(f"fVector slider changed to: {round(new_fvector, 3), round(fVectorb.value, 3), round(fVectorc.value,3)}")
        is_updating_color = False
    if source_widget is fVectorb and is_updating_color is False:
        is_updating_color = True
        with out: 
            out.clear_output()
            logger.info(f"fVector slider changed to: {round(fVectora.value,3), round(new_fvector, 3), round(fVectorc.value,3)}")
        is_updating_color = False
    if source_widget is fVectorc and is_updating_color is False:
        is_updating_color = True
        with out:
            out.clear_output()
            logger.info(f"fVector slider changed to: {round(fVectora.value,3), round(fVectorb.value,3), round(new_fvector, 3)}")
        is_updating_color = False


    tf.system.context_make_current()
    if source_widget is colorpicker:
        color=tf.FVector3(colorConverter(new_fvector))
    elif source_widget in [fVector1, fVectora]:
        color = tf.FVector3([new_fvector, fVector2.value, fVector3.value])
    elif source_widget in [fVector2, fVectorb]:
        color = tf.FVector3([fVector1.value, new_fvector, fVector3.value])
    elif source_widget in [fVector3, fVectorc]:
        color = tf.FVector3([fVector1.value, fVector2.value, new_fvector])

    tf.system.set_background_color(color=color)
    tf.system.context_release()


fVector1.observe(fVector_update, names='value')
fVector2.observe(fVector_update, names='value')
fVector3.observe(fVector_update, names='value')

fVectora.observe(fVector_update, names='value')
fVectorb.observe(fVector_update, names='value')
fVectorc.observe(fVector_update, names='value')

colorpicker.observe(fVector_update, names='value')


tf.show()