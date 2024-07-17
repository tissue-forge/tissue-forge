import tissue_forge as tf
import ipywidgets as widgets
# from IPython.display import display
from ipyfilechooser import FileChooser
import os

def colorConverter(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def default_color_picker(concise=False, description='Pick a color', value='blue', disabled=False, continuous_update=True):
    return widgets.ColorPicker(
                concise=concise,
                description=description,
                value=value,
                disabled=disabled,
                continuous_update=continuous_update)

def default_fvector_text(value=None):
    return widgets.FloatText(
            value= value,
            min=0,
            max=1,
            # description='First fVector value',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout={'width': '25%'})

def default_fvector_slider(value=None):
    return widgets.FloatSlider(
                value = value,
                min=0,
                max=1,
                # description='First fVector value',
                disabled=False,
                continuous_update=True,
                msg_throttle =0.05,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
                layout={'width': '25%'})

def colorPicker_set_background():
    color_picker= default_color_picker()
    def backgroundUpdate(_change):
        tf.system.context_make_current()
        tf.system.set_background_color(color=tf.FVector3(colorConverter(_change.new))/255)
        tf.system.context_release()
    color_picker.observe(backgroundUpdate, names='value')
    print("New background has been set!")
    return color_picker

def fvectorText_set_background():
    fVector1 = default_fvector_text(value = 0.3499999940395355)
    fVector2 = default_fvector_text(value = 0.3499999940395355)
    fVector3 = default_fvector_text(value = 0.3499999940395355)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])

    def fVector_update(change):
        tf.system.context_make_current()
        tf.system.set_background_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox

def fvectorSlider_set_background():
    fVector1 = default_fvector_slider(value = 0.3499999940395355)
    fVector2 = default_fvector_slider(value = 0.3499999940395355)
    fVector3 = default_fvector_slider(value = 0.3499999940395355)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])

    throttle_interval = 0.05

    def fVector_update(_change):
        tf.system.context_make_current()
        tf.system.set_background_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox


def colorPicker_set_grid():
    color_picker= default_color_picker()
    def grid_update(_change):
        tf.system.context_make_current()
        tf.system.set_grid_color(color=tf.FVector3(colorConverter(_change.new))/255)
        tf.system.context_release()
    color_picker.observe(grid_update, names='value')
    print("New grid has been set!")
    return color_picker

def fvectorText_set_grid():
    fVector1 = default_fvector_text(value= 1)
    fVector2 = default_fvector_text(value= 1)
    fVector3 = default_fvector_text(value= 1)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])

    def fVector_update(change):
        tf.system.context_make_current()
        tf.system.set_grid_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox

def fvectorSlider_set_grid():
    fVector1 = default_fvector_slider(value= 1)
    fVector2 = default_fvector_slider(value= 1)
    fVector3 = default_fvector_slider(value= 1)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])
    throttle_interval = 0.05

    def fVector_update(_change):
        tf.system.context_make_current()
        tf.system.set_grid_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox



def colorPicker_set_boarders():
    color_picker= default_color_picker()
    def border_update(_change):
        tf.system.context_make_current()
        tf.system.set_scene_box_color(color=tf.FVector3(colorConverter(_change.new))/255)
        tf.system.context_release()
    color_picker.observe(border_update, names='value')
    print('New border has been set!')
    return color_picker

def fvectorText_set_boarders():
    fVector1 = default_fvector_text(value= 1)
    fVector2 = default_fvector_text(value= 1)
    fVector3 = default_fvector_text(value= 0)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])

    def fVector_update(change):
        tf.system.context_make_current()
        tf.system.set_scene_box_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox

def fvectorSlider_set_boarders():
    fVector1 = default_fvector_slider(value= 1)
    fVector2 = default_fvector_slider(value= 1)
    fVector3 = default_fvector_slider(value= 0)
    style = {'description_width': 'initial'}
    hbox = widgets.HBox([widgets.Label('Input your fVectors:', style=style), fVector1, fVector2, fVector3])

    throttle_interval = 0.05

    def fVector_update(_change):
        tf.system.context_make_current()
        tf.system.set_scene_box_color(color=tf.FVector3([fVector1.value, fVector2.value, fVector3.value]))
        tf.system.context_release()
    fVector1.observe(fVector_update, names='value')
    fVector2.observe(fVector_update, names='value')
    fVector3.observe(fVector_update, names='value')
    return hbox
