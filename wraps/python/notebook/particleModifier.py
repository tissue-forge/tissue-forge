from IPython.display import display
import ipywidgets as ipw
import tissue_forge as tf
import numpy as np
from . import widgets as tfnw
from typing import Any, Dict

LABELTEXT_PTYPE_COLOR = 'Color for type '
LABELTEXT_PTYPE_MASS = 'Mass for type '
LABELTEXT_PTYPE_NUMBER = 'Particle # for type '
LABELTEXT_PTYPE_RADIUS = 'Radius for type '


def _safe_callback(_ptype: tf.ParticleType, _callback):

    name = _ptype.name

    def _inner(*args, **kwargs):

        _pt = tf.ParticleType_FindFromName(name)

        if _pt is not None:
            return _callback(_pt, *args, **kwargs)

    return _inner


def _update_ptype_color(_ptype: tf.ParticleType, _color: tf.FVector3):
    _ptype.style.color = _color
    for p in _ptype:
        if p.style is not None:
            p.style.color = _color


def _update_ptype_mass(_ptype: tf.ParticleType, _mass: float):
    _ptype.mass = _mass
    for p in _ptype:
        p.mass = _mass


def _update_ptype_number(_ptype: tf.ParticleType, _number: int):
    current = len(_ptype.parts)
    delta_particles = _number - current
    if delta_particles > 0:
        _ptype.factory(nr_parts=delta_particles)
    elif delta_particles < 0:
        for i in reversed(range(min(-delta_particles, current))):
            _ptype[np.random.randint(0, i+1)].destroy()


def _update_ptype_radius(_ptype: tf.ParticleType, _radius: float):
    _ptype.radius = _radius
    for p in _ptype:
        p.radius = _radius


def set_ptype_color_picker(ptype: tf.ParticleType,
                           show=False,
                           **kwargs):
    def_kwargs = dict(
        description=LABELTEXT_PTYPE_COLOR + ptype.name,
        value=tfnw.color_inverter(*ptype.style.color.as_list())
    )
    def_kwargs.update(kwargs)
    widget = tfnw.color_picker(_safe_callback(ptype, _update_ptype_color), **def_kwargs)
    if show:
        display(widget)
    return widget


def set_ptype_color_slider(ptype: tf.ParticleType,
                           show=False,
                           label_kwargs: Dict[str, Any] = None,
                           field_kwargs: Dict[int, Dict[str, Any]] = None):
    def_label_kwargs = dict(
        style=dict(
            description_width='initial'
        )
    )
    if label_kwargs is not None:
        def_label_kwargs.update(label_kwargs)
    def_field_kwargs = {i: dict(
        min=0.0,
        max=1.0,
        step=0.01
    ) for i in range(3)}
    if field_kwargs is not None:
        for k, v in field_kwargs.items():
            def_field_kwargs[k].update(v)
    label = ipw.Label(LABELTEXT_PTYPE_COLOR + ptype.name, **def_label_kwargs)
    box, widgets = tfnw.vector_slider(ndim=3,
                                      dtype=float,
                                      callback=_safe_callback(ptype, _update_ptype_color),
                                      initial_values=tf.system.get_background_color().as_list(),
                                      label=label,
                                      field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_ptype_color_text(ptype: tf.ParticleType,
                         show=False,
                         label_kwargs: Dict[str, Any] = None,
                         field_kwargs: Dict[int, Dict[str, Any]] = None):
    def_label_kwargs = dict(
        style=dict(
            description_width='initial'
        )
    )
    if label_kwargs is not None:
        def_label_kwargs.update(label_kwargs)
    def_field_kwargs = {i: dict(
        min=0.0,
        max=1.0,
        step=0.01
    ) for i in range(3)}
    if field_kwargs is not None:
        for k, v in field_kwargs.items():
            def_field_kwargs[k].update(v)
    label = ipw.Label(LABELTEXT_PTYPE_COLOR + ptype.name, **def_label_kwargs)
    box, widgets = tfnw.vector_textb(ndim=3,
                                     dtype=float,
                                     callback=_safe_callback(ptype, _update_ptype_color),
                                     initial_values=tf.system.get_background_color().as_list(),
                                     label=label,
                                     field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_ptype_mass_slider(ptype: tf.ParticleType,
                          show=False,
                          **kwargs):
    def_kwargs = dict(
        min=ptype.mass * 1E-1,
        max=ptype.mass * 1E1,
        description=LABELTEXT_PTYPE_MASS + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = (def_kwargs['max'] - def_kwargs['min']) / 100

    box, widget = tfnw.scalar_slider(float,
                                     _safe_callback(ptype, _update_ptype_mass),
                                     initial_value=ptype.mass,
                                     field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget


def set_ptype_mass_text(ptype: tf.ParticleType,
                        show=False,
                        **kwargs):
    def_kwargs = dict(
        value=ptype.mass,
        min=ptype.mass * 1E-1,
        max=ptype.mass * 1E1,
        description=LABELTEXT_PTYPE_MASS + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = (def_kwargs['max'] - def_kwargs['min']) / 100

    box, widget = tfnw.scalar_text(float,
                                   _safe_callback(ptype, _update_ptype_mass),
                                   initial_value=ptype.mass,
                                   field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget


def set_ptype_number_slider(ptype: tf.ParticleType,
                            show=False,
                            **kwargs):
    def_kwargs = dict(
        min=0,
        max=10000,
        description=LABELTEXT_PTYPE_NUMBER + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = int((def_kwargs['max'] - def_kwargs['min']) / 100)

    box, widget = tfnw.scalar_slider(int,
                                     _safe_callback(ptype, _update_ptype_number),
                                     initial_value=len(ptype.parts),
                                     field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget


def set_ptype_number_text(ptype: tf.ParticleType,
                          show=False,
                          **kwargs):
    def_kwargs = dict(
        min=0,
        max=10000,
        description=LABELTEXT_PTYPE_NUMBER + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = int((def_kwargs['max'] - def_kwargs['min']) / 100)

    box, widget = tfnw.scalar_text(int,
                                   _safe_callback(ptype, _update_ptype_number),
                                   initial_value=len(ptype.parts),
                                   field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget


def set_ptype_radius_slider(ptype: tf.ParticleType,
                            show=False,
                            **kwargs):
    def_kwargs = dict(
        min=ptype.minimum_radius,
        max=min(tf.Universe.dim.as_list()),
        description=LABELTEXT_PTYPE_RADIUS + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = (def_kwargs['max'] - def_kwargs['min']) / 100

    box, widget = tfnw.scalar_slider(float,
                                     _safe_callback(ptype, _update_ptype_radius),
                                     initial_value=ptype.radius,
                                     field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget


def set_ptype_radius_text(ptype: tf.ParticleType,
                          show=False,
                          **kwargs):
    def_kwargs = dict(
        min=ptype.minimum_radius,
        max=min(tf.Universe.dim.as_list()),
        description=LABELTEXT_PTYPE_RADIUS + ptype.name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f'
    )
    def_kwargs.update(kwargs)
    if 'step' not in def_kwargs:
        def_kwargs['step'] = (def_kwargs['max'] - def_kwargs['min']) / 100

    box, widget = tfnw.scalar_text(float,
                                   _safe_callback(ptype, _update_ptype_radius),
                                   initial_value=ptype.radius,
                                   field_kwargs=def_kwargs)
    if show:
        display(box)
    return box, widget
