from IPython.display import display
import ipywidgets as ipw
from typing import Any, Dict

import tissue_forge as tf
from . import widgets as tfnw


LABELTEXT_BACKGROUND = 'Background:'
LABELTEXT_BORDER = 'Borders:'
LABELTEXT_GRID = 'Grid:'


def _background_update(color: tf.FVector3):
    tf.system.context_make_current()
    tf.system.set_background_color(color=color)
    tf.system.context_release()


def _grid_update(color: tf.FVector3):
    tf.system.context_make_current()
    tf.system.set_grid_color(color=color)
    tf.system.context_release()


def _border_update(color: tf.FVector3):
    tf.system.context_make_current()
    tf.system.set_scene_box_color(color=color)
    tf.system.context_release()


def set_background_picker(show=False,
                          **kwargs):
    def_kwargs = dict(
        value=tfnw.color_inverter(*tf.system.get_background_color().as_list())
    )
    def_kwargs.update(kwargs)
    widget = tfnw.color_picker(_background_update, **def_kwargs)
    if show:
        display(widget)
    return widget


def set_background_text(show=False,
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
    label = ipw.Label(LABELTEXT_BACKGROUND, **def_label_kwargs)
    box, widgets = tfnw.vector_textb(ndim=3,
                                     dtype=float,
                                     callback=_background_update,
                                     initial_values=tf.system.get_background_color().as_list(),
                                     label=label,
                                     field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_background_slider(show=False,
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
    label = ipw.Label(LABELTEXT_BACKGROUND, **def_label_kwargs)
    box, widgets = tfnw.vector_slider(ndim=3,
                                      dtype=float,
                                      callback=_background_update,
                                      initial_values=tf.system.get_background_color().as_list(),
                                      label=label,
                                      field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_grid_picker(show=False,
                    **kwargs):
    def_kwargs = dict(
        value=tfnw.color_inverter(*tf.system.get_grid_color().as_list())
    )
    def_kwargs.update(kwargs)
    widget = tfnw.color_picker(_grid_update, **def_kwargs)
    if show:
        display(widget)
    return widget


def set_grid_text(show=False,
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
    label = ipw.Label(LABELTEXT_GRID, **def_label_kwargs)
    box, widgets = tfnw.vector_textb(ndim=3,
                                     dtype=float,
                                     callback=_grid_update,
                                     initial_values=tf.system.get_grid_color().as_list(),
                                     label=label,
                                     field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_grid_slider(show=False,
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
    label = ipw.Label(LABELTEXT_GRID, **def_label_kwargs)
    box, widgets = tfnw.vector_slider(ndim=3,
                                      dtype=float,
                                      callback=_grid_update,
                                      initial_values=tf.system.get_grid_color().as_list(),
                                      label=label,
                                      field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_borders_picker(show=False,
                       **kwargs):
    def_kwargs = dict(
        value=tfnw.color_inverter(*tf.system.get_scene_box_color().as_list())
    )
    def_kwargs.update(kwargs)
    widget = tfnw.color_picker(_border_update, **def_kwargs)
    if show:
        display(widget)
    return widget


def set_borders_text(show=False,
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
    label = ipw.Label(LABELTEXT_BORDER, **def_label_kwargs)
    box, widgets = tfnw.vector_textb(ndim=3,
                                     dtype=float,
                                     callback=_border_update,
                                     initial_values=tf.system.get_scene_box_color().as_list(),
                                     label=label,
                                     field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets


def set_borders_slider(show=False,
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
    label = ipw.Label(LABELTEXT_BORDER, **def_label_kwargs)
    box, widgets = tfnw.vector_slider(ndim=3,
                                      dtype=float,
                                      callback=_border_update,
                                      initial_values=tf.system.get_scene_box_color().as_list(),
                                      label=label,
                                      field_kwargs=def_field_kwargs)
    if show:
        display(box)
    return box, *widgets
