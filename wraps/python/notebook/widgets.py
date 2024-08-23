"""Suite of widgets for basic Tissue Forge types"""

import ipywidgets as ipw
from typing import Any, Callable, Dict, Type

import tissue_forge as tf

DEF_KEY = 'default'


_vwidgets_slider: Dict[Type, Type] = {
    int: ipw.IntSlider,
    float: ipw.FloatSlider
}

_vwidgets_textb: Dict[Type, Type] = {
    int: ipw.BoundedIntText,
    float: ipw.BoundedFloatText
}

_vwidgets_text: Dict[Type, Type] = {
    int: ipw.IntText,
    float: ipw.FloatText
}

_boxes: Dict[str, Type] = {
    DEF_KEY: ipw.HBox,
    'horizontal': ipw.HBox,
    'vertical': ipw.VBox
}

_mtypes: Dict[Type, Dict[int, Type]] = {
    float: {
        3: tf.FMatrix3,
        4: tf.FMatrix4
    }
}

_vtypes: Dict[Type, Dict[int, Type]] = {
    int: {
        2: tf.iVector2,
        3: tf.iVector3,
        4: tf.iVector4
    },
    float: {
        2: tf.FVector2,
        3: tf.FVector3,
        4: tf.FVector4
    }
}


def color_converter(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def color_inverter(r: float, g: float, b: float):
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def _get_matrix_type(_ndim: int, _dtype: Type):
    try:
        mtypes = _mtypes[_dtype]
    except KeyError:
        raise ValueError('Invalid type selection')
    try:
        return mtypes[_ndim]
    except KeyError:
        raise ValueError('Invalid dimension')


def _get_vector_type(_ndim: int, _dtype: Type):
    try:
        vtypes = _vtypes[_dtype]
    except KeyError:
        raise ValueError('Invalid type selection')
    try:
        return vtypes[_ndim]
    except KeyError:
        raise ValueError('Invalid dimension')


def _default_vector_slider(_dtype: Type, **kwargs):
    def_kwargs = dict(
        disabled=False,
        continuous_update=True,
        msg_throttle=0.05,
        orientation='horizontal',
        readout=True
    )
    def_kwargs.update(kwargs)
    return _vwidgets_slider[_dtype](**def_kwargs)


def _default_vector_textb(_dtype: Type, **kwargs):
    def_kwargs = dict(
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True
    )
    def_kwargs.update(kwargs)
    return _vwidgets_textb[_dtype](**def_kwargs)


def _default_vector_text(_dtype: Type, **kwargs):
    def_kwargs = dict(
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True
    )
    def_kwargs.update(kwargs)
    return _vwidgets_text[_dtype](**def_kwargs)


def color_picker(callback: Callable[[Any], None] = None,
                 **kwargs):
    def_kwargs = dict(
        concise=False,
        description='Pick a color',
        value='blue',
        disabled=False,
        continuous_update=True
    )
    def_kwargs.update(kwargs)
    widget = ipw.ColorPicker(**def_kwargs)

    def _impl(_callback):

        def _inner(_change):
            _callback(tf.FVector3(color_converter(_change.new)))

        return _inner

    if callback is not None:
        widget.observe(_impl(callback), names='value')

    return widget


def scalar_slider(dtype=float,
                  callback: Callable[[Any], None] = None,
                  label=None,
                  layout=DEF_KEY,
                  initial_value=None,
                  field_kwargs: Dict[str, Any] = None):
    if field_kwargs is None:
        field_kwargs = {}
    widget = _default_vector_slider(dtype, **field_kwargs)
    if initial_value is not None:
        widget.value = initial_value
    widgets = [widget] if label is None else [label, widget]
    box = _boxes[layout](widgets)

    def _impl(_dtype, _widget, _callback):
        def _inner(*args, **kwargs):
            _callback(_dtype(_widget.value))

        return _inner

    if callback is not None:
        widget.observe(_impl(dtype, widget, callback))

    return box, widget


def scalar_text(dtype=float,
                callback: Callable[[Any], None] = None,
                label=None,
                layout=DEF_KEY,
                initial_value=None,
                field_kwargs: Dict[str, Any] = None):
    if field_kwargs is None:
        field_kwargs = {}
    widget = _default_vector_text(dtype, **field_kwargs)
    if initial_value is not None:
        widget.value = initial_value
    widgets = [widget] if label is None else [label, widget]
    box = _boxes[layout](widgets)

    def _impl(_dtype, _widget, _callback):
        def _inner(*args, **kwargs):
            _callback(_dtype(_widget.value))

        return _inner

    if callback is not None:
        widget.observe(_impl(dtype, widget, callback))

    return box, widget


def scalar_textb(dtype=float,
                 callback: Callable[[Any], None] = None,
                 label=None,
                 layout=DEF_KEY,
                 initial_value=None,
                 field_kwargs: Dict[str, Any] = None):
    if field_kwargs is None:
        field_kwargs = {}
    widget = _default_vector_textb(dtype, **field_kwargs)
    if initial_value is not None:
        widget.value = initial_value
    widgets = [widget] if label is None else [label, widget]
    box = _boxes[layout](widgets)

    def _impl(_dtype, _widget, _callback):
        def _inner(*args, **kwargs):
            _callback(_dtype(_widget.value))

        return _inner

    if callback is not None:
        widget.observe(_impl(dtype, widget, callback))

    return box, widget


def vector_slider(ndim=3,
                  dtype=float,
                  callback: Callable[[Any], None] = None,
                  label=None,
                  layout=DEF_KEY,
                  initial_values: list = None,
                  field_kwargs: Dict[int, Dict[str, Any]] = None):
    if field_kwargs is None:
        field_kwargs = {}
    vtype = _get_vector_type(ndim, dtype)
    vector_widgets = [_default_vector_slider(dtype, **field_kwargs.get(i, {})) for i in range(ndim)]
    if initial_values is not None:
        for i, v in enumerate(initial_values):
            vector_widgets[i].value = v
    widgets = vector_widgets if label is None else ([label] + vector_widgets)
    box = _boxes[layout](widgets)

    def _impl(_vtype, _vector_widgets, _callback):
        def _inner(*args, **kwargs):
            _callback(_vtype(*[w.value for w in _vector_widgets]))

        return _inner

    if callback is not None:
        [w.observe(_impl(vtype, vector_widgets, callback), names='value') for w in vector_widgets]

    return box, vector_widgets


def vector_text(ndim=3,
                dtype=float,
                callback: Callable[[Any], None] = None,
                label=None,
                layout=DEF_KEY,
                initial_values: list = None,
                field_kwargs: Dict[int, Dict[str, Any]] = None):
    if field_kwargs is None:
        field_kwargs = {}
    vtype = _get_vector_type(ndim, dtype)
    vector_widgets = [_default_vector_text(dtype, **field_kwargs.get(i, {})) for i in range(ndim)]
    if initial_values is not None:
        for i, v in enumerate(initial_values):
            vector_widgets[i].value = v
    widgets = vector_widgets if label is None else ([label] + vector_widgets)
    box = _boxes[layout](widgets)

    def _impl(_vtype, _vector_widgets, _callback):

        def _inner(*args, **kwargs):
            _callback(_vtype(*[w.value for w in _vector_widgets]))

        return _inner

    if callback is not None:
        [w.observe(_impl(vtype, vector_widgets, callback), names='value') for w in vector_widgets]

    return box, vector_widgets


def vector_textb(ndim=3,
                 dtype=float,
                 callback: Callable[[Any], None] = None,
                 label=None,
                 layout=DEF_KEY,
                 initial_values: list = None,
                 field_kwargs: Dict[int, Dict[str, Any]] = None):
    if field_kwargs is None:
        field_kwargs = {}
    vtype = _get_vector_type(ndim, dtype)
    vector_widgets = [_default_vector_textb(dtype, **field_kwargs.get(i, {})) for i in range(ndim)]
    if initial_values is not None:
        for i, v in enumerate(initial_values):
            vector_widgets[i].value = v
    widgets = vector_widgets if label is None else ([label] + vector_widgets)
    box = _boxes[layout](widgets)

    def _impl(_vtype, _vector_widgets, _callback):

        def _inner(*args, **kwargs):
            _callback(_vtype(*[w.value for w in _vector_widgets]))

        return _inner

    if callback is not None:
        [w.observe(_impl(vtype, vector_widgets, callback), names='value') for w in vector_widgets]

    return box, vector_widgets
