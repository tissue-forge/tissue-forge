import tissue_forge as tf
import random

# new simulator
tf.init(cutoff=3)

# enabling exception logging 
# tf.Logger.enableConsoleLogging(tf.Logger.ERROR)

# defining particle types 
class AType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'red'}


class BType(AType):
    style = {'color': 'blue'}

A = AType.get()
B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = tf.Potential.morse(d=3, a=5, max=3)
pot_bb = tf.Potential.morse(d=3, a=5, max=3)
pot_ab = tf.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
tf.bind.types(pot_aa, A, A)
tf.bind.types(pot_bb, B, B)
tf.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = tf.Force.random(mean=0, std=50)

# bind it just like any other force
tf.bind.force(rforce, A)
tf.bind.force(rforce, B)

# Display the simulation time in the widget
tf.system.show_widget_time()

# Display the number of particles in the simulation in the widget
tf.system.show_widget_particle_number()

# Display the number of bonds in the simulation in the widget
# tf.system.show_widget_bond_number()

# Add a widget output to display the current standard deviation of the random force (noise level)
idx_noise = tf.system.add_widget_output_float(rforce.std, 'Noise')

def update_display_noise():
    """
    Update the displayed noise level in the widget.
    This function sets the widget output to the current standard deviation of the random force.
    """
    tf.system.set_widget_output_float(idx_noise, rforce.std)

def increase_noise():
    """
    Increase the noise level in the simulation.
    This function increments the standard deviation of the random force and updates the widget display.
    """
    rforce.std += 1
    update_display_noise()

def decrease_noise():
    """
    Decrease the noise level in the simulation.
    This function decrements the standard deviation of the random force, ensuring it does not go below a minimal value,
    and updates the widget display.
    """
    rforce_std = rforce.std
    rforce.std = max(1E-12, rforce.std - 1)  # Ensure noise does not go below a very small positive value
    if rforce_std != rforce.std:
        update_display_noise()


# Default number of particles to add or remove in each operation
part_incr = 1000

def set_part_incr(val: int):
    """
    Set the number of particles to add or remove in each operation.
    This function updates the global `part_incr` variable to the specified value if it is positive.
    """
    global part_incr
    if val > 0:
        part_incr = val

# Add an input field to the UI to set the number of particles to add or remove
tf.system.add_widget_input_int(set_part_incr, part_incr, 'Part incr.')

def add_parts():
    """
    Add particles to the simulation.
    This function adds `part_incr` particles of type `A` or `B` to the simulation.
    """
    for _ in range(part_incr):
        [A, B][0 if random.random() < 0.5 else 1]()

def rem_parts():
    """
    Remove particles from the simulation.
    This function removes up to `part_incr` particles from the simulation, shuffling the particle list to randomize
    which particles are removed.
    """
    _part_incr = min(part_incr, len(tf.Universe.particles))  # Ensure not to remove more particles than present
    if _part_incr == 0:
        return
    parts = list(tf.Universe.particles)
    random.shuffle(parts)
    for ph in parts[:_part_incr]:
        ph.destroy()


class WidgetRendererError(Exception):
    """Base class for exceptions in WidgetRenderer."""
    pass

class FontSizeError(WidgetRendererError):
    """Exception raised for errors in setting font size."""
    pass

class ColorValueError(WidgetRendererError):
    """Exception raised for errors in setting color values."""
    pass

class ButtonError(WidgetRendererError):
    """Exception raised for errors in button handling."""
    pass

class FieldError(WidgetRendererError):
    """Exception raised for errors in field handling."""
    pass

def set_widget_font_size(size):
    """
    Set the font size for the widget text.
    This function attempts to set the widget's font size using the provided `size` parameter.
    If the operation fails (indicated by a non-zero result), a FontSizeError is raised.

    Args:
        size (float): The desired font size for the widget.

    Raises:
        FontSizeError: If setting the font size fails.
    """
    try:
        result = tf.system.set_widget_font_size(size)
        print(f"Result of setting font: {result}")
        if result != 0:  # Assuming 0 indicates success (S_OK)
            raise FontSizeError(f"Failed to set font size: {size}. Ensure size is between 5 and 30.")
    except FontSizeError as e:
        print(e)
        raise e

def set_widget_text_color(color_name):
    """
    Set the text color for the widget.
    This function attempts to set the widget's text color using the provided `color_name` parameter.
    If the operation fails (indicated by a non-zero result), a ColorValueError is raised.

    Args:
        color_name (str): The desired text color for the widget, specified by name.

    Raises:
        ColorValueError: If setting the text color fails.
    """
    try:
        result = tf.system.set_widget_text_color(color_name)
        print(f"Result of setting text: {result}")
        if result != 0:  # Assuming 0 indicates success (S_OK)
            raise ColorValueError(f"Failed to set text color: {color_name}.")
    except ColorValueError as e:
        print(e)
        raise e

def set_widget_background_color(color_name):
    """
    Set the background color for the widget.
    This function attempts to set the widget's background color using the provided `color_name` parameter.
    If the operation fails (indicated by a non-zero result), a ColorValueError is raised.

    Args:
        color_name (str): The desired background color for the widget, specified by name.

    Raises:
        ColorValueError: If setting the background color fails.
    """
    try:
        result = tf.system.set_widget_background_color(color_name)
        print(f"Result of setting color: {result}")
        if result > 0:  # Assuming 0 indicates success (S_OK)
            raise ColorValueError(f"Failed to set background color: {color_name}.")
    except ColorValueError as e:
        print(e)
        raise e

def add_button(cb, label):
    """
    Add a button to the widget interface.
    This function attempts to add a button with the specified `label` and associate it with a callback `cb`.
    If the operation fails (indicated by a negative result), a ButtonError is raised.

    Args:
        cb (function): The callback function to execute when the button is clicked.
        label (str): The label text for the button.

    Raises:
        ButtonError: If adding the button fails.
    """
    try:
        result = tf.system.add_widget_button(cb, label)
        if result < 0:  # Assuming negative values indicate errors
            raise ButtonError(f"Failed to add button with label: {label}")
    except ButtonError as e:
        print(e)
        raise e

def add_output_field(val, label=None):
    """
    Add an output field to the widget interface.
    This function attempts to add an output field to the UI, optionally with a label. 
    The output field displays data that can be updated dynamically to reflect changes in the system.

    Args:
        val (any): The value to be displayed in the output field.
        label (str, optional): The label text for the output field. Defaults to None.

    Raises:
        FieldError: If adding the output field fails.
    """
    try:
        if label:
            # Add an output field with a label
            result = tf.system.add_widget_output_float(val, label)
        else:
            # Add an output field without a label
            result = tf.system.add_widget_output_float(val)
        
        # Check if the result indicates an error (assuming negative values indicate errors)
        if result < 0:
            raise FieldError(f"Failed to add output field with value: {val} and label: {label}")
    except FieldError as e:
        # Handle and log the error
        print(e)
        raise e


def add_input_field(cb, val, label=None):
    """
    Add an input field to the widget interface.
    This function attempts to add an input field with the specified value and optional label,
    associating it with a callback function. The input field allows users to enter data that 
    can be processed or used within the application.

    Args:
        cb (function): The callback function to execute when the input field's value changes.
        val (any): The default value for the input field.
        label (str, optional): The label text for the input field. Defaults to None.

    Raises:
        FieldError: If adding the input field fails.
    """
    try:
        if label:
            # Add an input field with a label
            result = tf.system.add_widget_input_int(cb, val, label)
        else:
            # Add an input field without a label
            result = tf.system. add_widget_input_int(cb, val)
        
        # Check if the result indicates an error (assuming negative values indicate errors)
        if result < 0:
            raise FieldError(f"Failed to add input field with value: {val} and label: {label}")
    except FieldError as e:
        # Handle and log the error
        print(e)
        raise e

# Example Usage
add_button(increase_noise, '+Noise')
add_button(decrease_noise, '-Noise')
add_output_field(rforce.std, 'Noise')

add_input_field(set_part_incr, part_incr, 'Part incr.')

add_button(add_parts, '+Parts')
add_button(rem_parts, '-Parts')

set_widget_font_size(10)  # Valid size
# set_widget_font_size(50)  # Invalid size, should raise FontSizeError

# set_widget_text_color("red")  # Assuming valid color name
set_widget_text_color("invalid_color")  # Invalid, should raise ColorValueError

set_widget_background_color("blue")  # Assuming valid color name
# set_widget_background_color("invalid_color")  # Invalid, should raise ColorValueError


# run the simulator
tf.step(20*tf.Universe.dt)


def test_pass():
    pass


