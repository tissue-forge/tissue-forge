import tissue_forge as tf
import random

tf.init(cutoff=3)
tf.Logger.enableConsoleLogging(tf.Logger.ERROR)


class AType(tf.ParticleTypeSpec):
    mass = 40
    radius = 0.4
    dynamics = tf.Overdamped
    style = {'color': 'red'}


class BType(AType):
    style = {'color': 'blue'}


A = AType.get()
B = BType.get()

pot_aa = tf.Potential.morse(d=3, a=5, min=-0.8, max=2)
pot_bb = tf.Potential.morse(d=3, a=5, min=-0.8, max=2)
pot_ab = tf.Potential.morse(d=0.3, a=5, min=-0.8, max=2)

tf.bind.types(pot_aa, A, A)
tf.bind.types(pot_bb, B, B)
tf.bind.types(pot_ab, A, B)

rforce = tf.Force.random(mean=0.0, std=50)
tf.bind.force(rforce, A)
tf.bind.force(rforce, B)

tf.system.show_widget_time()
tf.system.show_widget_particle_number()
# tf.system.show_widget_bond_number()

idx_noise = tf.system.add_widget_output_float(rforce.std, 'Noise')


def update_display_noise():
    tf.system.set_widget_output_float(idx_noise, rforce.std)


def increase_noise():
    rforce.std += 1
    update_display_noise()


def decrease_noise():
    rforce_std = rforce.std
    rforce.std = max(1E-12, rforce.std - 1)
    if rforce_std != rforce.std:
        update_display_noise()


tf.system.add_widget_button(increase_noise, '+Noise')
tf.system.add_widget_button(decrease_noise, '-Noise')


part_incr = 1000


def set_part_incr(val: int):
    global part_incr
    if val > 0:
        part_incr = val


tf.system.add_widget_input_int(set_part_incr, part_incr, 'Part incr.')


def add_parts():
    for _ in range(part_incr):
        [A, B][0 if random.random() < 0.5 else 1]()


def rem_parts():
    _part_incr = min(part_incr, len(tf.Universe.particles))
    if _part_incr == 0:
        return
    parts = list(tf.Universe.particles)
    random.shuffle(parts)
    for ph in parts[:_part_incr]:
        ph.destroy()


tf.system.add_widget_button(add_parts, '+Parts')
tf.system.add_widget_button(rem_parts, '-Parts')

tf.system.set_widget_font_size(15)
tf.Logger.enableConsoleLogging(tf.Logger.ERROR)

tf.system.set_widget_text_color('RED')
tf.system.set_widget_background_color("blue")
tf.run()
