# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022, 2023 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# ******************************************************************************

from . import tissue_forge as tf
from . import rendering
from . import state


def _colormap_from_dict(d: dict, type_instance):
    args = ['Rainbow', 0.0, 1.0]
    if 'map' in d.keys():
        args[0] = d['map']
    if 'range' in d.keys():
        r = d['range']
        args[1] = r[0]
        args[2] = r[1]
    mapper = rendering.ColorMapper(*args)

    if 'position' in d.keys():
        if d['position'].lower() == 'x':
            mapper.set_map_particle_position_x()
        elif d['position'].lower() == 'y':
            mapper.set_map_particle_position_y()
        elif d['position'].lower() == 'z':
            mapper.set_map_particle_position_z()

    elif 'velocity' in d.keys():
        if d['velocity'].lower() == 'x':
            mapper.set_map_particle_velocity_x()
        elif d['velocity'].lower() == 'y':
            mapper.set_map_particle_velocity_y()
        elif d['velocity'].lower() == 'z':
            mapper.set_map_particle_velocity_z()
        elif d['velocity'].lower() == 'speed':
            mapper.set_map_particle_speed()

    elif 'force' in d.keys():
        if d['force'].lower() == 'x':
            mapper.set_map_particle_force_x()
        elif d['force'].lower() == 'y':
            mapper.set_map_particle_force_y()
        elif d['force'].lower() == 'z':
            mapper.set_map_particle_force_z()

    elif 'species' in d.keys():
        mapper.set_map_particle_species(type_instance, d['species'])

    return mapper


class ParticleTypeSpec:
    """
    Interface for class-centric design of ParticleType
    """

    mass = None
    """Particle type mass"""

    charge = None
    """Particle type charge"""

    radius = None
    """Particle type radius"""

    target_energy = None
    """Particle type target energy"""

    minimum_radius = None
    """Particle type minimum radius"""

    eps = None
    """Particle type nonbonded interaction parameter"""

    rmin = None
    """Particle type nonbonded interaction parameter"""

    dynamics = None
    """Particle type dynamics flag"""

    frozen = None
    """Particle type frozen flag"""

    name = None
    """Particle type name"""

    name2 = None
    """Particle type second name"""

    style = None
    """
    Particle type style dictionary specification. 
    
    Basic rendering details can be specified as a dictionary, like color and visibility, 
    
    .. code:: python
    
        style = {'color': 'CornflowerBlue', 'visible': False}

    This declaration is the same as performing operations on a type after registration, 
    
    .. code:: python
    
        ptype: ParticleType
        ptype.style.setColor('CornflowerBlue')
        ptype.style.setVisible(False)
    
    Rendering instead by species and species amount uses specification for a color mapper, 
    
    .. code:: python
    
        style = {'colormap': {'species': 'S1', 'map': 'rainbow', 'range': (0, 10)}}

    This declaration is the same as performing operations on a type after registration, 
    
    .. code:: python
    
        ptype: ParticleType
        ptype.style.mapper = rendering.ColorMapper('rainbow', 0, 10)
        ptype.style.mapper.set_map_particle_species(ptype, 'S1')

    Valid entries for data mapping are as follows, 

    * 'force': 'x', 'y' or 'z' for x-, y- or z-component of force
    * 'position': 'x', 'y' or 'z' for x-, y- or z-component of position
    * 'species': Name of a species attached to a particle type
    * 'velocity': 'x', 'y', 'z' or 'speed' for x-, y-, z-component or magnitude of velocity
    
    """

    species = None
    """
    Particle type list of species by name, if any. Species are automatically created and populated in the state 
    vector of the type and all created particles. 
    """

    __tf_properties__ = [
        'mass',
        'charge',
        'radius',
        'target_energy',
        'minimum_radius',
        'eps',
        'rmin',
        'dynamics',
        'frozen',
        'name',
        'name2',
        'style',
        'species'
    ]
    """All defined particle type properties"""

    @classmethod
    def get(cls):
        """
        Get the engine type that corresponds to this class.

        The type is automatically registered as necessary.

        :return: registered type instance
        :rtype: tissue_forge.ParticleType
        """

        name = cls.name
        if name is None:
            name = cls.__name__

        type_instance = tf.ParticleType_FindFromName(name)
        if type_instance is not None:
            return type_instance

        type_instance = tf.ParticleType(noReg=True)

        props_to_copy = [n for n in cls.__tf_properties__ if n not in ['name', 'species', 'style', 'types']]
        props_to_assign = {prop_name: getattr(cls, prop_name) for prop_name in props_to_copy}
        props_to_assign['name'] = name

        for prop_name, prop_value in props_to_assign.items():
            if prop_value is not None:
                setattr(type_instance, prop_name, prop_value)

        if cls.species is not None:
            type_instance.species = state.SpeciesList()
            for s in cls.species:
                type_instance.species.insert(s)

        if cls.style is not None:
            cls.style: dict
            if 'color' in cls.style.keys():
                type_instance.style.setColor(cls.style['color'])
            if 'visible' in cls.style.keys():
                type_instance.style.setVisible(cls.style['visible'])
            if 'colormap' in cls.style.keys():
                type_instance.style.mapper = _colormap_from_dict(cls.style['colormap'], type_instance)

        type_instance.registerType()
        type_instance = type_instance.get()

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance


class ClusterTypeSpec(ParticleTypeSpec):
    """
    Interface for class-centric design of ClusterParticleType
    """

    types = None
    """List of constituent types of the cluster, if any"""

    __tf_properties__ = ParticleTypeSpec.__tf_properties__ + [
        'types'
    ]
    """All defined cluster type properties"""

    @classmethod
    def get(cls):
        """
        Get the engine type that corresponds to this class.

        The type is automatically registered as necessary.

        :return: registered type instance
        :rtype: tissue_forge.ClusterParticleType
        """

        name = cls.name
        if name is None:
            name = cls.__name__

        type_instance = tf.ClusterParticleType_FindFromName(name)
        if type_instance is not None:
            return type_instance

        type_instance = tf.ClusterParticleType(noReg=True)

        props_to_copy = [n for n in cls.__tf_properties__ if n not in ['name', 'species', 'style', 'types']]
        props_to_assign = {prop_name: getattr(cls, prop_name) for prop_name in props_to_copy}
        props_to_assign['name'] = name

        for prop_name, prop_value in props_to_assign.items():
            if prop_value is not None:
                setattr(type_instance, prop_name, prop_value)

        if cls.species is not None:
            type_instance.species = state.SpeciesList()
            for s in cls.species:
                type_instance.species.insert(s)

        if cls.style is not None:
            cls.style: dict
            if 'color' in cls.style.keys():
                type_instance.style.setColor(cls.style['color'])
            if 'visible' in cls.style.keys():
                type_instance.style.setVisible(cls.style['visible'])
            if 'colormap' in cls.style.keys():
                type_instance.style.mapper = _colormap_from_dict(cls.style['colormap'], type_instance)

        type_instance.registerType()
        type_instance = type_instance.get()

        if cls.types is not None:
            for t in cls.types:
                type_instance.types.insert(t.id)

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance
