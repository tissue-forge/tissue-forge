# ******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

from tissue_forge.tissue_forge import FVector3
from tissue_forge.tissue_forge import _vertex_solver_bind_body_type as bind_body_type
from tissue_forge.tissue_forge import _vertex_solver_bind_surface_type as bind_surface_type
from tissue_forge.tissue_forge import _vertex_solver_bind_types as bind_types
from tissue_forge.tissue_forge import _vertex_solver_MeshSolver as MeshSolver
from tissue_forge.tissue_forge import _vertex_solver_BodyType as BodyType
from tissue_forge.tissue_forge import _vertex_solver_SurfaceType as SurfaceType
from tissue_forge.tissue_forge import _vertex_solver_Adhesion as Adhesion
from tissue_forge.tissue_forge import _vertex_solver_BodyForce as BodyForce
from tissue_forge.tissue_forge import _vertex_solver_EdgeTension as EdgeTension
from tissue_forge.tissue_forge import _vertex_solver_NormalStress as NormalStress
from tissue_forge.tissue_forge import _vertex_solver_SurfaceAreaConstraint as SurfaceAreaConstraint
from tissue_forge.tissue_forge import _vertex_solver_SurfaceTraction as SurfaceTraction
from tissue_forge.tissue_forge import _vertex_solver_VolumeConstraint as VolumeConstraint

from typing import Dict, List, Optional, Type, Union


class _TypeSpecBase:

    name: Optional[str] = None
    """Type name"""

    @classmethod
    def _bind_type(cls, actor, inst):
        raise NotImplementedError

    @classmethod
    def _bind_generator(cls, _prop, _inst):
        actor = _prop()
        if actor is not None:
            actor.thisown = 0
            cls._bind_type(actor, _inst)

    @classmethod
    def get_name(cls) -> str:
        return cls.name if cls.name is not None else cls.__name__


class SurfaceTypeSpec(_TypeSpecBase):
    """
    Interface for class-centric design of SurfaceType
    """

    style: Optional[dict] = None
    """
    Surface type style dictionary specification. 
    
    Basic rendering details can be specified as a dictionary, like color and visibility, 
    
    .. code:: python
    
        style = {'color': 'CornflowerBlue', 'visible': False}

    This declaration is the same as performing operations on a type after registration, 
    
    .. code:: python
    
        stype: SurfaceType
        stype.style.setColor('CornflowerBlue')
        stype.style.setVisible(False)
    """

    edge_tension_lam: Optional[float] = None
    """Edge tension Lagrange multiplier"""

    edge_tension_order: Optional[int] = None
    """Edge tension order"""

    normal_stress_mag: Optional[float] = None
    """Normal stress magnitude"""

    surface_area_lam: Optional[float] = None
    """Surface area constraint Lagrange multiplier"""

    surface_area_val: Optional[float] = None
    """Surface area constraint target value"""

    surface_traction_comps: Optional[Union[FVector3, List[float]]] = None
    """Surface traction components"""

    adhesion: Optional[Dict[str, float]] = None
    """Adhesion by name and parameter"""

    @classmethod
    def _bind_type(cls, actor, inst):
        return bind_surface_type(actor, inst)

    @classmethod
    def get(cls) -> SurfaceType:

        solver: MeshSolver = MeshSolver.get()
        if solver is None:
            raise RuntimeError('Solver unavailable')

        name = cls.get_name()
        type_instance = SurfaceType.find_from_name(name)
        if type_instance is not None:
            return type_instance

        type_instance = SurfaceType(True)

        # process style

        if cls.style is not None:
            if 'color' in cls.style.keys():
                type_instance.style.setColor(cls.style['color'])
            if 'visible' in cls.style.keys():
                type_instance.style.setVisible(cls.style['visible'])

        # process actor generators

        cls._bind_generator(cls.edge_tension, type_instance)
        cls._bind_generator(cls.normal_stress, type_instance)
        cls._bind_generator(cls.surface_area_constaint, type_instance)
        cls._bind_generator(cls.surface_traction, type_instance)

        solver.register_type(type_instance)
        type_instance.name = name
        type_instance.thisown = 0

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance

    @classmethod
    def edge_tension(cls) -> Optional[EdgeTension]:
        if cls.edge_tension_lam is None:
            return None

        actor_args = [cls.edge_tension_lam]
        if cls.edge_tension_order is not None:
            actor_args.append(cls.edge_tension_order)
        return EdgeTension(*actor_args)

    @classmethod
    def normal_stress(cls) -> Optional[NormalStress]:
        if cls.normal_stress_mag is None:
            return None
        return NormalStress(cls.normal_stress)

    @classmethod
    def surface_area_constaint(cls) -> Optional[SurfaceAreaConstraint]:
        if cls.surface_area_lam is None or cls.surface_area_val is None:
            return None

        return SurfaceAreaConstraint(cls.surface_area_lam, cls.surface_area_val)

    @classmethod
    def surface_traction(cls) -> Optional[SurfaceTraction]:
        if cls.surface_traction_comps is None:
            return None

        surface_traction_comps = FVector3(*cls.surface_traction_comps) if isinstance(cls.surface_traction_comps, list) else cls.surface_traction_comps
        return SurfaceTraction(surface_traction_comps)

    @staticmethod
    def bind_adhesion(specs: List[Type]) -> Dict[str, Dict[str, Adhesion]]:
        specs: List[Type[SurfaceTypeSpec]]

        result = dict()

        for i in range(len(specs)):

            ti = specs[i]
            ti_name = ti.get_name()
            ti_instance = SurfaceType.find_from_name(ti_name)
            if ti_instance is None:
                continue

            for j in range(i, len(specs)):

                tj = specs[j]
                if tj.adhesion is None:
                    continue

                tj_name = tj.get_name()
                tj_instance = SurfaceType.find_from_name(tj_name)
                if tj_instance is None:
                    continue

                for name, val in tj.adhesion.items():
                    if name == ti_name:
                        actor = Adhesion(val)
                        actor.thisown = 0
                        bind_types(actor, ti_instance, tj_instance)

                        try:
                            result[ti_name][tj_name] = actor
                        except KeyError:
                            result[ti_name] = {tj_name: actor}
                        try:
                            result[tj_name][ti_name] = actor
                        except KeyError:
                            result[tj_name] = {ti_name: actor}

        return result


class BodyTypeSpec(_TypeSpecBase):
    """
    Interface for class-centric design of BodyType
    """

    density: Optional[float] = None
    """Mass density"""

    body_force_comps: Optional[Union[FVector3, List[float]]] = None
    """Body force components"""

    surface_area_lam: Optional[float] = None
    """Surface area constraint Lagrange multiplier"""

    surface_area_val: Optional[float] = None
    """Surface area constraint target value"""

    volume_lam: Optional[float] = None
    """Volume constraint Lagrange multiplier"""

    volume_val: Optional[float] = None
    """Volume constraint target value"""

    adhesion: Optional[Dict[str, float]] = None
    """Adhesion by name and parameter"""

    @classmethod
    def _bind_type(cls, actor, inst):
        return bind_body_type(actor, inst)

    @classmethod
    def get(cls) -> BodyType:

        solver: MeshSolver = MeshSolver.get()
        if solver is None:
            raise RuntimeError('Solver unavailable')

        name = cls.get_name()
        type_instance = BodyType.find_from_name(name)
        if type_instance is not None:
            return type_instance

        type_instance = BodyType(True)

        if cls.density is not None:
            type_instance.density = cls.density

        # process actor generators

        cls._bind_generator(cls.body_force, type_instance)
        cls._bind_generator(cls.surface_area_constaint, type_instance)
        cls._bind_generator(cls.volume_constraint, type_instance)

        solver.register_type(type_instance)
        type_instance.name = name
        type_instance.thisown = 0

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance

    @classmethod
    def body_force(cls) -> Optional[BodyForce]:
        if cls.body_force_comps is None:
            return None

        body_force_comps = FVector3(*cls.body_force_comps) if isinstance(cls.body_force_comps, list) else cls.body_force_comps
        return BodyForce(body_force_comps)

    @classmethod
    def surface_area_constaint(cls) -> Optional[SurfaceAreaConstraint]:
        if cls.surface_area_lam is None or cls.surface_area_val is None:
            return None

        return SurfaceAreaConstraint(cls.surface_area_lam, cls.surface_area_val)

    @classmethod
    def volume_constraint(cls) -> Optional[VolumeConstraint]:
        if cls.volume_lam is None or cls.volume_val is None:
            return None

        return VolumeConstraint(cls.volume_lam, cls.volume_val)

    @staticmethod
    def bind_adhesion(specs: List[Type]) -> Dict[str, Dict[str, Adhesion]]:
        specs: List[Type[BodyTypeSpec]]

        result = dict()

        for i in range(len(specs)):

            ti = specs[i]
            ti_name = ti.get_name()
            ti_instance = BodyType.find_from_name(ti_name)
            if ti_instance is None:
                continue

            for j in range(i, len(specs)):

                tj = specs[j]
                if tj.adhesion is None:
                    continue

                tj_name = tj.get_name()
                tj_instance = BodyType.find_from_name(tj_name)
                if tj_instance is None:
                    continue

                for name, val in tj.adhesion.items():
                    if name == ti_name:
                        actor = Adhesion(val)
                        actor.thisown = 0
                        bind_types(actor, ti_instance, tj_instance)

                        try:
                            result[ti_name][tj_name] = actor
                        except KeyError:
                            result[ti_name] = {tj_name: actor}
                        try:
                            result[tj_name][ti_name] = actor
                        except KeyError:
                            result[tj_name] = {ti_name: actor}

        return result
