/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%{

#include "tf_util.h"

%}


%ignore Differentiator;
%ignore WallTime;
%ignore PerformanceTimer;
%ignore color3_Names;
%ignore TissueForge::RandomType;
%ignore Color3_Parse;
%ignore randomEngine();
%ignore aligned_Malloc(size_t, size_t);
%ignore nextPrime(const uint64_t&);
%ignore findPrimes(uint64_t, int, uint64_t*);
%ignore aligned_Free(void*);

%rename(_icosphere) TissueForge::icosphere;
%rename(_get_features_map) TissueForge::util::getFeaturesMap;

%rename(_util_CompileFlags) CompileFlags;

%include "tf_util.h"

%pythoncode %{
    
    from enum import Enum as EnumPy

    class PointsType(EnumPy):
        Sphere = PointsType_Sphere
        """Unit sphere

        :meta hide-value:
        """

        SolidSphere = PointsType_SolidSphere
        """Unit sphere shell

        :meta hide-value:
        """

        Disk = PointsType_Disk
        """Unit disk

        :meta hide-value:
        """

        Cube = PointsType_Cube
        """Unit hollow cube

        :meta hide-value:
        """

        SolidCube = PointsType_SolidCube
        """Unit solid cube

        :meta hide-value:
        """

        Ring = PointsType_Ring
        """Unit ring

        :meta hide-value:
        """

    def random_point(kind: int, dr: float = None, phi0: float = None, phi1: float = None):
        """
        Get the coordinates of a random point in a kind of shape.
    
        Currently supports :attr:`PointsType.Sphere`, :attr:`PointsType.Disk`, :attr:`PointsType.SolidCube` and :attr:`PointsType.SolidSphere`.
    
        :param kind: kind of shape
        :param dr: thickness parameter; only applicable to solid sphere kind
        :param phi0: angle lower bound; only applicable to solid sphere kind
        :param phi1: angle upper bound; only applicable to solid sphere kind
        :return: coordinates of random points
        :rtype: :class:`FVector3`
        """
        
        args = [kind]
        if dr is not None:
            args.append(dr)
            if phi0 is not None:
                args.append(phi0)
                if phi1 is not None:
                    args.append(phi1)
        return randomPoint(*args)

    def random_points(kind: int, n: int = 1, dr: float = None, phi0: float = None, phi1: float = None):
        """
        Get the coordinates of random points in a kind of shape.
    
        Currently supports :attr:`PointsType.Sphere`, :attr:`PointsType.Disk`, :attr:`PointsType.SolidCube` and :attr:`PointsType.SolidSphere`.
    
        :param kind: kind of shape
        :param n: number of points
        :param dr: thickness parameter; only applicable to solid sphere kind
        :param phi0: angle lower bound; only applicable to solid sphere kind
        :param phi1: angle upper bound; only applicable to solid sphere kind
        :return: coordinates of random points
        :rtype: list of :class:`FVector3`
        """
        
        args = [kind, n]
        if dr is not None:
            args.append(dr)
            if phi0 is not None:
                args.append(phi0)
                if phi1 is not None:
                    args.append(phi1)
        return list(randomPoints(*args))

    def filled_cube_uniform(corner1,
                            corner2,
                            num_parts_x: int = 2,
                            num_parts_y: int = 2,
                            num_parts_z: int = 2):
        """
        Get the coordinates of a uniformly filled cube.
    
        :param corner1: first corner of cube
        :type corner1: list of float or :class:`FVector3`
        :param corner2: second corner of cube
        :type corner2: list of float or :class:`FVector3`
        :param num_parts_x: number of particles along x-direction of filling axes (>=2)
        :param num_parts_y: number of particles along y-direction of filling axes (>=2)
        :param num_parts_z: number of particles along z-direction of filling axes (>=2)
        :return: coordinates of uniform points
        :rtype: list of :class:`FVector3`
        """
    
        return list(filledCubeUniform(FVector3(corner1), FVector3(corner2), num_parts_x, num_parts_y, num_parts_z))

    def filled_cube_random(corner1, corner2, num_particles: int):
        """
        Get the coordinates of a randomly filled cube.
    
        :param corner1: first corner of cube
        :type corner1: list of float or :class:`FVector3`
        :param corner2: second corner of cube
        :type corner2: list of float or :class:`FVector3`
        :param num_particles: number of particles
        :return: coordinates of random points
        :rtype: list of :class:`FVector3`
        """
    
        return list(filledCubeUniform(FVector3(corner1), FVector3(corner2), num_particles))

    def icosphere(subdivisions: int, phi0: float, phi1: float):
        """
        Get the coordinates of an icosphere.
    
        :param subdivisions: number of subdivisions
        :param phi0: angle lower bound
        :param phi1: angle upper bound
        :return: vertices and indices
        :rtype: (list of :class:`FVector3`, list of int)
        """

        verts = vectorFVector3()
        inds = vectorl()
        _icosphere(subdivisions, phi0, phi1, verts, inds)
        return list(verts), list(inds)

    def color3_names():
        """
        Get the names of all available colors

        :rtype: list of str
        """
    
        return list(color3Names())

    random_vector = randomVector

    random_unit_vector = randomUnitVector

    plane_equation = planeEquation

    get_seed = getSeed

    set_seed = setSeed

    def _util_get_features_map():
        return _get_features_map.asdict()
%}
