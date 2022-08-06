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

/**
 * @file tfC_util.h
 * 
 */

#ifndef _WRAPS_C_TFC_UTIL_H_
#define _WRAPS_C_TFC_UTIL_H_

#include "tf_port_c.h"

// Handles

struct CAPI_EXPORT tfPointsTypeHandle {
    unsigned int Sphere;
    unsigned int SolidSphere;
    unsigned int Disk;
    unsigned int SolidCube;
    unsigned int Cube;
    unsigned int Ring;
};


////////////////
// PointsType //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfPointsType_init(struct tfPointsTypeHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the current seed for the pseudo-random number generator
 * 
 */
CAPI_FUNC(HRESULT) tfGetSeed(unsigned int *seed);

/**
 * @brief Set the current seed for the pseudo-random number generator
 * 
 * @param seed new seed value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfSetSeed(unsigned int seed);

/**
 * @brief Get the names of all available colors
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfColor3Names(char ***names, unsigned int *numNames);

/**
 * @brief Calculate the coefficients of a plane equation from a point and normal of the plane
 * 
 * @param point point on the plane
 * @param normal normal of the plane
 * @param planeEq plane equation coefficients
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPlaneEquationFPN(tfFloatP_t *point, tfFloatP_t *normal, tfFloatP_t **planeEq);

/**
 * @brief Calculate a point and normal of a plane equation
 * 
 * @param planeEq coefficients of the plane equation
 * @param point point on the plane
 * @param normal normal of the plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPlaneEquationTPN(tfFloatP_t *planeEq, tfFloatP_t **point, tfFloatP_t **normal);

/**
 * @brief Get the coordinates of a random point in a kind of shape. 
 * 
 * Currently supports sphere, disk, solid cube and solid sphere. 
 * 
 * @param kind kind of shape
 * @param dr thickness parameter; only applicable to solid sphere kind
 * @param phi0 angle lower bound; only applicable to solid sphere kind
 * @param phi1 angle upper bound; only applicable to solid sphere kind
 * @param x x-coordinate of random point
 * @param y y-coordinate of random point
 * @param z z-coordinate of random point
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRandomPoint(unsigned int kind, tfFloatP_t dr, tfFloatP_t phi0, tfFloatP_t phi1, tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z);

/**
 * @brief Get the coordinates of random points in a kind of shape. 
 * 
 * Currently supports sphere, disk, solid cube and solid sphere.
 * 
 * @param kind kind of shape
 * @param n number of points
 * @param dr thickness parameter; only applicable to solid sphere kind
 * @param phi0 angle lower bound; only applicable to solid sphere kind
 * @param phi1 angle upper bound; only applicable to solid sphere kind
 * @param x coordinates of random points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRandomPoints(unsigned int kind, int n, tfFloatP_t dr, tfFloatP_t phi0, tfFloatP_t phi1, tfFloatP_t **x);

/**
 * @brief Get the coordinates of uniform points in a kind of shape. 
 * 
 * Currently supports ring and sphere. 
 * 
 * @param kind kind of shape
 * @param n number of points
 * @param x coordinates of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfPoints(unsigned int kind, int n, tfFloatP_t **x);

/**
 * @brief Get the coordinates of a uniformly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticlesX number of particles along x-direction of filling axes (>=2)
 * @param nParticlesY number of particles along y-direction of filling axes (>=2)
 * @param nParticlesZ number of particles along z-direction of filling axes (>=2)
 * @param x coordinates of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFilledCubeUniform(
    tfFloatP_t *corner1, 
    tfFloatP_t *corner2, 
    unsigned int nParticlesX, 
    unsigned int nParticlesY, 
    unsigned int nParticlesZ, 
    tfFloatP_t **x
);

/**
 * @brief Get the coordinates of a randomly filled cube. 
 * 
 * @param corner1 first corner of cube
 * @param corner2 second corner of cube
 * @param nParticles number of points in the cube
 * @param x coordinates of points
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFilledCubeRandom(tfFloatP_t *corner1, tfFloatP_t *corner2, int nParticles, tfFloatP_t **x);

/**
 * @brief Get the coordinates of an icosphere. 
 * 
 * @param subdivisions number of subdivisions
 * @param phi0 angle lower bound
 * @param phi1 angle upper bound
 * @param verts returned vertices
 * @param numVerts number of vertices
 * @param inds returned indices
 * @param numInds number of indices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIcosphere(
    unsigned int subdivisions, 
    tfFloatP_t phi0, 
    tfFloatP_t phi1,
    tfFloatP_t **verts, 
    unsigned int *numVerts,
    int **inds, 
    unsigned int *numInds
);

/**
 * @brief Generates a randomly oriented vector with random magnitude 
 * with given mean and standard deviation according to a normal 
 * distribution.
 * 
 * @param mean magnitude mean
 * @param std magnitude standard deviation
 * @param x x-component of vector
 * @param y y-component of vector
 * @param z z-component of vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRandomVector(tfFloatP_t mean, tfFloatP_t std, tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z);

/**
 * @brief Generates a randomly oriented unit vector.
 * 
 * @param x x-component of vector
 * @param y y-component of vector
 * @param z z-component of vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRandomUnitVector(tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z);

/**
 * @brief Get the compiler features names and flags
 * 
 * @param names feature names
 * @param flags feature flags
 * @param numFeatures number of features
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUtilGetFeaturesMap(char ***names, bool **flags, unsigned int *numFeatures);

/**
 * @brief Get the current wall time
 * 
 * @param wtime wall time
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUtilWallTime(double *wtime);

/**
 * @brief Get the current CPU time
 * 
 * @param cputime CPU time
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfUtilCPUTime(double *cputime);

#endif // _WRAPS_C_TFC_UTIL_H_