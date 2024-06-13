/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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
 * @file tf_metrics.h
 * 
 */

#ifndef _SOURCE_TF_METRICS_H_
#define _SOURCE_TF_METRICS_H_

#include "TissueForge_private.h"
#include "tfParticle.h"
#include <set>
#include <vector>


namespace TissueForge::metrics {


    /**
     * @brief Computes the relative position with respect to an origin while 
     * optionally account for boundary conditions. 
     * 
     * If boundaries along a dimension are periodic, then this chooses the 
     * relative coordinate closest to the origin. 
     * 
     * @param pos absolute position
     * @param origin origin
     * @param comp_bc flag to compensate for boundary conditions; default true
     * @return FVector3 relative position with respect to the given origin
     */
    CPPAPI_FUNC(FVector3) relativePosition(const FVector3 &pos, const FVector3 &origin, const bool &comp_bc=true);

    /**
     * find all particles in a neighborhood defined by a point and distance
     */
    CPPAPI_FUNC(ParticleList) neighborhoodParticles(const FVector3 &position, const FloatP_t &dist, const bool &comp_bc=true);

    /**
     * @origin [in] origin of the sphere where we will comptute
     * the local virial tensor.
     * @radius [in] include all partices a given radius in calculation. 
     * @typeIds [in] vector of type ids to indlude in calculation,
     * if empty, includes all particles.
     * @tensor [out] result vector, writes a 3x3 matrix in a row-major in the given
     * location.
     *
     * If periodoc, we don't include the periodic image cells, because we only
     * calculate the forces within the simulation volume.
     */
    CPPAPI_FUNC(HRESULT) calculateVirial(
        FloatP_t *origin,
        FloatP_t radius,
        const std::set<short int> &typeIds,
        FloatP_t *tensor
    );

    /**
     * calculate the virial tensor for a specific list of particles.
     * currently uses center of mass as origin, may change in the
     * future with different flags.
     *
     * flags currently ignored.
     */
    CAPI_FUNC(HRESULT) particlesVirial(
        int32_t *parts,
        uint16_t nr_parts,
        uint32_t flags,
        FloatP_t *tensor
    );

    /**
     * @param result: pointer to float to store result.
     */
    CAPI_FUNC(HRESULT) particlesRadiusOfGyration(int32_t *parts, uint16_t nr_parts, FloatP_t* result);

    /**
     * @param result: pointer to float[3] to store result
     */
    CAPI_FUNC(HRESULT) particlesCenterOfMass(int32_t *parts, uint16_t nr_parts, FloatP_t* result);

    /**
     * @param result: pointer to float[3] to store result.
     */
    CAPI_FUNC(HRESULT) particlesCenterOfGeometry(int32_t *parts, uint16_t nr_parts, FloatP_t* result);

    /**
     * @param result: pointer to float[9] to store result.
     */
    CAPI_FUNC(HRESULT) particlesMomentOfInertia(int32_t *parts, uint16_t nr_parts, FloatP_t* result);

    /**
     * converts cartesian to spherical, writes spherical
     * coords in to result array.
     * return FVector3{radius, theta, phi};
     */
    CPPAPI_FUNC(FVector3) cartesianToSpherical(const FVector3& postion, const FVector3& origin);

    /**
     * Searches and enumerates a location of space for all particles there.
     *
     * Allocates a buffer, and stores the results there.
     *
     * @param part the particle
     * @param radius [optional] the radius of the neighborhood
     * @param typeIds [optional] set of type ids to include. If not given, gets all other parts within radius.
     * @param nr_parts [out] number of parts
     * @param parts [out] newly allocated buffer of particle ids.
     */
    CAPI_FUNC(HRESULT) particleNeighbors(
        struct Particle *part,
        FloatP_t radius,
        const std::set<short int> *typeIds,
        uint16_t *nr_parts,
        int32_t **parts
    );


    /**
     * Creates an array of ParticleList objects.
     */
    std::vector<std::vector<std::vector<ParticleList> > > particleGrid(const iVector3 &shape);

    CAPI_FUNC(HRESULT) particleGrid(const iVector3 &shape, ParticleList *result);

    /**
     * @brief Compute the eigenvalues of a 3x3 matrix
     * 
     * @param mat the matrix
     * @param symmetric flag signifying whether the matrix is symmetric
     * @return eigenvalues
     */
    CPPAPI_FUNC(FVector3) eigenVals(const FMatrix3 &mat, const bool &symmetric=false);

    /**
     * @brief Compute the eigenvalues of a 4x4 matrix
     * 
     * @param mat the matrix
     * @param symmetric flag signifying whether the matrix is symmetric
     * @return eigenvalues
     */
    CPPAPI_FUNC(FVector4) eigenVals(const FMatrix4 &mat, const bool &symmetric=false);

    /**
     * @brief Compute the eigenvectors and eigenvalues of a 3x3 matrix
     * 
     * @param mat the matrix
     * @param symmetric flag signifying whether the matrix is symmetric
     * @return eigenvalues, eigenvectors
     */
    CPPAPI_FUNC(std::pair<FVector3, FMatrix3>) eigenVecsVals(const FMatrix3 &mat, const bool &symmetric=false);

    /**
     * @brief Compute the eigenvectors and eigenvalues of a 4x4 matrix
     * 
     * @param mat the matrix
     * @param symmetric flag signifying whether the matrix is symmetric
     * @return eigenvalues, eigenvectors 
     */
    CPPAPI_FUNC(std::pair<FVector4, FMatrix4>) eigenVecsVals(const FMatrix4 &mat, const bool &symmetric=false);

};


#endif // _SOURCE_TF_METRICS_H_