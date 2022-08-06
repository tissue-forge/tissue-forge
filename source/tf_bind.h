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
 * @file tf_bind.h
 * 
 */

#pragma once
#ifndef _SOURCE_TF_BIND_H_
#define _SOURCE_TF_BIND_H_

#include <tfBoundaryConditions.h>
#include <tfForce.h>
#include <tfPotential.h>
#include <tfBond.h>

#include <utility>


namespace TissueForge::bind { 


    /**
     * @brief Bind a potential to a pair of particles
     * 
     * @param p The potential
     * @param a The first particle
     * @param b The second particle
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) particles(Potential *p, Particle *a, Particle *b);

    /**
     * @brief Bind a potential to a pair of particle types. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param a The first type
     * @param b The second type
     * @param bound Flag signifying whether this potential exclusively operates on particles of different clusters, optional
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) types(Potential *p, ParticleType *a, ParticleType *b, bool bound=false);

    /**
     * @brief Bind a potential to a pair of particle type and all boundary conditions. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param t The particle type
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) boundaryConditions(Potential *p, ParticleType *t);
    
    /**
     * @brief Bind a potential to a pair of particle type and a boundary conditions. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param bc The boundary condition
     * @param t The particle type
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) boundaryCondition(Potential *p, BoundaryCondition *bc, ParticleType *t);

    /**
     * @brief Bind a force to a particle type
     * 
     * @param force The force
     * @param a_type The particle type
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) force(Force *force, ParticleType *a_type);

    /**
     * @brief Bind a force to a particle type with magnitude proportional to a species amount
     * 
     * @param force The force
     * @param a_type The particle type
     * @param coupling_symbol The symbol of the species
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) force(Force *force, ParticleType *a_type, const std::string& coupling_symbol);

    /**
     * @brief Create bonds for a set of pairs of particles
     * 
     * @param potential The bond potential
     * @param particles The list of particles
     * @param cutoff Interaction cutoff
     * @param pairs Pairs to bind
     * @param half_life Bond half life
     * @param bond_energy Bond dissociation energy
     * @param flags Bond flags
     * @param out List of created bonds
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) bonds(
        Potential* potential,
        ParticleList *particles, 
        const FloatP_t &cutoff, 
        std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs=NULL, 
        const FloatP_t &half_life=std::numeric_limits<FloatP_t>::max(), 
        const FloatP_t &bond_energy=std::numeric_limits<FloatP_t>::max(), 
        uint32_t flags=0, 
        std::vector<BondHandle*> **out=NULL
    );

    CPPAPI_FUNC(HRESULT) sphere(
        Potential *potential,
        const int &n,
        FVector3 *center=NULL,
        const FloatP_t &radius=1.0,
        std::pair<FloatP_t, FloatP_t> *phi=NULL, 
        ParticleType *type=NULL, 
        std::pair<ParticleList*, std::vector<BondHandle*>*> **out=NULL
    );

};

#endif // _SOURCE_TF_BIND_H_