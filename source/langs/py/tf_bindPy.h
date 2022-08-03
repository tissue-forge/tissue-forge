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

#ifndef _SOURCE_LANGS_PY_TF_BINDPY_H_
#define _SOURCE_LANGS_PY_TF_BINDPY_H_

#include "tf_py.h"
#include "tfBoundaryConditionsPy.h"
#include "tfForcePy.h"
#include "tfPotentialPy.h"

#include <tfParticle.h>


namespace TissueForge::py {


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
    CPPAPI_FUNC(HRESULT) boundary_conditions(Potential *p, ParticleType *t);
    
    /**
     * @brief Bind a potential to a pair of particle type and a boundary conditions. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param bcs The boundary condition
     * @param t The particle type
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) boundary_condition(Potential *p, BoundaryCondition *bc, ParticleType *t);

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
     * @return std::vector<BondHandle*> 
     */
    CPPAPI_FUNC(std::vector<BondHandle*>) _bondsPy(
        Potential* potential,
        ParticleList *particles, 
        const FloatP_t &cutoff, 
        std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs=NULL, 
        FloatP_t *half_life=NULL, 
        FloatP_t *bond_energy=NULL, 
        uint32_t flags=0
    );

    CPPAPI_FUNC(std::vector<BondHandle*>) _spherePy(
        ParticleList *particles,
        Potential *potential, 
        const int &n, 
        FVector3 *center=NULL, 
        const FloatP_t &radius=1.0, 
        FloatP_t *phi0=NULL, 
        FloatP_t *phi1=NULL, 
        ParticleType *type=NULL
    );

}

// note: bonds and sphere -> _bondsPy and _spherePy

#endif // _SOURCE_LANGS_PY_TF_BINDPY_H_