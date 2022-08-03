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

#ifndef _WRAPS_C_TFC_BIND_H_
#define _WRAPS_C_TFC_BIND_H_

#include "tf_port_c.h"

#include "tfCBond.h"
#include "tfCBoundaryConditions.h"
#include "tfCForce.h"
#include "tfCParticle.h"
#include "tfCPotential.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Bind a potential to a pair of particles
 * 
 * @param p The potential
 * @param a The first particle
 * @param b The second particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindParticles(struct tfPotentialHandle *p, struct tfParticleHandleHandle *a, struct tfParticleHandleHandle *b);

/**
 * @brief Bind a potential to a pair of particle types. 
 * 
 * @param p The potential
 * @param a The first type
 * @param b The second type
 * @param bound Flag signifying whether this potential exclusively operates on particles of different clusters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindTypes(struct tfPotentialHandle *p, struct tfParticleTypeHandle *a, struct tfParticleTypeHandle *b, bool bound);

/**
 * @brief Bind a potential to a pair of particle type and all boundary conditions. 
 * 
 * @param p The potential
 * @param t The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindBoundaryConditions(struct tfPotentialHandle *p, struct tfParticleTypeHandle *t);

/**
 * @brief Bind a potential to a pair of particle type and a boundary conditions. 
 * 
 * @param p The potential
 * @param bcs The boundary condition
 * @param t The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindBoundaryCondition(struct tfPotentialHandle *p, struct tfBoundaryConditionHandle *bc, struct tfParticleTypeHandle *t);

/**
 * @brief Bind a force to a particle type
 * 
 * @param force The force
 * @param a_type The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindForce(struct tfForceHandle *force, struct tfParticleTypeHandle *a_type);

/**
 * @brief Bind a force to a particle type with magnitude proportional to a species amount
 * 
 * @param force The force
 * @param a_type The particle type
 * @param coupling_symbol The symbol of the species
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindForceS(struct tfForceHandle *force, struct tfParticleTypeHandle *a_type, const char *coupling_symbol);

/**
 * @brief Create bonds for a set of pairs of particles
 * 
 * @param potential The bond potential
 * @param particles The list of particles
 * @param cutoff Interaction cutoff
 * @param ppairsA first elements of type pairs that are bonded, optional
 * @param ppairsB second elements of type pairs that are bonded, optional
 * @param numTypes number of passed type pairs
 * @param half_life Bond half life, optional
 * @param bond_energy Bond dissociation energy, optional
 * @param out List of created bonds, optional
 * @param numOut Number of created bonds, optional
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBindBonds(
    struct tfPotentialHandle *potential,
    struct tfParticleListHandle *particles, 
    tfFloatP_t cutoff, 
    struct tfParticleTypeHandle **ppairsA, 
    struct tfParticleTypeHandle **ppairsB, 
    unsigned int numTypes, 
    tfFloatP_t *half_life, 
    tfFloatP_t *bond_energy, 
    struct tfBondHandleHandle **out, 
    unsigned int *numOut
);

#endif // _WRAPS_C_TFC_BIND_H_