/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tf_bindPy.h"
#include <tf_bind.h>


namespace TissueForge::py {


    HRESULT particles(Potential *p, Particle *a, Particle *b) { return bind::particles(p, a, b); }

    HRESULT types(Potential *p, ParticleType *a, ParticleType *b, bool bound) { return bind::types(p, a, b, bound); }

    HRESULT boundary_conditions(Potential *p, ParticleType *t) { return bind::boundaryConditions(p, t); }
    
    HRESULT boundary_condition(Potential *p, BoundaryCondition *bc, ParticleType *t) { return bind::boundaryCondition(p, bc, t); }

    HRESULT force(Force *force, ParticleType *a_type) { return bind::force(force, a_type); }

    HRESULT force(Force *force, ParticleType *a_type, const std::string& coupling_symbol) { return bind::force(force, a_type, coupling_symbol); }

    std::vector<BondHandle> _bondsPy(
        Potential* potential,
        ParticleList *particles, 
        const FloatP_t &cutoff, 
        std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs, 
        FloatP_t *half_life, 
        FloatP_t *bond_energy, 
        uint32_t flags) 
    {
        auto _half_life = half_life ? *half_life : std::numeric_limits<FloatP_t>::max();
        auto _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<FloatP_t>::max();

        std::vector<BondHandle> _result;
        bind::bonds(potential, *particles, cutoff, pairs, _half_life, _bond_energy, flags, &_result);
        return _result;
    }

    std::vector<BondHandle> _spherePy(
        ParticleList *particles,
        Potential *potential, 
        const int &n, 
        FVector3 *center, 
        const FloatP_t &radius, 
        FloatP_t *phi0, 
        FloatP_t *phi1, 
        ParticleType *type
    ) {
        std::pair<FloatP_t, FloatP_t> _phi, *phi = NULL;
        if(phi0 && phi1) {
            _phi = std::pair<FloatP_t, FloatP_t>(*phi0, *phi1);
            phi = &_phi;
        }

        std::vector<BondHandle> result;
        bind::sphere(potential, n, center,radius, phi, type, particles, &result);
        return result;
    }
}
