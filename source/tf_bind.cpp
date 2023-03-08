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

#include "tf_bind.h"
#include <tf_errs.h>
#include <tfParticle.h>
#include <tfEngine.h>
#include "tfError.h"
#include "tfLogger.h"
#include "tf_util.h"

#include <string>


using namespace TissueForge;


/* the error macro. */
#define error(id)   (tf_error(E_FAIL, errs_err_msg[id]))



HRESULT universe_bind_potential(Potential *p, Particle *a, Particle *b) {
    if (!a || !b) return S_OK;

    TF_Log(LOG_DEBUG) << p->name << ", " << a->id << ", " << b->id;

    auto bond = new BondHandle();
    bond->init(p, a->handle(), b->handle());
    return S_OK;
}

HRESULT universe_bind_potential(Potential *p, ParticleType *a, ParticleType *b, bool bound) {
    if (!a || !b) return S_OK;

    TF_Log(LOG_DEBUG) << p->name << ", " << a->name << ", " << b->name << ", " << bound;

    Potential *pot = NULL;

    if(p->create_func) {
        pot = p->create_func(p, a, b);
    }
    else {
        pot = p;
    }

    if(bound) {
        pot->flags = pot->flags | POTENTIAL_BOUND;
    }

    if(engine_addpot(&_Engine, pot, a->id, b->id) != S_OK) 
        return error(MDCERR_engine);
    
    return S_OK;
}

HRESULT universe_bind_potential(Potential *p, BoundaryConditions *bcs, ParticleType *t) {
    TF_Log(LOG_DEBUG) << p->name << ", " << t->name;
    bcs->set_potential(t, p);
    return S_OK;
}

HRESULT universe_bind_potential(Potential *p, BoundaryCondition *bc, ParticleType *t) {
    TF_Log(LOG_DEBUG) << p->name << ", " << t->name;
    bc->set_potential(t, p);
    return S_OK;
}

HRESULT universe_bind_force(Force *force, ParticleType *a_type, const std::string* coupling_symbol) {

    if(engine_add_singlebody_force(&_Engine, force, a_type->id) != S_OK) 
        return error(MDCERR_engine);
    
    if(coupling_symbol == NULL) {
        return S_OK;
    }
    
    if(!a_type->species) {
        std::string msg = "could not add force, given a coupling symbol, but the particle type ";
        msg += a_type->name;
        msg += " does not have a chemical species vector";
        return tf_error(E_FAIL, msg.c_str());
    }

    return force->bind_species(a_type, *coupling_symbol);
}

HRESULT bind::particles(Potential *p, Particle *a, Particle *b) {
    return universe_bind_potential(p, a, b);
}

HRESULT bind::types(Potential *p, ParticleType *a, ParticleType *b, bool bound) {
    return universe_bind_potential(p, a, b, bound);
}

HRESULT bind::boundaryConditions(Potential *p, ParticleType *t) {
    return universe_bind_potential(p, &_Engine.boundary_conditions, t);
}

HRESULT bind::boundaryCondition(Potential *p, BoundaryCondition *bc, ParticleType *t) {
    return universe_bind_potential(p, bc, t);
}

HRESULT bind::force(Force *force, ParticleType *a_type) {
    return universe_bind_force(force, a_type, 0);
}

HRESULT bind::force(Force *force, ParticleType *a_type, const std::string& coupling_symbol) {
    return universe_bind_force(force, a_type, &coupling_symbol);
}

HRESULT bind::bonds(
    Potential* potential,
    ParticleList &particles, 
    const FloatP_t &cutoff, 
    std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs, 
    const FloatP_t &half_life, 
    const FloatP_t &bond_energy, 
    uint32_t flags, 
    std::vector<BondHandle> *out) 
{ 
    TF_Log(LOG_DEBUG);
    auto result = BondHandle::pairwise(potential, particles, cutoff, pairs, half_life, bond_energy, flags);
    if (out) *out = result;
    return S_OK;
}

HRESULT bind::sphere(
    Potential *potential,
    const int &n,
    FVector3 *center,
    const FloatP_t &radius,
    std::pair<FloatP_t, FloatP_t> *phi, 
    ParticleType *type, 
    ParticleList *partList,
    std::vector<BondHandle> *bondList)
{
    TF_Log(LOG_TRACE);

    static const FloatP_t Pi = M_PI;


    // potential
    //*     number of subdivisions
    //*     tuple of starting / stopping theta (polar angle)
    //*     center of sphere
    //*     radius of sphere

    FloatP_t phi0 = 0;
    FloatP_t phi1 = Pi;

    if(phi) {
        phi0 = std::get<0>(*phi);
        phi1 = std::get<1>(*phi);

        if(phi0 < 0 || phi0 > Pi) return tf_error(E_FAIL, "phi_0 must be between 0 and pi");
        if(phi1 < 0 || phi1 > Pi) return tf_error(E_FAIL, "phi_1 must be between 0 and pi");
        if(phi1 < phi0) return tf_error(E_FAIL, "phi_1 must be greater than phi_0");
    }

    FVector3 _center =  center ? *center : engine_center();

    std::vector<FVector3> vertices;
    std::vector<int32_t> indices;

    FMatrix4 s = FMatrix4::scaling(FVector3{radius, radius, radius});
    FMatrix4 t = FMatrix4::translation(_center);
    FMatrix4 m = t * s;

    icosphere(n, phi0, phi1, vertices, indices);

    FVector3 velocity;

    ParticleList parts(vertices.size());
    parts.nr_parts = vertices.size();

    // Euler formula for graphs:
    // For a closed polygon -- non-manifold mesh: T−E+V=1 -> E = T + V - 1
    // for a sphere: T−E+V=2. -> E = T + V - 2

    int edges;
    if(phi0 <= 0 && phi1 >= Pi) {
        edges = vertices.size() + (indices.size() / 3) - 2;
    }
    else if(TissueForge::almost_equal(phi0, (FloatP_t)0.0) || TissueForge::almost_equal(phi1, Pi)) {
        edges = vertices.size() + (indices.size() / 3) - 1;
    }
    else {
        edges = vertices.size() + (indices.size() / 3);
    }

    if(edges <= 0) return tf_error(E_FAIL, "No edges resulted from input.");

    std::vector<BondHandle> bonds;

    for(int i = 0; i < vertices.size(); ++i) {
        FVector3 pos = m.transformPoint(vertices[i]);
        ParticleHandle *p = (*type)(&pos, &velocity);
        parts.parts[i] = p->id;
    }

    if(vertices.size() > 0 && indices.size() == 0) return tf_error(E_FAIL, "No vertices resulted from input.");

    int nbonds = 0;
    for(int i = 0; i < indices.size(); i += 3) {
        int a = indices[i];
        int b = indices[i+1];
        int c = indices[i+2];

        nbonds += insert_bond(bonds, a, b, potential, &parts);
        nbonds += insert_bond(bonds, b, c, potential, &parts);
        nbonds += insert_bond(bonds, c, a, potential, &parts);
    }

    if(nbonds != bonds.size()) {
        std::string msg = "unknown error in finding edges for sphere mesh, \n";
        msg += "vertices: " + std::to_string(vertices.size()) + "\n";
        msg += "indices: " + std::to_string(indices.size()) + "\n";
        msg += "expected edges: " + std::to_string(edges) + "\n";
        msg += "found edges: " + std::to_string(nbonds);
        tf_error(E_FAIL, msg.c_str());
    }

    if(partList) *partList = parts;
    if(bondList) *bondList = bonds;

    return S_OK;
}
