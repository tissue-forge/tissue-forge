/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <rendering/tfStyle.h>

/* Include some conditional headers. */
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfPotential.h>
#include "tf_potential_eval.h"
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfBond.h>

#include <tfLogger.h>
#include <tf_util.h>
#include <tfError.h>
#include <io/tfFIO.h>
#include <tf_mdcore_io.h>

#ifdef HAVE_CUDA
#include "tfBond_cuda.h"
#endif

#include <random>


using namespace TissueForge;


rendering::Style *Bond_StylePtr = new rendering::Style("lime");

/* Global variables. */
/** The ID of the last error. */
int TissueForge::bond_err = bond_err_ok;

/* the error macro. */
#define error(id)				(bond_err = errs_register(id, bond_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))

/* list of error messages. */
const char *bond_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
};


/**
 * check if a type pair is in a list of pairs
 * pairs has to be a vector of pairs of types
 */
static bool pair_check(std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs, short a_typeid, short b_typeid);

static bool Bond_decays(Bond *b, std::uniform_real_distribution<FPTYPE> *uniform01=NULL) {
    if(!b || b->half_life <= 0.0) return false;

    bool created = uniform01 == NULL;
    if(created) uniform01 = new std::uniform_real_distribution<FPTYPE>(0.0, 1.0);

    FPTYPE pr = 1.0 - std::pow(2.0, -_Engine.dt / b->half_life);
    RandomType &randEng = randomEngine();
    bool result = (*uniform01)(randEng) < pr;

    if(created) delete uniform01;

    return result;
}

/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int TissueForge::bond_eval(struct Bond *bonds, int N, struct engine *e, FPTYPE *epot_out) {

    #ifdef HAVE_CUDA
    if(e->bonds_cuda) {
        return cuda::engine_bond_eval_cuda(bonds, N, e, epot_out);
    }
    #endif

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *potb;
    std::vector<struct Potential *> pots;
    struct Bond *b;
    FPTYPE ee, r2, _r2, w, f[3];
    std::unordered_set<struct Bond*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<FPTYPE> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    struct Bond *bondq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if(bonds == NULL || e == NULL)
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for(bid = 0 ; bid < N ; bid++) {

        b = &bonds[bid];
        b->potential_energy = 0.0;

        if(Bond_decays(b, &uniform01)) {
            toDestroy.insert(b);
            continue;
        }

        if(!(b->flags & BOND_ACTIVE))
            continue;
    
        /* Get the particles involved. */
        pid = b->i; pjd = b->j;
        if((pi = partlist[ pid ]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        
        /* Skip if both ghosts. */
        if((pi->flags & PARTICLE_GHOST) && (pj->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        potb = b->potential;
        if(!potb) {
            continue;
        }
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
        }
        r2 = fptype_r2(pix, pj->x, dx);

        if(potb->kind == POTENTIAL_KIND_COMBINATION && potb->flags & POTENTIAL_SUM) {
            pots = potb->constituents();
            if(pots.size() == 0) pots = {potb};
        }
        else pots = {potb};

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(f), std::end(f), 0.0);
                pot->eval_byparts(pot, pi, pj, dx, r2, &ee, f);
                for(int i = 0; i < 3; ++i) {
                    pi->f[i] += f[i];
                    pj->f[i] -= f[i];
                }
                epot += ee;
                b->potential_energy += ee;
                if(b->potential_energy >= b->dissociation_energy)
                    toDestroy.insert(b);
            }
            else {

                _r2 = potential_eval_adjust_distance2(pot, pi->radius, pj->radius, r2);
                if(_r2 > pot->b * pot->b) 
                    continue;
                _r2 = FPTYPE_FMAX(_r2, pot->a * pot->a);

                #ifdef VECTORIZE
                    /* add this bond to the interaction queue. */
                    r2q[icount] = _r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = pi->f;
                    effj[icount] = pj->f;
                    potq[icount] = pot;
                    bondq[icount] = b;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single(potq, r2q, eeq, eff);
                            #else
                            potential_eval_vec_4single(potq, r2q, eeq, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double(potq, r2q, eeq, eff);
                            #else
                            potential_eval_vec_2double(potq, r2q, eeq, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            epot += eeq[l];
                            bondq[l]->potential_energy += eeq[l];
                            if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                                toDestroy.insert(bondq[l]);

                            for(k = 0 ; k < 3 ; k++) {
                                w = eff[l] * dxq[l*3+k];
                                effi[l][k] -= w;
                                effj[l][k] += w;
                            }
                        }

                        /* re-set the counter. */
                        icount = 0;

                    }
                #else // NOT VECTORIZE
                    /* evaluate the bond */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, _r2, &ee, &eff);
                    #else
                        potential_eval(pot, _r2, &ee, &eff);
                    #endif
                
                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        w = eff * dx[k];
                        pi->f[k] -= w;
                        pj->f[k] += w;
                    }
                    /* tabulate the energy */
                    epot += ee;
                    b->potential_energy += ee;
                    if(b->potential_energy >= b->dissociation_energy) 
                        toDestroy.insert(b);


                #endif

            }

        }

        } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {

            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single(potq, r2q, eeq, eff);
                #else
                potential_eval_vec_4single(potq, r2q, eeq, eff);
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, eeq, eff);
                #else
                potential_eval_vec_2double(potq, r2q, eeq, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += eeq[l];
                bondq[l]->potential_energy += eeq[l];
                if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                    toDestroy.insert(bondq[l]);

                for(k = 0 ; k < 3 ; k++) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto bi : toDestroy)
        Bond_Destroy(bi);
    
    /* Store the potential energy. */
    if(epot_out != NULL)
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
}



/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param bonds Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param forces An array of @c FPTYPE in which to aggregate the resulting forces.
 * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
 * 
 * This function differs from #bond_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int TissueForge::bond_evalf(struct Bond *bonds, int N, struct engine *e, FPTYPE *forces, FPTYPE *epot_out) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *potb;
    std::vector<struct Potential *> pots;
    struct Bond *b;
    FPTYPE ee, r2, _r2, w, f[3];
    std::unordered_set<struct Bond*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<FPTYPE> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    struct Bond *bondq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if(bonds == NULL || e == NULL || forces == NULL)
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for(bid = 0 ; bid < N ; bid++) {
        b = &bonds[bid];
        b->potential_energy = 0.0;

        if(Bond_decays(b, &uniform01)) {
            toDestroy.insert(b);
            continue;
        }
    
        /* Get the particles involved. */
        pid = b->i; pjd = b->j;
        if((pi = partlist[ pid ]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        
        /* Skip if both ghosts. */
        if(pi->flags & PARTICLE_GHOST && pj->flags & PARTICLE_GHOST)
            continue;
            
        /* Get the potential. */
        if((pot = b->potential) == NULL)
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
            }
        r2 = fptype_r2(pix, pj->x, dx);

        if(potb->kind == POTENTIAL_KIND_COMBINATION && potb->flags & POTENTIAL_SUM) {
            pots = potb->constituents();
            if(pots.size() == 0) pots = {potb};
        }
        else pots = {potb};

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(f), std::end(f), 0.0);
                pot->eval_byparts(pot, pi, pj, dx, r2, &ee, f);
                for(int i = 0; i < 3; ++i) {
                    pi->f[i] += f[i];
                    pj->f[i] -= f[i];
                }
                epot += ee;
                b->potential_energy += ee;
                if(b->potential_energy >= b->dissociation_energy)
                    toDestroy.insert(b);
            }
            else {

                _r2 = potential_eval_adjust_distance2(pot, pi->radius, pj->radius, r2);
                if(_r2 > pot->b * pot->b) 
                    continue;
                _r2 = FPTYPE_FMAX(_r2, pot->a * pot->a);

                #ifdef VECTORIZE
                    /* add this bond to the interaction queue. */
                    r2q[icount] = _r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = &(forces[ 4*pid ]);
                    effj[icount] = &(forces[ 4*pjd ]);
                    potq[icount] = pot;
                    bondq[icount] = b;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single(potq, r2q, eeq, eff);
                            #else
                            potential_eval_vec_4single(potq, r2q, eeq, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double(potq, r2q, eeq, eff);
                            #else
                            potential_eval_vec_2double(potq, r2q, eeq, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            epot += eeq[l];
                            bondq[l]->potential_energy += eeq[l];
                            if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                                toDestroy.insert(bondq[l]);

                            for(k = 0 ; k < 3 ; k++) {
                                w = eff[l] * dxq[l*3+k];
                                effi[l][k] -= w;
                                effj[l][k] += w;
                            }
                        }

                        /* re-set the counter. */
                        icount = 0;

                    }
                #else
                    /* evaluate the bond */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, _r2, &ee, &eff);
                    #else
                        potential_eval(pot, _r2, &ee, &eff);
                    #endif
                    b->potential_energy += ee;
                
                    if(b->potential_energy >= b->dissociation_energy) {
                        toDestroy.insert(b);
                    }
                    else {
                        /* update the forces */
                        for(k = 0 ; k < 3 ; k++) {
                            w = eff * dx[k];
                            forces[ 4*pid + k ] -= w;
                            forces[ 4*pjd + k ] += w;
                        }
                        /* tabulate the energy */
                        epot += ee;
                    }
                #endif

            }

        }

    } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {

            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
            }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single(potq, r2q, eeq, eff);
                #else
                potential_eval_vec_4single(potq, r2q, eeq, eff);
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, eeq, eff);
                #else
                potential_eval_vec_2double(potq, r2q, eeq, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += eeq[l];
                bondq[l]->potential_energy += eeq[l];
                if(bondq[l]->potential_energy >= bondq[l]->dissociation_energy)
                    toDestroy.insert(bondq[l]);

                for(k = 0 ; k < 3 ; k++) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                }
            }

        }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto bi : toDestroy)
        Bond_Destroy(bi);
    
    /* Store the potential energy. */
    if(epot_out != NULL)
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
}

int TissueForge::BondHandle::_init(
    uint32_t flags, 
    int32_t i, 
    int32_t j, 
    FPTYPE half_life, 
    FPTYPE bond_energy, 
    struct Potential *potential) 
{
    // check whether this handle has previously been initialized and return without error if so
    if (this->id > 0 && _Engine.nr_bonds > 0) return 0;

    Bond *bond = NULL;
    
    int result = engine_bond_alloc(&_Engine, &bond);
    
    if(result < 0) {
        throw std::runtime_error("could not allocate bond");
        return E_FAIL;
    }
    
    bond->flags = flags;
    bond->i = i;
    bond->j = j;
    bond->creation_time = _Engine.time;
    bond->half_life = half_life;
    bond->dissociation_energy = bond_energy;
    if (!bond->style) bond->style = Bond_StylePtr;
    
    if(bond->i >= 0 && bond->j >= 0) {
        bond->flags = bond->flags | BOND_ACTIVE;
        _Engine.nr_active_bonds++;
    }

    if(potential) {
        bond->potential = potential;
    }
    
    this->id = result;

    #ifdef HAVE_CUDA
    if(_Engine.bonds_cuda) 
        if(cuda::engine_cuda_add_bond(bond) < 0) 
            return error(engine_err);
    #endif

    TF_Log(LOG_TRACE) << "Created bond: " << this->id  << ", i: " << bond->i << ", j: " << bond->j;

    return 0;
}

rendering::Style *TissueForge::Bond::styleDef() {
    return Bond_StylePtr;
}

BondHandle *TissueForge::Bond::create(
    struct Potential *potential, 
    ParticleHandle *i, 
    ParticleHandle *j, 
    FPTYPE *half_life, 
    FPTYPE *bond_energy, 
    uint32_t flags)
{
    if(!potential || !i || !j) return NULL;

    auto _half_life = half_life ? *half_life : std::numeric_limits<FPTYPE>::max();
    auto _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<FPTYPE>::max();
    return new BondHandle(potential, i->id, j->id, _half_life, _bond_energy, flags);
}

std::string TissueForge::Bond::toString() {
    return io::toString(*this);
}

Bond *TissueForge::Bond::fromString(const std::string &str) {
    return new Bond(io::fromString<Bond>(str));
}

Bond *TissueForge::BondHandle::get() { 
    return &_Engine.bonds[this->id];
};

TissueForge::BondHandle::BondHandle(int id) {
    if(id >= 0 && id < _Engine.nr_bonds) this->id = id;
    else throw std::range_error("invalid id");
}

TissueForge::BondHandle::BondHandle(
    struct Potential *potential, 
    int32_t i, 
    int32_t j,
    FPTYPE half_life, 
    FPTYPE bond_energy, 
    uint32_t flags) : 
    BondHandle()
{
    _init(flags, i, j, half_life, bond_energy, potential);
}

int TissueForge::BondHandle::init(
    Potential *pot, 
    ParticleHandle *p1, 
    ParticleHandle *p2, 
    const FPTYPE &half_life, 
    const FPTYPE &bond_energy, 
    uint32_t flags) 
{

    TF_Log(LOG_DEBUG);

    try {
        return _init(
            flags, 
            p1->id, 
            p2->id, 
            half_life < FPTYPE_ZERO ? std::numeric_limits<FPTYPE>::max() : half_life, 
            bond_energy < FPTYPE_ZERO ? std::numeric_limits<FPTYPE>::max() : bond_energy, 
            pot
        );
    }
    catch (const std::exception &e) {
        return tf_exp(e);
    }
}

std::string TissueForge::BondHandle::str() {
    std::stringstream  ss;
    Bond *bond = &_Engine.bonds[this->id];
    
    ss << "Bond(i=" << bond->i << ", j=" << bond->j << ")";
    
    return ss.str();
}

bool TissueForge::BondHandle::check() {
    return (bool)this->get();
}

FPTYPE TissueForge::BondHandle::getEnergy()
{
    TF_Log(LOG_DEBUG);
    
    Bond *bond = this->get();
    FPTYPE energy = 0;
    
    Bond_Energy(bond, &energy);
    
    return energy;
}

std::vector<int32_t> TissueForge::BondHandle::getParts() {
    std::vector<int32_t> result;
    Bond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        result = std::vector<int32_t>{bond->i, bond->j};
    }
    return result;
}

Potential *TissueForge::BondHandle::getPotential() {
    Bond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        return bond->potential;
    }
    return NULL;
}

uint32_t TissueForge::BondHandle::getId() {
    Bond *bond = get();
    if(bond && bond->flags & BOND_ACTIVE) {
        return bond->id;
    }
    return NULL;
}

FPTYPE TissueForge::BondHandle::getDissociationEnergy() {
    FPTYPE result;
    Bond *bond = get();
    if (bond) result = bond->dissociation_energy;
    return result;
}

void TissueForge::BondHandle::setDissociationEnergy(const FPTYPE &dissociation_energy) {
    Bond *bond = get();
    if (bond) bond->dissociation_energy = dissociation_energy;
}

FPTYPE TissueForge::BondHandle::getHalfLife() {
    Bond *bond = get();
    if (bond) return bond->half_life;
    return NULL;
}

void TissueForge::BondHandle::setHalfLife(const FPTYPE &half_life) {
    Bond *bond = get();
    if (bond) bond->half_life = half_life;
}

bool TissueForge::BondHandle::getActive() {
    Bond *bond = get();
    if (bond) return (bool)(bond->flags & BOND_ACTIVE);
    return false;
}

rendering::Style *TissueForge::BondHandle::getStyle() {
    Bond *bond = get();
    if (bond) return bond->style;
    return NULL;
}

void TissueForge::BondHandle::setStyle(rendering::Style *style) {
    Bond *bond = get();
    if (bond) bond->style = style;
}

FPTYPE TissueForge::BondHandle::getAge() {
    Bond *bond = get();
    if (bond) return (_Engine.time - bond->creation_time) * _Engine.dt;
    return 0;
}

static void make_pairlist(
    const ParticleList *parts,
    FPTYPE cutoff, std::vector<std::pair<ParticleType*, ParticleType*>* > *paircheck_list,
    PairList& pairs) 
{
    int i, j;
    struct Particle *part_i, *part_j;
    FVector4 dx;
    FVector4 pix, pjx;
 
    /* get the space and cutoff */
    pix[3] = FPTYPE_ZERO;
    
    FPTYPE r2;
    
    FPTYPE c2 = cutoff * cutoff;
    
    // TODO: more effecient to caclulate everythign in reference frame
    // of outer particle.
    
    /* loop over all particles */
    for(i = 1 ; i < parts->nr_parts ; i++) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts->parts[i]];
        
        // global position
        FPTYPE *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        /* loop over all other particles */
        for(j = 0 ; j < i ; j++) {
            
            /* get the other particle */
            part_j = _Engine.s.partlist[parts->parts[j]];
            
            // global position
            FPTYPE *oj = _Engine.s.celllist[part_j->id]->origin;
            pjx[0] = part_j->x[0] + oj[0];
            pjx[1] = part_j->x[1] + oj[1];
            pjx[2] = part_j->x[2] + oj[2];
            
            /* get the distance between both particles */
            r2 = fptype_r2(pix.data(), pjx.data(), dx.data());
            
            if(r2 <= c2 && pair_check(paircheck_list, part_i->typeId, part_j->typeId)) {
                pairs.push_back(Pair{part_i->id,part_j->id});
            }
        } /* loop over all other particles */
    } /* loop over all particles */
}

static bool Bond_destroyingAll = false;

CAPI_FUNC(HRESULT) TissueForge::Bond_Destroy(struct Bond *b) {
    
    std::unique_lock<std::mutex> lock(_Engine.bonds_mutex);
    
    if(b->flags & BOND_ACTIVE) {
        #ifdef HAVE_CUDA
        if(_Engine.bonds_cuda && !Bond_destroyingAll) 
            if(cuda::engine_cuda_finalize_bond(b->id) < 0) 
                return E_FAIL;
        #endif

        // this clears the BOND_ACTIVE flag
        bzero(b, sizeof(Bond));
        _Engine.nr_active_bonds -= 1;
    }
    return S_OK;
}

HRESULT TissueForge::Bond_DestroyAll() {
    Bond_destroyingAll = true;

    #ifdef HAVE_CUDA
    if(_Engine.bonds_cuda) 
        if(cuda::engine_cuda_finalize_bonds_all(&_Engine) < 0) 
            return E_FAIL;
    #endif

    for(auto bh : BondHandle::items()) bh->destroy();

    Bond_destroyingAll = false;
    return S_OK;
}

std::vector<BondHandle*>* TissueForge::BondHandle::pairwise(
    struct Potential* pot,
    struct ParticleList *parts,
    const FPTYPE &cutoff,
    std::vector<std::pair<ParticleType*, ParticleType*>* > *ppairs,
    const FPTYPE &half_life,
    const FPTYPE &bond_energy,
    uint32_t flags) 
{
    
    PairList pairs;
    std::vector<BondHandle*> *bonds = new std::vector<BondHandle*>();
    
    try {
        make_pairlist(parts, cutoff, ppairs, pairs);
        
        for(auto &pair : pairs) {
            auto bond = new BondHandle(pot, pair.i, pair.j, half_life, bond_energy, flags);
            if(!bond) {
                throw std::logic_error("failed to allocated bond");
            }
            
            bonds->push_back(bond);
        }
        
        return bonds;
    }
    catch (const std::exception &e) {
        delete bonds;
        tf_exp(e);
    }
    return NULL;
}

HRESULT TissueForge::BondHandle::destroy()
{
    TF_Log(LOG_DEBUG);
    
    return Bond_Destroy(this->get());
}

std::vector<BondHandle*> TissueForge::BondHandle::bonds() {
    std::vector<BondHandle*> list;
    
    for(int i = 0; i < _Engine.nr_bonds; ++i) 
        list.push_back(new BondHandle(i));
    
    return list;
}

std::vector<BondHandle*> TissueForge::BondHandle::items() {
    return bonds();
}

bool TissueForge::BondHandle::decays() {
    return Bond_decays(&_Engine.bonds[this->id]);
}

ParticleHandle *TissueForge::BondHandle::operator[](unsigned int index) {
    auto *b = get();
    if(!b) {
        TF_Log(LOG_ERROR) << "Invalid bond handle";
        return NULL;
    }

    if(index == 0) return Particle_FromId(b->i)->handle();
    else if(index == 1) return Particle_FromId(b->j)->handle();
    
    tf_exp(std::range_error("Index out of range (must be 0 or 1)"));
    return NULL;
}

HRESULT TissueForge::Bond_Energy (Bond *b, FPTYPE *epot_out) {
    
    int pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct Potential *pot;
    FPTYPE r2, w;
    FPTYPE ee, eff, dx[4], pix[4];
    
    
    /* Get local copies of some variables. */
    s = &_Engine.s;
    partlist = s->partlist;
    celllist = s->celllist;

    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
    
    pix[3] = FPTYPE_ZERO;
    
    if(!(b->flags & BOND_ACTIVE))
        return -1;
    
    /* Get the particles involved. */
    pid = b->i; pjd = b->j;
    if((pi = partlist[ pid ]) == NULL)
        return -1;
    if((pj = partlist[ pjd ]) == NULL)
        return -1;
    
    /* Skip if both ghosts. */
    if((pi->flags & PARTICLE_GHOST) && (pj->flags & PARTICLE_GHOST))
        return 0;
    
    /* Get the potential. */
    pot = b->potential;
    if (!pot) {
        return 0;
    }
    
    /* get the distance between both particles */
    loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
    for(k = 0 ; k < 3 ; k++) {
        shift[k] = loci[k] - locj[k];
        if(shift[k] > 1)
            shift[k] = -1;
        else if(shift[k] < -1)
            shift[k] = 1;
        pix[k] = pi->x[k] + h[k]*shift[k];
    }
    r2 = fptype_r2(pix, pj->x, dx);
    
    if(r2 < pot->a*pot->a || r2 > pot->b*pot->b) {
        r2 = fmax(pot->a*pot->a, fmin(pot->b*pot->b, r2));
    }
    
    /* evaluate the bond */
    potential_eval(pot, r2, &ee, &eff);
    
    /* tabulate the energy */
    epot += ee;
    

    /* Store the potential energy. */
    if(epot_out != NULL)
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
}

std::vector<int32_t> TissueForge::Bond_IdsForParticle(int32_t pid) {
    std::vector<int32_t> bonds;
    for (int i = 0; i < _Engine.nr_bonds; ++i) {
        Bond *b = &_Engine.bonds[i];
        if((b->flags & BOND_ACTIVE) && (b->i == pid || b->j == pid)) {
            assert(i == b->id);
            bonds.push_back(b->id);
        }
    }
    return bonds;
}

bool pair_check(std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs, short a_typeid, short b_typeid) {
    if(!pairs) {
        return true;
    }
    
    auto *a = &_Engine.types[a_typeid];
    auto *b = &_Engine.types[b_typeid];
    
    for (int i = 0; i < (*pairs).size(); ++i) {
        std::pair<ParticleType*, ParticleType*> *o = (*pairs)[i];
        if((a == std::get<0>(*o) && b == std::get<1>(*o)) ||
            (b == std::get<0>(*o) && a == std::get<1>(*o))) {
            return true;
        }
    }
    return false;
}

bool TissueForge::contains_bond(const std::vector<BondHandle*> &bonds, int a, int b) {
    for(auto h : bonds) {
        Bond *bond = &_Engine.bonds[h->id];
        if((bond->i == a && bond->j == b) || (bond->i == b && bond->j == a)) {
            return true;
        }
    }
    return false;
}

int TissueForge::insert_bond(
    std::vector<BondHandle*> &bonds, 
    int a, 
    int b,
    Potential *pot, 
    ParticleList *parts) 
{
    int p1 = parts->parts[a];
    int p2 = parts->parts[b];
    if(!contains_bond(bonds, p1, p2)) {
        BondHandle *bond = new BondHandle(pot, p1, p2,
                                          std::numeric_limits<FPTYPE>::max(),
                                          std::numeric_limits<FPTYPE>::max(),
                                          0);
        bonds.push_back(bond);
        return 1;
    }
    return 0;
}


namespace TissueForge::io {


    #define TF_BONDIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_BONDIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(const Bond &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_BONDIOTOEASY(fe, "flags", dataElement.flags);
        TF_BONDIOTOEASY(fe, "i", dataElement.i);
        TF_BONDIOTOEASY(fe, "j", dataElement.j);
        TF_BONDIOTOEASY(fe, "id", dataElement.id);
        TF_BONDIOTOEASY(fe, "creation_time", dataElement.creation_time);
        TF_BONDIOTOEASY(fe, "half_life", dataElement.half_life);
        TF_BONDIOTOEASY(fe, "dissociation_energy", dataElement.dissociation_energy);
        TF_BONDIOTOEASY(fe, "potential_energy", dataElement.potential_energy);

        fileElement->type = "Bond";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Bond *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "flags", &dataElement->flags);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "i", &dataElement->i);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "j", &dataElement->j);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "creation_time", &dataElement->creation_time);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "half_life", &dataElement->half_life);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "dissociation_energy", &dataElement->dissociation_energy);
        TF_BONDIOFROMEASY(feItr, fileElement.children, metaData, "potential_energy", &dataElement->potential_energy);

        return S_OK;
    }

};
