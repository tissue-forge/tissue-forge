/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfRunner.h>
#include <tfParticle.h>
#include <tfForce.h>
#include <tfPotential.h>
#include "tf_potential_eval.h"
#include "tf_flux_eval.h"
#include "tf_smoothing_kernel.h"
#include "tf_dpd_eval.h"
#include "tf_boundary_eval.h"
#include <tfError.h>


using namespace TissueForge;


/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


extern unsigned int runner_rcount;

static std::mutex _mutexPrint;


__attribute__ ((flatten)) HRESULT TissueForge::runner_dopair(struct runner *r,
        struct space_cell *cell_i, struct space_cell *cell_j,
        int sid) {

    struct Particle *part_i, *part_j;
    struct space *s;
    int i, j;
    struct Particle *parts_i, *parts_j;
    struct Potential *pot;
    Fluxes *fluxes;
    struct engine *eng;
    int dmaxdist, dnshift;
    FPTYPE cutoff, cutoff2, r2;
    unsigned int *iparts, *jparts;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, bias;
    FPTYPE *pif;
    int pid, count_i, count_j;
    FPTYPE epot = 0.0f;
    FPTYPE number_density;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[3*VEC_SIZE];
#else
    FPTYPE dx[4], pix[4];
#endif
    
    /* break early if one of the cells is empty */
    if(cell_i->count == 0 || cell_j->count == 0)
        return S_OK;
    
    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    bias = sqrt(s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2]);
    dscale = (FPTYPE)SHRT_MAX / (2 * bias);
    dmaxdist = 2 + dscale * (cutoff + 2*s->maxdx);
    pix[3] = FPTYPE_ZERO;
    
    /* Get the sort ID. */
    sid = space_getsid(s, &cell_i, &cell_j, shift);
    
    /* Get the counts. */
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* Make local copies of the parts if requested. */
    if(r->e->flags & engine_flag_localparts) {
        parts_i = (struct Particle *)alloca(sizeof(struct Particle) * count_i);
        memcpy(parts_i, cell_i->parts, sizeof(struct Particle) * count_i);
        parts_j = (struct Particle *)alloca(sizeof(struct Particle) * count_j);
        memcpy(parts_j, cell_j->parts, sizeof(struct Particle) * count_j);
    }
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
    }
        
    /* Get the discretized shift norm. */
    nshift = sqrt(shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2]);
    dnshift = dscale * nshift;

    /* Get the pointers to the left and right particle data. */
    iparts = &cell_i->sortlist[ count_i * sid ];
    jparts = &cell_j->sortlist[ count_j * sid ];

    /* loop over the sorted list of particles in i */
    for(i = 0 ; i < count_i ; i++) {

        /* Quit early? */
        if((jparts[count_j-1] & 0xffff) + dnshift - (iparts[i] & 0xffff) > dmaxdist)
            break;

        /* get a handle on this particle */
        pid = iparts[i] >> 16;
        part_i = &(parts_i[pid]);
        pix[0] = part_i->x[0] - shift[0];
        pix[1] = part_i->x[1] - shift[1];
        pix[2] = part_i->x[2] - shift[2];
        pif = &(part_i->f[0]);

        /* loop over the left particles */
        for(j = count_j-1 ; j >= 0 && (jparts[j] & 0xffff) + dnshift - (iparts[i] & 0xffff) < dmaxdist ; j--) {

            /* get a handle on the second particle */
            part_j = &(parts_j[ jparts[j] >> 16 ]);

            /* get the distance between both particles */
            r2 = fptype_r2(pix, part_j->x, dx);

            /* is this within cutoff? */
            if (r2 > cutoff2)
                continue;
            
            number_density = W(r2, cutoff);
            part_i->number_density += number_density;
            part_j->number_density += number_density;
            
            /* fetch the potential, if any */
            pot = get_potential(part_i, part_j);
            fluxes = get_fluxes(part_i, part_j);

            if(pot == NULL && fluxes == NULL) 
                continue;

            #if defined(VECTORIZE)

            if(pot->kind == POTENTIAL_KIND_COMBINATION && pot->flags & POTENTIAL_SUM) {
                pots = pot->constituents();
            } 
            else {
                pots = {pot};
            }

            for(auto &p : pots) {
                _r2 = potential_eval_adjust_distance2(p, pi->radius, pj->radius, r2);
                if(_r2 > p->b * p->b) 
                    continue;
                _r2 = FPTYPE_FMAX(_r2, p->a * p->a);

                /* add this interaction to the interaction queue. */
                r2q[icount] = _r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = pif;
                effj[icount] = part_j->f;
                potq[icount] = pot;
                icount += 1;

                /* evaluate the interactions if the queue is full. */
                if(icount == VEC_SIZE) {

                    #if defined(FPTYPE_SINGLE)
                        #if VEC_SIZE==8
                        potential_eval_vec_8single(potq, r2q, e, f);
                        #else
                        potential_eval_vec_4single(potq, r2q, e, f);
                        #endif
                    #elif defined(FPTYPE_DOUBLE)
                        #if VEC_SIZE==4
                        potential_eval_vec_4double(potq, r2q, e, f);
                        #else
                        potential_eval_vec_2double(potq, r2q, e, f);
                        #endif
                    #endif

                    /* update the forces and the energy */
                    for(l = 0 ; l < VEC_SIZE ; l++) {
                        epot += e[l];
                        for(k = 0 ; k < 3 ; k++) {
                            w = f[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                }
            }
            #else
            
                /* eval the flux if we have any */
                if(fluxes) {
                    flux_eval_ex(fluxes, r2, part_i, part_j);
                }
            
                /* evaluate the interaction */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl(pot, r2, &e, &f);
                #else
            
                if(pot) {
                    
                    potential_eval_super_ex(cell_i, pot, part_i, part_j, dx,  r2, &epot);
            
                }
                #endif // EXPLICIT_POTENTIALS
            #endif // VECTORIZE

            }

        } /* loop over all particles */
            
        
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
                potential_eval_vec_8single(potq, r2q, e, f);
                #else
                potential_eval_vec_4single(potq, r2q, e, f);
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, e, f);
                #else
                potential_eval_vec_2double(potq, r2q, e, f);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += e[l];
                for(k = 0 ; k < 3 ; k++) {
                    w = f[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
        
    /* Store the potential energy to cell_i. */
    if(cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost)
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* Write local data back if needed. */
    if(r->e->flags & engine_flag_localparts) {
    
        /* copy the particle data back */
        for(i = 0 ; i < count_i ; i++) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
        }
        if(cell_i != cell_j) {
                for(i = 0 ; i < count_j ; i++) {
                    cell_j->parts[i].f[0] = parts_j[i].f[0];
                    cell_j->parts[i].f[1] = parts_j[i].f[1];
                    cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
            }
        }
        
    /* since nothing bad happened to us... */
    return S_OK;
}

__attribute__ ((flatten)) HRESULT TissueForge::runner_dopair_fluxonly(struct runner *r,
        struct space_cell *cell_i, struct space_cell *cell_j,
        int sid) {

    struct Particle *part_i, *part_j;
    struct space *s;
    int i, j;
    struct Particle *parts_i, *parts_j;
    Fluxes *fluxes;
    int dmaxdist, dnshift;
    FPTYPE cutoff, cutoff2, r2;
    unsigned int *iparts, *jparts;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, bias;
    int pid, count_i, count_j;
    FPTYPE dx[3], pix[3];
    
    /* break early if one of the cells is empty */
    if(cell_i->count == 0 || cell_j->count == 0)
        return S_OK;
    
    /* get the space and cutoff */
    s = &(r->e->s);
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    bias = sqrt(s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2]);
    dscale = (FPTYPE)SHRT_MAX / (2 * bias);
    dmaxdist = 2 + dscale * (cutoff + 2*s->maxdx);
    
    /* Get the sort ID. */
    sid = space_getsid(s, &cell_i, &cell_j, shift);
    
    /* Get the counts. */
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
        
    /* Get the discretized shift norm. */
    nshift = sqrt(shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2]);
    dnshift = dscale * nshift;

    /* Get the pointers to the left and right particle data. */
    iparts = &cell_i->sortlist[ count_i * sid ];
    jparts = &cell_j->sortlist[ count_j * sid ];

    /* loop over the sorted list of particles in i */
    for(i = 0 ; i < count_i ; i++) {

        /* Quit early? */
        if((jparts[count_j-1] & 0xffff) + dnshift - (iparts[i] & 0xffff) > dmaxdist)
            break;

        /* get a handle on this particle */
        pid = iparts[i] >> 16;
        part_i = &(parts_i[pid]);
        pix[0] = part_i->x[0] - shift[0];
        pix[1] = part_i->x[1] - shift[1];
        pix[2] = part_i->x[2] - shift[2];

        /* loop over the left particles */
        for(j = count_j-1 ; j >= 0 && (jparts[j] & 0xffff) + dnshift - (iparts[i] & 0xffff) < dmaxdist ; j--) {

            /* get a handle on the second particle */
            part_j = &(parts_j[ jparts[j] >> 16 ]);

            /* get the distance between both particles */
            r2 = fptype_r2(pix, part_j->x, dx);

            /* is this within cutoff? */
            if (r2 > cutoff2)
                continue;
            
            /* eval the flux if we have any */
            if((fluxes = get_fluxes(part_i, part_j))) {
                flux_eval_ex(fluxes, r2, part_i, part_j);
            }

        }

    } /* loop over all particles */
        
    /* since nothing bad happened to us... */
    return S_OK;
}

static inline HRESULT particle_largecell_force(Particle *p, struct space_cell *c, FPTYPE& epot) {
    FPTYPE w, r2, e, f, dx[4], pix[4];
    space_cell *large = &_Engine.s.largeparts;
    Potential *pot;
    int k;
    FPTYPE *pif = p->f;
    
    // TODO will be more local parts than large parts, so more efficient
    // to translate large parts to local coordinate system once.
    // but simpler to do this...
    
    for(k = 0; k < 3; ++k) {
        pix[k] = p->x[k] + c->origin[k];
    }
    
    
    /* loop over the left particles */
    for (int j = 0 ; j < large->count; ++j) {
        
        /* get a handle on the second particle */
        Particle *part_j = &large->parts[j];
        
        /* fetch the potential, if any */
        pot = get_potential(p, part_j);
        if(pot == NULL)
            continue;
        
        /* get the distance between both particles */
        r2 = fptype_r2(pix, part_j->x, dx);
        
        /* is this within cutoff? */
        // TODO add large particle cutoff...
        //if (r2 > cutoff2)
        //    continue;
        
#if defined(VECTORIZE)
        /* add this interaction to the interaction queue. */
        r2q[icount] = r2;
        dxq[icount*3] = dx[0];
        dxq[icount*3+1] = dx[1];
        dxq[icount*3+2] = dx[2];
        effi[icount] = pif;
        effj[icount] = part_j->f;
        potq[icount] = pot;
        icount += 1;
        
        /* evaluate the interactions if the queue is full. */
        if(icount == VEC_SIZE) {
            
#if defined(FPTYPE_SINGLE)
#if VEC_SIZE==8
            potential_eval_vec_8single(potq, r2q, e, f);
#else
            potential_eval_vec_4single(potq, r2q, e, f);
#endif
#elif defined(FPTYPE_DOUBLE)
#if VEC_SIZE==4
            potential_eval_vec_4double(potq, r2q, e, f);
#else
            potential_eval_vec_2double(potq, r2q, e, f);
#endif
#endif
            
            /* update the forces and the energy */
            for(l = 0 ; l < VEC_SIZE ; l++) {
                epot += e[l];
                for(k = 0 ; k < 3 ; k++) {
                    w = f[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                }
            }
            
            /* re-set the counter. */
            icount = 0;
        }
#else
        /* evaluate the interaction */
#ifdef EXPLICIT_POTENTIALS
        potential_eval_expl(pot, r2, &e, &f);
#else
        /* update the forces if part in range */
        if(potential_eval_super_ex(c, pot, p, part_j, dx,  r2, &e)) {
            /* tabulate the energy */
            epot += e;
        }
#endif // EXPLICIT_POTENTIALS
#endif // VECTORIZE
        
    }
    return S_OK;
}

__attribute__ ((flatten)) HRESULT TissueForge::runner_doself(struct runner *r, struct space_cell *c) {

    struct Particle *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j;
    struct Particle *parts;
    FPTYPE epot = 0.0f;
    struct Potential *pot;
    Fluxes *fluxes;
    // single body force and forces
    Force **forces, *force;
    struct engine *eng;
    FPTYPE cutoff, cutoff2, r2;
    FPTYPE *pif;
    std::vector<Potential*> pots;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    FPTYPE _r2;
#else
    FPTYPE number_density;
    FPTYPE dx[4], pix[4];
#endif
    
    const unsigned cell_flags = c->flags;
    
    const bool boundary = cell_flags & cell_active_any;
        
    //print_thread();
    
    /* break early if one of the cells is empty */
    count = c->count;
    if(count == 0)
        return S_OK;
    
    /* get some useful data */
    eng = r->e;
    s = &(eng->s);
    forces = eng->forces;
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    pix[3] = FPTYPE_ZERO;
    
    /* Make local copies of the parts if requested. */
    if(r->e->flags & engine_flag_localparts) {
        parts = (struct Particle *)alloca(sizeof(struct Particle) * count);
        memcpy(parts, c->parts, sizeof(struct Particle) * count);
    }
    else {
        parts = c->parts;
    }
    
    // loop over all particles, indexing here only calculates pairwise
    // interactions, and avoids self-interactions.
    for(i = 0 ; i < count ; i++) {

        /* get the particle */
        part_i = &(parts[i]);
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        pif = &(part_i->f[0]);

        // calculate single body force if any
        force = forces[part_i->typeId];
        if(force) {
            force->func(force, part_i, part_i->f);
        }
        
        // force between particle and large particles
        particle_largecell_force(part_i, c, epot);
        
        if(boundary) {
            boundary_eval(&_Engine.boundary_conditions, c, part_i, &epot);
        }
        
        /* loop over all other particles */
        for(j = i + 1 ; j < count ; j++) {

            /* get the other particle */
            part_j = &(parts[j]);
                        
            /* get the distance between both particles */
            r2 = fptype_r2(pix, part_j->x, dx);

            /* is this within cutoff? */
            /* potentials have cutoff also */
            // TODO move the square to the one-time potential init value.
            if(r2 > cutoff2)
                continue;
            
            number_density = W(r2, cutoff);
            part_i->number_density += number_density;
            part_j->number_density += number_density;
            
            pot = get_potential(part_i, part_j);
            fluxes = get_fluxes(part_i, part_j);
  
            if(pot == NULL && fluxes == NULL) 
                continue;

            #if defined(VECTORIZE)

            if(pot->kind == POTENTIAL_KIND_COMBINATION && pot->flags & POTENTIAL_SUM) {
                pots = pot->constituents();
            } 
            else {
                pots = {pot};
            }

            for(auto &p : pots) {
                _r2 = potential_eval_adjust_distance2(p, pi->radius, pj->radius, r2);
                if(_r2 > p->b * p->b) 
                    continue;
                _r2 = FPTYPE_FMAX(_r2, p->a * p->a);

                /* add this interaction to the interaction queue. */
                r2q[icount] = _r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = pif;
                effj[icount] = part_j->f;
                potq[icount] = p;
                icount += 1;

                /* evaluate the interactions if the queue is full. */
                if(icount == VEC_SIZE) {

                    /* evaluate the potentials */
                    #if defined(FPTYPE_SINGLE)
                        #if VEC_SIZE==8
                        potential_eval_vec_8single(potq, r2q, e, f);
                        #else
                        potential_eval_vec_4single(potq, r2q, e, f);
                        #endif
                    #elif defined(FPTYPE_DOUBLE)
                        #if VEC_SIZE==4
                        potential_eval_vec_4double(potq, r2q, e, f);
                        #else
                        potential_eval_vec_2double(potq, r2q, e, f);
                        #endif
                    #endif

                    /* update the forces and the energy */
                    for(l = 0 ; l < VEC_SIZE ; l++) {
                        epot += e[l];
                        for(k = 0 ; k < 3 ; k++) {
                            w = f[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                }
            }

            #else // defined(VECTORIZE)
            
                /* eval the flux if we have any */
                if(fluxes) {
                    flux_eval_ex(fluxes, r2, part_i, part_j);
                }
            
                /* evaluate the interaction */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl(pot, r2, &e, &f);
                #else
            
            /* update the forces if part in range */
            if(pot) {
                potential_eval_super_ex(c, pot, part_i, part_j, dx,  r2, &epot);
            }
                #endif // EXPLICIT_POTENTIALS
            #endif // VECTORIZE

            } /* loop over all other particles */
        } /* loop over all particles */
        

    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {

            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single(potq, r2q, e, f);
                #elif VEC_SIZE==4
                potential_eval_vec_4single(potq, r2q, e, f);
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, e, f);
                #elif VEC_SIZE==2
                potential_eval_vec_2double(potq, r2q, e, f);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += e[l];
                for(k = 0 ; k < 3 ; k++) {
                    w = f[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }
            }
    #endif // #if defined(VECTORIZE)
        
    /* Write local data back if needed. */
    if(r->e->flags & engine_flag_localparts) {
    
        /* copy the particle data back */
        for(i = 0 ; i < count ; i++) {
            c->parts[i].f[0] = parts[i].f[0];
            c->parts[i].f[1] = parts[i].f[1];
            c->parts[i].f[2] = parts[i].f[2];
        }
    }
        
    /* Store the potential energy to c. */
    c->epot += epot;
        
    /* since nothing bad happened to us... */
    return S_OK;
}

__attribute__ ((flatten)) HRESULT TissueForge::runner_doself_fluxonly(struct runner *r, struct space_cell *c) {

    struct Particle *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j;
    struct Particle *parts;
    Fluxes *fluxes;
    // single body force and forces
    FPTYPE cutoff, cutoff2, r2;
    FPTYPE dx[3], pix[3];
    
    /* break early if one of the cells is empty */
    count = c->count;
    if(count == 0)
        return S_OK;
    
    /* get some useful data */
    s = &(r->e->s);
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    
    parts = c->parts;
    
    // loop over all particles, indexing here only calculates pairwise
    // interactions, and avoids self-interactions.
    for(i = 0 ; i < count ; i++) {

        /* get the particle */
        part_i = &(parts[i]);
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        
        /* loop over all other particles */
        for(j = i + 1 ; j < count ; j++) {

            /* get the other particle */
            part_j = &(parts[j]);
                        
            /* get the distance between both particles */
            r2 = fptype_r2(pix, part_j->x, dx);

            /* is this within cutoff? */
            /* potentials have cutoff also */
            // TODO move the square to the one-time potential init value.
            if(r2 > cutoff2)
                continue;
            
            /* eval the flux if we have any */
            if((fluxes = get_fluxes(part_i, part_j))) {
                flux_eval_ex(fluxes, r2, part_i, part_j);
            }

        } /* loop over all other particles */
    } /* loop over all particles */
        
    /* since nothing bad happened to us... */
    return S_OK;
}
