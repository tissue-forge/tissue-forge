/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2011 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <tf_cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfPotential.h>
#include "tf_potential_eval.h"
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfExclusion.h>

#pragma clang diagnostic ignored "-Wwritable-strings"


using namespace TissueForge;


/* Global variables. */

#define error(id)          ( tf_error(E_FAIL, errs_err_msg[id]) )
    

HRESULT TissueForge::exclusion_eval(struct exclusion *b, int N, struct engine *e, FPTYPE *epot_out) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, **pots;
    FPTYPE r2, w, cutoff2;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff, dx[4];
#endif
    
    /* Check inputs. */
    if(b == NULL || e == NULL)
        return error(MDCERR_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    cutoff2 = s->cutoff2;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the exclusions. */
    for(bid = 0 ; bid < N ; bid++) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if((pi = partlist[ pid ]) == NULL ||
            (pj = partlist[ pjd ]) == NULL)
            continue;
        
        /* Skip if both ghosts. */
        if((pi->flags & PARTICLE_GHOST) && 
            (pj->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pot = pots[ pj->typeId*ld_pots + pi->typeId ]) == NULL)
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for(r2 = 0.0, k = 0 ; k < 3 ; k++) {
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            dx[k] = pi->x[k] - pj->x[k] + h[k]*shift[k];
            r2 += dx[k] * dx[k];
            }
        
        /* Out of range? */
        if(r2 > cutoff2)
            continue;

        #ifdef VECTORIZE
            /* add this exclusion to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if(icount == VEC_SIZE) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single(potq, r2q, ee, eff);
                    #else
                    potential_eval_vec_4single(potq, r2q, ee, eff);
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double(potq, r2q, ee, eff);
                    #else
                    potential_eval_vec_2double(potq, r2q, ee, eff);
                    #endif
                #endif

                /* update the forces and the energy */
                for(l = 0 ; l < VEC_SIZE ; l++) {
                    epot += ee[l];
                    for(k = 0 ; k < 3 ; k++) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] += w;
                        effj[l][k] -= w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the exclusion */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl(pot, r2, &ee, &eff);
            #else
                potential_eval(pot, r2, &ee, &eff);
            #endif

            /* update the forces */
            for(k = 0 ; k < 3 ; k++) {
                w = eff * dx[k];
                pi->f[k] += w;
                pj->f[k] -= w;
                }

            /* tabulate the energy */
            epot += ee;
            
        #endif

        } /* loop over exclusions. */
        
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
                potential_eval_vec_8single(potq, r2q, ee, eff);
                #else
                potential_eval_vec_4single(potq, r2q, ee, eff);
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, ee, eff);
                #else
                potential_eval_vec_2double(potq, r2q, ee, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += ee[l];
                for(k = 0 ; k < 3 ; k++) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] += w;
                    effj[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if(epot_out != NULL)
        *epot_out -= epot;
    
    /* We're done here. */
    return S_OK;
    
}


HRESULT TissueForge::exclusion_evalf(struct exclusion *b, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, **pots;
    FPTYPE dx[3], r2, w, cutoff2;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if(b == NULL || e == NULL)
        return error(MDCERR_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    cutoff2 = s->cutoff2;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the exclusions. */
    for(bid = 0 ; bid < N ; bid++) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if((pi = partlist[ pid ]) == NULL ||
            (pj = partlist[ pjd ]) == NULL)
            continue;
        
        /* Skip if both ghosts. */
        if((pi->flags & PARTICLE_GHOST) && 
            (pj->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pot = pots[ pj->typeId*ld_pots + pi->typeId ]) == NULL)
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for(r2 = 0.0, k = 0 ; k < 3 ; k++) {
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            dx[k] = pi->x[k] - pj->x[k] + h[k]*shift[k];
            r2 += dx[k] * dx[k];
            }
        
        /* Out of range? */
        if(r2 > cutoff2)
            continue;

        #ifdef VECTORIZE
            /* add this exclusion to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = &(f[ 4*pid ]);
            effj[icount] = &(f[ 4*pjd ]);
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if(icount == VEC_SIZE) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single(potq, r2q, ee, eff);
                    #else
                    potential_eval_vec_4single(potq, r2q, ee, eff);
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double(potq, r2q, ee, eff);
                    #else
                    potential_eval_vec_2double(potq, r2q, ee, eff);
                    #endif
                #endif

                /* update the forces and the energy */
                for(l = 0 ; l < VEC_SIZE ; l++) {
                    epot += ee[l];
                    for(k = 0 ; k < 3 ; k++) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] += w;
                        effj[l][k] -= w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the exclusion */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl(pot, r2, &ee, &eff);
            #else
                potential_eval(pot, r2, &ee, &eff);
            #endif

            /* update the forces */
            for(k = 0 ; k < 3 ; k++) {
                w = eff * dx[k];
                f[ 4*pid + k ] -= w;
                f[ 4*pjd + k ] += w;
                }

            /* tabulate the energy */
            epot += ee;
            
        #endif

        } /* loop over exclusions. */
        
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
                potential_eval_vec_8single(potq, r2q, ee, eff);
                #else
                potential_eval_vec_4single(potq, r2q, ee, eff);
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double(potq, r2q, ee, eff);
                #else
                potential_eval_vec_2double(potq, r2q, ee, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += ee[l];
                for(k = 0 ; k < 3 ; k++) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] += w;
                    effj[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if(epot_out != NULL)
        *epot_out -= epot;
    
    /* We're done here. */
    return S_OK;
    
}
