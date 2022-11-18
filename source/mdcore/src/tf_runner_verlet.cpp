/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>


#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define ceil128(v) (((v) + 127) & ~127)
#endif
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
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
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include "tf_potential_eval.h"
#include <tfEngine.h>
#include <tfRunner.h>
#include <tfError.h>


#ifdef CELL
    /* the SPU executeable */
    extern spe_program_handle_t runner_spu;
#endif


/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


using namespace TissueForge;


extern unsigned int runner_rcount;


__attribute__ ((flatten)) HRESULT TissueForge::runner_verlet_eval(struct runner *r, struct space_cell *c, FPTYPE *f_out) {

    struct space *s;
    struct Particle *part_i, *part_j;
    struct verlet_entry *verlet_list;
    struct Potential *pot;
    int pid, i, j, k, nrpairs;
    FPTYPE pix[4], pjx[4];
    FPTYPE cutoff2, r2, dx[4], w, h[3];
    FPTYPE epot = 0.0;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *pif;
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif

    /* Get a direct pointer on the space and some other useful things. */
    s = &(r->e->s);
    cutoff2 = s->cutoff2;
    h[0] = s->h[0]; h[1] = s->h[1]; h[2] = s->h[2];
    pix[3] = FPTYPE_ZERO;
    pjx[3] = FPTYPE_ZERO;
    
    /* Loop over all entries. */
    for(i = 0 ; i < c->count ; i++) {
    
        /* Get a hold of the ith particle. */
        part_i = &(c->parts[i]);
        pid = part_i->id;
        verlet_list = &(s->verlet_list[ pid * space_verlet_maxpairs ]);
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        nrpairs = s->verlet_nrpairs[ pid ];
        #if defined(VECTORIZE)
            pif = &(f_out[ pid*4 ]);
        #endif
        
        /* loop over all other particles */
        for(j = 0 ; j < nrpairs ; j++) {

            /* get the other particle */
            part_j = verlet_list[j].p;

            /* get the distance between both particles */
            for(k = 0 ; k < 3 ; k++)
                pjx[k] = part_j->x[k] + verlet_list[j].shift[k]*h[k];
            r2 = fptype_r2(pix, pjx, dx);

            /* is this within cutoff? */
            if(r2 > cutoff2)
                continue;
            // runner_rcount += 1;
                
            /* fetch the potential, should be non-NULL by design! */
            pot = verlet_list[j].pot;

            #ifdef VECTORIZE
                /* add this interaction to the interaction queue. */
                r2q[icount] = r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = pif;
                effj[icount] = &(f_out[part_j->id*4]);
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
                    potential_eval_expl(pot, r2, &ee, &eff);
                #else
                    potential_eval(pot, r2, &ee, &eff);
                #endif

                /* update the forces */
                for(k = 0 ; k < 3 ; k++) {
                    w = eff * dx[k];
                    f_out[i*4+k] -= w;
                    f_out[part_j->id*4+k] += w;
                    }

                /* tabulate the energy */
                epot += ee;
            #endif

            } /* loop over all other particles */
            
        }
        
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
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
        
    /* Store the accumulated potential energy. */
    r->epot += epot;

    /* All has gone well. */
    return S_OK;

}

__attribute__ ((flatten)) HRESULT TissueForge::runner_verlet_fill(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, FPTYPE *pshift) {

    struct Particle *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j, k;
    struct Particle *parts_i, *parts_j;
    struct Potential *pot, **pots;
    struct engine *eng;
    int emt, pioff, count_i, count_j;
    FPTYPE cutoff, cutoff2, skin, skin2, r2, dx[4], w;
    FPTYPE dscale;
    FPTYPE shift[3], inshift, nshift;
    FPTYPE pix[4], *pif;
    int pid, pind, ishift[3];
    unsigned int *parts, dskin;
    struct verlet_entry *vbuff;
    FPTYPE epot = 0.0;
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE e, f;
#endif
    
    /* break early if one of the cells is empty */
    count_i = cell_i->count;
    count_j = cell_j->count;
    if(count_i == 0 || count_j == 0 || (cell_i == cell_j && count_i < 2))
        return S_OK;
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    skin = fmin(s->h[0], fmin(s->h[1], s->h[2]));
    skin2 = skin * skin;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    dscale = (FPTYPE)SHRT_MAX / (3 * sqrt(s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2]));
    dskin = 1 + dscale*skin;
    pix[3] = FPTYPE_ZERO;
    
    /* Make local copies of the parts if requested. */
    if(r->e->flags & engine_flag_localparts) {
    
        /* set pointers to the particle lists */
        parts_i = (Particle *)alloca(sizeof(Particle) * count_i);
        memcpy(parts_i, cell_i->parts, sizeof(Particle) * count_i);
        if(cell_i != cell_j) {
            parts_j = (Particle *)alloca(sizeof(Particle) * count_j);
            memcpy(parts_j, cell_j->parts, sizeof(Particle) * count_j);
            }
        else
            parts_j = parts_i;
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* Is this a self interaction? */
    if(cell_i == cell_j) {
    
        /* loop over all particles */
        for(i = 1 ; i < count_i ; i++) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->typeId * emt;
            pid = part_i->id;
            pind = s->verlet_nrpairs[ pid ];
            vbuff = &(s->verlet_list[ pid * space_verlet_maxpairs ]);
            pif = &(part_i->f[0]);
        
            /* loop over all other particles */
            for(j = 0 ; j < i ; j++) {
            
                /* get the other particle */
                part_j = &(parts_j[j]);
                
                /* get the distance between both particles */
                r2 = fptype_r2(pix, part_j->x, dx);
                    
                /* is this within cutoff? */
                if(r2 > skin2)
                    continue;
                /* runner_rcount += 1; */
                    
                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->typeId ];
                if(pot == NULL)
                    continue;
                    
                /* Add this pair to the verlet list. */
                vbuff[pind].shift[0] = 0;
                vbuff[pind].shift[1] = 0;
                vbuff[pind].shift[2] = 0;
                vbuff[pind].pot = pot;
                vbuff[pind].p = &(cell_j->parts[j]);
                pind += 1;
                    
                /* is this within cutoff? */
                if(r2 > cutoff2)
                    continue;
                // runner_rcount += 1;

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
                        potential_eval(pot, r2, &e, &f);
                    #endif

                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        w = f * dx[k];
                        pif[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                } /* loop over all other particles */
                
            /* Adjust verlet_nrpairs. */
            if((s->verlet_nrpairs[pid] = pind) > space_verlet_maxpairs)
                return error(MDCERR_verlet_overflow);
        
            } /* loop over all particles */
    
        }
        
    /* No, genuine pair. */
    else {
    
        /* Get the integer shift. */
        ishift[0] = round(pshift[0] * s->ih[0]);
        ishift[1] = round(pshift[1] * s->ih[1]);
        ishift[2] = round(pshift[2] * s->ih[2]);
        
        /* Allocate work arrays on stack. */
        if((parts = (unsigned int*)alloca(sizeof(unsigned int) * (count_i + count_j))) == NULL)
            return error(MDCERR_malloc);
        
        /* start by filling the particle ids of both cells into ind and d */
        nshift = sqrt(pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2]);
        inshift = 1.0 / nshift;
        shift[0] = pshift[0]*inshift; shift[1] = pshift[1]*inshift; shift[2] = pshift[2]*inshift;
        for(i = 0 ; i < count_i ; i++) {
            part_i = &(parts_i[i]);
            parts[count] = (i << 16) |
                (unsigned int)(dscale * (nshift + part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2]));
            count += 1;
            }
        for(i = 0 ; i < count_j ; i++) {
            part_i = &(parts_j[i]);
            parts[count] = (i << 16) |
                (unsigned int)(dscale * (nshift + (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2]));
            count += 1;
            }

        /* Sort parts in cell_i in decreasing order. */
        runner_sort_descending(parts, count_i);

        /* Sort parts in cell_j in increasing order. */
        runner_sort_ascending(&parts[count_i], count_j);


        /* loop over the sorted list of particles in i */
        for(i = 0 ; i < count_i ; i++) {

            /* Quit early? */
            if((parts[count_i] & 0xffff) - (parts[i] & 0xffff) > dskin)
                break;

            /* get a handle on this particle */
            part_i = &(parts_i[ parts[i] >> 16 ]);
            pix[0] = part_i->x[0] - pshift[0];
            pix[1] = part_i->x[1] - pshift[1];
            pix[2] = part_i->x[2] - pshift[2];
            pioff = part_i->typeId * emt;
            pif = &(part_i->f[0]);
            pid = part_i->id;
            pind = s->verlet_nrpairs[ pid ];
            vbuff = &(s->verlet_list[ pid * space_verlet_maxpairs ]);

            /* loop over the left particles */
            for(j = 0 ; j < count_j && (parts[count_i+j] & 0xffff) - (parts[i] & 0xffff) < dskin ; j++) {

                /* get a handle on the second particle */
                part_j = &(parts_j[ parts[count_i+j] >> 16 ]);
                    
                /* get the distance between both particles */
                r2 = fptype_r2(pix, part_j->x, dx);

                /* is this within cutoff? */
                if(r2 > skin2)
                    continue;
                /* runner_rcount += 1; */

                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->typeId ];
                if(pot == NULL)
                    continue;

                /* Add this pair to the verlet list. */
                vbuff[pind].shift[0] = ishift[0];
                vbuff[pind].shift[1] = ishift[1];
                vbuff[pind].shift[2] = ishift[2];
                vbuff[pind].pot = pot;
                vbuff[pind].p = &(cell_j->parts[ parts[count_i+j] >> 16 ]);
                pind += 1;

                /* is this within cutoff? */
                if(r2 > cutoff2)
                    continue;
                // runner_rcount += 1;

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
                        potential_eval(pot, r2, &e, &f);
                    #endif

                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        w = f * dx[k];
                        pif[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                } /* loop over particles in cell_j. */

            /* Adjust verlet_nrpairs. */
            if((s->verlet_nrpairs[pid] = pind) > space_verlet_maxpairs)
                return error(MDCERR_verlet_overflow);
        
            } /* loop over all particles in cell_i. */

        }
        
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
        if(cell_i != cell_j)
            for(i = 0 ; i < count_j ; i++) {
                cell_j->parts[i].f[0] = parts_j[i].f[0];
                cell_j->parts[i].f[1] = parts_j[i].f[1];
                cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
        }
        
    /* since nothing bad happened to us... */
    return S_OK;

}
