/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#include "mdcore_config.h"
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
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfRigid.h>
#include <tfError.h>


using namespace TissueForge;


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))
    

HRESULT TissueForge::rigid_eval_shake(struct rigid *rs, int N, struct engine *e) {

    int iter, rid, k, j, pid, pjd, nr_parts, nr_constr, shift;
    struct Particle *p[rigid_maxparts], **partlist;
    struct space_cell *c[rigid_maxparts], **celllist;
    struct rigid *r;
    FPTYPE dt, idt;
    FPTYPE xp[3*rigid_maxparts], xp_old[3*rigid_maxparts], h[3];
    FPTYPE res[rigid_maxconstr];
    FPTYPE m[rigid_maxparts], tol, lambda, w;
    FPTYPE vc[3*rigid_maxconstr], max_res;
    FPTYPE wvc[3*rigid_maxconstr];
    // FPTYPE vcom[3];

    /* Check for bad input. */
    partlist = e->s.partlist;
    celllist = e->s.celllist;
    tol = e->tol_rigid;
    dt = e->dt;
    idt = 1.0/dt;
    if(rs == NULL || e == NULL)
        return error(MDCERR_null);
        
    /* Get some local values. */
    for(k = 0 ; k < 3 ; k++)
        h[k] = e->s.h[k];
        
    /* Loop over the rigid constraints. */
    for(rid = 0 ; rid < N ; rid++) {
    
        /* Get some local values we'll be re-usnig quite a bit. */
        r = &rs[rid];
        nr_parts = r->nr_parts;
        nr_constr = r->nr_constr;
    
        /* Check if the particles are local, if not bail. */
        for(k = 0 ; k < nr_parts ; k++) {
            if((p[k] = partlist[ r->parts[k] ]) == NULL)
                break;
            c[k] = celllist[ r->parts[k] ];
            m[k] = e->types[ p[k]->typeId ].mass;
        }
        if(k < nr_parts)
            continue;
            
        /* Are all the parts ghosts? */
        for(k = 0 ; k < nr_parts && (p[k]->flags & PARTICLE_GHOST) ; k++);
        if(k == nr_parts)
            continue;
            
        /* Load the particle positions relative to the first particle's cell. */
        for(k = 0 ; k < nr_parts ; k++)
            if(c[k] != c[0])
                for(j = 0 ; j < 3 ; j++) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if(shift > 1)
                        shift = -1;
                    else if(shift < -1)
                        shift = 1;
                    xp[3*k+j] = p[k]->x[j] + h[j]*shift;
                }
            else
                for(j = 0 ; j < 3 ; j++)
                    xp[3*k+j] = p[k]->x[j];
                    
        /* Get the particle positions before the step. */
        for(k = 0 ; k < nr_parts ; k++)
            for(j = 0 ; j < 3 ; j++)
                xp_old[ k*3 + j ] = xp[ k*3 + j ] - dt*p[k]->v[j];
                    
        /* Create the gradient vectors. */
        for(k = 0 ; k < nr_constr ; k++) {
            pid = r->constr[k].i;
            pjd = r->constr[k].j;
            for(j = 0 ; j < 3 ; j++) {
                vc[k*3+j] = xp_old[3*pid+j] - xp_old[3*pjd+j];
                wvc[k*3+j] = FPTYPE_ONE /(m[pid] + m[pjd]) * vc[3*k+j];
            }
        }
                    
        /* Main SHAKE loop. */
        for(iter = 0 ; iter < rigid_maxiter ; iter++) {
        
            /* Compute the residues (squared). */
            for(max_res = 0.0, k = 0 ; k < nr_constr ; k++) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                res[k] = r->constr[k].d2;
                for(j = 0 ; j < 3 ; j++)
                    res[k] -=(xp[3*pid+j] - xp[3*pjd+j]) *(xp[3*pid+j] - xp[3*pjd+j]);
                if(fabs(res[k]) > max_res)
                    max_res = fabs(res[k]);
            }
            
            /* Are we done? */
            if(max_res < tol)
                break;
                
            /* Adjust each constraint. */
            for(k = 0 ; k < nr_constr ; k++) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                lambda = 0.5 * res[k] /((xp[3*pid] - xp[3*pjd])*vc[3*k] + (xp[3*pid+1] - xp[3*pjd+1])*vc[3*k+1] + (xp[3*pid+2] - xp[3*pjd+2])*vc[3*k+2]);
                for(j = 0 ; j < 3 ; j++) {
                    w = lambda * wvc[3*k+j];
                    xp[3*pid+j] += w * m[pjd];
                    xp[3*pjd+j] -= w * m[pid];
                }
            }
        
        } /* Main SHAKE loop. */
            
        /* Did we fail to converge? */
        if(iter == rigid_maxiter) {
            printf("rigid_eval_shake: rigid %i failed to converge in less than %i iterations.\n", rid, rigid_maxiter);
            for(k = 0 ; k < nr_constr ; k++) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                printf("rigid_eval_shake: constr %i between parts %i and %i, d=%e.\n", k, r->parts[pid], r->parts[pjd], sqrt(r->constr[k].d2));
                printf("rigid_eval_shake: res[%i]=%e.\n", k, res[k]);
            }
        }
            
        for(k = 0 ; k < nr_parts ; k++) {
            if(c[k] != c[0])
                for(j = 0 ; j < 3 ; j++) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if(shift > 1)
                        shift = -1;
                    else if(shift < -1)
                        shift = 1;
                    p[k]->v[j] += idt *(xp[3*k+j] - h[j]*shift - p[k]->x[j]);
                    p[k]->x[j] = xp[3*k+j] - h[j]*shift;
                }
            else
                for(j = 0 ; j < 3 ; j++) {
                    p[k]->v[j] += idt *(xp[3*k+j] - p[k]->x[j]);
                    p[k]->x[j] = xp[3*k+j];
                }
        }
    
    } /* Loop over the constraints. */
        
    /* Bail quitely. */
    return S_OK;
        
}

HRESULT TissueForge::rigid_eval_pshake(struct rigid *rs, int N, struct engine *e, int a_update) {

    int iter, rid, k, j, i, pid, pjd, nr_parts, nr_constr, shift;
    struct Particle *p[rigid_maxparts], **partlist;
    struct space_cell *c[rigid_maxparts], **celllist;
    struct rigid *r;
    FPTYPE dt, idt;
    FPTYPE xp[3*rigid_maxparts], xp_old[3*rigid_maxparts], h[3];
    FPTYPE res[rigid_maxconstr];
    FPTYPE lambda, m[rigid_maxparts], tol, max_res;
    FPTYPE vc[3*rigid_maxconstr*rigid_maxparts], dv[3];
    // FPTYPE vcom[3];

    /* Check for bad input. */
    partlist = e->s.partlist;
    celllist = e->s.celllist;
    tol = e->tol_rigid;
    dt = e->dt;
    idt = 1.0/dt;
    if(rs == NULL || e == NULL)
        return error(MDCERR_null);
        
    /* Get some local values. */
    for(k = 0 ; k < 3 ; k++)
        h[k] = e->s.h[k];
        
    /* Loop over the rigid constraints. */
    for(rid = 0 ; rid < N ; rid++) {
    
        /* Get some local values we'll be re-using quite a bit. */
        r = &rs[rid];
        nr_parts = r->nr_parts;
        nr_constr = r->nr_constr;
    
        /* Check if the particles are local, if not bail. */
        for(k = 0 ; k < nr_parts ; k++) {
            if((p[k] = partlist[ r->parts[k] ]) == NULL)
                break;
            c[k] = celllist[ r->parts[k] ];
            m[k] = e->types[ p[k]->typeId ].mass;
        }
        if(k < nr_parts)
            continue;
            
        /* Are all the parts ghosts? */
        for(k = 0 ; k < nr_parts && (p[k]->flags & PARTICLE_GHOST) ; k++);
        if(k == nr_parts)
            continue;
            
        /* Load the particle positions relative to the first particle's cell. */
        for(k = 0 ; k < nr_parts ; k++)
            if(c[k] != c[0])
                for(j = 0 ; j < 3 ; j++) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if(shift > 1)
                        shift = -1;
                    else if(shift < -1)
                        shift = 1;
                    xp[3*k+j] = p[k]->x[j] + h[j]*shift;
                }
            else
                for(j = 0 ; j < 3 ; j++)
                    xp[3*k+j] = p[k]->x[j];
                    
        /* Get the particle positions before the step. */
        for(k = 0 ; k < nr_parts ; k++)
            for(j = 0 ; j < 3 ; j++)
                xp_old[ k*3 + j ] = xp[ k*3 + j ] - dt*p[k]->v[j];
                    
        /* Create the gradient vectors. */
        bzero(vc, sizeof(FPTYPE) * 3 * nr_constr * nr_parts);
        for(k = 0 ; k < nr_constr ; k++) {
            pid = r->constr[k].i;
            pjd = r->constr[k].j;
            for(j = 0 ; j < 3 ; j++)
                dv[j] =(xp_old[3*pid+j] - xp_old[3*pjd+j]) /(m[pid] + m[pjd]);
            for(i = 0 ; i < nr_constr ; i++)
                if(r->a[ k*nr_constr + i ] != 0.0f)
                    for(j = 0 ; j < 3 ; j++) {
                        vc[ i*nr_parts*3 + pid*3+j ] +=  r->a[ k*nr_constr + i ] * dv[j] * m[pjd];
                        vc[ i*nr_parts*3 + pjd*3+j ] += -r->a[ k*nr_constr + i ] * dv[j] * m[pid];
                    }
        }
                    
        /* Main P-SHAKE loop. */
        for(iter = 0 ; iter < rigid_maxiter ; iter++) {
        
            /* Loop over the constraints... */
            for(max_res = 0.0, k = 0 ; k < nr_constr ; k++) {
            
                /* Parts in this constraint? */
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
        
                /* Compute the residue (squared). */
                res[k] = r->constr[k].d2;
                for(j = 0 ; j < 3 ; j++)
                    res[k] -=(xp[3*pid+j] - xp[3*pjd+j]) *(xp[3*pid+j] - xp[3*pjd+j]);
                if(fabs(res[k]) > max_res)
                    max_res = fabs(res[k]);
                
                // printf("rigid_eval_pshake: res[%i]=%e.\n", k, res[k]);
                    
                /* Compute the correction. */
                lambda = 0.5 * res[k] /((xp[3*pid+0] - xp[3*pjd+0])*(vc[k*3*nr_parts+3*pid+0] - vc[k*3*nr_parts+3*pjd+0]) + 
                                          (xp[3*pid+1] - xp[3*pjd+1])*(vc[k*3*nr_parts+3*pid+1] - vc[k*3*nr_parts+3*pjd+1]) +
                                          (xp[3*pid+2] - xp[3*pjd+2])*(vc[k*3*nr_parts+3*pid+2] - vc[k*3*nr_parts+3*pjd+2]));
            
                /* Adjust the particle positions. */
                for(j = 0 ; j < 3*nr_parts ; j++)
                    xp[j] += lambda * vc[k*3*nr_parts+j];
                    
            } /* loop over the constraints. */
        
            /* Are we done? */
            if(max_res < tol)
                break;
                
        } /* Main SHAKE loop. */
            
        /* Dump data if failed. */
        if(iter == rigid_maxiter) {
            printf("rigid_eval_pshake: rigid %i failed to converge in less than %i iterations.\n", rid, iter);
            for(k = 0 ; k < nr_constr ; k++) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                printf("rigid_eval_pshake: constr %i between parts %i and %i, d=%e.\n", k, r->parts[pid], r->parts[pjd], sqrt(r->constr[k].d2));
                printf("rigid_eval_pshake: res[%i]=%e.\n", k, res[k]);
            }
        }
            
        /* Adjust weights? */
        if(nr_constr > 1 &&(a_update || iter > rigid_pshake_refine)) {
        
            /* Some strictly local stuff. */
            FPTYPE a_new[ nr_constr * nr_constr ], tmp[ nr_constr * nr_constr ];
            FPTYPE w, dx[3], max_alpha = 0.0f;
                
            /* Compute the entries of a_new. */
            for(i = 0 ; i < nr_constr ; i++) {
                pid = r->constr[i].i;
                pjd = r->constr[i].j;
                for(j = 0 ; j < 3 ; j++)
                    dx[j] = xp[3*pid+j] - xp[3*pjd+j];
                w = -1.0 /(dx[0]*(vc[ i*3*nr_parts + pid*3 + 0 ] - vc[ i*3*nr_parts + pjd*3 + 0 ]) +
                             dx[1]*(vc[ i*3*nr_parts + pid*3 + 1 ] - vc[ i*3*nr_parts + pjd*3 + 1 ]) +
                             dx[2]*(vc[ i*3*nr_parts + pid*3 + 2 ] - vc[ i*3*nr_parts + pjd*3 + 2 ]));
                for(j = i+1 ; j < nr_constr ; j++) {
                    a_new[ i + j*nr_constr ] = w *(dx[0]*(vc[ j*3*nr_parts + pid*3 + 0 ] - vc[ j*3*nr_parts + pjd*3 + 0 ]) +
                                                     dx[1]*(vc[ j*3*nr_parts + pid*3 + 1 ] - vc[ j*3*nr_parts + pjd*3 + 1 ]) +
                                                     dx[2]*(vc[ j*3*nr_parts + pid*3 + 2 ] - vc[ j*3*nr_parts + pjd*3 + 2 ]));
                    a_new[ j + i*nr_constr ] = a_new[ i + j*nr_constr ];
                    max_alpha = FPTYPE_FMAX(max_alpha, FPTYPE_FABS(a_new[ i + j*nr_constr ]));
                }
                a_new[ i + i*nr_constr ] = 1.0f;
            }
                
            /* Re-scale? */
            if(max_alpha > rigid_pshake_maxalpha) {
                w = 0.1 / max_alpha;
                for(i = 0 ; i < nr_constr * nr_constr ; i++)
                    a_new[ i ] *= w;
                for(i = 0 ; i < nr_constr ; i++)
                    a_new[ i*nr_constr + i ] = 1.0f;
            }
                
            /* tmp = r->a * a_new. */
            for(i = 0 ; i < nr_constr ; i++)
                for(j = 0 ; j < nr_constr ; j++) {
                    tmp[ j*nr_constr + i ] = 0.0;
                    for(k = 0 ; k < nr_constr ; k++)
                        tmp[ j*nr_constr + i ] += r->a[ k*nr_constr + i ] * a_new[ j*nr_constr + k ];
                }
            memcpy(r->a, tmp, sizeof(FPTYPE) * nr_constr * nr_constr);
        
        }

        for(k = 0 ; k < nr_parts ; k++) {
            if(c[k] != c[0])
                for(j = 0 ; j < 3 ; j++) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if(shift > 1)
                        shift = -1;
                    else if(shift < -1)
                        shift = 1;
                    p[k]->v[j] += idt *(xp[3*k+j] - h[j]*shift - p[k]->x[j]);
                    p[k]->x[j] = xp[3*k+j] - h[j]*shift;
                }
            else
                for(j = 0 ; j < 3 ; j++) {
                    p[k]->v[j] += idt *(xp[3*k+j] - p[k]->x[j]);
                    p[k]->x[j] = xp[3*k+j];
                }
        }
    
    } /* Loop over the constraints. */
        
    /* Bail quitely. */
    return S_OK;
        
}
