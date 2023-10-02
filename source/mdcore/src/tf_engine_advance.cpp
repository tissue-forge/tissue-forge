/*******************************************************************************
 * This file is part of mdcore.
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

/* Include conditional headers. */
#include <mdcore_config.h>
#include <tfEngine.h>
#include "tf_engine_advance.h"
#include <tf_errs.h>
#include <tfCluster.h>
#include <tfFlux.h>
#include "tf_boundary_eval.h"
#include <tfError.h>
#include <tfLogger.h>
#include <tfTaskScheduler.h>

#include <math.h>

#include <sstream>
#pragma clang diagnostic ignored "-Wwritable-strings"
#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef WITH_METIS
#include <metis.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif


using namespace TissueForge;


/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


static HRESULT engine_advance_forward_euler(struct engine *e);
static HRESULT engine_advance_runge_kutta_4(struct engine *e);



static HRESULT _toofast_error(Particle *p, int line, const char* func) {
    std::stringstream ss;
    ss << "ERROR, particle moving too fast, p: {" << std::endl;
    ss << "\tid: " << p->id << ", " << std::endl;
    ss << "\ttype: " << _Engine.types[p->typeId].name << "," << std::endl;
    ss << "\tx: [" << p->x[0] << ", " << p->x[1] << ", " << p->x[2] << "], " << std::endl;
    ss << "\tv: [" << p->v[0] << ", " << p->v[1] << ", " << p->v[2] << "], " << std::endl;
    ss << "\tf: [" << p->f[0] << ", " << p->f[1] << ", " << p->f[2] << "], " << std::endl;
    ss << "}";
    
    return tf_error(E_FAIL, ss.str().c_str());
}

#define toofast_error(p) _toofast_error(p, __LINE__, TF_FUNCTION)


HRESULT TissueForge::engine_advance(struct engine *e) {
    if(e->integrator == EngineIntegrator::FORWARD_EULER) {
        return engine_advance_forward_euler(e);
    }
    else {
        return engine_advance_runge_kutta_4(e);
    }
}

FQuaternion integrate_angular_velocity_exact_1(const FVector3 &em, FPTYPE deltaTime)
{
    FVector3 ha = em * deltaTime * 0.5; // vector of half angle
    FPTYPE len = ha.length(); // magnitude
    if (len > 0) {
        ha *= std::sin(len) / len;
        FPTYPE w = std::cos(len);
        return FQuaternion(ha, w);
    } else {
        return FQuaternion(ha, 1.0);
    }
}

static FQuaternion integrate_angular_velocity_2(const FVector3 &av, FPTYPE dt) {
    FPTYPE len = av.length();
    FPTYPE theta = len * dt * 0.5;
    if (len > 1.0e-12) {
        FPTYPE w = std::cos(theta);
        FPTYPE s = std::sin(theta) / len;
        return  FQuaternion(av * s, w);
    } else {
        return FQuaternion({0.f, 0.f, 0.f}, 1.f);
    }
}

static int *cell_staggered_ids(space *s) { 
    int ind = 0;
    int *ids = (int*)malloc(sizeof(int) * s->nr_real);
    for(int ii = 0; ii < 3; ii++) 
        for(int jj = 0; jj < 3; jj++) 
            for(int kk = 0; kk < 3; kk++) 
                for(int i = ii; i < s->cdim[0]; i += 3) 
                    for(int j = jj; j < s->cdim[1]; j += 3) 
                        for(int k = kk; k < s->cdim[2]; k += 3) {
                            int cid = space_cellid(s, i, j, k);
                            if(!(s->cells[cid].flags & cell_flag_ghost)) 
                                ids[ind++] = cid;
                        }
    return ids;
}

// FPTYPE dt, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.

static inline void cell_advance_forward_euler(const FPTYPE dt, const FPTYPE h[3], const FPTYPE h2[3],
                   const FPTYPE maxv[3], const FPTYPE maxv2[3], const FPTYPE maxx[3],
                   const FPTYPE maxx2[3], int cid)
{
    space *s = &_Engine.s;
    int pid = 0;
    
    struct space_cell *c = &(s->cells[ cid ]);
    int cdim[] = {s->cdim[0], s->cdim[1], s->cdim[2]};
    FPTYPE computed_volume = 0;
    
    while(pid < c->count) {
        Particle *p = &(c->parts[pid]);
        
        if(p->flags & PARTICLE_CLUSTER || (
                                           (p->flags & PARTICLE_FROZEN_X) &&
                                           (p->flags & PARTICLE_FROZEN_Y) &&
                                           (p->flags & PARTICLE_FROZEN_Z)
                                          )) {
            pid++;
            continue;
        }
        
        FPTYPE mask[] = {
            (p->flags & PARTICLE_FROZEN_X) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Y) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Z) ? 0.0f : 1.0f
        };
        
        int delta[3];
        if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
            for(int k = 0 ; k < 3 ; k++) {
                FPTYPE v = mask[k] * (p->v[k] + dt * p->f[k] * p->imass);
                p->v[k] = v * v <= maxv2[k] ? v : v / abs(v) * maxv[k];
                p->x[k] += dt * p->v[k];
                delta[k] = std::isgreaterequal(p->x[k], h[k]) - std::isless(p->x[k], 0.0);
            }
        }
        else {
            for(int k = 0 ; k < 3 ; k++) {
                FPTYPE dx = mask[k] * (dt * p->f[k] * p->imass);
                dx = dx * dx <= maxx2[k] ? dx : dx / abs(dx) * maxx[k];
                p->v[k] = dx / dt;
                p->x[k] += dx;
                delta[k] = std::isgreaterequal(p->x[k], h[k]) - std::isless(p->x[k], 0.0);
            }
        }
        
        p->inv_number_density = p->number_density > 0.f ? 1.f / p->number_density : 0.f;
        
        /* do we have to move this particle? */
        // TODO: consolidate moving to one method.
        
        // if delta is non-zero, need to check boundary conditions, and
        // if moved out of cell, or out of bounds.
        if((delta[0] != 0) ||(delta[1] != 0) ||(delta[2] != 0)) {
            
            // if we enforce boundary, reflect back into same cell
            if(apply_update_pos_vel(p, c, h, delta)) {
                pid += 1;
            }
            // otherwise queue move to different cell
            else {
                for(int k = 0 ; k < 3 ; k++) {
                    if(p->x[k] >= h2[k] || p->x[k] <= -h[k]) {
                        toofast_error(p);
                        break;
                    }
                    FPTYPE dx = - delta[k] * h[k];
                    p->x[k] += dx;
                    p->p0[k] += dx;
                }
                struct space_cell *c_dest = &(s->cells[ celldims_cellid(cdim,
                                                   (c->loc[0] + delta[0] + cdim[0]) % cdim[0],
                                                   (c->loc[1] + delta[1] + cdim[1]) % cdim[1],
                                                   (c->loc[2] + delta[2] + cdim[2]) % cdim[2]) ]);
                
                // update any state variables on the object accordign to the boundary conditions
                // since we might be moving across periodic boundaries.
                apply_boundary_particle_crossing(p, delta, s->celllist[ p->id ], c_dest);
                
                pthread_mutex_lock(&c_dest->cell_mutex);
                space_cell_add_incomming(c_dest, p);
                c_dest->computed_volume += p->inv_number_density;
                pthread_mutex_unlock(&c_dest->cell_mutex);
                
                s->celllist[ p->id ] = c_dest;
                
                // remove a particle from a cell. if the part was the last in the
                // cell, simply dec the count, otherwise, move the last part
                // in the cell to the ejected part's prev loc.
                c->count -= 1;
                if(pid < c->count) {
                    c->parts[pid] = c->parts[c->count];
                    s->partlist[ c->parts[pid].id ] = &(c->parts[pid]);
                }
            }
        }
        else {
            computed_volume += p->inv_number_density;
            pid += 1;
        }
        
        assert(p->verify());
    }

    c->computed_volume = computed_volume;
}

static inline void cell_advance_forward_euler_cluster(const FPTYPE h[3], int cid) 
{
    space *s = &_Engine.s;
    space_cell *c = &(s->cells[ s->cid_real[cid] ]);
    int pid = 0;
    
    while(pid < c->count) {
        Particle *p = &(c->parts[pid]);
        if((p->flags & PARTICLE_CLUSTER) && p->nr_parts > 0) {
            
            Cluster_ComputeAggregateQuantities((Cluster*)p);
            
            int delta[3];
            for(int k = 0 ; k < 3 ; k++) {
                delta[k] = std::isgreaterequal(p->x[k], h[k]) - std::isless(p->x[k], 0.0);
            }
            
            /* do we have to move this particle? */
            // TODO: consolidate moving to one method.
            if((delta[0] != 0) ||(delta[1] != 0) ||(delta[2] != 0)) {
                for(int k = 0 ; k < 3 ; k++) {
                    p->x[k] -= delta[k] * h[k];
                    p->p0[k] -= delta[k] * h[k];
                }
                
                space_cell *c_dest = &(s->cells[ space_cellid(s,
                                                    (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0],
                                                    (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1],
                                                    (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2]) ]);
                
                pthread_mutex_lock(&c_dest->cell_mutex);
                space_cell_add_incomming(c_dest, p);
                pthread_mutex_unlock(&c_dest->cell_mutex);
                
                s->celllist[ p->id ] = c_dest;
                
                // remove a particle from a cell. if the part was the last in the
                // cell, simply dec the count, otherwise, move the last part
                // in the cell to the ejected part's prev loc.
                c->count -= 1;
                if(pid < c->count) {
                    c->parts[pid] = c->parts[c->count];
                    s->partlist[ c->parts[pid].id ] = &(c->parts[pid]);
                }
            }
            else {
                pid += 1;
            }
        }
        else {
            pid += 1;
        }
    }
}

/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 */
HRESULT engine_advance_forward_euler(struct engine *e) {

    TF_Log(LOG_TRACE);

    // do flux substeps, if any
    if(!(e->flags & engine_flag_cuda) && e->nr_fluxsteps > 1) {
        auto func = [](int cid) -> void { Fluxes_integrate(&_Engine.s.cells[cid], _Engine.dt_flux); };
        for(e->step_flux = 0; e->step_flux < e->nr_fluxsteps - 1; e->step_flux++) {
            if(engine_fluxonly_eval(e) != S_OK) 
                return error(MDCERR_engine);
            parallel_for(e->s.nr_real, func);
        }

        TF_Log(LOG_TRACE);
    }

    // set the integrator flag to set any persistent forces
    // forward euler is a single step, so alwasy set this flag
    e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;

    if (engine_force(e) != S_OK) {
        TF_Log(LOG_CRITICAL);
        return error(MDCERR_engine);
    }

    ticks tic = getticks();

    int cid, pid, k, delta[3];
    struct space_cell *c, *c_dest;
    struct Particle *p;
    struct space *s;
    FPTYPE dt, time, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    FPTYPE epot = 0.0, computed_volume = 0.0;

#ifdef HAVE_OPENMP

    int step;
    FPTYPE epot_local;

#endif

    /* Get a grip on the space. */
    s = &(e->s);
    time = e->time;
    dt = e->dt;
    for(k = 0 ; k < 3 ; k++) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];
        
        maxx[k] = h[k] * e->particle_max_dist_fraction;
        maxx2[k] = maxx[k] * maxx[k];
        
        // max velocity and step, as a fraction of cell size.
        maxv[k] = maxx[k] / dt;
        maxv2[k] = maxv[k] * maxv[k];
    }

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for(cid = 0 ; cid < s->nr_ghost ; cid++)
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#ifdef HAVE_OPENMP
#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for(pid = 0 ; pid < c->count ; pid++) {
                    p = &(c->parts[pid]);

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for(k = 0 ; k < 3 ; k++) {
                            p->v[k] += p->f[k] * dt * p->imass;
                            p->x[k] += p->v[k] * dt;
                        }
                    }
                    else {
                        for(k = 0 ; k < 3 ; k++) {
                            p->v[k] = p->f[k] * p->imass;
                            p->x[k] += p->v[k] * dt;
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
#else
        auto func_update_parts = [&](int _cid) -> void {
            space_cell *_c = &(s->cells[ s->cid_real[_cid] ]);
            for(int _pid = 0 ; _pid < _c->count ; _pid++) {
                Particle *_p = &(_c->parts[_pid]);

                if(engine::types[_p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                    for(int _k = 0 ; _k < 3 ; _k++) {
                        _p->v[_k] += _p->f[_k] * dt * _p->imass;
                        _p->x[_k] += _p->v[_k] * dt;
                    }
                }
                else {
                    for(int _k = 0 ; _k < 3 ; _k++) {
                        _p->v[_k] = _p->f[_k] * _p->imass;
                        _p->x[_k] += _p->v[_k] * dt;
                    }
                }
            }
        };
        parallel_for(s->nr_real, func_update_parts);
        for(cid = 0; cid < s->nr_real; cid++) {
            epot += s->cells[ s->cid_real[cid] ].epot;
            computed_volume += s->cells[s->cid_real[cid]].computed_volume;
        }
#endif
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {
        
        // make a lambda function that we run in parallel, capture local vars.
        // we use the same lambda in both parallel and serial versions to
        // make sure same code gets exercized.
        //
        // cell_advance_forward_euler(const FPTYPE dt, const FPTYPE h[3], const FPTYPE h2[3],
        // const FPTYPE maxv[3], const FPTYPE maxv2[3], const FPTYPE maxx[3],
        // const FPTYPE maxx2[3], FPTYPE *total_pot, int cid)

        static int *staggered_ids = cell_staggered_ids(s);
        
        auto func = [dt, &h, &h2, &maxv, &maxv2, &maxx, &maxx2](int cid) -> void {
            int _cid = staggered_ids[cid];
            cell_advance_forward_euler(dt, h, h2, maxv, maxv2, maxx, maxx2, _cid);
            Fluxes_integrate(&_Engine.s.cells[_cid], _Engine.dt_flux);
        };
        
        parallel_for(s->nr_real, func);

        auto func_advance_clusters = [&h](int _cid) -> void {
            cell_advance_forward_euler_cluster(h, staggered_ids[_cid]);
        };
        parallel_for(s->nr_real, func_advance_clusters);

        auto func_space_cell_welcome = [&](int _cid) -> void {
            space_cell_welcome(&(s->cells[ s->cid_marked[_cid] ]), s->partlist);
        };
        parallel_for(s->nr_marked, func_space_cell_welcome);

        /* Collect potential energy and computed volume */
        for(cid = 0; cid < s->nr_cells; cid++) {
            epot += s->cells[cid].epot;
            computed_volume += s->cells[cid].computed_volume;
        }

        TF_Log(LOG_TRACE) << "step: " << time  << ", computed volume: " << computed_volume;
    } // endif NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot += epot;
    s->epot_nonbond += epot;
    e->computed_volume = computed_volume;

    VERIFY_PARTICLES();

    e->timers[engine_timer_advance] += getticks() - tic;

    TF_Log(LOG_TRACE);

    /* return quietly */
    return S_OK;
}

#define CHECK_TOOFAST(p, h, h2) \
{\
    for(int _k = 0; _k < 3; _k++) {\
        if (p->x[_k] >= h2[_k] || p->x[_k] <= -h[_k]) {\
            return toofast_error(p);\
        }\
    }\
}\



/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 */

HRESULT engine_advance_runge_kutta_4(struct engine *e) {

    int cid, pid, k, delta[3], step;
    struct space_cell *c, *c_dest;
    struct Particle *p;
    struct space *s;
    FPTYPE dt, w, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    FPTYPE epot = 0.0, epot_local;
    int toofast;

    /* Get a grip on the space. */
    s = &(e->s);
    dt = e->dt;
    for(k = 0 ; k < 3 ; k++) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];

        maxv[k] = h[k] / (e->particle_max_dist_fraction * dt);
        maxv2[k] = maxv[k] * maxv[k];

        maxx[k] = h[k] / (e->particle_max_dist_fraction);
        maxx2[k] = maxx[k] * maxx[k];
    }

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for(cid = 0 ; cid < s->nr_ghost ; cid++)
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for(pid = 0 ; pid < c->count ; pid++) {
                    p = &(c->parts[pid]);
                    w = dt * p->imass;

                    toofast = 0;
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for(k = 0 ; k < 3 ; k++) {
                            p->v[k] += dt * p->f[k] * w;
                            p->x[k] += dt * p->v[k];
                            delta[k] = isgreaterequal(p->x[k], h[k]) - isless(p->x[k], 0.0);
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                    else {
                        for(k = 0 ; k < 3 ; k++) {
                            p->v[k] = p->f[k] * w;
                            p->x[k] += dt * p->v[k];
                            delta[k] = isgreaterequal(p->x[k], h[k]) - isless(p->x[k], 0.0);
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

        /* Collect potential energy from ghosts. */
        for(cid = 0 ; cid < s->nr_ghost ; cid++) {
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        }

        // **  get K1, calculate forces at current position **
        // set the integrator flag to set any persistent forces
        e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;
        if (engine_force(e) != S_OK) {
            return error(MDCERR_engine);
        }
        e->integrator_flags &= ~INTEGRATOR_UPDATE_PERSISTENTFORCE;

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            toofast = 0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                
                while(pid < c->count) {
                    TF_Log(LOG_TRACE);

                    p = &(c->parts[pid]);
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[0] = p->force * p->imass;
                        p->xk[0] = p->velocity;
                    }
                    else {
                        p->xk[0] = p->force * p->imass;
                    }

                    // update position for k2
                    p->p0 = p->position;
                    p->v0 = p->velocity;
                    p->position = p->p0 + 0.5 * dt * p->xk[0];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K2, calculate forces at x0 + 1/2 dt k1 **
        if (engine_force_prep(e) != S_OK || engine_force(e) != S_OK) {
            return error(MDCERR_engine);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while(pid < c->count) {
                    TF_Log(LOG_TRACE);

                    p = &(c->parts[pid]);

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[1] = p->force * p->imass;
                        p->xk[1] = p->v0 + 0.5 * dt * p->vk[0];
                    }
                    else {
                        p->xk[1] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + 0.5 * dt * p->xk[1];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K3, calculate forces at x0 + 1/2 dt k2 **
        if (engine_force_prep(e) != S_OK || engine_force(e) != S_OK) {
            return error(MDCERR_engine);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while(pid < c->count) {
                    TF_Log(LOG_TRACE);

                    p = &(c->parts[pid]);

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[2] = p->force * p->imass;
                        p->xk[2] = p->v0 + 0.5 * dt * p->vk[1];
                    }
                    else {
                        p->xk[2] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + dt * p->xk[2];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K4, calculate forces at x0 + dt k3, final position calculation **
        if (engine_force_prep(e) != S_OK || engine_force(e) != S_OK) {
            return error(MDCERR_engine);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while(pid < c->count) {
                    TF_Log(LOG_TRACE);

                    p = &(c->parts[pid]);
                    toofast = 0;

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[3] = p->imass * p->force;
                        p->xk[3] = p->v0 + dt * p->vk[2];
                        p->velocity = p->v0 + (dt/6.) * (p->vk[0] + 2*p->vk[1] + 2 * p->vk[2] + p->vk[3]);
                    }
                    else {
                        p->xk[3] = p->imass * p->force;
                    }
                    
                    p->position = p->p0 + (dt/6.) * (p->xk[0] + 2*p->xk[1] + 2 * p->xk[2] + p->xk[3]);

                    for(int k = 0; k < 3; ++k) {
                        delta[k] = std::isgreaterequal(p->x[k], h[k]) - std::isless(p->x[k], 0.0);
                        toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                    }

                    if(toofast) {
                        return toofast_error(p);
                    }

                    /* do we have to move this particle? */
                    if((delta[0] != 0) ||(delta[1] != 0) ||(delta[2] != 0)) {
                        for(k = 0 ; k < 3 ; k++) {
                            p->x[k] -= delta[k] * h[k];
                        }

                        c_dest = &(s->cells[ space_cellid(s,
                                (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0],
                                (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1],
                                (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2]) ]);

                        pthread_mutex_lock(&c_dest->cell_mutex);
                        space_cell_add_incomming(c_dest, p);
                        pthread_mutex_unlock(&c_dest->cell_mutex);

                        s->celllist[ p->id ] = c_dest;

                        // remove a particle from a cell. if the part was the last in the
                        // cell, simply dec the count, otherwise, move the last part
                        // in the cell to the ejected part's prev loc.
                        c->count -= 1;
                        if(pid < c->count) {
                            c->parts[pid] = c->parts[c->count];
                            s->partlist[ c->parts[pid].id ] = &(c->parts[pid]);
                        }
                    }
                    else {
                        pid += 1;
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }

        /* Welcome the new particles in each cell. */
#pragma omp parallel for schedule(static)
        for(cid = 0 ; cid < s->nr_marked ; cid++) {
            space_cell_welcome(&(s->cells[ s->cid_marked[cid] ]), s->partlist);
        }

    } // endif  NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot_nonbond += epot;
    s->epot += epot;

    /* return quietly */
    return S_OK;
}
