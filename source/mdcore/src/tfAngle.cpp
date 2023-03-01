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

#include <tfAngle.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <unordered_set>

/* Include some conditional headers. */
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <tfError.h>
#include <tf_util.h>
#include <tfLogger.h>
#include "cycle.h"
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include "tf_potential_eval.h"
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <io/tfFIO.h>
#include <rendering/tfStyle.h>
#include <tf_mdcore_io.h>
#include <tf_metrics.h>

#ifdef HAVE_CUDA
#include "tfAngle_cuda.h"
#endif

#include <iostream>
#include <random>


using namespace TissueForge;


rendering::Style *Angle_StylePtr = new rendering::Style("aqua");


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


static bool Angle_decays(Angle *a, std::uniform_real_distribution<FPTYPE> *uniform01=NULL) {
    if(!a || a->half_life <= 0.0) return false;

    bool created = uniform01 == NULL;
    if(created) uniform01 = new std::uniform_real_distribution<FPTYPE>(0.0, 1.0);

    FPTYPE pr = 1.0 - std::pow(2.0, -_Engine.dt / a->half_life);
    RandomType &randEng = randomEngine();
    bool result = (*uniform01)(randEng) < pr;

    if(created) delete uniform01;

    return result;
}


HRESULT TissueForge::angle_eval(struct Angle *a, int N, struct engine *e, FPTYPE *epot_out) { 

    #ifdef HAVE_CUDA
    if(e->angles_cuda) {
        return cuda::engine_angle_eval_cuda(a, N, e, epot_out);
    }
    #endif

    Angle *angle;
    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, *pk, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *pota;
    std::vector<struct Potential *> pots;
    FVector3 xi, xj, xk, dxi, dxk;
    FPTYPE ctheta, wi, wk, fi[3], fk[3], fic, fkc;
    FVector3 rji, rjk;
    FPTYPE ee, inji, injk, dprod;
    std::unordered_set<struct Angle*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<FPTYPE> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE];
    FPTYPE cthetaq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], dkq[VEC_SIZE*3];
    struct Angle *angleq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff;
#endif
    
    /* Check inputs. */
    if(a == NULL || e == NULL) 
        return error(MDCERR_null);

    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for(aid = 0 ; aid < N ; aid++) {
        angle = &a[aid];
        angle->potential_energy = 0.0;

        if(Angle_decays(angle)) {
            toDestroy.insert(angle);
            continue;
        }

        if(!(a->flags & ANGLE_ACTIVE))
            continue;
    
        /* Get the particles involved. */
        pid = angle->i; pjd = angle->j; pkd = angle->k;
        if((pi = partlist[ pid]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        if((pk = partlist[ pkd ]) == NULL)
            continue;
        
        /* Skip if all three are ghosts. */
        if((pi->flags & PARTICLE_GHOST) && (pj->flags & PARTICLE_GHOST) && (pk->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pota = angle->potential) == NULL)
            continue;
    
        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
        
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if(shift > 1)
                shift = -1;
            else if(shift < -1)
                shift = 1;
            xi[k] = pi->x[k] + shift*h[k];
            shift = lock[k] - locj[k];
            if(shift > 1)
                shift = -1;
            else if(shift < -1)
                shift = 1;
            xk[k] = pk->x[k] + shift*h[k];
        }
            
        /* Get the angle rays. */
        for(k = 0 ; k < 3 ; k++) {
            rji[k] = xi[k] - xj[k];
            rjk[k] = xk[k] - xj[k];
        }
            
        /* Compute some quantities we will re-use. */
        dprod = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];
        inji = FPTYPE_ONE / FPTYPE_SQRT(rji[0]*rji[0] + rji[1]*rji[1] + rji[2]*rji[2]);
        injk = FPTYPE_ONE / FPTYPE_SQRT(rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2]);
        
        /* Compute the cosine. */
        ctheta = FPTYPE_FMAX(-FPTYPE_ONE, FPTYPE_FMIN(FPTYPE_ONE, dprod * inji * injk));
        
        // Set the derivatives.
        // particles could be perpenducular, then plan is undefined, so
        // choose a random orientation plane
        if(ctheta == 0 || ctheta == -1) {
            std::uniform_real_distribution<FPTYPE> dist{-1, 1};
            // make a random vector
            RandomType &randEng = randomEngine();
            FVector3 x{dist(randEng), dist(randEng), dist(randEng)};
            
            // vector between outer particles
            FVector3 vik = xi - xk;
            
            // make it orthogonal to rji
            x = x - Magnum::Math::dot(x, vik) * vik;
            
            // normalize it.
            dxi = dxk = x.normalized();
        } else {
            for(k = 0 ; k < 3 ; k++) {
                dxi[k] = (rjk[k]*injk - ctheta * rji[k]*inji) * inji;
                dxk[k] = (rji[k]*inji - ctheta * rjk[k]*injk) * injk;
            }
        }

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fk), std::end(fk), 0.0);
                pot->eval_byparts3(pot, pi, pj, pk, ctheta, &ee, fi, fk);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pk->f[i] += (fkc = fk[i]);
                    pj->f[i] -= fic + fkc;
                }
                epot += ee;
                angle->potential_energy += ee;
                if(angle->potential_energy >= angle->dissociation_energy)
                    toDestroy.insert(angle);
            }
            else {

                if(ctheta > pot->b) 
                    continue;
            
                #ifdef VECTORIZE
                    /* add this angle to the interaction queue. */
                    cthetaq[icount] = FPTYPE_FMAX(ctheta, pot->a);
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    dkq[icount*3] = dxk[0];
                    dkq[icount*3+1] = dxk[1];
                    dkq[icount*3+2] = dxk[2];
                    effi[icount] = pi->f;
                    effj[icount] = pj->f;
                    effk[icount] = pk->f;
                    potq[icount] = pot;
                    angleq[icount] = angle;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r(potq, cthetaq, eeq, eff);
                            #else
                            potential_eval_vec_4single_r(potq, cthetaq, eeq, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r(potq, cthetaq, eeq, eff);
                            #else
                            potential_eval_vec_2double_r(potq, cthetaq, eeq, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            for(k = 0 ; k < 3 ; k++) {
                                effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                                effk[l][k] -= (wk = eff[l] * dkq[3*l+k]);
                                effj[l][k] += wi + wk;
                            }
                            epot += eeq[l];
                            angleq[l]->potential_energy += eeq[l];
                            if(angleq[l]->potential_energy >= angleq[l]->dissociation_energy)
                                toDestroy.insert(angleq[l]);
                        }

                        /* re-set the counter. */
                        icount = 0;

                        }
                #else
                    /* evaluate the angle */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, FPTYPE_FMAX(ctheta, pot->a), &ee, &eff);
                    #else
                        potential_eval_r(pot, FPTYPE_FMAX(ctheta, pot->a), &ee, &eff);
                    #endif
                    
                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        pi->f[k] -= (wi = eff * dxi[k]);
                        pk->f[k] -= (wk = eff * dxk[k]);
                        pj->f[k] += wi + wk;
                    }

                    /* tabulate the energy */
                    epot += ee;
                    angle->potential_energy += ee;
                    if(angle->potential_energy >= angle->dissociation_energy)
                        toDestroy.insert(angle);
                #endif

            }

        }
        
    } /* loop over angles. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {

            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
            }

            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r(potq, cthetaq, eeq, eff);
                #else
                potential_eval_vec_4single_r(potq, cthetaq, eeq, eff);
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r(potq, cthetaq, eeq, eff);
                #else
                potential_eval_vec_2double_r(potq, cthetaq, eeq, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                for(k = 0 ; k < 3 ; k++) {
                    effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                    effk[l][k] -= (wk = eff[l] * dkq[3*l+k]);
                    effj[l][k] += wi + wk;
                }
                epot += eeq[l];
                angleq[l]->potential_energy += eeq[l];
                if(angleq[l]->potential_energy >= angleq[l]->dissociation_energy)
                    toDestroy.insert(angleq[l]);
            }

        }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto ai : toDestroy)
        Angle_Destroy(ai);
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return S_OK;
    
}


HRESULT TissueForge::angle_evalf(struct Angle *a, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out) {

    Angle *angle;
    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, *pk, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *pota;
    std::vector<struct Potential *> pots;
    FPTYPE ee, xi[3], xj[3], xk[3], dxi[3], dxk[3], ctheta, wi, wk, fi[3], fk[3], fic, fkc;
    FPTYPE t1, t10, t11, t12, t13, t21, t22, t23, t24, t25, t26, t27, t3,
        t5, t6, t7, t8, t9, t4, t14, t2;
    std::unordered_set<struct Angle*> toDestroy;
    toDestroy.reserve(N);
    std::uniform_real_distribution<FPTYPE> uniform01(0.0, 1.0);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE];
    FPTYPE cthetaq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eeq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], dkq[VEC_SIZE*3];
    struct Angle *angleq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE eff;
#endif
    
    /* Check inputs. */
    if(a == NULL || e == NULL)
        return error(MDCERR_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for(aid = 0 ; aid < N ; aid++) {
        angle = &a[aid];
        angle->potential_energy = 0.0;

        if(Angle_decays(angle)) {
            toDestroy.insert(angle);
            continue;
        }
    
        /* Get the particles involved. */
        pid = angle->i; pjd = angle->j; pkd = angle->k;
        if((pi = partlist[ pid]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        if((pk = partlist[ pkd ]) == NULL)
            continue;
        
        /* Skip if all three are ghosts. */
        if((pi->flags & PARTICLE_GHOST) && (pj->flags & PARTICLE_GHOST) && (pk->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pota = angle->potential) == NULL)
            continue;
    
        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
        
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if(shift > 1)
                shift = -1;
            else if(shift < -1)
                shift = 1;
            xi[k] = pi->x[k] + h[k]*shift;
            shift = lock[k] - locj[k];
            if(shift > 1)
                shift = -1;
            else if(shift < -1)
                shift = 1;
            xk[k] = pk->x[k] + h[k]*shift;
        }
            
        /* This is Maple-generated code, see "angles.maple" for details. */
        t2 = xj[2]*xj[2];
        t4 = xj[1]*xj[1];
        t14 = xj[0]*xj[0];
        t21 = t2+t4+t14;
        t24 = -FPTYPE_TWO*xj[2];
        t25 = -FPTYPE_TWO*xj[1];
        t26 = -FPTYPE_TWO*xj[0];
        t6 = (t24+xi[2])*xi[2]+(t25+xi[1])*xi[1]+(t26+xi[0])*xi[0]+t21;
        t3 = FPTYPE_ONE/sqrt(t6);
        t10 = xk[0]-xj[0];
        t11 = xi[2]-xj[2];
        t12 = xi[1]-xj[1];
        t13 = xi[0]-xj[0];
        t8 = xk[2]-xj[2];
        t9 = xk[1]-xj[1];
        t7 = t13*t10+t12*t9+t11*t8;
        t27 = t3*t7;
        t5 = (t24+xk[2])*xk[2]+(t25+xk[1])*xk[1]+(t26+xk[0])*xk[0]+t21;
        t1 = FPTYPE_ONE/sqrt(t5);
        t23 = t1/t5*t7;
        t22 = FPTYPE_ONE/t6*t27;
        dxi[0] = (t10*t3-t13*t22)*t1;
        dxi[1] = (t9*t3-t12*t22)*t1;
        dxi[2] = (t8*t3-t11*t22)*t1;
        dxk[0] = (t13*t1-t10*t23)*t3;
        dxk[1] = (t12*t1-t9*t23)*t3;
        dxk[2] = (t11*t1-t8*t23)*t3;
        ctheta = FPTYPE_FMAX(-FPTYPE_ONE, FPTYPE_FMIN(FPTYPE_ONE, t1*t27));

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fk), std::end(fk), 0.0);
                pot->eval_byparts3(pot, pi, pj, pk, ctheta, &ee, fi, fk);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pk->f[i] += (fkc = fk[i]);
                    pj->f[i] -= fic + fkc;
                }
                epot += ee;
                angle->potential_energy += ee;
                if(angle->potential_energy >= angle->dissociation_energy)
                    toDestroy.insert(angle);
            }
            else {

                if(ctheta > pot->b) 
                    continue;

                #ifdef VECTORIZE
                    /* add this angle to the interaction queue. */
                    cthetaq[icount] = FPTYPE_FMAX(ctheta, pot->a);
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    dkq[icount*3] = dxk[0];
                    dkq[icount*3+1] = dxk[1];
                    dkq[icount*3+2] = dxk[2];
                    effi[icount] = &f[ 4*pid ];
                    effj[icount] = &f[ 4*pjd ];
                    effk[icount] = &f[ 4*pkd ];
                    potq[icount] = pot;
                    angleq[icount] = angle;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r(potq, cthetaq, eeq, eff);
                            #else
                            potential_eval_vec_4single_r(potq, cthetaq, eeq, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r(potq, cthetaq, eeq, eff);
                            #else
                            potential_eval_vec_2double_r(potq, cthetaq, eeq, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            for(k = 0 ; k < 3 ; k++) {
                                effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                                effk[l][k] -= (wk = eff[l] * dkq[3*l+k]);
                                effj[l][k] += wi + wk;
                            }
                            epot += eeq[l];
                            angleq[l] += eeq[l];
                            if(angleq[l]->potential_energy >= angleq[l]->dissociation_energy)
                                toDestroy.insert(angleq[l]);
                        }

                        /* re-set the counter. */
                        icount = 0;

                    }
                #else
                    /* evaluate the angle */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, FPTYPE_FMAX(ctheta, pot->a), &ee, &eff);
                    #else
                        potential_eval_r(pot, FPTYPE_FMAX(ctheta, pot->a), &ee, &eff);
                    #endif
                    
                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        f[4*pid+k] -= (wi = eff * dxi[k]);
                        f[4*pkd+k] -= (wk = eff * dxk[k]);
                        f[4*pjd+k] += wi + wk;
                    }

                    /* tabulate the energy */
                    epot += ee;
                    angle->potential_energy += ee;
                    if(angle->potential_energy >= angle->dissociation_energy)
                        toDestroy.insert(angle);
                #endif

            }

        }
        
    } /* loop over angles. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {

            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
            }

            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r(potq, cthetaq, eeq, eff);
                #else
                potential_eval_vec_4single_r(potq, cthetaq, eeq, eff);
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r(potq, cthetaq, eeq, eff);
                #else
                potential_eval_vec_2double_r(potq, cthetaq, eeq, eff);
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                for(k = 0 ; k < 3 ; k++) {
                    effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                    effk[l][k] -= (wk = eff[l] * dkq[3*l+k]);
                    effj[l][k] += wi + wk;
                    }
                }
                epot += eeq[l];
                angleq[l] += eeq[l];
                if(angleq[l]->potential_energy >= angleq[l]->dissociation_energy)
                    toDestroy.insert(angleq[l]);

            }
    #endif

    // Destroy every bond scheduled for destruction
    for(auto ai : toDestroy)
        Angle_Destroy(ai);
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return S_OK;
    
}

static bool Angle_destroyingAll = false;

HRESULT TissueForge::Angle_Destroy(Angle *a) {
    if(!a) return error(MDCERR_null);

    if(a->flags & ANGLE_ACTIVE) {
        #ifdef HAVE_CUDA
        if(_Engine.angles_cuda && !Angle_destroyingAll) 
            if(cuda::engine_cuda_finalize_angle(a->id) < 0) 
                return error(MDCERR_cuda);
        #endif

        bzero(a, sizeof(Angle));
        _Engine.nr_active_angles -= 1;
    }

    return S_OK;
};

HRESULT TissueForge::Angle_DestroyAll() {
    Angle_destroyingAll = true;

    #ifdef HAVE_CUDA
    if(_Engine.angles_cuda) 
        if(cuda::engine_cuda_finalize_angles_all(&_Engine) < 0) 
            return error(MDCERR_cuda);
    #endif

    for(auto ah: TissueForge::AngleHandle::items()) ah.destroy();

    Angle_destroyingAll = false;
    return S_OK;
}

rendering::Style *TissueForge::Angle::styleDef() {
    return Angle_StylePtr;
}

void TissueForge::Angle::init(Potential *potential, ParticleHandle *p1, ParticleHandle *p2, ParticleHandle *p3, uint32_t flags) {
    this->potential = potential;
    this->i = p1->id;
    this->j = p2->id;
    this->k = p3->id;
    this->flags = flags;

    this->creation_time = _Engine.time;
    this->dissociation_energy = std::numeric_limits<FPTYPE>::max();
    this->half_life = 0.0;

    if(!this->style) this->style = Angle_StylePtr;
}

AngleHandle *TissueForge::Angle::create(Potential *potential, ParticleHandle *p1, ParticleHandle *p2, ParticleHandle *p3, uint32_t flags) {
    if(potential->flags & POTENTIAL_SCALED || potential->flags & POTENTIAL_SHIFTED) {
        error(MDCERR_angsspot);
        return NULL;
    }

    Angle *angle = NULL;

    auto id = engine_angle_alloc(&_Engine, &angle);
    
    if(!angle) {
        error(MDCERR_null);
        return NULL;
    }

    angle->init(potential, p1, p2, p3, flags);
    if(angle->i >=0 && angle->j >=0 && angle->k >=0) {
        angle->flags = angle->flags | ANGLE_ACTIVE;
        _Engine.nr_active_angles++;
    }
    
    AngleHandle *handle = new AngleHandle(id);

    #ifdef HAVE_CUDA
    if(_Engine.angles_cuda) 
        cuda::engine_cuda_add_angle(handle);
    #endif

    TF_Log(LOG_TRACE) << "Created angle: " << angle->id  << ", i: " << angle->i << ", j: " << angle->j << ", k: " << angle->k;

    return handle;
}

std::string TissueForge::Angle::toString() {
    return io::toString(*this);
}

Angle *TissueForge::Angle::fromString(const std::string &str) {
    return new Angle(io::fromString<Angle>(str));
}

Angle *TissueForge::AngleHandle::get() {
    if(id < 0 || id >= _Engine.angles_size) {
        error(MDCERR_id);
        return NULL;
    }
    Angle *r = &_Engine.angles[this->id];
    return r && r->flags & BOND_ACTIVE ? r : NULL;
}

static std::string AngleHandle_str(const AngleHandle *h) {
    std::stringstream ss;

    ss << "AngleHandle(id=" << h->id;
    if(h->id >= 0) {
        const Angle *o = AngleHandle(h->id).get();
        if(o->flags & ANGLE_ACTIVE) 
            ss << ", i=" << o->i << ", j=" << o->j << ", k=" << o->k;
    }
    ss << ")";
    
    return ss.str();
}

std::string TissueForge::AngleHandle::str() const {
    return AngleHandle_str(this);
}

bool TissueForge::AngleHandle::check() {
    return (bool)this->get();
}

HRESULT TissueForge::AngleHandle::destroy() {
    #ifdef HAVE_CUDA
    if(_Engine.angles_cuda && !Angle_destroyingAll) 
        cuda::engine_cuda_finalize_angle(this->id);
    #endif

    Angle *o = this->get();
    return o ? Angle_Destroy(o) : error(MDCERR_null);
}

std::vector<AngleHandle> TissueForge::AngleHandle::items() {
    std::vector<AngleHandle> list;
    list.reserve(_Engine.nr_active_angles);

    for(int i = 0; i < _Engine.nr_angles; ++i)
        if((&_Engine.angles[i])->flags & ANGLE_ACTIVE) 
            list.emplace_back(i);

    return list;
}

bool TissueForge::AngleHandle::decays() {
    return Angle_decays(&_Engine.angles[this->id]);
}

ParticleHandle *TissueForge::AngleHandle::operator[](unsigned int index) {
    auto *a = get();
    if(!a) {
        error(MDCERR_null);
        return NULL;
    }

    if(index == 0) return Particle_FromId(a->i)->handle();
    else if(index == 1) return Particle_FromId(a->j)->handle();
    else if(index == 2) return Particle_FromId(a->k)->handle();
    
    error(MDCERR_range);
    return NULL;
}

bool TissueForge::AngleHandle::has(const int32_t &pid) {
    return getPartList().has(pid);
}

bool TissueForge::AngleHandle::has(ParticleHandle *part) {
    return part ? getPartList().has(part) : false;
}

FloatP_t TissueForge::AngleHandle::getAngle() {
    FloatP_t result = 0;
    Angle *a = this->get();
    if(a) { 
        ParticleHandle pi(a->i), pj(a->j), pk(a->k);
        FVector3 ri = pi.getPosition();
        FVector3 rj = pj.getPosition();
        FVector3 rk = pk.getPosition();
        FVector3 rij = metrics::relativePosition(ri, rj);
        FVector3 rkj = metrics::relativePosition(rk, rj);
        result = rij.angle(rkj);
    }
    return result;
}

FPTYPE TissueForge::AngleHandle::getEnergy() {

    Angle *a = this->get();
    if(!a) {
        error(MDCERR_null);
        return 0;
    }
    Angle angles[] = {*a};
    FPTYPE f[] = {0.0, 0.0, 0.0};
    FPTYPE epot_out = 0.0;
    angle_evalf(angles, 1, &_Engine, f, &epot_out);
    return epot_out;
}

std::vector<int32_t> TissueForge::AngleHandle::getParts() {
    std::vector<int32_t> result;
    Angle *a = this->get();
    if(a) {
        result = std::vector<int32_t>{a->i, a->j, a->k};
    }
    return result;
}

ParticleList TissueForge::AngleHandle::getPartList() {
    ParticleList result;
    Angle *a = this->get();
    if(a) {
        result.insert(a->i);
        result.insert(a->j);
        result.insert(a->k);
    }
    return result;
}

Potential *TissueForge::AngleHandle::getPotential() {
    Angle *a = this->get();
    if(a) {
        return a->potential;
    }
    return NULL;
}

uint32_t TissueForge::AngleHandle::getId() {
    return this->id;
}

FPTYPE TissueForge::AngleHandle::getDissociationEnergy() {
    auto *a = this->get();
    return a ? a->dissociation_energy : FPTYPE_ZERO;
}

void TissueForge::AngleHandle::setDissociationEnergy(const FPTYPE &dissociation_energy) {
    auto *a = this->get();
    if (a) a->dissociation_energy = dissociation_energy;
}

FPTYPE TissueForge::AngleHandle::getHalfLife() {
    auto *a = this->get();
    return a ? a->half_life : FPTYPE_ZERO;
}

void TissueForge::AngleHandle::setHalfLife(const FPTYPE &half_life) {
    auto *a = this->get();
    if (a) a->half_life = half_life;
}

rendering::Style *TissueForge::AngleHandle::getStyle() {
    auto *a = this->get();
    return a ? a->style : NULL;
}

void TissueForge::AngleHandle::setStyle(rendering::Style *style) {
    auto *a = this->get();
    if (a) a->style = style;
}

FPTYPE TissueForge::AngleHandle::getAge() {
    auto *a = this->get();
    return a ? (_Engine.time - a->creation_time) * _Engine.dt : 0;
}

std::vector<int32_t> TissueForge::Angle_IdsForParticle(int32_t pid) {
    std::vector<int32_t> angles;
    for (int i = 0; i < _Engine.nr_angles; ++i) {
        Angle *a = &_Engine.angles[i];
        if((a->flags & ANGLE_ACTIVE) && (a->i == pid || a->j == pid || a->k == pid)) {
            assert(i == a->id);
            angles.push_back(a->id);
        }
    }
    return angles;
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const Angle &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "flags", dataElement.flags);
        TF_IOTOEASY(fileElement, metaData, "i", dataElement.i);
        TF_IOTOEASY(fileElement, metaData, "j", dataElement.j);
        TF_IOTOEASY(fileElement, metaData, "k", dataElement.k);
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);
        TF_IOTOEASY(fileElement, metaData, "creation_time", dataElement.creation_time);
        TF_IOTOEASY(fileElement, metaData, "half_life", dataElement.half_life);
        TF_IOTOEASY(fileElement, metaData, "dissociation_energy", dataElement.dissociation_energy);
        TF_IOTOEASY(fileElement, metaData, "potential_energy", dataElement.potential_energy);

        fileElement.get()->type = "Angle";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Angle *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "flags", &dataElement->flags);
        TF_IOFROMEASY(fileElement, metaData, "i", &dataElement->i);
        TF_IOFROMEASY(fileElement, metaData, "j", &dataElement->j);
        TF_IOFROMEASY(fileElement, metaData, "k", &dataElement->k);
        TF_IOFROMEASY(fileElement, metaData, "id", &dataElement->id);
        TF_IOFROMEASY(fileElement, metaData, "creation_time", &dataElement->creation_time);
        TF_IOFROMEASY(fileElement, metaData, "half_life", &dataElement->half_life);
        TF_IOFROMEASY(fileElement, metaData, "dissociation_energy", &dataElement->dissociation_energy);
        TF_IOFROMEASY(fileElement, metaData, "potential_energy", &dataElement->potential_energy);

        return S_OK;
    }

};
