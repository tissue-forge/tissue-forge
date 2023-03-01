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

/* Include configuration header */
#include <mdcore_config.h>

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
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfPotential.h>
#include <tf_util.h>
#include <tfLogger.h>
#include "tf_potential_eval.h"
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfDihedral.h>
#include <io/tfFIO.h>
#include <rendering/tfStyle.h>

#include <random>


using namespace TissueForge;


rendering::Style *Dihedral_StylePtr = new rendering::Style("gold");


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


static bool Dihedral_decays(Dihedral *d, std::uniform_real_distribution<FPTYPE> *uniform01=NULL) {
    if(!d || d->half_life <= 0.0) return false;

    bool created = uniform01 == NULL;
    if(created) uniform01 = new std::uniform_real_distribution<FPTYPE>(0.0, 1.0);

    FPTYPE pr = 1.0 - std::pow(2.0, -_Engine.dt / d->half_life);
    RandomType &randEng = randomEngine();
    bool result = (*uniform01)(randEng) < pr;

    if(created) delete uniform01;

    return result;
}

HRESULT TissueForge::dihedral_eval(struct Dihedral *d, int N, struct engine *e, FPTYPE *epot_out) {

    Dihedral *dihedral;
    int did, pid, pjd, pkd, pld, k;
    int *loci, *locj, *lock, *locl, shift[3];
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, *pk, *pl, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *pota;
    std::vector<struct Potential *> pots;
    FPTYPE xi[3], xj[3], xk[3], xl[3], dxi[3], dxj[3], dxl[3], cphi;
    FPTYPE wi, wj, wl, fi[3], fl[3], fic, flc;
    FPTYPE t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21,
        t22, t24, t26, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
        t41, t42, t43, t44, t45, t46, t47, t5, t6, t7, t8, t9,
        t2, t4, t23, t25, t27, t28, t51, t52, t53, t54, t59;
    std::unordered_set<struct Dihedral*> toDestroy;
    toDestroy.reserve(N);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE], *effl[VEC_SIZE];
    FPTYPE cphiq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], djq[VEC_SIZE*3], dlq[VEC_SIZE*3];
    struct Dihedral *dihedralq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if(d == NULL || e == NULL)
        return error(MDCERR_null);

    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the dihedrals. */
    for(did = 0 ; did < N ; did++) {

        dihedral = &d[did];
        dihedral->potential_energy = 0.0;

        if(Dihedral_decays(dihedral)) {
            toDestroy.insert(dihedral);
            continue;
        }
    
        /* Get the particles involved. */
        pid = dihedral->i; pjd = dihedral->j; pkd = dihedral->k; pld = dihedral->l;
        if((pi = partlist[ pid ]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        if((pk = partlist[ pkd ]) == NULL)
            continue;
        if((pl = partlist[ pld ]) == NULL)
            continue;
            
        /* Skip if all three are ghosts. */
        if((pi->flags & PARTICLE_GHOST) && 
            (pj->flags & PARTICLE_GHOST) &&
            (pk->flags & PARTICLE_GHOST) &&
            (pl->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pota = d->potential) == NULL)
            continue;

        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
    
        /* Get positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        locl = celllist[ pld ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            xj[k] = pj->x[k];
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xi[k] = pi->x[k] + h[k]*shift[k];
            shift[k] = lock[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xk[k] = pk->x[k] + h[k]*shift[k];
            shift[k] = locl[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xl[k] = pl->x[k] + h[k]*shift[k];
            }
            
        /* This is Maple-generated code, see "dihedral.maple" for details. */
        t16 = xl[2]-xk[2];
        t17 = xl[1]-xk[1];
        t18 = xl[0]-xk[0];
        t2 = t18*t18;
        t4 = t17*t17;
        t23 = t16*t16;
        t10 = t2+t4+t23;
        t19 = xk[2]-xj[2];
        t20 = xk[1]-xj[1];
        t21 = xk[0]-xj[0];
        t25 = t21*t21;
        t27 = t20*t20;
        t28 = t19*t19;
        t11 = t25+t27+t28;
        t7 = t18*t21+t17*t20+t16*t19;
        t51 = t7*t7;
        t5 = t11*t10-t51;
        t22 = xi[2]-xj[2];
        t24 = xi[1]-xj[1];
        t26 = xi[0]-xj[0];
        t52 = t26*t26;
        t53 = t24*t24;
        t54 = t22*t22;
        t12 = t52+t53+t54;
        t9 = -t26*t21-t24*t20-t22*t19;
        t59 = t9*t9;
        t6 = t12*t11-t59;
        t3 = t6*t5;
        t1 = FPTYPE_ONE/FPTYPE_SQRT(t3);
        t8 = -t26*t18-t24*t17-t22*t16;
        t47 = (t9*t7-t8*t11)*t1;
        t46 = FPTYPE_TWO*t8;
        t45 = t6*t7;
        t44 = t9*t5;
        t43 = t6*t10;
        t42 = -t9-t11;
        t41 = t22*t11;
        t40 = t24*t11;
        t39 = t26*t11;
        t38 = FPTYPE_ONE/t3*t47;
        t37 = -t7*t19+t16*t11;
        t36 = -t7*t20+t17*t11;
        t35 = -t7*t21+t18*t11;
        t34 = t9*t19+t41;
        t33 = t9*t20+t40;
        t32 = t9*t21+t39;
        t31 = t5*t38;
        t30 = t6*t38;
        t15 = xk[0]-FPTYPE_TWO*xj[0]+xi[0];
        t14 = xk[1]-FPTYPE_TWO*xj[1]+xi[1];
        t13 = xk[2]-FPTYPE_TWO*xj[2]+xi[2];
        dxi[0] = t35*t1-t32*t31;
        dxi[1] = t36*t1-t33*t31;
        dxi[2] = t37*t1-t34*t31;
        dxj[0] = (t15*t7+t21*t46+t42*t18)*t1-(-t15*t44+t18*t45+(-t39-t12*t21)*t5-t21*t43)*t38;
        dxj[1] = (t14*t7+t20*t46+t42*t17)*t1-(-t14*t44+t17*t45+(-t40-t12*t20)*t5-t20*t43)*t38;
        dxj[2] = (t13*t7+t19*t46+t42*t16)*t1-(-t13*t44+t16*t45+(-t41-t12*t19)*t5-t19*t43)*t38;
        dxl[0] = t32*t1-t35*t30;
        dxl[1] = t33*t1-t36*t30;
        dxl[2] = t34*t1-t37*t30;
        cphi = FPTYPE_FMAX(-FPTYPE_ONE, FPTYPE_FMIN(FPTYPE_ONE, t47));

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fl), std::end(fl), 0.0);
                pot->eval_byparts4(pot, pi, pj, pk, pl, cphi, &ee, fi, fl);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pl->f[i] += (flc = fl[i]);
                    pj->f[i] -= fic;
                    pk->f[i] -= flc;
                }
                epot += ee;
                dihedral->potential_energy += ee;
                if(dihedral->potential_energy >= dihedral->dissociation_energy)
                    toDestroy.insert(dihedral);
            }
            else {

                #ifdef VECTORIZE
                    if(cphi > pot->b) 
                        continue;

                    /* add this dihedral to the interaction queue. */
                    cphiq[icount] = FPTYPE_FMAX(cphi, pot->a);
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    djq[icount*3] = dxj[0];
                    djq[icount*3+1] = dxj[1];
                    djq[icount*3+2] = dxj[2];
                    dlq[icount*3] = dxl[0];
                    dlq[icount*3+1] = dxl[1];
                    dlq[icount*3+2] = dxl[2];
                    effi[icount] = &pi->f[0];
                    effj[icount] = &pj->f[0];
                    effk[icount] = &pk->f[0];
                    effl[icount] = &pl->f[0];
                    potq[icount] = pot;
                    dihedralq[icount] = dihedral;
                    icount += 1;
                
                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r(potq, cphiq, ee, eff);
                            #else
                            potential_eval_vec_4single_r(potq, cphiq, ee, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r(potq, cphiq, ee, eff);
                            #else
                            potential_eval_vec_2double_r(potq, cphiq, ee, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            epot += ee[l];
                            dihedralq[l]->potential_energy += ee[l];
                            for(k = 0 ; k < 3 ; k++) {
                                effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                                effj[l][k] -= (wj = eff[l] * djq[3*l+k]);
                                effl[l][k] -= (wl = eff[l] * dlq[3*l+k]);
                                effk[l][k] += wi + wj + wl;
                                }
                            if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                                toDestroy.insert(dihedralq[l]);
                            }

                        /* re-set the counter. */
                        icount = 0;

                        }
                #else
                    /* evaluate the dihedral */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, cphi, &ee, &eff);
                    #else
                        potential_eval_r(pot, cphi, &ee, &eff);
                    #endif
                    
                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        pi->f[k] -= (wi = eff * dxi[k]);
                        pj->f[k] -= (wj = eff * dxj[k]);
                        pl->f[k] -= (wl = eff * dxl[k]);
                        pk->f[k] += wi + wj + wl;
                        }

                    /* tabulate the energy */
                    epot += ee;
                    dihedral->potential_energy += ee;
                    if(dihedral->potential_energy >= dihedral->dissociation_energy)
                        toDestroy.insert(dihedral);
                #endif

            }

        }
        
        } /* loop over dihedrals. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {
    
            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                cphiq[k] = cphiq[0];
                }
    
            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r(potq, cphiq, ee, eff);
                #else
                potential_eval_vec_4single_r(potq, cphiq, ee, eff);
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r(potq, cphiq, ee, eff);
                #else
                potential_eval_vec_2double_r(potq, cphiq, ee, eff);
                #endif
            #endif
    
            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += ee[l];
                dihedralq[l]->potential_energy += ee[l];
                for(k = 0 ; k < 3 ; k++) {
                    effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                    effj[l][k] -= (wj = eff[l] * djq[3*l+k]);
                    effl[l][k] -= (wl = eff[l] * dlq[3*l+k]);
                    effk[l][k] += wi + wj + wl;
                    }
                if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                    toDestroy.insert(dihedralq[l]);
                }
    
            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return S_OK;
    
    }


HRESULT TissueForge::dihedral_evalf(struct Dihedral *d, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out) {

    Dihedral *dihedral;
    int did, pid, pjd, pkd, pld, k;
    int *loci, *locj, *lock, *locl, shift[3];
    FPTYPE h[3], epot = 0.0;
    struct space *s;
    struct Particle *pi, *pj, *pk, *pl, **partlist;
    struct space_cell **celllist;
    struct Potential *pot, *pota;
    std::vector<struct Potential *> pots;
    FPTYPE xi[3], xj[3], xk[3], xl[3], dxi[3], dxj[3], dxl[3], cphi;
    FPTYPE wi, wj, wl, fi[3], fl[3], fic, flc;
    FPTYPE t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21,
        t22, t24, t26, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
        t41, t42, t43, t44, t45, t46, t47, t5, t6, t7, t8, t9,
        t2, t4, t23, t25, t27, t28, t51, t52, t53, t54, t59;
    std::unordered_set<struct Dihedral*> toDestroy;
    toDestroy.reserve(N);
#if defined(VECTORIZE)
    struct Potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE], *effl[VEC_SIZE];
    FPTYPE cphiq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], djq[VEC_SIZE*3], dlq[VEC_SIZE*3];
    struct Dihedral *dihedralq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if(d == NULL || e == NULL)
        return error(MDCERR_null);

    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for(k = 0 ; k < 3 ; k++)
        h[k] = s->h[k];
        
    /* Loop over the dihedrals. */
    for(did = 0 ; did < N ; did++) {

        dihedral = &d[did];
        dihedral->potential_energy = 0.0;

        if(Dihedral_decays(dihedral)) {
            toDestroy.insert(dihedral);
            continue;
        }
    
        /* Get the particles involved. */
        pid = dihedral->i; pjd = dihedral->j; pkd = dihedral->k; pld = dihedral->l;
        if((pi = partlist[ pid]) == NULL)
            continue;
        if((pj = partlist[ pjd ]) == NULL)
            continue;
        if((pk = partlist[ pkd ]) == NULL)
            continue;
        if((pl = partlist[ pld ]) == NULL)
            continue;
        
        /* Skip if all three are ghosts. */
        if((pi->flags & PARTICLE_GHOST) && 
            (pj->flags & PARTICLE_GHOST) &&
            (pk->flags & PARTICLE_GHOST) &&
            (pl->flags & PARTICLE_GHOST))
            continue;
            
        /* Get the potential. */
        if((pota = d->potential) == NULL)
            continue;

        if(pota->kind == POTENTIAL_KIND_COMBINATION && pota->flags & POTENTIAL_SUM) {
            pots = pota->constituents();
            if(pots.size() == 0) pots = {pota};
        }
        else pots = {pota};
    
        /* Get positions relative to pj. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        locl = celllist[ pld ]->loc;
        for(k = 0 ; k < 3 ; k++) {
            xj[k] = pj->x[k];
            shift[k] = loci[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xi[k] = pi->x[k] + h[k]*shift[k];
            shift[k] = lock[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xk[k] = pk->x[k] + h[k]*shift[k];
            shift[k] = locl[k] - locj[k];
            if(shift[k] > 1)
                shift[k] = -1;
            else if(shift[k] < -1)
                shift[k] = 1;
            xl[k] = pl->x[k] + h[k]*shift[k];
            }
            
        /* This is Maple-generated code, see "dihedral.maple" for details. */
        t16 = xl[2]-xk[2];
        t17 = xl[1]-xk[1];
        t18 = xl[0]-xk[0];
        t2 = t18*t18;
        t4 = t17*t17;
        t23 = t16*t16;
        t10 = t2+t4+t23;
        t19 = xk[2]-xj[2];
        t20 = xk[1]-xj[1];
        t21 = xk[0]-xj[0];
        t25 = t21*t21;
        t27 = t20*t20;
        t28 = t19*t19;
        t11 = t25+t27+t28;
        t7 = t18*t21+t17*t20+t16*t19;
        t51 = t7*t7;
        t5 = t11*t10-t51;
        t22 = xi[2]-xj[2];
        t24 = xi[1]-xj[1];
        t26 = xi[0]-xj[0];
        t52 = t26*t26;
        t53 = t24*t24;
        t54 = t22*t22;
        t12 = t52+t53+t54;
        t9 = -t26*t21-t24*t20-t22*t19;
        t59 = t9*t9;
        t6 = t12*t11-t59;
        t3 = t6*t5;
        t1 = FPTYPE_ONE/FPTYPE_SQRT(t3);
        t8 = -t26*t18-t24*t17-t22*t16;
        t47 = (t9*t7-t8*t11)*t1;
        t46 = FPTYPE_TWO*t8;
        t45 = t6*t7;
        t44 = t9*t5;
        t43 = t6*t10;
        t42 = -t9-t11;
        t41 = t22*t11;
        t40 = t24*t11;
        t39 = t26*t11;
        t38 = FPTYPE_ONE/t3*t47;
        t37 = -t7*t19+t16*t11;
        t36 = -t7*t20+t17*t11;
        t35 = -t7*t21+t18*t11;
        t34 = t9*t19+t41;
        t33 = t9*t20+t40;
        t32 = t9*t21+t39;
        t31 = t5*t38;
        t30 = t6*t38;
        t15 = xk[0]-FPTYPE_TWO*xj[0]+xi[0];
        t14 = xk[1]-FPTYPE_TWO*xj[1]+xi[1];
        t13 = xk[2]-FPTYPE_TWO*xj[2]+xi[2];
        dxi[0] = t35*t1-t32*t31;
        dxi[1] = t36*t1-t33*t31;
        dxi[2] = t37*t1-t34*t31;
        dxj[0] = (t15*t7+t21*t46+t42*t18)*t1-(-t15*t44+t18*t45+(-t39-t12*t21)*t5-t21*t43)*t38;
        dxj[1] = (t14*t7+t20*t46+t42*t17)*t1-(-t14*t44+t17*t45+(-t40-t12*t20)*t5-t20*t43)*t38;
        dxj[2] = (t13*t7+t19*t46+t42*t16)*t1-(-t13*t44+t16*t45+(-t41-t12*t19)*t5-t19*t43)*t38;
        dxl[0] = t32*t1-t35*t30;
        dxl[1] = t33*t1-t36*t30;
        dxl[2] = t34*t1-t37*t30;
        cphi = FPTYPE_FMAX(-FPTYPE_ONE, FPTYPE_FMIN(FPTYPE_ONE, t47));

        for(int i = 0; i < pots.size(); i++) {
            pot = pots[i];

            if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
                std::fill(std::begin(fi), std::end(fi), 0.0);
                std::fill(std::begin(fl), std::end(fl), 0.0);
                pot->eval_byparts4(pot, pi, pj, pk, pl, cphi, &ee, fi, fl);
                for (int i = 0; i < 3; ++i) {
                    pi->f[i] += (fic = fi[i]);
                    pl->f[i] += (flc = fl[i]);
                    pj->f[i] -= fic;
                    pk->f[i] -= flc;
                }
                epot += ee;
                dihedral->potential_energy += ee;
                if(dihedral->potential_energy >= dihedral->dissociation_energy)
                    toDestroy.insert(dihedral);
            }
            else {
            
                #ifdef VECTORIZE
                    if(cphi > pot->b) 
                        continue;

                    /* add this dihedral to the interaction queue. */
                    cphiq[icount] = FPTYPE_FMAX(cphi, pot->a);
                    diq[icount*3] = dxi[0];
                    diq[icount*3+1] = dxi[1];
                    diq[icount*3+2] = dxi[2];
                    djq[icount*3] = dxj[0];
                    djq[icount*3+1] = dxj[1];
                    djq[icount*3+2] = dxj[2];
                    dlq[icount*3] = dxl[0];
                    dlq[icount*3+1] = dxl[1];
                    dlq[icount*3+2] = dxl[2];
                    effi[icount] = &f[ 4*pid ];
                    effj[icount] = &f[ 4*pjd ];
                    effk[icount] = &f[ 4*pkd ];
                    effl[icount] = &f[ 4*pld ];
                    potq[icount] = pot;
                    dihedralq[icount] = dihedral;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if(icount == VEC_SIZE) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single_r(potq, cphiq, ee, eff);
                            #else
                            potential_eval_vec_4single_r(potq, cphiq, ee, eff);
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double_r(potq, cphiq, ee, eff);
                            #else
                            potential_eval_vec_2double_r(potq, cphiq, ee, eff);
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for(l = 0 ; l < VEC_SIZE ; l++) {
                            epot += ee[l];
                            dihedralq[l]->potential_energy += ee[l];
                            for(k = 0 ; k < 3 ; k++) {
                                effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                                effj[l][k] -= (wj = eff[l] * djq[3*l+k]);
                                effl[l][k] -= (wl = eff[l] * dlq[3*l+k]);
                                effk[l][k] += wi + wj + wl;
                                }
                            if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                                toDestroy.insert(dihedralq[l]);
                            }

                        /* re-set the counter. */
                        icount = 0;

                        }
                #else
                    /* evaluate the dihedral */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl(pot, cphi, &ee, &eff);
                    #else
                        potential_eval_r(pot, cphi, &ee, &eff);
                    #endif
                    
                    /* update the forces */
                    for(k = 0 ; k < 3 ; k++) {
                        f[4*pid+k] -= (wi = eff * dxi[k]);
                        f[4*pjd+k] -= (wj = eff * dxj[k]);
                        f[4*pld+k] -= (wl = eff * dxl[k]);
                        f[4*pkd+k] += wi + wj + wl;
                        }

                    /* tabulate the energy */
                    epot += ee;
                    dihedral->potential_energy += ee;
                    if(dihedral->potential_energy >= dihedral->dissociation_energy)
                        toDestroy.insert(dihedral);
                #endif

            }

        }
        
        } /* loop over dihedrals. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if(icount > 0) {
    
            /* copy the first potential to the last entries */
            for(k = icount ; k < VEC_SIZE ; k++) {
                potq[k] = potq[0];
                cphiq[k] = cphiq[0];
                }
    
            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r(potq, cphiq, ee, eff);
                #else
                potential_eval_vec_4single_r(potq, cphiq, ee, eff);
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r(potq, cphiq, ee, eff);
                #else
                potential_eval_vec_2double_r(potq, cphiq, ee, eff);
                #endif
            #endif
    
            /* for each entry, update the forces and energy */
            for(l = 0 ; l < icount ; l++) {
                epot += ee[l];
                dihedralq[l]->potential_energy += ee[l];
                for(k = 0 ; k < 3 ; k++) {
                    effi[l][k] -= (wi = eff[l] * diq[3*l+k]);
                    effj[l][k] -= (wj = eff[l] * djq[3*l+k]);
                    effl[l][k] -= (wl = eff[l] * dlq[3*l+k]);
                    effk[l][k] += wi + wj + wl;
                    }
                if(dihedralq[l]->potential_energy >= dihedralq[l]->dissociation_energy)
                    toDestroy.insert(dihedralq[l]);
                }
    
            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return S_OK;
    
}

rendering::Style *TissueForge::Dihedral::styleDef() {
    return Dihedral_StylePtr;
}

void TissueForge::Dihedral::init(Potential *potential, 
                                 ParticleHandle *p1, 
                                 ParticleHandle *p2, 
                                 ParticleHandle *p3, 
                                 ParticleHandle *p4) 
{
    this->potential = potential;
    this->i = p1->id;
    this->j = p2->id;
    this->k = p3->id;
    this->l = p4->id;

    this->creation_time = _Engine.time;
    this->dissociation_energy = std::numeric_limits<FPTYPE>::max();
    this->half_life = 0.0;

    if(!this->style) this->style = Dihedral_StylePtr;
}

DihedralHandle *TissueForge::Dihedral::create(Potential *potential, 
                                              ParticleHandle *p1, 
                                              ParticleHandle *p2, 
                                              ParticleHandle *p3, 
                                              ParticleHandle *p4) 
{
    if(potential->flags & POTENTIAL_SCALED || potential->flags & POTENTIAL_SHIFTED) {
        error(MDCERR_dihsspot);
        return NULL;
    }

    Dihedral *dihedral = NULL;

    int id = engine_dihedral_alloc(&_Engine, &dihedral);

    if(id < 0) {
        error(MDCERR_malloc);
        return NULL;
    }

    dihedral->init(potential, p1, p2, p3, p4);
    if(dihedral->i >= 0 && dihedral->j >= 0 && dihedral->k >= 0 && dihedral->l >= 0) {
        dihedral->flags |= DIHEDRAL_ACTIVE;
        _Engine.nr_active_dihedrals++;
    }

    DihedralHandle *handle = new DihedralHandle(id);

    return handle;
}

std::string TissueForge::Dihedral::toString() {
    return io::toString(*this);
}

Dihedral *TissueForge::Dihedral::fromString(const std::string &str) {
    return new Dihedral(io::fromString<Dihedral>(str));
}

Dihedral *TissueForge::DihedralHandle::get() {
    if(id < 0 || id >= _Engine.dihedrals_size) {
        error(MDCERR_id);
        return NULL;
    }
    Dihedral *r = &_Engine.dihedrals[this->id];
    return r && r->flags & DIHEDRAL_ACTIVE ? r : NULL;
}

static std::string DihedralHandle_str(const DihedralHandle *h) {
    std::stringstream ss;

    ss << "DihedralHandle(id=" << h->id;
    if(h->id >= 0) {
        const Dihedral *o = DihedralHandle(h->id).get();
        if(o) 
            ss << ", i=" << o->i << ", j=" << o->j << ", k=" << o->k << ", l=" << o->l;
    }
    ss << ")";
    
    return ss.str();
}

std::string TissueForge::DihedralHandle::str() const {
    return DihedralHandle_str(this);
}

bool TissueForge::DihedralHandle::check() {
    return (bool)this->get();
}

HRESULT TissueForge::DihedralHandle::destroy() {
    Dihedral *o = this->get();
    return o ? Dihedral_Destroy(o) : error(MDCERR_null);
}

std::vector<DihedralHandle> TissueForge::DihedralHandle::items() {
    std::vector<DihedralHandle> list;
    list.reserve(_Engine.nr_active_dihedrals);

    for(int i = 0; i < _Engine.nr_dihedrals; ++i)
        if((&_Engine.dihedrals[i])->flags & DIHEDRAL_ACTIVE) 
            list.emplace_back(i);

    return list;
}

bool TissueForge::DihedralHandle::decays() {
    Dihedral *o = this->get();
    return o ? Dihedral_decays(o) : error(MDCERR_null);
}

ParticleHandle *TissueForge::DihedralHandle::operator[](unsigned int index) {
    auto *d = this->get();
    if(!d) {
        error(MDCERR_null);
        return NULL;
    }

    if(index == 0) return Particle_FromId(d->i)->handle();
    else if(index == 1) return Particle_FromId(d->j)->handle();
    else if(index == 2) return Particle_FromId(d->k)->handle();
    else if(index == 3) return Particle_FromId(d->l)->handle();
    
    error(MDCERR_range);
    return NULL;
}

bool TissueForge::DihedralHandle::has(const int32_t &pid) {
    return getPartList().has(pid);
}

bool TissueForge::DihedralHandle::has(ParticleHandle *part) {
    return part ? getPartList().has(part) : false;
}

FloatP_t TissueForge::DihedralHandle::getAngle() {
    FloatP_t result = 0;
    Dihedral *d = this->get();
    if(d) { 
        ParticleHandle pi(d->i), pj(d->j), pk(d->k), pl(d->l);
        FVector3 ri = pi.getPosition();
        FVector3 rj = pj.getPosition();
        FVector3 rk = pk.getPosition();
        FVector3 rl = pl.getPosition();
        FVector3 nlijk = FVector4::planeEquation(ri, rj, rk).xyz();
        FVector3 nljkl = FVector4::planeEquation(rj, rk, rl).xyz();
        result = nlijk.angle(nljkl);
    }
    return result;
}

FPTYPE TissueForge::DihedralHandle::getEnergy() {

    Dihedral *d = this->get();
    if(!d) {
        error(MDCERR_null);
        return FPTYPE_ZERO;
    }

    Dihedral dihedrals[] = {*d};
    FPTYPE f[] = {0.0, 0.0, 0.0};
    FPTYPE epot_out = 0.0;
    dihedral_evalf(dihedrals, 1, &_Engine, f, &epot_out);
    return epot_out;
}

std::vector<int32_t> TissueForge::DihedralHandle::getParts() {
    std::vector<int32_t> result;
    Dihedral *d = this->get();
    if(d) 
        result = std::vector<int32_t>{d->i, d->j, d->k, d->l};
    return result;
}

ParticleList TissueForge::DihedralHandle::getPartList() {
    ParticleList result;
    Dihedral *d = this->get();
    if(d) {
        result.insert(d->i);
        result.insert(d->j);
        result.insert(d->k);
        result.insert(d->l);
    }
    return result;
}

Potential *TissueForge::DihedralHandle::getPotential() {
    Dihedral *d = this->get();
    return d ? d->potential : NULL;
}

uint32_t TissueForge::DihedralHandle::getId() {
    return this->id;
}

FPTYPE TissueForge::DihedralHandle::getDissociationEnergy() {
    Dihedral *d = this->get();
    return d ? d->dissociation_energy : FPTYPE_ZERO;
}

void TissueForge::DihedralHandle::setDissociationEnergy(const FPTYPE &dissociation_energy) {
    auto *d = this->get();
    if(d) d->dissociation_energy = dissociation_energy;
}

FPTYPE TissueForge::DihedralHandle::getHalfLife() {
    Dihedral *d = this->get();
    return d ? d->half_life : FPTYPE_ZERO;
}

void TissueForge::DihedralHandle::setHalfLife(const FPTYPE &half_life) {
    auto *d = this->get();
    if(d) d->half_life = half_life;
}

rendering::Style *TissueForge::DihedralHandle::getStyle() {
    Dihedral *d = this->get();
    return d ? d->style : NULL;
}

void TissueForge::DihedralHandle::setStyle(rendering::Style *style) {
    auto *d = this->get();
    if(d) d->style = style;
}

FPTYPE TissueForge::DihedralHandle::getAge() {
    Dihedral *d = this->get();
    return d ? (_Engine.time - this->get()->creation_time) * _Engine.dt : FPTYPE_ZERO;
}

HRESULT TissueForge::Dihedral_Destroy(Dihedral *d) {
    if(!d) return error(MDCERR_null);

    if(d->flags & DIHEDRAL_ACTIVE) {
        bzero(d, sizeof(Dihedral));
        _Engine.nr_active_dihedrals -= 1;
    }

    return S_OK;
}

HRESULT TissueForge::Dihedral_DestroyAll() {
    for (auto dh: TissueForge::DihedralHandle::items()) dh.destroy();
    return S_OK;
}

std::vector<int32_t> TissueForge::Dihedral_IdsForParticle(int32_t pid) {
    std::vector<int32_t> dihedrals;
    for (int i = 0; i < _Engine.nr_dihedrals; ++i) {
        Dihedral *d = &_Engine.dihedrals[i];
        if((d->flags & DIHEDRAL_ACTIVE) && (d->i == pid || d->j == pid || d->k == pid || d->l == pid)) {
            assert(i == d->id);
            dihedrals.push_back(d->id);
        }
    }
    return dihedrals;
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const Dihedral &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "flags", dataElement.flags);
        TF_IOTOEASY(fileElement, metaData, "i", dataElement.i);
        TF_IOTOEASY(fileElement, metaData, "j", dataElement.j);
        TF_IOTOEASY(fileElement, metaData, "k", dataElement.k);
        TF_IOTOEASY(fileElement, metaData, "l", dataElement.l);
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);
        TF_IOTOEASY(fileElement, metaData, "creation_time", dataElement.creation_time);
        TF_IOTOEASY(fileElement, metaData, "half_life", dataElement.half_life);
        TF_IOTOEASY(fileElement, metaData, "dissociation_energy", dataElement.dissociation_energy);
        TF_IOTOEASY(fileElement, metaData, "potential_energy", dataElement.potential_energy);

        fileElement.get()->type = "Dihedral";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Dihedral *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "flags", &dataElement->flags);
        TF_IOFROMEASY(fileElement, metaData, "i", &dataElement->i);
        TF_IOFROMEASY(fileElement, metaData, "j", &dataElement->j);
        TF_IOFROMEASY(fileElement, metaData, "k", &dataElement->k);
        TF_IOFROMEASY(fileElement, metaData, "l", &dataElement->l);
        TF_IOFROMEASY(fileElement, metaData, "id", &dataElement->id);
        TF_IOFROMEASY(fileElement, metaData, "creation_time", &dataElement->creation_time);
        TF_IOFROMEASY(fileElement, metaData, "half_life", &dataElement->half_life);
        TF_IOFROMEASY(fileElement, metaData, "dissociation_energy", &dataElement->dissociation_energy);
        TF_IOFROMEASY(fileElement, metaData, "potential_energy", &dataElement->potential_energy);

        return S_OK;
    }

};
