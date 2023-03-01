/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

#ifndef _MDCORE_SOURCE_TF_POTENTIAL_EVAL_H_
#define _MDCORE_SOURCE_TF_POTENTIAL_EVAL_H_

#include <tfPotential.h>
#include <tfEngine.h>
#include "tf_dpd_eval.h"
#include <tfDPDPotential.h>

/* This file contains the potential evaluation function als "extern inline",
   such that they can be inlined in the respective modules.

   If your code wants to call any potential_eval functions, you must include
   this file.
*/

#include <iostream>


namespace TissueForge { 


    TF_ALWAYS_INLINE FPTYPE potential_eval_adjust_distance(struct Potential *p, FPTYPE ri, FPTYPE rj, FPTYPE r) {
        if(p->flags & POTENTIAL_SCALED) {
            r = r / (ri + rj);
        }
        else if(p->flags & POTENTIAL_SHIFTED) {
            r = r - (ri + rj) + p->r0_plusone;
        }
        return r;
    }

    TF_ALWAYS_INLINE FPTYPE potential_eval_adjust_distance2(struct Potential *p, FPTYPE ri, FPTYPE rj, FPTYPE r2) {
        FPTYPE r = potential_eval_adjust_distance(p, ri, rj, FPTYPE_SQRT(r2));
        return r * r;
    }


    /**
     * @brief Evaluates the given potential at the given point (interpolated).
     *
     * @param p The #potential to be evaluated.
     * @param r2 The radius at which it is to be evaluated, squared.
     * @param e Pointer to a floating-point value in which to store the
     *      interaction energy.
     * @param f Pointer to a floating-point value in which to store the
     *      magnitude of the interaction force divided by r.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     */

    TF_ALWAYS_INLINE void potential_eval(struct Potential *p, FPTYPE r2, FPTYPE *e, FPTYPE *f) {

        int ind, k;
        FPTYPE x, ee, eff, *c, r;

        /* Get r for the right type. */
        r = FPTYPE_SQRT(r2);

        /* compute the index */
        ind = FPTYPE_FMAX(FPTYPE_ZERO, p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]));

        /* get the table offset */
        c = &(p->c[ind * potential_chunk]);

        /* adjust x to the interval */
        x = (r - c[0]) * c[1];

        /* compute the potential and its derivative */
        ee = c[2] * x + c[3];
        eff = c[2];
        for(k = 4 ; k < potential_chunk ; k++) {
            eff = eff * x + ee;
            ee = ee * x + c[k];
        }

        /* store the result */
        *e = ee; *f = eff * c[1] / r;

    }

    TF_ALWAYS_INLINE bool potential_eval_ex(
        struct Potential *p, FPTYPE ri, FPTYPE rj, FPTYPE r2, FPTYPE *e, FPTYPE *f) {

        unsigned ind, k;
        FPTYPE x, ee, eff, *c, r, ro;
        
        static const FPTYPE epsilon = std::numeric_limits<FPTYPE>::epsilon();

        /* Get r for the right type. */
        r = FPTYPE_SQRT(r2);
        ro = r < epsilon ? epsilon : r;
        
        r = potential_eval_adjust_distance(p, ri, rj, r);
        
        // cutoff min value, eval at lowest func interpolation.
        r = r < p->a ? p->a : r;

        /* compute the index */
        ind = std::max(FPTYPE_ZERO, p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]));

        if(r > p->b || ind > p->n) {
            return false;
        }

        /* get the table offset */
        c = &(p->c[ind * potential_chunk]);

        /* adjust x to the interval */
        x = (r - c[0]) * c[1];

        /* compute the potential and its derivative */
        ee = c[2] * x + c[3];
        eff = c[2];
        for(k = 4 ; k < potential_chunk ; k++) {
            eff = eff * x + ee;
            ee = ee * x + c[k];
        }

        /* store the result */
        *e = ee; *f = eff * c[1] / ro;

        return true;
    }



    /**
     * @brief Evaluates the given potential at the given point (interpolated).
     *
     * @param p The #potential to be evaluated.
     * @param r The radius at which it is to be evaluated.
     * @param e Pointer to a floating-point value in which to store the
     *      interaction energy.
     * @param f Pointer to a floating-point value in which to store the
     *      magnitude of the interaction force.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     */

    TF_ALWAYS_INLINE void potential_eval_r (struct Potential *p, FPTYPE r, FPTYPE *e, FPTYPE *f) {

        int ind, k;
        FPTYPE x, ee, eff, *c;

        /* compute the index */
        ind = FPTYPE_FMAX(FPTYPE_ZERO, p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]));

        if(ind > p->n) {
            return;
        }

        /* get the table offset */
        c = &(p->c[ind * potential_chunk]);

        /* adjust x to the interval */
        x = (r - c[0]) * c[1];

        /* compute the potential and its derivative */
        ee = c[2] * x + c[3];
        eff = c[2];
        for(k = 4 ; k < potential_chunk ; k++) {
            eff = eff * x + ee;
            ee = ee * x + c[k];
            }

        /* store the result */
        *e = ee; *f = eff * c[1];

    }

    TF_ALWAYS_INLINE bool potential_eval_super_ex(const space_cell *cell,
                                Potential *pot, Particle *part_i, Particle *part_j,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot);

    static bool _potential_eval_super_ex(const space_cell *cell,
                                Potential *pot, Particle *part_i, Particle *part_j,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot) 
    {
        return potential_eval_super_ex(cell, pot, part_i, part_j, dx, r2, epot);
    }

    bool potential_eval_super_ex(const space_cell *cell,
                                Potential *pot, Particle *part_i, Particle *part_j,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot) {
        
        if(pot->kind == POTENTIAL_KIND_COMBINATION) {
            if(pot->flags & POTENTIAL_SUM) {
                return _potential_eval_super_ex(cell, pot->pca, part_i, part_j, dx, r2, epot) || _potential_eval_super_ex(cell, pot->pcb, part_i, part_j, dx, r2, epot);
            }
        }
        
        FPTYPE e;
        bool result = false;
        FPTYPE _dx[3], _r2;
        
        if(pot->flags & POTENTIAL_PERIODIC) {
            // Assuming elsewhere there's a corresponding potential in the opposite direction
            _r2 = fptype_r2(dx, pot->offset, _dx);
        }
        else {
            _r2 = r2;
            for (int k = 0; k < 3; k++) _dx[k] = dx[k];
        }
        
        if(pot->kind == POTENTIAL_KIND_DPD) {
            /* update the forces if part in range */
            if (dpd_eval((DPDPotential*)pot, space_cell_gaussian(cell->id), part_i, part_j, _dx, _r2, &e)) {
                
                /* tabulate the energy */
                *epot += e;
                result = true;
            }
        }
        else if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
            FPTYPE fv[3] = {0., 0., 0.};

            pot->eval_byparts(pot, part_i, part_j, _dx, _r2, &e, fv);

            for (int k = 0 ; k < 3 ; k++) {
                part_i->f[k] += fv[k];
                part_j->f[k] -= fv[k];
            }
            
            /* tabulate the energy */
            *epot += e;
            result = true;
        }
        else if(pot->kind != POTENTIAL_KIND_COMBINATION) {
            FPTYPE f;
        
            /* update the forces if part in range */
            if (potential_eval_ex(pot, part_i->radius, part_j->radius, _r2, &e, &f)) {
                
                for (int k = 0 ; k < 3 ; k++) {
                    FPTYPE w = f * _dx[k];
                    part_i->f[k] -= w;
                    part_j->f[k] += w;
                }
                
                /* tabulate the energy */
                *epot += e;
                result = true;
            }
        }

        return result;
    }


    /**
     * @brief Evaluates the given potential at the given radius explicitly.
     *
     * @param p The #potential to be evaluated.
     * @param r2 The radius squared.
     * @param e A pointer to a floating point value in which to store the
     *      interaction energy.
     * @param f A pointer to a floating point value in which to store the
     *      magnitude of the interaction force
     *
     * Assumes that the parameters for the potential forms given in the value
     * @c flags of the #potential @c p are stored in the array @c alpha of
     * @c p.
     *
     * This way of evaluating a potential is not extremely efficient and is
     * intended for comparison and debugging purposes.
     *
     * Note that for performance reasons, this function does not check its input
     * arguments for @c NULL.
     */

    TF_ALWAYS_INLINE void potential_eval_expl(struct Potential *p, FPTYPE r2, FPTYPE *e, FPTYPE *f) {

        const FPTYPE isqrtpi = 0.56418958354775628695;
        const FPTYPE kappa = 3.0;
        FPTYPE r = sqrt(r2), ir = 1.0 / r, ir2 = ir * ir, ir4, ir6, ir12, t1, t2;
        FPTYPE ee = 0.0, eff = 0.0;

        /* Do we have a Lennard-Jones interaction? */
        if(p->flags & POTENTIAL_LJ126) {

            /* init some variables */
            ir4 = ir2 * ir2; ir6 = ir4 * ir2; ir12 = ir6 * ir6;

            /* compute the energy and the force */
            ee = (p->alpha[0] * ir12 - p->alpha[1] * ir6);
            eff = 6.0 * ir * (-2.0 * p->alpha[0] * ir12 + p->alpha[1] * ir6);

            }

        /* Do we have an Ewald short-range part? */
        if(p->flags & POTENTIAL_EWALD) {

            /* get some values we will re-use */
            t2 = r * kappa;
            t1 = erfc(t2);

            /* compute the energy and the force */
            ee += p->alpha[2] * t1 * ir;
            eff += p->alpha[2] * (-2.0 * isqrtpi * exp(-t2 * t2) * kappa * ir - t1 * ir2);

            }

        /* Do we have a Coulomb interaction? */
        if(p->flags & POTENTIAL_COULOMB) {

            /* compute the energy and the force */
            ee += potential_escale * p->alpha[2] * ir;
            eff += -potential_escale * p->alpha[2] * ir2;

            }

        /* store the potential and force. */
        *e = ee;
        *f = eff;

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r2 Pointer to an array of the radii at which the potentials
     *      are to be evaluated, squared.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes four single-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE or AltiVec
     * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
     * this function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_4single(struct Potential *p[4], float *r2, float *e, float *f) {

    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        // int j, k;
        union {
            __v4sf v;
            __m128i m;
            float f[4];
            int i[4];
            } alpha[4], mi, hi, x, ee, eff, c[6], r, ind, t[8];
        // float *data[4];

        /* Get r . */
        r.v = _mm_sqrt_ps(_mm_load_ps(r2));

        /* compute the index */
        alpha[0].v = _mm_load_ps(p[0]->alpha);
        alpha[1].v = _mm_load_ps(p[1]->alpha);
        alpha[2].v = _mm_load_ps(p[2]->alpha);
        alpha[3].v = _mm_load_ps(p[3]->alpha);
        t[0].m = _mm_unpacklo_epi32(alpha[0].m, alpha[1].m);
        t[1].m = _mm_unpacklo_epi32(alpha[2].m, alpha[3].m);
        t[2].m = _mm_unpackhi_epi32(alpha[0].m, alpha[1].m);
        t[3].m = _mm_unpackhi_epi32(alpha[2].m, alpha[3].m);
        alpha[0].m = _mm_unpacklo_epi64(t[0].m, t[1].m);
        alpha[1].m = _mm_unpackhi_epi64(t[0].m, t[1].m);
        alpha[2].m = _mm_unpacklo_epi64(t[2].m, t[3].m);
        ind.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha[0].v, _mm_mul_ps(r.v, _mm_add_ps(alpha[1].v, _mm_mul_ps(r.v, alpha[2].v))))));

        /* Unpack/transpose the coefficient data. */
        mi.v = _mm_load_ps(&p[0]->c[ ind.i[0] * potential_chunk ]);
        hi.v = _mm_load_ps(&p[1]->c[ ind.i[1] * potential_chunk ]);
        c[0].v = _mm_load_ps(&p[2]->c[ ind.i[2] * potential_chunk ]);
        c[1].v = _mm_load_ps(&p[3]->c[ ind.i[3] * potential_chunk ]);
        _MM_TRANSPOSE4_PS(mi.v, hi.v, c[0].v, c[1].v);
        c[2].v = _mm_load_ps(&p[0]->c[ ind.i[0] * potential_chunk + 4 ]);
        c[3].v = _mm_load_ps(&p[1]->c[ ind.i[1] * potential_chunk + 4 ]);
        c[4].v = _mm_load_ps(&p[2]->c[ ind.i[2] * potential_chunk + 4 ]);
        c[5].v = _mm_load_ps(&p[3]->c[ ind.i[3] * potential_chunk + 4 ]);
        _MM_TRANSPOSE4_PS(c[2].v, c[3].v, c[4].v, c[5].v);

        /* adjust x to the interval */
        x.v = _mm_mul_ps(_mm_sub_ps(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = c[0].v;
        ee.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), c[1].v);
        eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
        ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c[2].v);
        eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
        ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c[3].v);
        eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
        ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c[4].v);
        eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
        ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c[5].v);

        /* store the result */
        _mm_store_ps(e, ee.v);
        _mm_store_ps(f, _mm_mul_ps(eff.v, _mm_div_ps(hi.v, r.v)));

    #elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
        int j, k;
        union {
            vector float v;
            float f[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r;
        union {
            vector unsigned int v;
            unsigned int i[4];
            } ind;
        float *data[4];

        /* Get r . */
        r.v = vec_sqrt(*((vector float *)r2));

        /* compute the index (vec_ctu maps negative floats to 0) */
        alpha0.v = vec_load4(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = vec_load4(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = vec_load4(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        ind.v = vec_ctu(vec_madd(r.v, vec_madd(r.v, alpha2.v, alpha1.v), alpha0.v), 0);

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = vec_load4(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = vec_load4(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = vec_mul(vec_sub(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = vec_load4(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = vec_load4(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = vec_madd(eff.v, x.v, c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = vec_load4(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = vec_madd(eff.v, x.v, ee.v);
            ee.v = vec_madd(ee.v, x.v, c.v);
            }

        /* store the result */
        *((vector float *)e) = ee.v;
        *((vector float *)f) = vec_mul(eff.v, vec_div(hi.v, r.v));

    #else
        int k;
        FPTYPE ee, eff;
        for(k = 0 ; k < 4 ; k++) {
            potential_eval_r(p[k], r2[k], &ee, &eff);
            e[k] = ee; f[k] = eff;
            }
    #endif

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r2 Pointer to an array of the radii at which the potentials
     *      are to be evaluated, squared.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes four single-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE or AltiVec
     * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
     * this function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_4single_old(struct Potential *p[4], float *r2, float *e, float *f) {

    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        int j, k;
        union {
            __v4sf v;
            __m128i m;
            float f[4];
            int i[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
        float *data[4];

        /* Get r . */
        r.v = _mm_sqrt_ps(_mm_load_ps(r2));

        /* compute the index */
        alpha0.v = _mm_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = _mm_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = _mm_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        ind.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0.v, _mm_mul_ps(r.v, _mm_add_ps(alpha1.v, _mm_mul_ps(r.v, alpha2.v))))));

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = _mm_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = _mm_mul_ps(_mm_sub_ps(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = _mm_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
            ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm_store_ps(e, ee.v);
        _mm_store_ps(f, _mm_mul_ps(eff.v, _mm_div_ps(hi.v, r.v)));

    #elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
        int j, k;
        union {
            vector float v;
            float f[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r;
        union {
            vector unsigned int v;
            unsigned int i[4];
            } ind;
        float *data[4];

        /* Get r . */
        r.v = vec_sqrt(*((vector float *)r2));

        /* compute the index (vec_ctu maps negative floats to 0) */
        alpha0.v = vec_load4(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = vec_load4(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = vec_load4(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        ind.v = vec_ctu(vec_madd(r.v, vec_madd(r.v, alpha2.v, alpha1.v), alpha0.v), 0);

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = vec_load4(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = vec_load4(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = vec_mul(vec_sub(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = vec_load4(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = vec_load4(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = vec_madd(eff.v, x.v, c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = vec_load4(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = vec_madd(eff.v, x.v, ee.v);
            ee.v = vec_madd(ee.v, x.v, c.v);
            }

        /* store the result */
        *((vector float *)e) = ee.v;
        *((vector float *)f) = vec_mul(eff.v, vec_div(hi.v, r.v));

    #else
        int k;
        FPTYPE ee, eff;
        for(k = 0 ; k < 4 ; k++) {
            potential_eval_r(p[k], r2[k], &ee, &eff);
            e[k] = ee; f[k] = eff;
            }
    #endif

        }


    #ifdef STILL_NOT_READY_FOR_PRIME_TIME
    TF_ALWAYS_INLINE void potential_eval_vec_4single_gccvec(struct Potential *p[4], float *r2, float *e, float *f) {

        int j, k;
        union {
            vector(4,float) v;
            float f[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r;
        union {
            vector(4,int) v;
            int i[4];
            } ind;
        FPTYPE *data[4];

        /* Get r . */
        r.v = sqrtf(*((vector(4,float) *)r2));

        /* compute the index */
        for(k = 0 ; k < 4 ; k++) {
            alpha0.f[k] = p[k]->alpha[0];
            alpha1.f[k] = p[k]->alpha[1];
            alpha2.f[k] = p[k]->alpha[2];
            }
        ind.v = max((vector(4,int)){0,0,0,0}, (vector(4,int))(alpha0.v + r.v*(alpha1.v + r.v*alpha2.v)));

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        for(k = 0 ; k < 4 ; k++) {
            mi.f[k] = data[k][0];
            hi.f[k] = data[k][1];
            }
        x.v = (r.v - mi.v) * hi.v;

        /* compute the potential and its derivative */
        for(k = 0 ; k < 4 ; k++) {
            eff.f[k] = data[k][2];
            c.f[k] = data[k][3];
            }
        ee.v = eff.v*x.v + c.v;
        for(j = 4 ; j < potential_chunk ; j++) {
                for(k = 0 ; k < 4 ; k++)
                c.f[k] = data[k][j];
            eff.v = eff.v*x.v + ee.v;
            ee.v = ee.v*x.v + c.v;
            }

        /* store the result */
        *((vector(4,float) *)e) = ee.v;
        *((vector(4,float) *)f) = eff.v*(hi.v / r.v);

        }
    #endif


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r_in Pointer to an array of the radii at which the potentials
     *      are to be evaluated.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes four single-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE or AltiVec
     * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
     * this function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_4single_r(struct Potential *p[4], float *r_in, float *e, float *f) {

    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        int j, k;
        union {
            __v4sf v;
            __m128i m;
            float f[4];
            int i[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
        float *data[4];

        /* Get r . */
        r.v = _mm_load_ps(r_in);

        /* compute the index */
        alpha0.v = _mm_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = _mm_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = _mm_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        ind.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0.v, _mm_mul_ps(r.v, _mm_add_ps(alpha1.v, _mm_mul_ps(r.v, alpha2.v))))));

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = _mm_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = _mm_mul_ps(_mm_sub_ps(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = _mm_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = _mm_add_ps(_mm_mul_ps(eff.v, x.v), ee.v);
            ee.v = _mm_add_ps(_mm_mul_ps(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm_store_ps(e, ee.v);
        _mm_store_ps(f, _mm_mul_ps(eff.v, hi.v));

    #elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
        int j, k;
        union {
            vector float v;
            float f[4];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r;
        union {
            vector unsigned int v;
            unsigned int i[4];
            } ind;
        float *data[4];

        /* Get r . */
        r.v = *((vector float *)r_in);

        /* compute the index (vec_ctu maps negative floats to 0) */
        alpha0.v = vec_load4(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = vec_load4(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = vec_load4(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        ind.v = vec_ctu(vec_madd(r.v, vec_madd(r.v, alpha2.v, alpha1.v), alpha0.v), 0);

        /* get the table offset */
        for(k = 0 ; k < 4 ; k++)
            data[k] = &(p[k]->c[ ind.i[k] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = vec_load4(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = vec_load4(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = vec_mul(vec_sub(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = vec_load4(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = vec_load4(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = vec_madd(eff.v, x.v, c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = vec_load4(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = vec_madd(eff.v, x.v, ee.v);
            ee.v = vec_madd(ee.v, x.v, c.v);
            }

        /* store the result */
        *((vector float *)e) = ee.v;
        *((vector float *)f) = vec_mul(eff.v, hi.v);

    #else
        int k;
        FPTYPE ee, eff;
        for(k = 0 ; k < 4 ; k++) {
            potential_eval(p[k], r_in[k], &ee, &eff);
            e[k] = ee; f[k] = eff;
            }
    #endif

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r2 Pointer to an array of the radii at which the potentials
     *      are to be evaluated, squared.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes eight single-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE or AltiVec
     * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
     * this function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_8single(struct Potential *p[8], float *r2, float *e, float *f) {

    #if defined(__AVX__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            __v8sf v;
            __m256i m;
            float f[8];
            int i[8];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
        float *data[8];

        /* Get r . */
        r.v = _mm256_sqrt_ps(_mm256_load_ps(r2));

        /* compute the index */
        alpha0.v = _mm256_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0], p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1.v = _mm256_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1], p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2.v = _mm256_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2], p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind.m = _mm256_cvttps_epi32(_mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(alpha0.v, _mm256_mul_ps(r.v, _mm256_add_ps(alpha1.v, _mm256_mul_ps(r.v, alpha2.v))))));

        /* get the table offset */
        data[0] = &(p[0]->c[ ind.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind.i[4] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind.i[5] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind.i[6] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind.i[7] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm256_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0], data[4][0], data[5][0], data[6][0], data[7][0]);
        hi.v = _mm256_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1], data[4][1], data[5][1], data[6][1], data[7][1]);
        x.v = _mm256_mul_ps(_mm256_sub_ps(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm256_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2], data[4][2], data[5][2], data[6][2], data[7][2]);
        c.v = _mm256_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3], data[4][3], data[5][3], data[6][3], data[7][3]);
        ee.v = _mm256_add_ps(_mm256_mul_ps(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm256_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j], data[4][j], data[5][j], data[6][j], data[7][j]);
            eff.v = _mm256_add_ps(_mm256_mul_ps(eff.v, x.v), ee.v);
            ee.v = _mm256_add_ps(_mm256_mul_ps(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm256_store_ps(e, ee.v);
        _mm256_store_ps(f, _mm256_mul_ps(eff.v, _mm256_div_ps(hi.v, r.v)));

    #elif defined(__SSE__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            __v4sf v;
            __m128i m;
            float f[4];
            int i[4];
            } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
            alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
        float *data[8];

        /* Get r . */
        r_1.v = _mm_sqrt_ps(_mm_load_ps(&r2[0]));
        r_2.v = _mm_sqrt_ps(_mm_load_ps(&r2[4]));

        /* compute the index */
        alpha0_1.v = _mm_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_1.v = _mm_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_1.v = _mm_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        alpha0_2.v = _mm_setr_ps(p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1_2.v = _mm_setr_ps(p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2_2.v = _mm_setr_ps(p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind_1.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0_1.v, _mm_mul_ps(r_1.v, _mm_add_ps(alpha1_1.v, _mm_mul_ps(r_1.v, alpha2_1.v))))));
        ind_2.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0_2.v, _mm_mul_ps(r_2.v, _mm_add_ps(alpha1_2.v, _mm_mul_ps(r_2.v, alpha2_2.v))))));

        /* get the table offset */
        data[0] = &(p[0]->c[ ind_1.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind_1.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind_1.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind_1.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind_2.i[0] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind_2.i[1] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind_2.i[2] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind_2.i[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = _mm_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi_1.v = _mm_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1]);
        mi_2.v = _mm_setr_ps(data[4][0], data[5][0], data[6][0], data[7][0]);
        hi_2.v = _mm_setr_ps(data[4][1], data[5][1], data[6][1], data[7][1]);
        x_1.v = _mm_mul_ps(_mm_sub_ps(r_1.v, mi_1.v), hi_1.v);
        x_2.v = _mm_mul_ps(_mm_sub_ps(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = _mm_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2]);
        eff_2.v = _mm_setr_ps(data[4][2], data[5][2], data[6][2], data[7][2]);
        c_1.v = _mm_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3]);
        c_2.v = _mm_setr_ps(data[4][3], data[5][3], data[6][3], data[7][3]);
        ee_1.v = _mm_add_ps(_mm_mul_ps(eff_1.v, x_1.v), c_1.v);
        ee_2.v = _mm_add_ps(_mm_mul_ps(eff_2.v, x_2.v), c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = _mm_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j]);
            c_2.v = _mm_setr_ps(data[4][j], data[5][j], data[6][j], data[7][j]);
            eff_1.v = _mm_add_ps(_mm_mul_ps(eff_1.v, x_1.v), ee_1.v);
            eff_2.v = _mm_add_ps(_mm_mul_ps(eff_2.v, x_2.v), ee_2.v);
            ee_1.v = _mm_add_ps(_mm_mul_ps(ee_1.v, x_1.v), c_1.v);
            ee_2.v = _mm_add_ps(_mm_mul_ps(ee_2.v, x_2.v), c_2.v);
            }

        /* store the result */
        _mm_store_ps(&e[0], ee_1.v);
        _mm_store_ps(&e[4], ee_2.v);
        _mm_store_ps(&f[0], _mm_mul_ps(eff_1.v, _mm_div_ps(hi_1.v, r_1.v)));
        _mm_store_ps(&f[4], _mm_mul_ps(eff_2.v, _mm_div_ps(hi_2.v, r_2.v)));

    #elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            vector float v;
            vector unsigned int m;
            float f[4];
            unsigned int i[4];
            } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
            alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
        float *data[8];

        /* Get r . */
        r_1.v = vec_sqrt(*((vector float *)&r2[0]));
        r_2.v = vec_sqrt(*((vector float *)&r2[4]));

        /* compute the index */
        alpha0_1.v = vec_load4(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_1.v = vec_load4(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_1.v = vec_load4(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        alpha0_2.v = vec_load4(p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1_2.v = vec_load4(p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2_2.v = vec_load4(p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind_1.m = vec_ctu(vec_madd(r_1.v, vec_madd(r_1.v, alpha2_1.v, alpha1_1.v), alpha0_1.v), 0);
        ind_2.m = vec_ctu(vec_madd(r_2.v, vec_madd(r_2.v, alpha2_2.v, alpha1_2.v), alpha0_2.v), 0);

        /* get the table offset */
        data[0] = &(p[0]->c[ ind_1.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind_1.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind_1.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind_1.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind_2.i[0] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind_2.i[1] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind_2.i[2] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind_2.i[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = vec_load4(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi_1.v = vec_load4(data[0][1], data[1][1], data[2][1], data[3][1]);
        mi_2.v = vec_load4(data[4][0], data[5][0], data[6][0], data[7][0]);
        hi_2.v = vec_load4(data[4][1], data[5][1], data[6][1], data[7][1]);
        x_1.v = vec_mul(vec_sub(r_1.v, mi_1.v), hi_1.v);
        x_2.v = vec_mul(vec_sub(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = vec_load4(data[0][2], data[1][2], data[2][2], data[3][2]);
        eff_2.v = vec_load4(data[4][2], data[5][2], data[6][2], data[7][2]);
        c_1.v = vec_load4(data[0][3], data[1][3], data[2][3], data[3][3]);
        c_2.v = vec_load4(data[4][3], data[5][3], data[6][3], data[7][3]);
        ee_1.v = vec_madd(eff_1.v, x_1.v, c_1.v);
        ee_2.v = vec_madd(eff_2.v, x_2.v, c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = vec_load4(data[0][j], data[1][j], data[2][j], data[3][j]);
            c_2.v = vec_load4(data[4][j], data[5][j], data[6][j], data[7][j]);
            eff_1.v = vec_madd(eff_1.v, x_1.v, ee_1.v);
            eff_2.v = vec_madd(eff_2.v, x_2.v, ee_2.v);
            ee_1.v = vec_madd(ee_1.v, x_1.v, c_1.v);
            ee_2.v = vec_madd(ee_2.v, x_2.v, c_2.v);
            }

        /* store the result */
        eff_1.v = vec_mul(eff_1.v, vec_div(hi_1.v, r_1.v));
        eff_2.v = vec_mul(eff_2.v, vec_div(hi_2.v, r_2.v));
        memcpy(&e[0], &ee_1, sizeof(vector float));
        memcpy(&f[0], &eff_1, sizeof(vector float));
        memcpy(&e[4], &ee_2, sizeof(vector float));
        memcpy(&f[4], &eff_2, sizeof(vector float));

    #else
        int k;
        FPTYPE ee, eff;
        for(k = 0 ; k < 8 ; k++) {
            potential_eval(p[k], r2[k], &ee, &eff);
            e[k] = ee; f[k] = eff;
            }
    #endif

        }


    TF_ALWAYS_INLINE void potential_eval_vec_8single_r(struct Potential *p[8], float *r2, float *e, float *f) {

    #if defined(__AVX__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            __v8sf v;
            __m256i m;
            float f[8];
            int i[8];
            } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
        float *data[8];

        /* Get r . */
        r.v = _mm256_load_ps(r2);

        /* compute the index */
        alpha0.v = _mm256_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0], p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1.v = _mm256_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1], p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2.v = _mm256_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2], p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind.m = _mm256_cvttps_epi32(_mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(alpha0.v, _mm256_mul_ps(r.v, _mm256_add_ps(alpha1.v, _mm256_mul_ps(r.v, alpha2.v))))));

        /* get the table offset */
        data[0] = &(p[0]->c[ ind.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind.i[4] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind.i[5] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind.i[6] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind.i[7] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm256_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0], data[4][0], data[5][0], data[6][0], data[7][0]);
        hi.v = _mm256_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1], data[4][1], data[5][1], data[6][1], data[7][1]);
        x.v = _mm256_mul_ps(_mm256_sub_ps(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm256_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2], data[4][2], data[5][2], data[6][2], data[7][2]);
        c.v = _mm256_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3], data[4][3], data[5][3], data[6][3], data[7][3]);
        ee.v = _mm256_add_ps(_mm256_mul_ps(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm256_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j], data[4][j], data[5][j], data[6][j], data[7][j]);
            eff.v = _mm256_add_ps(_mm256_mul_ps(eff.v, x.v), ee.v);
            ee.v = _mm256_add_ps(_mm256_mul_ps(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm256_store_ps(e, ee.v);
        _mm256_store_ps(f, _mm256_mul_ps(eff.v, hi.v));

    #elif defined(__SSE__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            __v4sf v;
            __m128i m;
            float f[4];
            int i[4];
            } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
            alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
        float *data[8];

        /* Get r . */
        r_1.v = _mm_load_ps(&r2[0]);
        r_2.v = _mm_load_ps(&r2[4]);

        /* compute the index */
        alpha0_1.v = _mm_setr_ps(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_1.v = _mm_setr_ps(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_1.v = _mm_setr_ps(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        alpha0_2.v = _mm_setr_ps(p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1_2.v = _mm_setr_ps(p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2_2.v = _mm_setr_ps(p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind_1.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0_1.v, _mm_mul_ps(r_1.v, _mm_add_ps(alpha1_1.v, _mm_mul_ps(r_1.v, alpha2_1.v))))));
        ind_2.m = _mm_cvttps_epi32(_mm_max_ps(_mm_setzero_ps(), _mm_add_ps(alpha0_2.v, _mm_mul_ps(r_2.v, _mm_add_ps(alpha1_2.v, _mm_mul_ps(r_2.v, alpha2_2.v))))));

        /* get the table offset */
        data[0] = &(p[0]->c[ ind_1.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind_1.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind_1.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind_1.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind_2.i[0] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind_2.i[1] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind_2.i[2] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind_2.i[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = _mm_setr_ps(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi_1.v = _mm_setr_ps(data[0][1], data[1][1], data[2][1], data[3][1]);
        mi_2.v = _mm_setr_ps(data[4][0], data[5][0], data[6][0], data[7][0]);
        hi_2.v = _mm_setr_ps(data[4][1], data[5][1], data[6][1], data[7][1]);
        x_1.v = _mm_mul_ps(_mm_sub_ps(r_1.v, mi_1.v), hi_1.v);
        x_2.v = _mm_mul_ps(_mm_sub_ps(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = _mm_setr_ps(data[0][2], data[1][2], data[2][2], data[3][2]);
        eff_2.v = _mm_setr_ps(data[4][2], data[5][2], data[6][2], data[7][2]);
        c_1.v = _mm_setr_ps(data[0][3], data[1][3], data[2][3], data[3][3]);
        c_2.v = _mm_setr_ps(data[4][3], data[5][3], data[6][3], data[7][3]);
        ee_1.v = _mm_add_ps(_mm_mul_ps(eff_1.v, x_1.v), c_1.v);
        ee_2.v = _mm_add_ps(_mm_mul_ps(eff_2.v, x_2.v), c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = _mm_setr_ps(data[0][j], data[1][j], data[2][j], data[3][j]);
            c_2.v = _mm_setr_ps(data[4][j], data[5][j], data[6][j], data[7][j]);
            eff_1.v = _mm_add_ps(_mm_mul_ps(eff_1.v, x_1.v), ee_1.v);
            eff_2.v = _mm_add_ps(_mm_mul_ps(eff_2.v, x_2.v), ee_2.v);
            ee_1.v = _mm_add_ps(_mm_mul_ps(ee_1.v, x_1.v), c_1.v);
            ee_2.v = _mm_add_ps(_mm_mul_ps(ee_2.v, x_2.v), c_2.v);
            }

        /* store the result */
        _mm_store_ps(&e[0], ee_1.v);
        _mm_store_ps(&e[4], ee_2.v);
        _mm_store_ps(&f[0], _mm_mul_ps(eff_1.v, hi_1.v));
        _mm_store_ps(&f[4], _mm_mul_ps(eff_2.v, hi_2.v));

    #elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
        int j;
        union {
            vector float v;
            vector unsigned int m;
            float f[4];
            unsigned int i[4];
            } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
            alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
        float *data[8];

        /* Get r . */
        r_1.v = *((vector float *)&r2[0]);
        r_2.v = *((vector float *)&r2[4]);

        /* compute the index */
        alpha0_1.v = vec_load4(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_1.v = vec_load4(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_1.v = vec_load4(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        alpha0_2.v = vec_load4(p[4]->alpha[0], p[5]->alpha[0], p[6]->alpha[0], p[7]->alpha[0]);
        alpha1_2.v = vec_load4(p[4]->alpha[1], p[5]->alpha[1], p[6]->alpha[1], p[7]->alpha[1]);
        alpha2_2.v = vec_load4(p[4]->alpha[2], p[5]->alpha[2], p[6]->alpha[2], p[7]->alpha[2]);
        ind_1.m = vec_ctu(vec_madd(r_1.v, vec_madd(r_1.v, alpha2_1.v, alpha1_1.v), alpha0_1.v), 0);
        ind_2.m = vec_ctu(vec_madd(r_2.v, vec_madd(r_2.v, alpha2_2.v, alpha1_2.v), alpha0_2.v), 0);

        /* get the table offset */
        data[0] = &(p[0]->c[ ind_1.i[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind_1.i[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind_1.i[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind_1.i[3] * potential_chunk ]);
        data[4] = &(p[4]->c[ ind_2.i[0] * potential_chunk ]);
        data[5] = &(p[5]->c[ ind_2.i[1] * potential_chunk ]);
        data[6] = &(p[6]->c[ ind_2.i[2] * potential_chunk ]);
        data[7] = &(p[7]->c[ ind_2.i[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = vec_load4(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi_1.v = vec_load4(data[0][1], data[1][1], data[2][1], data[3][1]);
        mi_2.v = vec_load4(data[4][0], data[5][0], data[6][0], data[7][0]);
        hi_2.v = vec_load4(data[4][1], data[5][1], data[6][1], data[7][1]);
        x_1.v = vec_mul(vec_sub(r_1.v, mi_1.v), hi_1.v);
        x_2.v = vec_mul(vec_sub(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = vec_load4(data[0][2], data[1][2], data[2][2], data[3][2]);
        eff_2.v = vec_load4(data[4][2], data[5][2], data[6][2], data[7][2]);
        c_1.v = vec_load4(data[0][3], data[1][3], data[2][3], data[3][3]);
        c_2.v = vec_load4(data[4][3], data[5][3], data[6][3], data[7][3]);
        ee_1.v = vec_madd(eff_1.v, x_1.v, c_1.v);
        ee_2.v = vec_madd(eff_2.v, x_2.v, c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = vec_load4(data[0][j], data[1][j], data[2][j], data[3][j]);
            c_2.v = vec_load4(data[4][j], data[5][j], data[6][j], data[7][j]);
            eff_1.v = vec_madd(eff_1.v, x_1.v, ee_1.v);
            eff_2.v = vec_madd(eff_2.v, x_2.v, ee_2.v);
            ee_1.v = vec_madd(ee_1.v, x_1.v, c_1.v);
            ee_2.v = vec_madd(ee_2.v, x_2.v, c_2.v);
            }

        /* store the result */
        eff_1.v = vec_mul(eff_1.v, hi_1.v);
        eff_2.v = vec_mul(eff_2.v, hi_2.v);
        memcpy(&e[0], &ee_1, sizeof(vector float));
        memcpy(&f[0], &eff_1, sizeof(vector float));
        memcpy(&e[4], &ee_2, sizeof(vector float));
        memcpy(&f[4], &eff_2, sizeof(vector float));

    #else
        int k;
        FPTYPE ee, eff;
        for(k = 0 ; k < 8 ; k++) {
            potential_eval(p[k], r2[k], &ee, &eff);
            e[k] = ee; f[k] = eff;
            }
    #endif

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r2 Pointer to an array of the radii at which the potentials
     *      are to be evaluated, squared.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes two double-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE2 and
     * double precision! If @c mdcore was not compiled with SSE2 enabled, this
     * function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_2double(struct Potential *p[2], FPTYPE *r2, FPTYPE *e, FPTYPE *f) {

    #if defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        int ind[2], j;
        union {
            __v2df v;
            double f[2];
            } alpha0, alpha1, alpha2, rind, mi, hi, x, ee, eff, c, r;
        double *data[2];

        /* Get r . */
        r.v = _mm_sqrt_pd(_mm_load_pd(r2));

        /* compute the index */
        alpha0.v = _mm_setr_pd(p[0]->alpha[0], p[1]->alpha[0]);
        alpha1.v = _mm_setr_pd(p[0]->alpha[1], p[1]->alpha[1]);
        alpha2.v = _mm_setr_pd(p[0]->alpha[2], p[1]->alpha[2]);
        rind.v = _mm_max_pd(_mm_setzero_pd(), _mm_add_pd(alpha0.v, _mm_mul_pd(r.v, _mm_add_pd(alpha1.v, _mm_mul_pd(r.v, alpha2.v)))));
        ind[0] = rind.f[0];
        ind[1] = rind.f[1];

        /* get the table offset */
        data[0] = &(p[0]->c[ ind[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind[1] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm_setr_pd(data[0][0], data[1][0]);
        hi.v = _mm_setr_pd(data[0][1], data[1][1]);
        x.v = _mm_mul_pd(_mm_sub_pd(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm_setr_pd(data[0][2], data[1][2]);
        c.v = _mm_setr_pd(data[0][3], data[1][3]);
        ee.v = _mm_add_pd(_mm_mul_pd(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm_setr_pd(data[0][j], data[1][j]);
            eff.v = _mm_add_pd(_mm_mul_pd(eff.v, x.v), ee.v);
            ee.v = _mm_add_pd(_mm_mul_pd(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm_store_pd(e, ee.v);
        _mm_store_pd(f, _mm_mul_pd(eff.v, _mm_div_pd(hi.v, r.v)));

    #else
        int k;
        for(k = 0 ; k < 2 ; k++)
            potential_eval(p[k], r2[k], &e[k], &f[k]);
    #endif

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r2 Pointer to an array of the radii at which the potentials
     *      are to be evaluated, squared.
     * @param e Pointer to an array of floating-point values in which to store the
     *      interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the
     *      magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes four double-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE2 and
     * double precision! If @c mdcore was not compiled with SSE2 enabled, this
     * function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_4double(struct Potential *p[4], FPTYPE *r2, FPTYPE *e, FPTYPE *f) {

    #if defined(__AVX__) && defined(FPTYPE_DOUBLE)
        int ind[4], j;
        union {
            __v4df v;
            double f[4];
            } alpha0, alpha1, alpha2, rind, mi, hi, x, ee, eff, c, r;
        double *data[4];

        /* Get r . */
        r.v = _mm256_sqrt_pd(_mm256_load_pd(r2));

        /* compute the index */
        alpha0.v = _mm256_setr_pd(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = _mm256_setr_pd(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = _mm256_setr_pd(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        rind.v = _mm256_max_pd(_mm256_setzero_pd(), _mm256_add_pd(alpha0.v, _mm256_mul_pd(r.v, _mm256_add_pd(alpha1.v, _mm256_mul_pd(r.v, alpha2.v)))));
        ind[0] = rind.f[0];
        ind[1] = rind.f[1];
        ind[2] = rind.f[2];
        ind[3] = rind.f[3];

        /* get the table offset */
        data[0] = &(p[0]->c[ ind[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm256_setr_pd(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = _mm256_setr_pd(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = _mm256_mul_pd(_mm256_sub_pd(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm256_setr_pd(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = _mm256_setr_pd(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = _mm256_add_pd(_mm256_mul_pd(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm256_setr_pd(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = _mm256_add_pd(_mm256_mul_pd(eff.v, x.v), ee.v);
            ee.v = _mm256_add_pd(_mm256_mul_pd(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm256_store_pd(e, ee.v);
        _mm256_store_pd(f, _mm256_mul_pd(eff.v, _mm256_div_pd(hi.v, r.v)));

    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        int ind[4], j;
        union {
            __v2df v;
            double f[2];
            } alpha0_1, alpha1_1, alpha2_1, rind_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1,
            alpha0_2, alpha1_2, alpha2_2, rind_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2;
        double *data[4];

        /* Get r . */
        r_1.v = _mm_sqrt_pd(_mm_load_pd(&r2[0]));
        r_2.v = _mm_sqrt_pd(_mm_load_pd(&r2[2]));

        /* compute the index */
        alpha0_1.v = _mm_setr_pd(p[0]->alpha[0], p[1]->alpha[0]);
        alpha1_1.v = _mm_setr_pd(p[0]->alpha[1], p[1]->alpha[1]);
        alpha2_1.v = _mm_setr_pd(p[0]->alpha[2], p[1]->alpha[2]);
        alpha0_2.v = _mm_setr_pd(p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_2.v = _mm_setr_pd(p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_2.v = _mm_setr_pd(p[2]->alpha[2], p[3]->alpha[2]);
        rind_1.v = _mm_max_pd(_mm_setzero_pd(), _mm_add_pd(alpha0_1.v, _mm_mul_pd(r_1.v, _mm_add_pd(alpha1_1.v, _mm_mul_pd(r_1.v, alpha2_1.v)))));
        rind_2.v = _mm_max_pd(_mm_setzero_pd(), _mm_add_pd(alpha0_2.v, _mm_mul_pd(r_2.v, _mm_add_pd(alpha1_2.v, _mm_mul_pd(r_2.v, alpha2_2.v)))));
        ind[0] = rind_1.f[0];
        ind[1] = rind_1.f[1];
        ind[2] = rind_2.f[0];
        ind[3] = rind_2.f[1];

        /* get the table offset */
        data[0] = &(p[0]->c[ ind[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = _mm_setr_pd(data[0][0], data[1][0]);
        hi_1.v = _mm_setr_pd(data[0][1], data[1][1]);
        mi_2.v = _mm_setr_pd(data[2][0], data[3][0]);
        hi_2.v = _mm_setr_pd(data[2][1], data[3][1]);
        x_1.v = _mm_mul_pd(_mm_sub_pd(r_1.v, mi_1.v), hi_1.v);
        x_2.v = _mm_mul_pd(_mm_sub_pd(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = _mm_setr_pd(data[0][2], data[1][2]);
        eff_2.v = _mm_setr_pd(data[2][2], data[3][2]);
        c_1.v = _mm_setr_pd(data[0][3], data[1][3]);
        c_2.v = _mm_setr_pd(data[2][3], data[3][3]);
        ee_1.v = _mm_add_pd(_mm_mul_pd(eff_1.v, x_1.v), c_1.v);
        ee_2.v = _mm_add_pd(_mm_mul_pd(eff_2.v, x_2.v), c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = _mm_setr_pd(data[0][j], data[1][j]);
            c_2.v = _mm_setr_pd(data[2][j], data[3][j]);
            eff_1.v = _mm_add_pd(_mm_mul_pd(eff_1.v, x_1.v), ee_1.v);
            eff_2.v = _mm_add_pd(_mm_mul_pd(eff_2.v, x_2.v), ee_2.v);
            ee_1.v = _mm_add_pd(_mm_mul_pd(ee_1.v, x_1.v), c_1.v);
            ee_2.v = _mm_add_pd(_mm_mul_pd(ee_2.v, x_2.v), c_2.v);
            }

        /* store the result */
        _mm_store_pd(&e[0], ee_1.v);
        _mm_store_pd(&f[0], _mm_mul_pd(eff_1.v, _mm_div_pd(hi_1.v, r_1.v)));
        _mm_store_pd(&e[2], ee_2.v);
        _mm_store_pd(&f[2], _mm_mul_pd(eff_2.v, _mm_div_pd(hi_2.v, r_2.v)));

    #else
        int k;
        for(k = 0 ; k < 4 ; k++)
            potential_eval(p[k], r2[k], &e[k], &f[k]);
    #endif

        }


    /**
     * @brief Evaluates the given potential at a set of points (interpolated).
     *
     * @param p Pointer to an array of pointers to the #potentials to be evaluated.
     * @param r_in Pointer to an array of the radii at which the potentials are to be evaluated.
     * @param e Pointer to an array of floating-point values in which to store the interaction energies.
     * @param f Pointer to an array of floating-point values in which to store the magnitude of the interaction forces.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     *
     * Computes four double-precision interactions simultaneously using vectorized
     * instructions.
     *
     * This function is only available if mdcore was compiled with SSE2 and
     * double precision! If @c mdcore was not compiled with SSE2 enabled, this
     * function simply calls #potential_eval on each entry.
     */

    TF_ALWAYS_INLINE void potential_eval_vec_4double_r(struct Potential *p[4], FPTYPE *r_in, FPTYPE *e, FPTYPE *f) {

    #if defined(__AVX__) && defined(FPTYPE_DOUBLE)
        int ind[4], j;
        union {
            __m256d v;
            double f[4];
            } alpha0, alpha1, alpha2, rind, mi, hi, x, ee, eff, c, r;
        double *data[4];

        /* Get r . */
        r.v = _mm256_load_pd(r_in);

        /* compute the index */
        alpha0.v = _mm256_setr_pd(p[0]->alpha[0], p[1]->alpha[0], p[2]->alpha[0], p[3]->alpha[0]);
        alpha1.v = _mm256_setr_pd(p[0]->alpha[1], p[1]->alpha[1], p[2]->alpha[1], p[3]->alpha[1]);
        alpha2.v = _mm256_setr_pd(p[0]->alpha[2], p[1]->alpha[2], p[2]->alpha[2], p[3]->alpha[2]);
        rind.v = _mm256_max_pd(_mm256_setzero_pd(), _mm256_add_pd(alpha0.v, _mm256_mul_pd(r.v, _mm256_add_pd(alpha1.v, _mm256_mul_pd(r.v, alpha2.v)))));
        ind[0] = rind.f[0];
        ind[1] = rind.f[1];
        ind[2] = rind.f[2];
        ind[3] = rind.f[3];

        /* get the table offset */
        data[0] = &(p[0]->c[ ind[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi.v = _mm256_setr_pd(data[0][0], data[1][0], data[2][0], data[3][0]);
        hi.v = _mm256_setr_pd(data[0][1], data[1][1], data[2][1], data[3][1]);
        x.v = _mm256_mul_pd(_mm256_sub_pd(r.v, mi.v), hi.v);

        /* compute the potential and its derivative */
        eff.v = _mm256_setr_pd(data[0][2], data[1][2], data[2][2], data[3][2]);
        c.v = _mm256_setr_pd(data[0][3], data[1][3], data[2][3], data[3][3]);
        ee.v = _mm256_add_pd(_mm256_mul_pd(eff.v, x.v), c.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c.v = _mm256_setr_pd(data[0][j], data[1][j], data[2][j], data[3][j]);
            eff.v = _mm256_add_pd(_mm256_mul_pd(eff.v, x.v), ee.v);
            ee.v = _mm256_add_pd(_mm256_mul_pd(ee.v, x.v), c.v);
            }

        /* store the result */
        _mm256_store_pd(e, ee.v);
        _mm256_store_pd(f, _mm256_mul_pd(eff.v, hi.v));

    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        int ind[4], j;
        union {
            __v2df v;
            double f[2];
            } alpha0_1, alpha1_1, alpha2_1, rind_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1,
            alpha0_2, alpha1_2, alpha2_2, rind_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2;
        double *data[4];

        /* Get r . */
        r_1.v = _mm_load_pd(&r_in[0]);
        r_2.v = _mm_load_pd(&r_in[2]);

        /* compute the index */
        alpha0_1.v = _mm_setr_pd(p[0]->alpha[0], p[1]->alpha[0]);
        alpha1_1.v = _mm_setr_pd(p[0]->alpha[1], p[1]->alpha[1]);
        alpha2_1.v = _mm_setr_pd(p[0]->alpha[2], p[1]->alpha[2]);
        alpha0_2.v = _mm_setr_pd(p[2]->alpha[0], p[3]->alpha[0]);
        alpha1_2.v = _mm_setr_pd(p[2]->alpha[1], p[3]->alpha[1]);
        alpha2_2.v = _mm_setr_pd(p[2]->alpha[2], p[3]->alpha[2]);
        rind_1.v = _mm_max_pd(_mm_setzero_pd(), _mm_add_pd(alpha0_1.v, _mm_mul_pd(r_1.v, _mm_add_pd(alpha1_1.v, _mm_mul_pd(r_1.v, alpha2_1.v)))));
        rind_2.v = _mm_max_pd(_mm_setzero_pd(), _mm_add_pd(alpha0_2.v, _mm_mul_pd(r_2.v, _mm_add_pd(alpha1_2.v, _mm_mul_pd(r_2.v, alpha2_2.v)))));
        ind[0] = rind_1.f[0];
        ind[1] = rind_1.f[1];
        ind[2] = rind_2.f[0];
        ind[3] = rind_2.f[1];

        /* get the table offset */
        data[0] = &(p[0]->c[ ind[0] * potential_chunk ]);
        data[1] = &(p[1]->c[ ind[1] * potential_chunk ]);
        data[2] = &(p[2]->c[ ind[2] * potential_chunk ]);
        data[3] = &(p[3]->c[ ind[3] * potential_chunk ]);

        /* adjust x to the interval */
        mi_1.v = _mm_setr_pd(data[0][0], data[1][0]);
        hi_1.v = _mm_setr_pd(data[0][1], data[1][1]);
        mi_2.v = _mm_setr_pd(data[2][0], data[3][0]);
        hi_2.v = _mm_setr_pd(data[2][1], data[3][1]);
        x_1.v = _mm_mul_pd(_mm_sub_pd(r_1.v, mi_1.v), hi_1.v);
        x_2.v = _mm_mul_pd(_mm_sub_pd(r_2.v, mi_2.v), hi_2.v);

        /* compute the potential and its derivative */
        eff_1.v = _mm_setr_pd(data[0][2], data[1][2]);
        eff_2.v = _mm_setr_pd(data[2][2], data[3][2]);
        c_1.v = _mm_setr_pd(data[0][3], data[1][3]);
        c_2.v = _mm_setr_pd(data[2][3], data[3][3]);
        ee_1.v = _mm_add_pd(_mm_mul_pd(eff_1.v, x_1.v), c_1.v);
        ee_2.v = _mm_add_pd(_mm_mul_pd(eff_2.v, x_2.v), c_2.v);
        for(j = 4 ; j < potential_chunk ; j++) {
            c_1.v = _mm_setr_pd(data[0][j], data[1][j]);
            c_2.v = _mm_setr_pd(data[2][j], data[3][j]);
            eff_1.v = _mm_add_pd(_mm_mul_pd(eff_1.v, x_1.v), ee_1.v);
            eff_2.v = _mm_add_pd(_mm_mul_pd(eff_2.v, x_2.v), ee_2.v);
            ee_1.v = _mm_add_pd(_mm_mul_pd(ee_1.v, x_1.v), c_1.v);
            ee_2.v = _mm_add_pd(_mm_mul_pd(ee_2.v, x_2.v), c_2.v);
            }

        /* store the result */
        _mm_store_pd(&e[0], ee_1.v);
        _mm_store_pd(&f[0], _mm_mul_pd(eff_1.v, hi_1.v));
        _mm_store_pd(&e[2], ee_2.v);
        _mm_store_pd(&f[2], _mm_mul_pd(eff_2.v, hi_2.v));

    #else
        int k;
        for(k = 0 ; k < 4 ; k++)
            potential_eval_r(p[k], r_in[k], &e[k], &f[k]);
    #endif

    }

    TF_ALWAYS_INLINE Potential *get_potential(const Particle *a, const Particle *b) {
        int index = _Engine.max_type * a->typeId + b->typeId;
        if ((a->flags & b->flags & PARTICLE_BOUND) && (a->clusterId == b->clusterId)) {
            return _Engine.p_cluster[index];
        }
        else {
            return _Engine.p[index];
        }
    }

};

#endif // _MDCORE_SOURCE_TF_POTENTIAL_EVAL_H_