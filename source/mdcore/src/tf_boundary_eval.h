/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TF_BOUNDARY_EVAL_H_
#define _MDCORE_SOURCE_TF_BOUNDARY_EVAL_H_

#include "tfParticle.h"
#include "tfPotential.h"
#include "tfSpace_cell.h"
#include "tfEngine.h"
#include "tf_dpd_eval.h"
#include "tf_potential_eval.h"

#include <iostream>


// velocity boundary conditions:
//
// r_new = r_old + 2 d n_w,
// where d is distance particle penetrated into wall, and
// n_w is normal vector into simulation domain.
//
// v_new = 2 U_wall - v_old
// where U_wall is wall velocity.


namespace TissueForge { 


    TF_ALWAYS_INLINE bool apply_update_pos_vel(Particle *p, space_cell *c, const FPTYPE* h, int* delta) {
        
        #define ENFORCE_FREESLIP_LOW(i)                                                     \
            p->position[i] = std::max<FPTYPE>(-p->position[i] * restitution, FPTYPE_ZERO);  \
            p->velocity[i] *= -restitution;                                                 \
            enforced = true;                                                                \

        #define ENFORCE_FREESLIP_HIGH(i)                                                                                \
            p->position[i] = c->dim[i] - std::max<FPTYPE>((p->position[i] - c->dim[i]) * restitution, FPTYPE_EPSILON);  \
            p->velocity[i] *= -restitution;                                                                             \
            enforced = true;                                                                                            \

        #define ENFORCE_VELOCITY_LOW(i, bc)                                                 \
            p->position[i] = std::max<FPTYPE>(-p->position[i] * bc.restore, FPTYPE_ZERO);   \
            p->velocity = bc.velocity - ((p->velocity - bc.velocity) * bc.restore);         \
            enforced = true;                                                                \

        #define ENFORCE_VELOCITY_HIGH(i, bc)                                                                            \
            p->position[i] = c->dim[i] - std::max<FPTYPE>((p->position[i] - c->dim[i]) * bc.restore, FPTYPE_EPSILON);   \
            p->velocity = bc.velocity - ((p->velocity - bc.velocity) * bc.restore);                                     \
            enforced = true;                                                                                            \

        static const BoundaryConditions *bc = &_Engine.boundary_conditions;
        
        static const FPTYPE restitution = 1.0;
        
        /* Enforce particle position to be within the given boundary */
        bool enforced = false;

        if(c->flags & cell_active_left && p->x[0] <= 0) {
            if(bc->left.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_LOW(0);
            }
            else if(bc->left.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_LOW(0, bc->left);
            }
        }

        if(c->flags & cell_active_right && p->x[0] >= c->dim[0]) {
            if(bc->right.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_HIGH(0);
            }
            else if(bc->right.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_HIGH(0, bc->right);
            }
        }

        if(c->flags & cell_active_front && p->x[1] <= 0) {
            if(bc->front.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_LOW(1);
            }
            else if(bc->front.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_LOW(1, bc->front);
            }
        }

        if(c->flags & cell_active_back && p->x[1] >= c->dim[1]) {
            if(bc->back.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_HIGH(1);
            }
            else if(bc->back.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_HIGH(1, bc->back);
            }
        }

        if(c->flags & cell_active_bottom && p->x[2] <= 0) {
            if(bc->bottom.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_LOW(2);
            }
            else if(bc->bottom.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_LOW(2, bc->bottom);
            }
        }

        if(c->flags & cell_active_top && p->x[2] >= c->dim[2]) {
            if(bc->top.kind & BOUNDARY_FREESLIP) {
                ENFORCE_FREESLIP_HIGH(2);
            }
            else if(bc->top.kind & BOUNDARY_VELOCITY) {
                ENFORCE_VELOCITY_HIGH(2, bc->top);
            }
        }
        
        if(enforced) {
            for (int k = 0 ; k < 3 ; k++ ) {
                delta[k] = __builtin_isgreaterequal( p->x[k], h[k] ) - __builtin_isless( p->x[k], 0.0 );
            }
        }

        return delta[0] == 0 && delta[1] == 0 && delta[2] == 0;
    };

    TF_ALWAYS_INLINE bool boundary_potential_eval_ex(const struct space_cell *cell,
                                Potential *pot, Particle *part, BoundaryCondition *bc,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot);

    static bool _boundary_potential_eval_ex(const struct space_cell *cell,
                                Potential *pot, Particle *part, BoundaryCondition *bc,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot) 
    {
        return boundary_potential_eval_ex(cell, pot, part, bc, dx, r2, epot);
    }

    bool boundary_potential_eval_ex(const struct space_cell *cell,
                                Potential *pot, Particle *part, BoundaryCondition *bc,
                                FPTYPE *dx, FPTYPE r2, FPTYPE *epot) 
    {
        FPTYPE e = 0;
        bool result = false;
        
        if(pot->kind == POTENTIAL_KIND_DPD) {
            /* update the forces if part in range */
            if (dpd_boundary_eval((DPDPotential*)pot, space_cell_gaussian(cell->id), part, bc->radius, bc->velocity.data(), dx, r2, &e)) {
                /* tabulate the energy */
                *epot += e;
                result = true;
            }
        }
        else if(pot->kind == POTENTIAL_KIND_BYPARTICLES) {
            FPTYPE fv[3] = {0., 0., 0.};

            pot->eval_bypart(pot, part, dx, r2, &e, fv);

            for (int k = 0 ; k < 3 ; k++ ) {
                part->f[k] += fv[k];
            }
            
            /* tabulate the energy */
            *epot += e;
            result = true;
        }
        else if(pot->kind == POTENTIAL_KIND_COMBINATION) {
            if(pot->flags & POTENTIAL_SUM) {
                _boundary_potential_eval_ex(cell, pot->pca, part, bc, dx, r2, epot);
                _boundary_potential_eval_ex(cell, pot->pcb, part, bc, dx, r2, epot);
                result = true;
            }
        }
        else {
            FPTYPE f;

            /* update the forces if part in range */
            if (potential_eval_ex(pot, part->radius, bc->radius, r2, &e, &f )) {

                for (int k = 0 ; k < 3 ; k++ ) {
                    FPTYPE w = f * dx[k];
                    part->f[k] -= w;
                }

                /* tabulate the energy */
                *epot += e;
                result = true;
            }
        }
        return result;
    }


    TF_ALWAYS_INLINE bool boundary_eval(BoundaryConditions *bc, const struct space_cell *cell, Particle *part, FPTYPE *epot ) {
        
        Potential *pot;
        FPTYPE r;
        bool result = false;
        
        FPTYPE dx[3] = {0.f, 0.f, 0.f};
            
        if((cell->flags & cell_active_left) &&
        (pot = bc->left.potenntials[part->typeId]) &&
        ((r = part->x[0]) <= pot->b)) {
            dx[0] = r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->left, dx, r*r, epot);
        }
        
        if((cell->flags & cell_active_right) &&
        (pot = bc->right.potenntials[part->typeId]) &&
        ((r = cell->dim[0] - part->x[0]) <= pot->b)) {
            dx[0] = -r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->right, dx, r*r, epot);
        }
        
        if((cell->flags & cell_active_front) &&
        (pot = bc->front.potenntials[part->typeId]) &&
        ((r = part->x[1]) <= pot->b)) {
            dx[1] = r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->front, dx, r*r, epot);
        }
        
        if((cell->flags & cell_active_back) &&
        (pot = bc->back.potenntials[part->typeId]) &&
        ((r = cell->dim[1] - part->x[1]) <= pot->b)) {
            dx[1] = -r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->back, dx, r*r, epot);
        }
        
        if((cell->flags & cell_active_bottom) &&
        (pot = bc->bottom.potenntials[part->typeId]) &&
        ((r = part->x[2]) <= pot->b)) {
            dx[2] = r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->bottom, dx, r*r, epot);
        }
        
        if((cell->flags & cell_active_top) &&
        (pot = bc->top.potenntials[part->typeId]) &&
        ((r = cell->dim[2] - part->x[2]) <= pot->b)) {
            dx[2] = -r;
            result |= boundary_potential_eval_ex(cell, pot, part, &bc->top, dx, r*r, epot);
        }
        return result;
    }

};

#endif // _MDCORE_SOURCE_TF_BOUNDARY_EVAL_H_