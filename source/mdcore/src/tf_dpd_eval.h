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

#ifndef _MDCORE_SOURCE_TF_DPD_EVAL_H_
#define _MDCORE_SOURCE_TF_DPD_EVAL_H_

#include <tfPotential.h>
#include <tfDPDPotential.h>
#include "tf_smoothing_kernel.h"
#include <random>


namespace TissueForge { 


    TF_ALWAYS_INLINE bool dpd_eval(DPDPotential *p, FPTYPE gaussian,
                                   Particle *pi, Particle *pj, FPTYPE* dx, FPTYPE r2, FPTYPE *energy) 
    {
        
        static const FPTYPE delta = 1.f / std::sqrt(_Engine.dt);
        static const FPTYPE epsilon = std::numeric_limits<FPTYPE>::epsilon();
        
        FPTYPE r = std::sqrt(r2);
        FPTYPE ro = r < epsilon ? epsilon : r;

        r = p->flags & POTENTIAL_SHIFTED ? r - (pi->radius + pj->radius) : r;

        if(r > p->b) {
            return false;
        }
        r = r >= p->a ? r : p->a;
        
        // unit vector
        FVector3 e = {dx[0] / ro, dx[1] / ro, dx[2] / ro};
        
        FVector3 v = pi->velocity - pj->velocity;
        
        // conservative force
        FPTYPE omega_c = r < 0.f ?  1.f : (1 - r / p->b);
        
        FPTYPE fc = p->alpha * omega_c;
        
        // dissapative force
        FPTYPE omega_d = omega_c * omega_c;
        
        FPTYPE fd = -p->gamma * omega_d * e.dot(v);
        
        FPTYPE fr = p->sigma * omega_c * delta;
        
        FPTYPE f = fc + fd + fr;
        
        pj->force = {pj->f[0] - f * e[0], pj->f[1] - f * e[1], pj->f[2] - f * e[2] };
        
        pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
        
        // TODO: correct energy
        *energy = 0;
        
        return true;
    }

    TF_ALWAYS_INLINE bool dpd_boundary_eval(DPDPotential *p, FPTYPE gaussian,
                                            Particle *pi, FPTYPE &rj, const FPTYPE *velocity, const FPTYPE* dx, FPTYPE r2, FPTYPE *energy) 
    {
        
        static const FPTYPE delta = 1.f / std::sqrt(_Engine.dt);
        static const FPTYPE epsilon = std::numeric_limits<FPTYPE>::epsilon();
        
        FPTYPE r = std::sqrt(r2);
        FPTYPE ro = r < epsilon ? epsilon : r;

        r = p->flags & POTENTIAL_SHIFTED ? r - (pi->radius + rj) : r;

        if(r > p->b) {
            return false;
        }
        r = r >= p->a ? r : p->a;
        
        // unit vector
        FVector3 e = {dx[0] / ro, dx[1] / ro, dx[2] / ro};
        
        FVector3 v = {pi->velocity[0] - velocity[0], pi->velocity[1] - velocity[1], pi->velocity[2] - velocity[2]};
        
        // conservative force
        FPTYPE omega_c = r < 0.f ?  1.f : (1 - r / p->b);
        
        FPTYPE fc = p->alpha * omega_c;
        
        // dissapative force
        FPTYPE omega_d = omega_c * omega_c;
        
        FPTYPE fd = -p->gamma * omega_d * e.dot(v);
        
        FPTYPE fr = p->sigma * omega_c * delta;
        
        FPTYPE f = fc + fd + fr;
        
        pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
        
        // TODO: correct energy
        *energy = 0;
        
        return true;
    }

};

#endif // _MDCORE_SOURCE_TF_DPD_EVAL_H_