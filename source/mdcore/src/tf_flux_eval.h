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

#ifndef _MDCORE_SOURCE_TF_FLUX_EVAL_H_
#define _MDCORE_SOURCE_TF_FLUX_EVAL_H_

#include <tfFlux.h>
#include <state/tfStateVector.h>
#include <tfEngine.h>
#include <iostream>


namespace TissueForge { 


    TF_ALWAYS_INLINE FPTYPE flux_fick(Flux *flux, int i, FPTYPE si, FPTYPE sj) {
        return flux->coef[i] * (si - sj);
    }

    TF_ALWAYS_INLINE FPTYPE flux_secrete(Flux *flux, int i, FPTYPE si, FPTYPE sj) {
        FPTYPE q = flux->coef[i] * (si - flux->target[i]);
        FPTYPE scale = q > 0.f;  // forward only, 1 if > 0, 0 if < 0.
        return scale * q;
    }

    TF_ALWAYS_INLINE FPTYPE flux_uptake(Flux *flux, int i, FPTYPE si, FPTYPE sj) {
        FPTYPE q = flux->coef[i] * (flux->target[i] - sj) * si;
        FPTYPE scale = q > 0.f;
        return scale * q;
    }

    TF_ALWAYS_INLINE void flux_eval_ex(
        struct Fluxes *f, FPTYPE r2, Particle *part_i, Particle *part_j) 
    {
        
        Flux *flux = &f->fluxes[0];
        FPTYPE  r = std::sqrt(r2);
        FPTYPE term = (1. - r / _Engine.s.cutoff);
        term = term * term;
        
        for(int i = 0; i < flux->size; ++i) {
            // NOTE: order important here, type ids could be the same, i.e.
            // Fick flux, the true branch of each assignemnt gets evaluated.
            Particle *pi = part_i->typeId == flux->type_ids[i].a ? part_i : part_j;
            Particle *pj = part_j->typeId == flux->type_ids[i].b ? part_j : part_i;
            
            assert(pi->typeId == flux->type_ids[i].a);
            assert(pj->typeId == flux->type_ids[i].b);
            assert(pi != pj);
            
            FPTYPE *si = pi->state_vector->fvec;
            FPTYPE *sj = pj->state_vector->fvec;
            
            FPTYPE *qi = pi->state_vector->q;
            FPTYPE *qj = pj->state_vector->q;
            
            int32_t *ii = flux->indices_a;
            int32_t *ij = flux->indices_b;
            
            FPTYPE ssi = si[ii[i]];
            FPTYPE ssj = sj[ij[i]];
            FPTYPE q =  term;
            
            switch(flux->kinds[i]) {
                case FLUX_FICK:
                    q *= flux_fick(flux, i, ssi, ssj);
                    break;
                case FLUX_SECRETE:
                    q *= flux_secrete(flux, i, ssi, ssj);
                    break;
                case FLUX_UPTAKE:
                    q *= flux_uptake(flux, i, ssi, ssj);
                    break;
                default:
                    assert(0);
            }
            
            FPTYPE half_decay = flux->decay_coef[i] / 2.f;
            qi[ii[i]] = qi[ii[i]] - q - half_decay * ssi;
            qj[ij[i]] = qj[ij[i]] + q - half_decay * ssj;
        }
    }

    inline Fluxes *get_fluxes(const Particle *a, const Particle *b) {
        int index = _Engine.max_type * a->typeId + b->typeId;
        return _Engine.fluxes[index];
    }

};

#endif // _MDCORE_SOURCE_TF_FLUX_EVAL_H_