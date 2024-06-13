/*******************************************************************************
 * This file is part of mdcore.
 * Copyright (c) 2022-2024 T.J. Sego
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

#ifndef _MDCORE_SOURCE_TFBOND_CUDA_H_
#define _MDCORE_SOURCE_TFBOND_CUDA_H_

#include "tfPotential_cuda.h"
#include <tfBond.h>


namespace TissueForge::cuda { 


    struct Bond { 
        uint32_t flags;
        float dissociation_energy;
        float half_life;
        int2 pids;

        TissueForge::Potential p;

        __host__ 
        Bond();
        
        __host__ 
        Bond(TissueForge::Bond *b);

        __host__ 
        void finalize();
    };

    int Bond_setThreads(const unsigned int &nr_threads);
    int Bond_setBlocks(const unsigned int &nr_blocks);
    int Bond_getDevice();
    int Bond_setDevice(engine *e, const int &deviceId);
    int Bond_toDevice(engine *e);
    int Bond_fromDevice(engine *e);
    int Bond_refresh(engine *e);
    int Bond_refreshBond(engine *e, BondHandle *b);
    int Bond_refreshBonds(engine *e, BondHandle **bonds, int nr_bonds);

    int engine_cuda_add_bond(TissueForge::Bond *b);
    int engine_cuda_add_bonds(TissueForge::Bond *bonds, int nr_bonds);
    int engine_cuda_finalize_bond(int bind);
    int engine_cuda_finalize_bonds(engine *e, int *binds, int nr_bonds);
    int engine_cuda_finalize_bonds_all(engine *e);
    int engine_bond_eval_cuda(struct TissueForge::Bond *bonds, int N, struct engine *e, float *epot_out);

};

#endif // _MDCORE_SOURCE_TFBOND_CUDA_H_