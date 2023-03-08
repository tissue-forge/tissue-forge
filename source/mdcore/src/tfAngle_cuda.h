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

#ifndef _MDCORE_SOURCE_TFANGLE_CUDA_H_
#define _MDCORE_SOURCE_TFANGLE_CUDA_H_

#include "tfPotential_cuda.h"
#include <tfAngle.h>


namespace TissueForge::cuda { 


    struct Angle { 
        uint32_t flags;
        float dissociation_energy;
        float half_life;
        int3 pids;

        TissueForge::Potential p;

        __host__ 
        Angle();
        
        __host__ 
        Angle(TissueForge::Angle *a);

        __host__ 
        void finalize();
    };

    int Angle_setThreads(const unsigned int &nr_threads);
    int Angle_setBlocks(const unsigned int &nr_blocks);
    int Angle_getDevice();
    int Angle_toDevice(engine *e);
    int Angle_fromDevice(engine *e);
    int Angle_refresh(engine *e);
    int Angle_refreshAngle(engine *e, AngleHandle *a);
    int Angle_refreshAngles(engine *e, AngleHandle **angles, int nr_angles);

    int engine_cuda_add_angle(AngleHandle *ah);
    int engine_cuda_finalize_angle(int aind);
    int engine_cuda_finalize_angles(engine *e, int *ainds, int nr_angles);
    int engine_cuda_finalize_angles_all(engine *e);
    int engine_angle_eval_cuda(struct TissueForge::Angle *angles, int N, struct engine *e, float *epot_out);

};

#endif // _MDCORE_SOURCE_TFANGLE_CUDA_H_