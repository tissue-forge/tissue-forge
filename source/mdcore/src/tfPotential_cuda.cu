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

#include "tfPotential_cuda.h"


using namespace TissueForge;


#define cuda_call_pots_safe(func)                                       \
    if(func != cudaSuccess) {                                           \
        tf_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));    \
        return;                                                         \
    }

#define cuda_call_pots_safer(func, retval)                              \
    if(func != cudaSuccess) {                                           \
        tf_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));    \
        return retval;                                                  \
    }


Potential cuda::toCUDADevice(const TissueForge::Potential &p) {
    TissueForge::Potential p_d(p);

    // Alloc and copy coefficients
    cuda_call_pots_safer(cudaMalloc(&p_d.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk), p_d)
    cuda_call_pots_safer(cudaMemcpy(p_d.c, p.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk, cudaMemcpyHostToDevice), p_d)

    if(p.pca != NULL) { 
        TissueForge::Potential pca_d = cuda::toCUDADevice(*p.pca);
        cuda_call_pots_safer(cudaMalloc(&p_d.pca, sizeof(TissueForge::Potential)), p_d)
        cuda_call_pots_safer(cudaMemcpy(p_d.pca, &pca_d, sizeof(TissueForge::Potential), cudaMemcpyHostToDevice), p_d)
    }
    else 
        p_d.pca = NULL;
    if(p.pcb != NULL) { 
        TissueForge::Potential pcb_d = cuda::toCUDADevice(*p.pcb);
        cuda_call_pots_safer(cudaMalloc(&p_d.pcb, sizeof(TissueForge::Potential)), p_d)
        cuda_call_pots_safer(cudaMemcpy(p_d.pcb, &pcb_d, sizeof(TissueForge::Potential), cudaMemcpyHostToDevice), p_d)
    } 
    else 
        p_d.pcb = NULL;

    return p_d;
}

__host__ __device__ 
void cuda::cudaFree(TissueForge::Potential *p) {
    if(p == NULL || p->flags & POTENTIAL_NONE) 
        return;
    
    if(p->pca != NULL) {
        cuda::cudaFree(p->pca);
    }
    if(p->pcb != NULL) {
        cuda::cudaFree(p->pcb);
    }

    ::cudaFree(p->c);
    p->c = NULL;
}


__host__ 
cuda::PotentialData::PotentialData(TissueForge::Potential *p) : 
    PotentialData()
{
    if(p == NULL) 
        return;
    
    this->flags = p->flags;
    this->alpha = make_float4(p->alpha[0], p->alpha[1], p->alpha[2], p->alpha[3]);
    this->w = make_float3(p->a, p->b, p->r0_plusone);
    this->offset = make_float3(p->offset[0], p->offset[1], p->offset[2]);
    this->n = p->n;

    cuda_call_pots_safe(cudaMalloc(&this->c, sizeof(float) * (p->n + 1) * potential_chunk))
    cuda_call_pots_safe(cudaMemcpy(this->c, p->c, sizeof(float) * (p->n + 1) * potential_chunk, cudaMemcpyHostToDevice))
}

__host__ 
void cuda::PotentialData::finalize() {
    if(this->flags & POTENTIAL_NONE) 
        return;

    cuda_call_pots_safe(::cudaFree(this->c))
}

__host__ 
cuda::DPDPotentialData::DPDPotentialData(TissueForge::DPDPotential *p) : 
    cuda::DPDPotentialData()
{
    if(p == NULL) 
        return;

    this->flags = p->flags;
    this->w = make_float2(p->a, p->b);
    this->dpd_cfs = make_float3(p->alpha, p->gamma, p->sigma);
}

__host__ 
cuda::Potential::Potential(TissueForge::Potential *p) : 
    cuda::Potential()
{
    if(p == NULL) 
        return;

    std::vector<TissueForge::Potential*> pcs_pot;
    std::vector<TissueForge::DPDPotential*> pcs_dpd;
    if(p->kind == POTENTIAL_KIND_COMBINATION) {
        for(auto pc : p->constituents()) {
            if(pc->kind != POTENTIAL_KIND_COMBINATION) {
                if(pc->kind == POTENTIAL_KIND_POTENTIAL) {
                    pcs_pot.push_back(pc);
                }
                else {
                    pcs_dpd.push_back((TissueForge::DPDPotential*)pc);
                }
            }
        }
    }
    else if(p->kind == POTENTIAL_KIND_POTENTIAL) {
        pcs_pot.push_back(p);
    }
    else {
        pcs_dpd.push_back((TissueForge::DPDPotential*)p);
    }

    this->nr_dpds = pcs_dpd.size();
    this->nr_pots = pcs_pot.size();

    if(this->nr_pots == 0 && this->nr_dpds == 0) 
        return;
    
    cuda::PotentialData *data_h_pots = (cuda::PotentialData*)malloc(this->nr_pots * sizeof(cuda::PotentialData));
    cuda::DPDPotentialData *data_h_dpds = (cuda::DPDPotentialData*)malloc(this->nr_dpds * sizeof(cuda::DPDPotentialData));
    
    for(int i = 0; i < this->nr_pots; i++) {
        data_h_pots[i] = cuda::PotentialData(pcs_pot[i]);
    }
    for(int i = 0; i < this->nr_dpds; i++) {
        data_h_dpds[i] = cuda::DPDPotentialData(pcs_dpd[i]);
    }

    cuda_call_pots_safe(cudaMalloc(&this->data_pots, this->nr_pots * sizeof(cuda::PotentialData)))
    cuda_call_pots_safe(cudaMemcpy(this->data_pots, data_h_pots, this->nr_pots * sizeof(cuda::PotentialData), cudaMemcpyHostToDevice))

    cuda_call_pots_safe(cudaMalloc(&this->data_dpds, this->nr_dpds * sizeof(cuda::DPDPotentialData)))
    cuda_call_pots_safe(cudaMemcpy(this->data_dpds, data_h_dpds, this->nr_dpds * sizeof(cuda::DPDPotentialData), cudaMemcpyHostToDevice))

    free(data_h_pots);
    free(data_h_dpds);
}
