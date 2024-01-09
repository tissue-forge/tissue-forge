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

/**
 * @file tfFlux.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFFLUX_H_
#define _MDCORE_INCLUDE_TFFLUX_H_

#include "tf_platform.h"
#include <mdcore_config.h>
#include <io/tf_io.h>
#include "tfSpace_cell.h"

#include <string>


namespace TissueForge { 


    enum FluxKind {
        FLUX_FICK = 0,
        FLUX_SECRETE = 1,
        FLUX_UPTAKE = 2
    };


    // keep track of the ids of the particle types, to determine
    // the reaction direction.
    struct TypeIdPair {
        int16_t a;
        int16_t b;
    };

    struct CAPI_EXPORT Flux {
        int32_t       size; // temporary size until we get SIMD instructions.
        int8_t        kinds[TF_SIMD_SIZE];
        TypeIdPair    type_ids[TF_SIMD_SIZE];
        int32_t       indices_a[TF_SIMD_SIZE];
        int32_t       indices_b[TF_SIMD_SIZE];
        FPTYPE        coef[TF_SIMD_SIZE];
        FPTYPE        decay_coef[TF_SIMD_SIZE];
        FPTYPE        target[TF_SIMD_SIZE];
        FPTYPE        cutoff[TF_SIMD_SIZE];
    };

    struct ParticleType;

    /**
     * @brief A flux is defined between a pair of types, and acts on the
     * state vector between a pair of instances.
     *
     * The indices of the species in each state vector
     * are most likely different, so Tissue Forge tracks the
     * indices in each type, and the transport constatants.
     *
     * A flux between a pair of types, and pair of respective
     * species needs:
     *
     * (1) type A, (2) type B, (3) species id in A, (4) species id in B,
     * (5) transport constant.
     *
     * Allocates Flux as a single block, member pointers point to
     * offsets in these blocks.
     *
     * Allocated size is:
     * sizeof(Fluxes) + 2 * alloc_size * sizeof(int32) + alloc_size * sizeof(FPTYPE)
     */
    struct CAPI_EXPORT Fluxes
    {
        int32_t size;          // how many individual flux objects this has
        int32_t fluxes_size;   // how many fluxes (blocks) this has.
        // static int32_t init;
        Flux fluxes[];       // allocated in single block, this

        static Fluxes* newFluxes(int32_t init_size);
        static Fluxes *create(
            FluxKind kind, 
            ParticleType *a, 
            ParticleType *b,
            const std::string& name, 
            FPTYPE k, 
            FPTYPE decay, 
            FPTYPE target,
            FPTYPE cutoff=-FPTYPE_ONE
        );
        static Fluxes *addFlux(
            FluxKind kind, 
            Fluxes *fluxes,
            int16_t typeId_a, 
            int16_t typeId_b,
            int32_t index_a, 
            int32_t index_b,
            FPTYPE k, 
            FPTYPE decay, 
            FPTYPE target,
            FPTYPE cutoff
        );

        /**
         * @brief Creates and binds a Fickian diffusion flux. 
         * 
         * Fickian diffusion flux implements the analogous reaction: 
         * 
         * @f[
         *      a.S \leftrightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}} \right)\left(a.S - b.S\right) , 
         * @f]
         * 
         * @f[
         *      a.S \rightarrow 0   ; \frac{d}{2} a.S , 
         * @f]
         * 
         * @f[
         *      b.S \rightarrow 0   ; \frac{d}{2} b.S , 
         * @f]
         * 
         * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
         * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
         * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
         * distance, and @f$ d @f$ is an optional decay term. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param A first type
         * @param B second type
         * @param name name of species
         * @param k transport coefficient
         * @param decay optional decay. Defaults to 0.0. 
         * @param cutoff optional cutoff distance. Defaults to global cutoff
         * @return Fluxes* 
         */
        static Fluxes *fluxFick(
            ParticleType *A, 
            ParticleType *B, 
            const std::string &name, 
            const FPTYPE &k, 
            const FPTYPE &decay=FPTYPE_ZERO, 
            const FPTYPE &cutoff=-FPTYPE_ONE
        );

        /**
         * @brief Alias of fluxFick. 
         * 
         * @param A first type
         * @param B second type
         * @param name name of species
         * @param k transport coefficient
         * @param decay optional decay. Defaults to 0.0. 
         * @param cutoff optional cutoff distance. Defaults to global cutoff
         * @return Fluxes* 
         */
        static Fluxes *flux(
            ParticleType *A, 
            ParticleType *B, 
            const std::string &name, 
            const FPTYPE &k, 
            const FPTYPE &decay=FPTYPE_ZERO, 
            const FPTYPE &cutoff=-FPTYPE_ONE
        );

        /**
         * @brief Creates a secretion flux by active pumping. 
         * 
         * Secretion flux implements the analogous reaction: 
         * 
         * @f[
         *      a.S \rightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}} \right)\left(a.S - a.S_{target} \right) ,
         * @f]
         * 
         * @f[
         *      a.S \rightarrow 0   ; \frac{d}{2} a.S ,
         * @f]
         * 
         * @f[
         *      b.S \rightarrow 0   ; \frac{d}{2} b.S ,
         * @f]
         * 
         * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
         * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
         * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
         * distance, and @f$ d @f$ is an optional decay term. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param A first type
         * @param B second type
         * @param name name of species
         * @param k transport coefficient
         * @param target target concentration
         * @param decay optional decay. Defaults to 0.0 
         * @param cutoff optional cutoff distance. Defaults to global cutoff
         * @return Fluxes* 
         */
        static Fluxes *secrete(
            ParticleType *A, 
            ParticleType *B, 
            const std::string &name, 
            const FPTYPE &k, 
            const FPTYPE &target, 
            const FPTYPE &decay=FPTYPE_ZERO, 
            const FPTYPE &cutoff=-FPTYPE_ONE
        );

        /**
         * @brief Creates an uptake flux by active pumping. 
         * 
         * Uptake flux implements the analogous reaction: 
         * 
         * @f[
         *      a.S \rightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}}\right)\left(b.S - b.S_{target} \right)\left(a.S\right) ,
         * @f]
         * 
         * @f[
         *      a.S \rightarrow 0   ; \frac{d}{2} a.S ,
         * @f]
         * 
         * @f[
         *      b.S \rightarrow 0   ; \frac{d}{2} b.S ,
         * @f]
         * 
         * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
         * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
         * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
         * distance, and @f$ d @f$ is an optional decay term. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param A first type
         * @param B second type
         * @param name name of species
         * @param k transport coefficient
         * @param target target concentration
         * @param decay optional decay. Defaults to 0.0 
         * @param cutoff optional cutoff distance. Defaults to global cutoff
         * @return Fluxes* 
         */
        static Fluxes *uptake(
            ParticleType *A, 
            ParticleType *B, 
            const std::string &name, 
            const FPTYPE &k, 
            const FPTYPE &target, 
            const FPTYPE &decay=FPTYPE_ZERO, 
            const FPTYPE &cutoff=-FPTYPE_ONE
        );

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation
         * 
         * @param str 
         * @return Fluxes* 
         */
        static Fluxes *fromString(const std::string &str);
    };

    /**
     * integrate all of the fluxes for a space cell.
     */
    HRESULT Fluxes_integrate(space_cell *cell, FPTYPE dt=-1.0);

    /**
     * integrate all of the fluxes for a space cell.
     */
    HRESULT Fluxes_integrate(int cellId);


};

#endif // _MDCORE_INCLUDE_TFFLUX_H_