/*******************************************************************************
 * This file is part of Tissue Forge.
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

#ifndef _WRAPS_C_TFCFLUX_H_
#define _WRAPS_C_TFCFLUX_H_

#include "tf_port_c.h"

#include "tfCParticle.h"

// Handles

struct CAPI_EXPORT tfFluxKindHandle {
    unsigned int FLUX_FICK;
    unsigned int FLUX_SECRETE;
    unsigned int FLUX_UPTAKE;
};

/**
 * @brief Handle to a @ref Flux instance
 * 
 */
struct CAPI_EXPORT tfFluxHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref Fluxes instance
 * 
 */
struct CAPI_EXPORT tfFluxesHandle {
    void *tfObj;
};


//////////////
// FluxKind //
//////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfFluxKindHandle_init(struct tfFluxKindHandle *handle);


//////////
// Flux //
//////////


/**
 * @brief Get the size of the fluxes
 * 
 * @param handle populated handle
 * @param size flux size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getSize(struct tfFluxHandle *handle, unsigned int *size);

/**
 * @brief Get the kind of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param kind flux kind
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getKind(struct tfFluxHandle *handle, unsigned int index, unsigned int *kind);

/**
 * @brief Get the type ids of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param typeid_a id of first type
 * @param typeid_b id of second type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getTypeIds(struct tfFluxHandle *handle, unsigned int index, unsigned int *typeid_a, unsigned int *typeid_b);

/**
 * @brief Get the coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param coef flux coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *coef);

/**
 * @brief Set the coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param coef flux coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_setCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t coef);

/**
 * @brief Get the decay coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param decay_coef flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getDecayCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *decay_coef);

/**
 * @brief Set the decay coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param decay_coef flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_setDecayCoef(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t decay_coef);

/**
 * @brief Get the target of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param target flux target
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_getTarget(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t *target);

/**
 * @brief Set the target of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param target flux target
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFlux_setTarget(struct tfFluxHandle *handle, unsigned int index, tfFloatP_t target);


////////////
// Fluxes //
////////////


/**
 * @brief Get the number of individual flux objects
 * 
 * @param handle populated handle
 * @param size number of individual flux objects
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFluxes_getSize(struct tfFluxesHandle *handle, int *size);

/**
 * @brief Get a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param flux flux
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFluxes_getFlux(struct tfFluxesHandle *handle, unsigned int index, struct tfFluxHandle *flux);


//////////////////////
// Module functions //
//////////////////////


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
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFluxes_fluxFick(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t decay
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
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param target flux target
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFluxes_secrete(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t target, 
    tfFloatP_t decay
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
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param target flux target
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfFluxes_uptake(
    struct tfFluxesHandle *handle, 
    struct tfParticleTypeHandle *A, 
    struct tfParticleTypeHandle *B, 
    const char *name, 
    tfFloatP_t k, 
    tfFloatP_t target, 
    tfFloatP_t decay
);

#endif // _WRAPS_C_TFCFLUX_H_