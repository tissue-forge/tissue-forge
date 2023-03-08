/*******************************************************************************
 * This file is part of Tissue Forge.
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

/**
 * @file TissueForge.h
 * 
 */

#ifndef _INCLUDE_TISSUEFORGE_H_
#define _INCLUDE_TISSUEFORGE_H_

/** 
 * @namespace TissueForge 
 * @brief The root Tissue Forge namespace 
 */
namespace TissueForge {

    /**
     * @namespace TissueForge::cuda
     * @brief Tissue Forge GPU acceleration on CUDA-supporting devices
     */
    namespace cuda {};

    /**
     * @namespace TissueForge::event
     * @brief Tissue Forge event system
     */
    namespace event {};

    /**
     * @namespace TissueForge::io
     * @brief Tissue Forge I/O
     */
    namespace io {};

    /**
     * @namespace TissueForge::metrics
     * @brief Tissue Forge simulation metrics
     */
    namespace metrics {};

    /**
     * @namespace TissueForge::models
     * @brief Application-specific Tissue Forge models and methods
     */
    namespace models {};

    /**
     * @namespace TissueForge::py
     * @brief Tissue Forge Python language support
     */
    namespace py {};

    /**
     * @namespace TissueForge::rendering
     * @brief Tissue Forge rendering and visualization
     */
    namespace rendering {};

    /**
     * @namespace TissueForge::shaders
     * @brief Tissue Forge shaders
     */
    namespace shaders {};

    /**
     * @namespace TissueForge::state
     * @brief Tissue Forge state dynamics modeling features
     */
    namespace state {};

    /**
     * @namespace TissueForge::types
     * @brief Native Tissue Forge type definitions
     */
    namespace types {};

    /**
     * @namespace TissueForge::util
     * @brief Tissue Forge utilities
     */
    namespace util {};
};

#include <tf_port.h>
#include <tf_style.h>
#include <tf_runtime.h>


#endif // _INCLUDE_TISSUEFORGE_H_