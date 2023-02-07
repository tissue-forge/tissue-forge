/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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
 * @file tf_mesh_metrics.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESH_METRICS_H_
#define _MODELS_VERTEX_SOLVER_TFMESH_METRICS_H_


#include "tfVertex.h"
#include "tfSurface.h"

#include <tuple>


namespace TissueForge::models::vertex {


/**
 * @brief Calculate the strain in a edge defined by two vertices
 * 
 * @param v1 first vertex
 * @param v2 second vertex
 */
FMatrix3 edgeStrain(const VertexHandle &v1, const VertexHandle &v2);

/**
 * @brief Calculate the strain in a vertex. 
 * 
 * Uses a weighted average of edge strains, with a preference for closer measurements. 
 * 
 * @param v vertex
 */
FMatrix3 vertexStrain(const VertexHandle &v);


};


#endif // _MODELS_VERTEX_SOLVER_TFMESH_METRICS_H_