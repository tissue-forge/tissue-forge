/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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
 * @file tf_mesh_io.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_IO_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_IO_H_

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
#include "tfMesh.h"
#include "tfMeshQuality.h"
#include "actors/tf_actors.h"

#include <io/tf_io.h>


namespace TissueForge::io {


    template <>
    HRESULT toFile(TissueForge::models::vertex::Vertex *dataElement, const MetaData &metaData, IOElement &fileElement);

    /** Does not assemble mesh child connectivity */
    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Vertex **dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::VertexHandle &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::VertexHandle *dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::Surface *dataElement, const MetaData &metaData, IOElement &fileElement);

    /** Does not assemble mesh child connectivity */
    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Surface **dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::SurfaceHandle &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceHandle *dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::SurfaceType &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceType **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::Body *dataElement, const MetaData &metaData, IOElement &fileElement);

    /** Does not assemble mesh child connectivity */
    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Body **dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::BodyHandle &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::BodyHandle *dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::BodyType &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::BodyType **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::Mesh *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Mesh *dataElement);

    template <>
    HRESULT toFile(const TissueForge::models::vertex::MeshQuality &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::MeshQuality *dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::MeshObjActor *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::MeshObjActor **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::Adhesion *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Adhesion **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::BodyForce *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::BodyForce **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::ConvexPolygonConstraint *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::ConvexPolygonConstraint **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::EdgeTension *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::EdgeTension **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::FlatSurfaceConstraint *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::FlatSurfaceConstraint **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::NormalStress *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::NormalStress **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::PerimeterConstraint *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::PerimeterConstraint **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::SurfaceAreaConstraint *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceAreaConstraint **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::SurfaceTraction *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceTraction **dataElement);

    template <>
    HRESULT toFile(TissueForge::models::vertex::VolumeConstraint *dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::VolumeConstraint **dataElement);


}

#endif // _MODELS_VERTEX_SOLVER_TF_MESH_IO_H_