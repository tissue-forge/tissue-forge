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

#include <Magnum/Trade/MeshData.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cylinder.h>

#include "tfThreeDFMeshGenerator.h"


namespace TissueForge::io {


    // ThreeDFMeshGenerator


    ThreeDFMeshGenerator::ThreeDFMeshGenerator() {
        this->mesh = new ThreeDFMeshData();
    }

    ThreeDFMeshData *ThreeDFMeshGenerator::getMesh() {
        return this->mesh;
    }


    // Supporting functions


    HRESULT constructExplicitMesh(
        std::vector<FVector3> positions, 
        std::vector<FVector3> normals, 
        std::vector<std::vector<unsigned int> > indices, 
        std::vector<ThreeDFVertexData*> *vertices, 
        std::vector<ThreeDFEdgeData*> *edges, 
        std::vector<ThreeDFFaceData*> *faces, 
        ThreeDFMeshData *mesh) 
    { 

        unsigned int numVerts = positions.size();
        unsigned int numFaces = indices.size();

        vertices->reserve(numVerts);

        ThreeDFVertexData *vertex;
        ThreeDFEdgeData *edge;
        ThreeDFFaceData *face;

        // Construct vertices

        for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) 
            vertices->push_back(new ThreeDFVertexData(positions[vIdx]));
        
        for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

            // Construct face

            face = new ThreeDFFaceData();

            // Get vertices and normal of this face

            auto fIndices = indices[fIdx];

            std::vector<ThreeDFVertexData*> fverts;
            fverts.reserve(fIndices.size());
            FVector3 fn = {0.f, 0.f, 0.f};
            for(auto fInd : fIndices) {

                fverts.push_back((*vertices)[fInd]);
                fn += normals[fInd];

            }
            face->normal = fn / fn.length();

            // Find or create edges

            std::vector<ThreeDFEdgeData*> fedges(fIndices.size(), 0);

            for(unsigned int i = 0; i < fIndices.size(); i++) {
                auto va = fverts[i];
                auto vb = i == fIndices.size() - 1 ? fverts[0] : fverts[i + 1];

                for(auto e : va->getEdges()) {
                    if(e->va == vb || e->vb == vb) {
                        fedges[i] = e;
                        break;
                    }
                }

                if(fedges[i] == NULL) {
                    edge = new ThreeDFEdgeData(va, vb);
                    fedges[i] = edge;
                    edges->push_back(edge);
                }
            }

            // Add all edges to face without condition

            face->edges.reserve(fedges.size());
            for(auto e : fedges) {

                face->edges.push_back(e);
                e->faces.push_back(face);

            }

            // Connect mesh and face

            mesh->faces.push_back(face);
            face->meshes.push_back(mesh);
            faces->push_back(face);

        }

        return S_OK;
    }


    HRESULT generateBallMesh(
        ThreeDFMeshData *mesh, 
        std::vector<ThreeDFFaceData*> *faces, 
        std::vector<ThreeDFEdgeData*> *edges, 
        std::vector<ThreeDFVertexData*> *vertices, 
        std::vector<FVector3> *normals, 
        const FloatP_t &radius, 
        const FVector3 &offset, 
        const unsigned int &numDivs) 
    { 
        Magnum::Trade::MeshData icoSphere = Magnum::Primitives::icosphereSolid(numDivs);
        auto mg_positions = icoSphere.positions3DAsArray();
        auto mg_normals = icoSphere.normalsAsArray();
        auto mg_indices = icoSphere.indicesAsArray();

        unsigned int numVerts = mg_positions.size();
        unsigned int numFaces = mg_indices.size() / 3;
        
        std::vector<FVector3> positions;
        std::vector<std::vector<unsigned int> > assmIndices;

        vertices->reserve(numVerts);
        normals->reserve(numVerts);
        faces->reserve(numFaces);
        positions.reserve(numVerts);
        assmIndices.reserve(numFaces);

        // Generate transformation
        FMatrix4 transformation = FMatrix4::translation(offset) * FMatrix4::scaling(FVector3(radius));

        for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) {
            
            // Apply transformation

            fVector3 pos = mg_positions[vIdx];
            fVector3 normal = mg_normals[vIdx];
            FVector4 post = {pos.x(), pos.y(), pos.z(), 1.f};
            FVector3 position = (transformation * post).xyz();

            positions.push_back(position);
            normals->push_back(FVector3(normal));
            
        }

        for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

            auto bIndices = &mg_indices[3 * fIdx];
            assmIndices.push_back({bIndices[0], bIndices[1], bIndices[2]});

        }

        constructExplicitMesh(positions, *normals, assmIndices, vertices, edges, faces, mesh);

        return S_OK;
    }

    HRESULT generateCylinderMesh(
        ThreeDFMeshData *mesh, 
        std::vector<ThreeDFFaceData*> *faces, 
        std::vector<ThreeDFEdgeData*> *edges, 
        std::vector<ThreeDFVertexData*> *vertices, 
        std::vector<FVector3> *normals, 
        const FloatP_t &radius, 
        const FVector3 &startPt, 
        const FVector3 &endPt, 
        const unsigned int &numDivs) 
    {
        auto cylVec = endPt - startPt;
        FloatP_t cylLength = cylVec.length();
        FloatP_t cylHalfLength = 0.5 * cylLength / radius;

        Magnum::Trade::MeshData cylinder = Magnum::Primitives::cylinderSolid(1, 3 * (numDivs + 1), cylHalfLength);

        auto mg_positions = cylinder.positions3DAsArray();
        auto mg_normals = cylinder.normalsAsArray();
        auto mg_indices = cylinder.indicesAsArray();

        unsigned int numVerts = mg_positions.size();
        unsigned int numFaces = mg_indices.size() / 3;
        
        std::vector<FVector3> positions;
        std::vector<std::vector<unsigned int> > assmIndices;

        vertices->reserve(numVerts);
        normals->reserve(numVerts);
        faces->reserve(numFaces);
        positions.reserve(numVerts);
        assmIndices.reserve(numFaces);

        // Generate transformation
        FVector3 cylVec0 = {0.f, 1.f, 0.f};
        FVector3 rotVec = Magnum::Math::cross(cylVec0, cylVec);
        rotVec = rotVec / rotVec.length();
        FloatP_t rotAng = std::acos(cylVec.y() / cylLength);
        FMatrix4 tRotate = FMatrix4::rotation(rotAng, rotVec);
        FMatrix4 transformation = FMatrix4::scaling(FVector3(radius));
        transformation = FMatrix4::translation({0.f, 0.5f * cylLength, 0.f}) * transformation;
        transformation = tRotate * transformation;
        transformation = FMatrix4::translation(startPt) * transformation;

        for(unsigned int vIdx = 0; vIdx < numVerts; vIdx++) {
            
            // Apply transformation

            fVector3 pos = mg_positions[vIdx];
            FVector4 qt = {pos.x(), pos.y(), pos.z(), 1.f};
            FVector3 post = (transformation * qt).xyz();

            fVector3 norm = mg_normals[vIdx];
            qt = {norm.x(), norm.y(), norm.z(), 1.f};
            FVector3 normt = (tRotate * qt).xyz();

            positions.push_back(post);
            normals->push_back(normt);
            
        }

        for(unsigned int fIdx = 0; fIdx < numFaces; fIdx++) {

            auto bIndices = &mg_indices[3 * fIdx];
            assmIndices.push_back({bIndices[0], bIndices[1], bIndices[2]});

        }

        constructExplicitMesh(positions, *normals, assmIndices, vertices, edges, faces, mesh);

        return S_OK;
    }

};
