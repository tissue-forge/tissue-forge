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

#include <tfParticle.h>
#include <tf_util.h>
#include <rendering/tfStyle.h>

#include <io/tfThreeDFRenderData.h>

#include "tfThreeDFDihedralMeshGenerator.h"


static std::string _threeDFDihedralDefColor = "gold";


namespace TissueForge::io {


    // ThreeDFDihedralMeshGenerator


    HRESULT ThreeDFDihedralMeshGenerator::process() {

        this->mesh->name = "Dihedrals";

        // Generate render data

        this->mesh->renderData = new ThreeDFRenderData();
        this->mesh->renderData->color = util::Color3_Parse(_threeDFDihedralDefColor);

        for(auto dh : this->dihedrals) {

            FVector3 posi = dh[0]->getPosition();
            FVector3 posj = dh[1]->getPosition();
            FVector3 posk = dh[2]->getPosition();
            FVector3 posl = dh[3]->getPosition();

            FVector3 mik = 0.5 * (posi + posk);
            FVector3 mjl = 0.5 * (posj + posl);

            std::vector<ThreeDFFaceData*> faces;
            std::vector<ThreeDFEdgeData*> edges;
            std::vector<ThreeDFVertexData*> vertices;
            std::vector<FVector3> normals;

            generateCylinderMesh(this->mesh, 
                                &faces, 
                                &edges, 
                                &vertices, 
                                &normals, 
                                this->radius, 
                                posi, 
                                posk, 
                                this->pRefinements);

            faces.clear(); edges.clear(); vertices.clear(); normals.clear();
            generateCylinderMesh(this->mesh, 
                                &faces, 
                                &edges, 
                                &vertices, 
                                &normals, 
                                this->radius, 
                                posk, 
                                posj, 
                                this->pRefinements);

            faces.clear(); edges.clear(); vertices.clear(); normals.clear();
            generateCylinderMesh(this->mesh, 
                                &faces, 
                                &edges, 
                                &vertices, 
                                &normals, 
                                this->radius, 
                                posj, 
                                posl, 
                                this->pRefinements);

            faces.clear(); edges.clear(); vertices.clear(); normals.clear();
            generateCylinderMesh(this->mesh, 
                                &faces, 
                                &edges, 
                                &vertices, 
                                &normals, 
                                0.5f * this->radius, 
                                mik, 
                                mjl, 
                                this->pRefinements);

        }

        return S_OK;
    }

};
