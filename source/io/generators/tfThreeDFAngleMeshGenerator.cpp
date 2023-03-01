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
#include <rendering/tfStyle.h>
#include <tf_util.h>
#include <io/tfThreeDFRenderData.h>

#include "tfThreeDFAngleMeshGenerator.h"


static std::string _threeDFAngleDefColor = "aqua";


namespace TissueForge::io {


    // ThreeDFAngleMeshGenerator


    HRESULT ThreeDFAngleMeshGenerator::process() {

        this->mesh->name = "Angles";

        // Generate render data

        this->mesh->renderData = new ThreeDFRenderData();
        this->mesh->renderData->color = util::Color3_Parse(_threeDFAngleDefColor);

        for(auto ah : this->angles) {

            FVector3 posi = ah[0]->getPosition();
            FVector3 posj = ah[1]->getPosition();
            FVector3 posk = ah[2]->getPosition();

            FVector3 mij = 0.5 * (posi + posj);
            FVector3 mkj = 0.5 * (posk + posj);

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
                                posj, 
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
                                0.5f * this->radius, 
                                mij, 
                                mkj, 
                                this->pRefinements);

        }

        return S_OK;
    }

};
