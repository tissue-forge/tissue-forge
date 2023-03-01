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

#include <tf_util.h>

#include "tfThreeDFVertexData.h"
#include "tfThreeDFEdgeData.h"
#include "tfThreeDFFaceData.h"
#include "tfThreeDFMeshData.h"


namespace TissueForge::io {


    ThreeDFEdgeData::ThreeDFEdgeData(ThreeDFVertexData *_va, ThreeDFVertexData *_vb) : 
        va{_va}, 
        vb{_vb}
    {
        if(this->va == NULL || this->vb == NULL) 
            tf_error(E_FAIL, "Invalid vertex (NULL)");

        this->va->edges.push_back(this);
        this->vb->edges.push_back(this);
    }

    std::vector<ThreeDFVertexData*> ThreeDFEdgeData::getVertices() {
        return {this->va, this->vb};
    }

    std::vector<ThreeDFFaceData*> ThreeDFEdgeData::getFaces() {
        return this->faces;
    }

    std::vector<ThreeDFMeshData*> ThreeDFEdgeData::getMeshes() {
        std::vector<ThreeDFMeshData*> result;
        for(auto e : this->faces) {
            auto v = e->getMeshes();
            result.insert(result.end(), v.begin(), v.end());
        }
        return util::unique(result);
    }

    unsigned int ThreeDFEdgeData::getNumVertices() {
        return 2;
    }

    unsigned int ThreeDFEdgeData::getNumFaces() {
        return this->faces.size();
    }

    unsigned int ThreeDFEdgeData::getNumMeshes() {
        return this->getMeshes().size();
    }

    bool ThreeDFEdgeData::has(ThreeDFVertexData *v) {
        return v == this->va || v == this->vb;
    }

    bool ThreeDFEdgeData::in(ThreeDFFaceData *f) {
        auto itr = std::find(this->faces.begin(), this->faces.end(), f);
        return itr != std::end(this->faces);
    }

    bool ThreeDFEdgeData::in(ThreeDFMeshData *m) {
        for(auto f : this->faces) 
            if(f->in(m)) 
                return true;
        return false;
    }

    bool ThreeDFEdgeData::in(ThreeDFStructure *s) {
        return s == this->structure;
    }

};
