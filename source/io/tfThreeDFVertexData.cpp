/*******************************************************************************
 * This file is part of Tissue Forge.
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

#include <tf_util.h>

#include "tfThreeDFVertexData.h"
#include "tfThreeDFEdgeData.h"
#include "tfThreeDFFaceData.h"
#include "tfThreeDFMeshData.h"


namespace TissueForge::io {


    ThreeDFVertexData::ThreeDFVertexData(const FVector3 &_position, ThreeDFStructure *_structure) : 
        structure{_structure}, 
        position{_position}
    {}

    std::vector<ThreeDFEdgeData*> ThreeDFVertexData::getEdges() {
        return edges;
    }

    std::vector<ThreeDFFaceData*> ThreeDFVertexData::getFaces() {
        std::vector<ThreeDFFaceData*> result;
        for(auto e : this->edges) {
            auto v = e->getFaces();
            result.insert(result.end(), v.begin(), v.end());
        }
        return util::unique(result);
    }

    std::vector<ThreeDFMeshData*> ThreeDFVertexData::getMeshes() {
        std::vector<ThreeDFMeshData*> result;
        for(auto e : this->edges) {
            auto v = e->getMeshes();
            result.insert(result.end(), v.begin(), v.end());
        }
        return util::unique(result);
    }

    unsigned int ThreeDFVertexData::getNumEdges() {
        return this->edges.size();
    }

    unsigned int ThreeDFVertexData::getNumFaces() {
        return this->getFaces().size();
    }

    unsigned int ThreeDFVertexData::getNumMeshes() {
        return this->getMeshes().size();
    }

    bool ThreeDFVertexData::in(ThreeDFEdgeData *e) {
        auto itr = std::find(this->edges.begin(), this->edges.end(), e);
        return itr != std::end(this->edges);
    }

    bool ThreeDFVertexData::in(ThreeDFFaceData *f) {
        for(auto e : this->edges) 
            if(e->in(f)) 
                return true;
        return false;
    }

    bool ThreeDFVertexData::in(ThreeDFMeshData *m) {
        for(auto e : this->edges) 
            if(e->in(m)) 
                return true;
        return false;
    }

    bool ThreeDFVertexData::in(ThreeDFStructure *s) {
        return s == this->structure;
    }

};
